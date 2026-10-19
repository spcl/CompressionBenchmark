"""One unique timestep per rank; total bytes divided by synchronized elapsed time."""

import json
import multiprocessing as mp
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import xarray as xr

from compbench.compression import compressors
from compbench.compression.normalization import relative_roundtrip
from compbench.datasets import era5 as data
from compbench.results import save

BARRIER = None


def validate_trials(cfg, output, tier):
    expected = cfg["weak_scaling"]["sample_ranks" if tier == "sample" else "ranks"]
    available = sorted(int(p.stem.split("_")[1]) for p in output.glob("trials_*.json"))
    wanted = Counter(
        (m, t, r)
        for m in cfg["compressors"]
        for t in cfg["range_relative_error_targets"]
        for r in range(cfg["weak_scaling"]["repeats"])
    )
    for count in available:
        rows = json.loads((output / f"trials_{count}.json").read_text())
        actual = Counter((r["method"], r["target"], r["repeat"]) for r in rows)
        if actual != wanted or any(r["ranks"] != count or r["seconds"] <= 0 for r in rows):
            raise ValueError(f"Invalid scaling cohort: {count} ranks")
    return dict(complete=available == expected, ranks=available, expected_ranks=expected)


def initialize(barrier):
    global BARRIER
    BARRIER = barrier


def encode_timestep(path, method, target, cfg):
    total = 0
    with xr.open_dataset(path) as ds:
        # A variable slab is read once, then processed level by level.
        for variable in cfg["datasets"]["variables"]:
            values = np.asarray(ds[variable].values, dtype=np.float32)
            for field in values.reshape(-1, *values.shape[-2:]):
                relative_roundtrip(field, method, target, cfg, decode=False)
                total += field.nbytes
    return total


def local_worker(job):
    cfg, path, method, target = job
    warm = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    compressors.roundtrip(warm, method, 1.0, cfg, decode=False)
    BARRIER.wait(timeout=120)
    start = perf_counter()
    size = encode_timestep(path, method, target, cfg)
    return size, perf_counter() - start


def input_paths(cfg, opts, tier):
    paths = [
        data.folder(opts["data_root"], "weak_scaling", tier) / data.filename(t)
        for t in data.times(cfg, "weak_scaling", tier)
    ]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"{path}; fetch the scaling dataset first")
    return paths


def run_scaling(cfg, opts, tier, output):
    if tier == "full":
        raise ValueError("Use submit for the full weak-scaling cohort")
    paths = input_paths(cfg, opts, tier)
    for count in cfg["weak_scaling"]["sample_ranks"]:
        path = output / f"trials_{count}.json"
        if path.exists():
            continue
        rows = []
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            count,
            mp_context=context,
            initializer=initialize,
            initargs=(context.Barrier(count),),
        ) as pool:
            for method in cfg["compressors"]:
                for target in cfg["range_relative_error_targets"]:
                    for repeat in range(cfg["weak_scaling"]["repeats"]):
                        results = list(
                            pool.map(
                                local_worker, [(cfg, p, method, target) for p in paths[:count]]
                            )
                        )
                        size = sum(x[0] for x in results)
                        seconds = max(x[1] for x in results)
                        rows.append(
                            dict(
                                method=method,
                                target=target,
                                repeat=repeat,
                                ranks=count,
                                nbytes=size,
                                seconds=seconds,
                                throughput=size / seconds,
                            )
                        )
                        print(
                            f"Scaling {method} ranks={count} target={target} repeat={repeat}",
                            flush=True,
                        )
        save(path, rows)
    save(output / "validation.json", validate_trials(cfg, output, tier))


def aggregate_trial(records, method, target, repeat, count, barrier):
    from compbench.execution.synchronization import MAX_START_LAG

    if len(records) != count or {r["rank"] for r in records} != set(range(count)):
        raise ValueError("Missing or duplicate scaling worker records")
    for record in records:
        if (record["method"], record["target"], record["repeat"], record["ranks"]) != (
            method,
            target,
            repeat,
            count,
        ):
            raise ValueError("Mismatched scaling trial records")
        sync = record["synchronization"]
        if sync["barrier_id"] != str(barrier) or not 0 <= sync["start_lag_s"] <= MAX_START_LAG:
            raise ValueError("Mismatched or late scaling start")
        if not all(np.isfinite(record[k]) for k in ("start", "seconds", "nbytes")):
            raise ValueError("Non-finite scaling measurement")
        if record["seconds"] <= 0 or record["nbytes"] <= 0:
            raise ValueError("Invalid scaling measurement")
    if len({r["synchronization"]["scheduled_start"] for r in records}) != 1:
        raise ValueError("Mismatched scheduled starts")
    starts = [r["start"] for r in records]
    spread = max(starts) - min(starts)
    if spread > MAX_START_LAG:
        raise ValueError("Excessive scaling start skew")
    seconds = max(r["start"] + r["seconds"] for r in records) - min(starts)
    size = sum(r["nbytes"] for r in records)
    return dict(
        method=method,
        target=target,
        repeat=repeat,
        ranks=count,
        nbytes=size,
        seconds=seconds,
        throughput=size / seconds,
        start_spread_s=spread,
        synchronization="shared_filesystem_v1",
    )


def run_distributed(cfg, opts, tier, output, rank, count):
    from compbench.execution.synchronization import synchronized_start, wait_for_files

    ranks = cfg["weak_scaling"]["sample_ranks" if tier == "sample" else "ranks"]
    if count not in ranks or not 0 <= rank < count:
        raise ValueError("Slurm workers do not match the configured scaling grid")
    attempt = Path(os.environ["COMPBENCH_SCALING_ATTEMPT"])
    if not attempt.is_absolute() or not attempt.is_dir():
        raise ValueError("A fresh shared COMPBENCH_SCALING_ATTEMPT directory is required")
    path = output / f"trials_{count}.json"
    decision = attempt / "cohort.json"
    if rank == 0:
        # Only the coordinator decides whether every worker skips this cohort.
        complete = path.exists()
        if complete:
            validate_trials(cfg, output, tier)
        save(decision, dict(complete=complete))
    wait_for_files([decision])
    if json.loads(decision.read_text())["complete"]:
        return
    paths = input_paths(cfg, opts, tier)
    if len(paths) < count:
        raise ValueError("Not enough input timesteps for the scaling workers")
    rows = []
    trial = 0
    for method in cfg["compressors"]:
        for target in cfg["range_relative_error_targets"]:
            for repeat in range(cfg["weak_scaling"]["repeats"]):
                folder = attempt / f"trial_{trial}"
                barrier = folder / "start"
                warm = np.arange(4096, dtype=np.float32).reshape(64, 64)
                compressors.roundtrip(warm, method, 1.0, cfg, decode=False)
                actual, sync = synchronized_start(barrier, rank, count)
                start = perf_counter()
                size = encode_timestep(paths[rank], method, target, cfg)
                seconds = perf_counter() - start
                save(
                    folder / f"rank_{rank}.json",
                    dict(
                        rank=rank,
                        ranks=count,
                        method=method,
                        target=target,
                        repeat=repeat,
                        start=actual,
                        seconds=seconds,
                        nbytes=size,
                        synchronization=sync,
                    ),
                )
                if rank == 0:
                    files = [folder / f"rank_{r}.json" for r in range(count)]
                    wait_for_files(files)
                    rows.append(
                        aggregate_trial(
                            [json.loads(p.read_text()) for p in files],
                            method,
                            target,
                            repeat,
                            count,
                            barrier,
                        )
                    )
                    print(
                        f"Scaling {method} ranks={count} target={target} repeat={repeat}",
                        flush=True,
                    )
                trial += 1
    if rank == 0:
        save(path, rows)
        save(output / "validation.json", validate_trials(cfg, output, tier))
