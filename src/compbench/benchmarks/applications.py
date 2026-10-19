"""Resume ensemble application cases and summarize their downstream errors."""

import json
from pathlib import Path

import numpy as np
import xarray as xr

from compbench.benchmarks.energy_budget_closure import energy_member
from compbench.benchmarks.trajectory import trajectory_member
from compbench.compression.metrics import accepts_bounds
from compbench.datasets import ensemble
from compbench.results import save


def application_cases(cfg, workflow, tier):
    members = cfg["datasets"][workflow][tier]["members"]
    return [(f"member_{m}", m, None, None) for m in members] + [
        (f"{compressor}_{target:g}", 0, compressor, target)
        for compressor in cfg["compressors"]
        for target in cfg["range_relative_error_targets"]
    ]


def run_application(cfg, opts, workflow, tier, output, index=None):
    cases = application_cases(cfg, workflow, tier)
    if index is not None:
        cases = [cases[index]]
    ensemble.verify(cfg, opts, workflow, tier, member=cases[0][1] if index is not None else None)
    spec = cfg["datasets"][workflow]
    auxiliary = (
        ensemble.asset_path("weights.nc")
        if workflow == "energy_budget_closure"
        else Path(opts["data_root"]) / spec["initial_particles_file"]
    )
    if not auxiliary.exists():
        raise FileNotFoundError(f"Required {workflow} input: {auxiliary}")
    paths = sorted((Path(opts["data_root"]) / workflow / tier).glob("member_*/*.nc"))
    inputs = {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in [auxiliary, *paths]}
    receipt = output / "inputs.json"
    if receipt.exists() and json.loads(receipt.read_text()) != inputs:
        raise ValueError("Downstream inputs changed; choose a fresh --output-root")
    save(receipt, inputs)
    for name, member, method, target in cases:
        folder = output / name
        folder.mkdir(exist_ok=True)
        record_path = folder / "record.json"
        if not record_path.exists():
            if workflow == "energy_budget_closure":
                result, measurements = energy_member(cfg, opts, tier, member, method, target)
            else:
                result, measurements = trajectory_member(
                    cfg, opts, tier, member, folder, method, target
                )
            result.to_netcdf(folder / "analysis.nc")
            save(
                record_path, dict(method=method, member=member, target=target, chunks=measurements)
            )
        print(f"Measured {workflow} {name}", flush=True)


def summarize_application(cfg, workflow, tier, output):
    spec = cfg["datasets"][workflow]
    members = spec[tier]["members"]
    summary, comparisons = [], []
    reference = None
    for name, member, method, target in application_cases(cfg, workflow, tier):
        folder = output / name
        record_path = folder / "record.json"
        with xr.open_dataset(folder / "analysis.nc") as ds:
            result = ds.load()
        record = json.loads(record_path.read_text())
        chunks = record["chunks"]
        if member == 0 and method is None:
            reference = result
            continue
        key = "residual" if workflow == "energy_budget_closure" else "density"
        if result[key].shape != reference[key].shape:
            raise ValueError("Analysis coverage differs from member 0")
        rmse = float(np.sqrt(np.mean((result[key].values - reference[key].values) ** 2)))
        if not np.isfinite(rmse):
            raise ValueError("Nonfinite downstream RMSE")
        row = dict(
            name=name,
            method=method,
            member=member,
            target=target,
            rmse=rmse,
            cr=None
            if not chunks
            else float(sum(r["nbytes"] for r in chunks) / sum(r["size"] for r in chunks)),
            verified=all(r["verified"] for r in chunks),
            accepted=all(accepts_bounds(r) for r in chunks),
        )
        summary.append(row)
        if method is None:
            comparisons.append(rmse)
        print(f"Measured {workflow} {name}: RMSE={rmse:.6g}", flush=True)
    save(
        output / "measurements.json",
        dict(
            records=summary,
            uncertainty=min(comparisons),
            uncertainty_members=members[1:],
            sample_reference=tier == "sample",
            storage_interval_days=(60 if tier == "full" else 2 * spec[tier].get("window_days", 2))
            if workflow == "energy_budget_closure"
            else None,
        ),
    )
    save(
        output / "validation.json",
        dict(
            complete=True,
            bounds_accepted=all(r["accepted"] for r in summary),
            outside_tolerance=sum(not r["accepted"] for r in summary),
            tolerated_bound_violations=sum(not r["verified"] and r["accepted"] for r in summary),
            measurements=len(summary),
            tier=tier,
            bound_violations=sum(not r["verified"] for r in summary),
        ),
    )
