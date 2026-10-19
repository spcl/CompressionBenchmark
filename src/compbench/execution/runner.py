"""Dispatch benchmark runs locally and across worker processes."""

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from compbench.benchmarks.fields import measure_field
from compbench.datasets import era5 as data
from compbench.execution.validation import validate
from compbench.results import prepare_run


def run(cfg, opts, workflow, tier, selected=None, index=None):
    import os

    if workflow == "weak_scaling" and "SLURM_JOB_ID" in os.environ:
        from compbench.benchmarks.weak_scaling import run_distributed
        from compbench.execution.alps import worker

        output = Path(opts["output_root"]) / tier / workflow
        rank, count = worker()
        if rank == 0:
            prepare_run(cfg, opts, workflow, tier, selected)
        run_distributed(cfg, opts, tier, output, rank, count)
        return output
    output = prepare_run(cfg, opts, workflow, tier, selected)
    if workflow == "weak_scaling":
        from compbench.benchmarks.weak_scaling import run_scaling

        run_scaling(cfg, opts, tier, output)
    elif workflow == "residual_layer_ablation":
        from compbench.benchmarks.residual_layer_ablation import spike

        spike(cfg, output)
    elif workflow == "case_study_hurricane":
        from compbench.benchmarks.hurricane import hurricane

        hurricane(cfg, opts, tier, output)
    elif workflow in ("energy_budget_closure", "trajectory_simulation"):
        from compbench.benchmarks.applications import run_application

        run_application(cfg, opts, workflow, tier, output, index)
        if index is not None:
            return output
    else:
        hourly_task = workflow == "rate_distortion" and tier == "full" and index is not None
        timestamps = [data.times(cfg, workflow, tier)[index]] if hourly_task else None
        inventory = data.fields(
            cfg,
            opts["data_root"],
            workflow,
            tier,
            timestamps=timestamps,
        )
        if selected:
            inventory = [
                f
                for f in inventory
                if (f["variable"] + (str(f["level"]) if f["level"] else "")) in selected
            ]
            if not inventory:
                raise ValueError(f"No fields match {selected}")
        if index is not None and not hourly_task:
            inventory = [inventory[index]]
        jobs = [(cfg, workflow, f, str(output)) for f in inventory]
        workers = 1 if workflow == "sequential_throughput" else opts["workers"]
        if workers == 1:
            for job in jobs:
                print("Measured", measure_field(job), flush=True)
        else:
            with ProcessPoolExecutor(
                workers, mp_context=multiprocessing.get_context("spawn")
            ) as pool:
                for name in pool.map(measure_field, jobs):
                    print("Measured", name, flush=True)
        if index is not None:
            return output
    report = validate(cfg, opts, workflow, tier, selected)
    print(json.dumps(report), flush=True)
    return output
