"""Launch the installed CLI through the CSCS Alps Container Engine."""

import shlex
import subprocess
from pathlib import Path

from compbench.datasets import era5 as data
from compbench.execution.alps import layout

# Defaults for omitted settings: memory in MiB, wall times in HH:MM:SS.
SCHEDULER_DEFAULTS = dict(
    array_size=1000,
    array_concurrency=4,
    time="04:00:00",
    scaling_time="01:00:00",
    memory=460000,
    scaling_memory=460000,
    exclusive=True,
    requeue=True,
)


def submit(args, cfg, opts):
    scheduler = opts.get("scheduler")
    if not scheduler:
        raise ValueError("Choose an execution profile with scheduler settings")
    scheduler = {**SCHEDULER_DEFAULTS, **scheduler}
    for key, value in scheduler.items():
        if "\n" in str(value) or "\r" in str(value):
            raise ValueError(f"Invalid scheduler value: {key}")
    if any(not scheduler.get(k) or scheduler[k] == "CHANGE_ME" for k in ("account", "partition")):
        raise ValueError("Set scheduler account and partition in your profile")
    runtime = opts.get("runtime", {})
    if set(runtime) != {"environment"} or not isinstance(runtime["environment"], str):
        raise ValueError("Set runtime.environment to an Alps EDF name or absolute TOML path")
    environment = runtime["environment"]
    if not environment.strip() or any(c in environment for c in "\n\r"):
        raise ValueError("Invalid runtime.environment")
    if "/" in environment or environment.endswith(".toml"):
        environment = str(Path(environment).expanduser().resolve())
    for key in ("array_size", "array_concurrency"):
        value = scheduler[key]
        if type(value) is not int or value < 1:
            raise ValueError(f"scheduler.{key} must be a positive integer")
    for key in ("exclusive", "requeue"):
        if type(scheduler[key]) is not bool:
            raise ValueError(f"scheduler.{key} must be a boolean")
    root = Path(opts["output_root"]) / args.tier / "jobs"
    root.mkdir(parents=True, exist_ok=True)
    common = [
        "compbench",
        "run",
        args.workflow,
        "--config",
        str(Path(args.config).resolve()),
        "--profile",
        str(Path(args.profile).resolve()),
        "--tier",
        args.tier,
        "--workers",
        "1",
    ]
    for key in ("data_root", "output_root", "scratch_root"):
        common.extend(["--" + key.replace("_", "-"), opts[key]])
    if args.fields:
        common.extend(["--fields", *args.fields])
    hourly = args.workflow == "rate_distortion" and args.tier == "full"
    if args.workflow == "weak_scaling":
        cohorts = [
            (n, 0, 0)
            for n in cfg["weak_scaling"]["sample_ranks" if args.tier == "sample" else "ranks"]
        ]
    elif hourly:
        count = len(data.times(cfg, "rate_distortion", "full"))
        limit = scheduler["array_size"]
        cohorts = [(None, start, min(limit, count - start)) for start in range(0, count, limit)]
    else:
        cohorts = [(None, 0, 0)]
    scripts = []
    for ranks, offset, hours in cohorts:
        suffix = f"_{ranks}" if ranks is not None else (f"_hours_{offset}" if hourly else "")
        path = root / f"{args.workflow}{suffix}.sbatch"
        nodes, tasks, per_node = 1, 1, 1
        array_tasks = None
        if ranks is not None:
            nodes, tasks, per_node = layout(
                ranks,
                cfg["weak_scaling"]["ranks_per_node"],
            )
        elif args.workflow in ("energy_budget_closure", "trajectory_simulation"):
            from compbench.benchmarks.applications import application_cases

            array_tasks = len(application_cases(cfg, args.workflow, args.tier))
        elif hourly:
            array_tasks = hours
        elif args.workflow in (
            "rate_distortion",
            "error_statistics",
            "kappa_study",
            "quantization_residual",
            "sequential_throughput",
        ):
            fields = data.fields(
                cfg,
                opts["data_root"],
                args.workflow,
                args.tier,
            )
            if args.fields:
                fields = [
                    f
                    for f in fields
                    if f["variable"] + (str(f["level"]) if f["level"] else "") in args.fields
                ]
            if not fields:
                raise ValueError("No tasks to submit")
            if args.workflow != "sequential_throughput":
                array_tasks = len(fields)
        command = [
            "srun",
            f"--nodes={nodes}",
            f"--ntasks={tasks}",
            f"--ntasks-per-node={per_node}",
            "--cpus-per-task=1",
            "--cpu-bind=cores",
            "--distribution=block",
            "--kill-on-bad-exit=1",
            "--mpi=none",
            f"--environment={environment}",
        ]
        if ranks is not None:
            command += ["env", f"COMPBENCH_ALPS_RANKS={ranks}"]
        command += common
        scaling_setup = ""
        if ranks is not None:
            attempt_prefix = shlex.quote(str(root / f".scaling_{ranks}"))
            scaling_setup = f"""COMPBENCH_SCALING_ATTEMPT=$(mktemp -d {attempt_prefix}_job${{SLURM_JOB_ID}}_restart${{SLURM_RESTART_COUNT:-0}}_XXXXXX)
export COMPBENCH_SCALING_ATTEMPT
"""
        text = shlex.join(command)
        if args.workflow in (
            "rate_distortion",
            "error_statistics",
            "kappa_study",
            "quantization_residual",
            "energy_budget_closure",
            "trajectory_simulation",
        ):
            text += f' --task-index "$((SLURM_ARRAY_TASK_ID + {offset}))"'
        exclusive = "#SBATCH --exclusive\n" if scheduler["exclusive"] else ""
        requeue = "#SBATCH --requeue\n" if scheduler["requeue"] else ""
        array = (
            f"#SBATCH --array=0-{array_tasks - 1}%{scheduler['array_concurrency']}\n"
            if array_tasks is not None
            else ""
        )
        script = f"""#!/bin/bash
#SBATCH --job-name=cb-{args.workflow}{suffix}
#SBATCH --account={scheduler['account']}
#SBATCH --partition={scheduler['partition']}
#SBATCH --time={scheduler['scaling_time'] if ranks is not None else scheduler['time']}
#SBATCH --mem={scheduler['scaling_memory'] if ranks is not None else scheduler['memory']}
#SBATCH --output={shlex.quote(str(root / '%x-%A_%a.out'))}
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
{exclusive}{requeue}{array}#SBATCH --nodes={nodes}
#SBATCH --ntasks={tasks}
#SBATCH --ntasks-per-node={per_node}
set -euo pipefail
ulimit -c 0
export NVIDIA_VISIBLE_DEVICES=void MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OMP_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export BLOSC_NTHREADS=1 MGARD_X_DEVICE=serial OPJ_NUM_THREADS=0
unset COMPBENCH_ALPS_RANKS
{scaling_setup}{text}
"""
        path.write_text(script)
        scripts.append(str(path))
        if not args.dry_run:
            result = subprocess.run(
                ["sbatch", "--parsable", str(path)], capture_output=True, text=True, check=True
            )
            path.with_suffix(".jobid").write_text(result.stdout)
    return "\n".join(scripts)
