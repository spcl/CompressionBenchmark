"""One entry point for local runs, cluster jobs, data preparation and figures."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Set limits before importing numerical libraries, including in spawned workers.
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "compbench-matplotlib")
)

WORKFLOWS = [
    "rate_distortion",
    "error_statistics",
    "kappa_study",
    "quantization_residual",
    "residual_layer_ablation",
    "case_study_hurricane",
    "sequential_throughput",
    "weak_scaling",
    "energy_budget_closure",
    "trajectory_simulation",
]
DATASETS = [workflow for workflow in WORKFLOWS if workflow != "residual_layer_ablation"]


def default_config():
    local = Path(__file__).resolve().parents[2] / "configs/benchmark.yaml"
    return str(
        local if local.exists() else Path(sys.prefix) / "share/compbench/configs/benchmark.yaml"
    )


def parser():
    p = argparse.ArgumentParser(prog="compbench", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in [
        "doctor",
        "plan",
        "run",
        "reproduce",
        "plot",
        "validate",
        "submit",
        "export-figures",
        "datasets",
    ]:
        command = sub.add_parser(name)
        command.add_argument(
            "--config", default=default_config(), help="benchmark YAML (scientific settings)"
        )
        command.add_argument("--profile", help="execution YAML (paths, workers, scheduler)")
        command.add_argument("--tier", choices=["sample", "full"], default="sample")
        command.add_argument("--data-root")
        command.add_argument("--output-root")
        command.add_argument("--scratch-root")
        command.add_argument("--workers", type=int)
        if name in ["plan", "run", "plot", "validate", "submit"]:
            command.add_argument("workflow", choices=WORKFLOWS, help="benchmark name")
            command.add_argument(
                "--fields", nargs="+", help="explicit diagnostic subset, e.g. t2m z500"
            )
        if name == "plot":
            command.add_argument(
                "--allow-bound-violations",
                action="store_true",
                help="compatibility flag; bound violations are always recorded and nonblocking",
            )
        if name == "run":
            command.add_argument("--task-index", type=int, help=argparse.SUPPRESS)
        if name == "reproduce":
            command.add_argument("--workflows", nargs="+", choices=WORKFLOWS, default=WORKFLOWS)
        if name == "submit":
            command.add_argument("--dry-run", action="store_true")
        if name == "export-figures":
            command.add_argument("--destination", required=True)
        if name == "datasets":
            command.add_argument(
                "action",
                choices=["fetch", "prepare", "verify", "list"],
                help="dataset action (verify is only available for ensemble datasets)",
            )
            command.add_argument(
                "dataset",
                nargs="?",
                choices=DATASETS,
            )
            command.add_argument(
                "--source", help="public Zarr URL, local Zarr mirror, or ensemble NetCDF glob"
            )
    return p


def doctor(cfg):
    import importlib.metadata

    from compbench.compression import compressors

    result = {}
    for name in cfg["compressors"]:
        try:
            if name == "ebcc":
                compressors.ebcc()
                detail = importlib.metadata.version("ebcc")
            elif name == "hpez":
                detail = compressors.hpez_library()._name
            elif name == "mgard":
                detail = compressors.mgard_library()._name
            else:
                from compbench.compression.plugins import library

                detail = library(name)._name
            result[name] = dict(available=True, detail=detail)
        except (ImportError, OSError, AttributeError, RuntimeError) as exc:
            result[name] = dict(available=False, error=str(exc))
    result["caltra"] = dict(available=shutil.which("caltra") is not None)
    print(json.dumps(result, indent=2))
    return 0 if all(result[n]["available"] for n in cfg["compressors"]) else 1


def plan(cfg, opts, workflow, tier, selected):
    from compbench.datasets import era5 as data

    count = None
    if workflow == "residual_layer_ablation":
        count = 1  # One generated Gaussian field.
    elif workflow not in ("energy_budget_closure", "trajectory_simulation", "weak_scaling"):
        fields = data.configured_fields(cfg, workflow)
        if selected:
            fields = [
                (name, level)
                for name, level in fields
                if name + (str(level) if level is not None else "") in selected
            ]
        count = len(fields) * len(data.times(cfg, workflow, tier))
    result = dict(
        workflow=workflow,
        tier=tier,
        fields=count,
        compressors=cfg["compressors"],
        range_relative_error_targets=cfg["range_relative_error_targets"],
        data_root=opts["data_root"],
        output=str(Path(opts["output_root"]) / tier / workflow),
        workers=opts["workers"],
        diagnostic=bool(selected),
    )
    if workflow in ("kappa_study", "quantization_residual"):
        result["range_relative_error_targets"] = cfg[workflow]["range_relative_error_targets"]
        result["pressure_levels"] = cfg[workflow]["pressure_levels"]
        if workflow == "kappa_study":
            result["compressors"] = ["ebcc"]
            result["kappa_values"] = cfg[workflow]["kappa_values"]
            variants = len(result["kappa_values"])
        else:
            result["compressors"] = ["ebcc", "sperr"]
            result["variants"] = ["ebcc", "rate_sample", "fixed_wavelet", "fixed_sample", "sperr"]
            variants = len(result["variants"])
        result["measurements"] = count * variants * len(result["range_relative_error_targets"])
    elif workflow in ("energy_budget_closure", "trajectory_simulation"):
        from compbench.benchmarks.applications import application_cases

        result.pop("fields")
        result["cases"] = len(application_cases(cfg, workflow, tier))
        result["selection"] = cfg["datasets"][workflow][tier]
    elif workflow == "weak_scaling":
        result["fields_per_rank"] = len(data.configured_fields(cfg, "rate_distortion"))
        result["ranks"] = cfg["weak_scaling"]["sample_ranks" if tier == "sample" else "ranks"]
        result["input_bytes"] = data.input_bytes(cfg, workflow, tier)
        result.pop("fields")
    elif workflow == "rate_distortion":
        result["input_bytes"] = data.input_bytes(cfg, workflow, tier)
    return result


def list_datasets(cfg):
    from compbench.datasets import ensemble, era5, format_bytes

    for dataset in DATASETS:
        inputs = ensemble if dataset in ("energy_budget_closure", "trajectory_simulation") else era5
        sizes = "; ".join(
            f"{tier}: {format_bytes(inputs.input_bytes(cfg, dataset, tier))}"
            for tier in ("sample", "full")
        )
        print(f"{dataset}  {sizes}")


def main(argv=None):
    args = parser().parse_args(argv)
    from compbench import results
    from compbench.config import load
    from compbench.datasets import era5 as data
    from compbench.execution import runner, validation

    try:
        cfg, opts = load(args.config, args.profile)
        for key in ("data_root", "output_root", "scratch_root", "workers"):
            value = getattr(args, key)
            if value is not None:
                opts[key] = str(Path(value).resolve()) if key.endswith("_root") else value
        if opts["workers"] < 1:
            raise ValueError("workers must be positive")
        Path(opts["scratch_root"]).mkdir(parents=True, exist_ok=True)
        os.environ["TMPDIR"] = opts["scratch_root"]
        import tempfile

        tempfile.tempdir = opts["scratch_root"]
        if args.command == "doctor":
            return doctor(cfg)
        if args.command == "plan":
            print(json.dumps(plan(cfg, opts, args.workflow, args.tier, args.fields), indent=2))
            return 0
        if args.command == "datasets":
            if args.action == "list":
                list_datasets(cfg)
                return 0
            if not args.dataset:
                raise ValueError("Select a dataset")
            if args.dataset in ("energy_budget_closure", "trajectory_simulation"):
                from compbench.datasets.ensemble import ensemble_data

                result = ensemble_data(cfg, opts, args.dataset, args.tier, args.action, args.source)
            else:
                if args.action == "verify":
                    raise ValueError(
                        "datasets verify supports only energy_budget_closure and trajectory_simulation"
                    )
                if args.action == "prepare" and not args.source:
                    raise ValueError(
                        "Provide a local Zarr mirror with --source, or use datasets fetch"
                    )
                result = data.fetch(cfg, opts["data_root"], args.dataset, args.tier, args.source)
            print(json.dumps(result, indent=2))
            return 0
        if args.command == "submit":
            from compbench.execution.slurm import submit

            print(submit(args, cfg, opts))
            return 0
        if args.command == "export-figures":
            from compbench.figures.export import export_figures

            print(export_figures(cfg, opts, args.tier, args.destination))
            return 0
        if args.command == "run":
            print(runner.run(cfg, opts, args.workflow, args.tier, args.fields, args.task_index))
            return 0
        if args.command == "validate":
            print(
                json.dumps(
                    validation.validate(cfg, opts, args.workflow, args.tier, args.fields), indent=2
                )
            )
            return 0
        if args.command == "plot":
            from compbench.figures.render import plot

            validation.validate(
                cfg, opts, args.workflow, args.tier, args.fields, args.allow_bound_violations
            )
            print(plot(cfg, opts, args.workflow, args.tier))
            return 0
        if args.command == "reproduce":
            from compbench.figures.render import plot

            report = {}
            errors = (
                ValueError,
                FileNotFoundError,
                RuntimeError,
                OSError,
                subprocess.SubprocessError,
            )
            # Figure 6 combines two workflows: finish measurements before rendering.
            for workflow in args.workflows:
                try:
                    runner.run(cfg, opts, workflow, args.tier)
                    report[workflow] = dict(complete=True)
                except errors as exc:
                    report[workflow] = dict(complete=False, error=str(exc))
                    print(f"{workflow}: {exc}", file=sys.stderr, flush=True)
            for workflow in args.workflows:
                if not report[workflow]["complete"]:
                    continue
                try:
                    report[workflow]["figures"] = str(plot(cfg, opts, workflow, args.tier))
                except errors as exc:
                    report[workflow] = dict(complete=False, error=str(exc))
                    print(f"{workflow}: {exc}", file=sys.stderr, flush=True)
            results.save(Path(opts["output_root"]) / args.tier / "reproduction.json", report)
            return int(not all(item["complete"] for item in report.values()))
    except (
        ValueError,
        KeyError,
        FileNotFoundError,
        RuntimeError,
        OSError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"compbench: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
