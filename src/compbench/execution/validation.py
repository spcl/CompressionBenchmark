"""Validate configuration, measurement coverage and numerical error bounds."""

import json
import sys
from collections import Counter
from pathlib import Path

from compbench.compression.metrics import accepts_bounds
from compbench.datasets import era5 as data
from compbench.results import record_batches, same_specification, save, specification


def check_matrix(rows, cfg, workflow, field_ids):
    """Reject duplicated settings as well as missing ones, independently per field."""
    if workflow == "kappa_study":
        settings = [
            ("ebcc", t, k)
            for k in cfg["kappa_study"]["kappa_values"]
            for t in cfg["kappa_study"]["range_relative_error_targets"]
        ]

        def key(r):
            return (r["method"], r["target"], r["kappa"])
    elif workflow == "quantization_residual":
        settings = [
            (m, t)
            for m in ("ebcc", "rate_sample", "fixed_wavelet", "fixed_sample", "sperr")
            for t in cfg["quantization_residual"]["range_relative_error_targets"]
        ]

        def key(r):
            return (r["method"], r["target"])
    elif workflow == "error_statistics":
        settings = [(m, t) for m in cfg["compressors"] for t in cfg["error_statistics"]["ratios"]]

        def key(r):
            return (r["method"], r["requested_cr"])
    else:
        settings = [(m, t) for m in cfg["compressors"] for t in cfg["range_relative_error_targets"]]

        def key(r):
            return (r["method"], r["target"])

    expected = Counter((field, setting) for field in field_ids for setting in settings)
    actual = Counter((row["field"]["id"], key(row)) for row in rows)
    if actual != expected:
        raise ValueError(f"Incomplete or duplicated setting matrix in {workflow}")


def warn_bounds(report, workflow):
    if report.get("bound_violations"):
        print(
            f"Warning: {workflow}: {report['bound_violations']} exact bound violations; "
            f"{report.get('outside_tolerance', 0)} outside sparse/small tolerance. "
            "Results retained; pipeline continues.",
            file=sys.stderr,
        )


def validate(cfg, opts, workflow, tier, selected=None, allow_bound_violations=False):
    output = Path(opts["output_root"]) / tier / workflow
    run = json.loads((output / "run.json").read_text())
    if not same_specification(run["specification"], specification(cfg, workflow, tier, selected)):
        raise ValueError(f"Validation configuration differs from {output / 'run.json'}")
    if workflow == "weak_scaling":
        from compbench.benchmarks.weak_scaling import validate_trials

        save(output / "validation.json", validate_trials(cfg, output, tier))
    if workflow in ("energy_budget_closure", "trajectory_simulation"):
        from compbench.benchmarks.applications import summarize_application

        summarize_application(cfg, workflow, tier, output)
    if workflow in (
        "residual_layer_ablation",
        "case_study_hurricane",
        "energy_budget_closure",
        "trajectory_simulation",
        "weak_scaling",
    ):
        report = json.loads((output / "validation.json").read_text())
        if not report["complete"]:
            raise ValueError(f"{workflow} has incomplete or invalid results")
        warn_bounds(report, workflow)
        return report
    fields = data.fields(cfg, opts["data_root"], workflow, tier)
    if selected:
        fields = [
            f for f in fields if f["variable"] + (str(f["level"]) if f["level"] else "") in selected
        ]
    wanted = {f["id"] for f in fields}
    actual = {p.stem for p in (output / "tasks").glob("*.json")}
    if actual != wanted:
        raise ValueError(
            f"Incomplete {workflow}: {len(actual & wanted)}/{len(wanted)} fields; unexpected={actual - wanted}"
        )
    violations, rejected, count = 0, 0, 0
    for field_id, batch in record_batches(output):
        check_matrix(batch, cfg, workflow, [field_id])
        violations += sum(not row["verified"] for row in batch)
        rejected += sum(not accepts_bounds(row) for row in batch)
        count += len(batch)
    report = dict(
        complete=True,
        bounds_accepted=rejected == 0,
        outside_tolerance=rejected,
        tolerated_bound_violations=violations - rejected,
        fields=len(fields),
        measurements=count,
        bound_violations=violations,
        tier=tier,
        diagnostic=bool(selected),
    )
    save(output / "validation.json", report)
    warn_bounds(report, workflow)
    return report
