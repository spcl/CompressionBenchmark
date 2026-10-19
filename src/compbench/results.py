"""Atomic result storage, run identity, and saved measurement readers."""

import json
import platform
import tempfile
import time
from pathlib import Path

import numpy as np


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        tmp = Path(handle.name)
        handle.write(
            json.dumps(
                value,
                indent=2,
                allow_nan=False,
                default=lambda x: x.item() if isinstance(x, np.generic) else str(x),
            )
            + "\n"
        )
    tmp.replace(path)


def specification(cfg, workflow, tier, fields=None):
    from compbench.compression.metrics import MAX_BOUND_FRACTION, MAX_VIOLATION_FRACTION

    spec = dict(config=cfg, tier=tier, workflow=workflow, fields=fields, measurement_version=4)
    spec["bound_acceptance"] = dict(
        max_violation_fraction=MAX_VIOLATION_FRACTION,
        max_bound_fraction=MAX_BOUND_FRACTION,
        scope="per_horizontal_slab",
        action="warn",
    )
    if workflow == "weak_scaling":
        spec["scaling_protocol"] = "shared_filesystem_v1"
    if workflow == "trajectory_simulation":
        spec["trajectory_solver"] = "LAGRANTO ECMWF 2.7 (CDO)"
    return spec


def same_specification(left, right):
    """Error-statistics field selection and ratios apply only to that workflow."""
    if left.get("workflow") == right.get("workflow") and left.get("workflow") != "error_statistics":
        left = dict(
            left, config={k: v for k, v in left["config"].items() if k != "error_statistics"}
        )
        right = dict(
            right, config={k: v for k, v in right["config"].items() if k != "error_statistics"}
        )
    return left == right


def prepare_run(cfg, opts, workflow, tier, fields=None):
    output = Path(opts["output_root"]) / tier / workflow
    output.mkdir(parents=True, exist_ok=True)
    resolved = specification(cfg, workflow, tier, fields)
    path = output / "run.json"
    if path.exists():
        old = json.loads(path.read_text())
        if not same_specification(old["specification"], resolved):
            raise ValueError(f"Configuration differs from {path}; choose a fresh --output-root")
    else:
        save(
            path,
            dict(
                specification=resolved,
                hardware=platform.platform(),
                machine=platform.machine(),
                workers=opts["workers"],
                started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            ),
        )
    return output


def record_batches(output):
    for path in sorted((Path(output) / "tasks").glob("*.json")):
        record = json.loads(path.read_text())
        yield path.stem, [dict(field=record["field"], **row) for row in record["records"]]


def records(output):
    return [row for _, batch in record_batches(output) for row in batch]
