"""Field-wise quality, parameter and sequential-throughput measurements."""

import json
from pathlib import Path

from compbench.compression import compressors, metrics
from compbench.compression.normalization import relative_roundtrip
from compbench.datasets import era5 as data
from compbench.results import save


def measurement(x, method, target, cfg, *, ssim=False, **options):
    result = relative_roundtrip(x, method, target, cfg, **options)
    bound = float(result.pop("bounds").item())
    decoded = result.pop("decoded")
    payload = result.pop("payload", None)
    row = dict(
        method=method,
        target=float(target),
        nbytes=x.nbytes,
        **result,
        **metrics.distortion(x, decoded, bound),
    )
    if ssim:
        row["ssim"] = 1.0 if result.get("constant") else metrics.dssim(x, decoded, factor=4)
    if method == "ebcc" and not result.get("constant") and payload and x.ndim == 2:
        _, _, parts = compressors.ebcc().base(payload, x.shape)
        row.update(parts)
    return row


def measure_field(job):
    cfg, workflow, field, output = job
    output = Path(output)
    record_path = output / "tasks" / (field["id"] + ".json")
    info = Path(field["path"]).stat()
    identity = dict(size=info.st_size, modified_ns=info.st_mtime_ns)
    if record_path.exists():
        old = json.loads(record_path.read_text())
        if old["input"] != identity:
            raise ValueError(f"Input changed; use a fresh output directory: {field['path']}")
        return field["id"]
    x = data.load_field(field)
    rows = []
    if workflow == "kappa_study":
        for kappa in cfg["kappa_study"]["kappa_values"]:
            for target in cfg["kappa_study"]["range_relative_error_targets"]:
                row = measurement(x, "ebcc", target, cfg, kappa=kappa)
                rows.append(dict(kappa=kappa, **row))
    elif workflow == "quantization_residual":
        for name, fixed, sample in [
            ("ebcc", False, False),
            ("rate_sample", False, True),
            ("fixed_wavelet", True, False),
            ("fixed_sample", True, True),
            ("sperr", False, False),
        ]:
            for target in cfg["quantization_residual"]["range_relative_error_targets"]:
                row = measurement(
                    x,
                    "sperr" if name == "sperr" else "ebcc",
                    target,
                    cfg,
                    fixed=fixed,
                    sample=sample,
                )
                row["method"] = name
                rows.append(row)
    elif workflow == "error_statistics":
        from compbench.benchmarks.error_statistics import measure_histograms

        rows = measure_histograms(x, field, cfg, output)
    else:
        # Warm each compressor before sequential measurements; no repetition enters the results.
        if workflow == "sequential_throughput":
            for method in cfg["compressors"]:
                relative_roundtrip(x, method, 0.01, cfg)
        for method in cfg["compressors"]:
            for target in cfg["range_relative_error_targets"]:
                rows.append(
                    measurement(x, method, target, cfg, ssim=workflow == "rate_distortion")
                )
    save(record_path, dict(field=field, input=identity, records=rows))
    return field["id"]
