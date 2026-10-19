"""Compress and reconstruct ensemble variables with per-level error bounds."""

import numpy as np

from compbench.compression.metrics import bound_statistics
from compbench.compression.normalization import relative_roundtrip


def reconstruct(ds, method, target, cfg):
    if method is None:
        return ds, []
    output = ds.copy(deep=True)
    rows = []
    for name, field in ds.data_vars.items():
        if "latitude" not in field.dims:
            continue
        x = np.ascontiguousarray(field.squeeze("time").values, dtype=np.float32)
        result = relative_roundtrip(x, method, target, cfg)
        decoded = result["decoded"]
        rows.append(
            dict(
                variable=name,
                nbytes=x.nbytes,
                size=result["size"],
                compressor_bytes=result["compressor_bytes"],
                normalization_bytes=result["normalization_bytes"],
                **bound_statistics(decoded.astype(float) - x, result["bounds"]),
            )
        )
        output[name].data = decoded[None, ...]
    return output, rows
