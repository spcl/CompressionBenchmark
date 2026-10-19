"""Shared range-relative bounds and reversible per-level 3D normalization."""

import numpy as np

from compbench.compression import compressors


def relative_roundtrip(x, method, target, cfg, *, decode=True, **options):
    x = np.ascontiguousarray(x, dtype=np.float32)
    if x.ndim not in (2, 3) or not x.size or not np.isfinite(x).all():
        raise ValueError("Expected nonempty finite 2D/3D float32 input")
    if not np.isfinite(target) or not 0 < target <= 1:
        raise ValueError("Relative error target must be in (0, 1]")
    low = x.min(axis=(-2, -1), keepdims=True).astype(np.float64)
    span = x.max(axis=(-2, -1), keepdims=True).astype(np.float64) - low
    metadata = b""
    if not np.any(span):
        # A slab can have a different constant on each level: retain every value.
        payload = low.astype("<f4").tobytes()
        result = dict(payload=payload, size=len(payload), comp_time=0.0, constant=True)
        if decode:
            result.update(decoded=x.copy(), decomp_time=0.0)
    else:
        if x.ndim == 3:
            work = np.asarray((x - low) / np.where(span > 0, span, 1), dtype=np.float32)
            bound = target
            # These are exactly the float64 offsets and scales used for reconstruction.
            metadata = np.stack((low, span)).astype("<f8").tobytes()
        else:
            work, bound = x, float(span.item()) * target
        result = compressors.roundtrip(work, method, bound, cfg, decode=decode, **options)
        if decode and x.ndim == 3:
            result["decoded"] = (result["decoded"].astype(np.float64) * span + low).astype(
                np.float32
            )
    result["compressor_bytes"] = result["size"]
    result["normalization_bytes"] = len(metadata)
    result["payload"] = metadata + result["payload"]
    result["size"] = len(result["payload"])
    result["bounds"] = target * span
    return result
