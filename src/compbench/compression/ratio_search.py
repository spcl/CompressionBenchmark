"""Search for the requested compression ratio without exceeding it."""

import numpy as np

from compbench.compression import metrics
from compbench.compression.normalization import relative_roundtrip


def match_ratio(x, method, ratio, cfg, *, tolerance=0.05):
    """Ten geometric search steps; retain the best ratio below the requested ratio."""
    lo, hi = (1e-8 if method == "mgard" else 0.0005), 0.5
    span = float(x.max()) - float(x.min())
    if span <= 0:
        raise ValueError("A compression-ratio search needs a nonconstant field")
    trials = []

    def trial(target):
        result = relative_roundtrip(x, method, target, cfg)
        row = dict(
            target=target,
            requested_cr=ratio,
            cr=x.nbytes / result["size"],
            method=method,
            nbytes=x.nbytes,
            size=result["size"],
            **metrics.distortion(x, result["decoded"], target * span),
        )
        trials.append(row)
        return result["decoded"], row

    best = trial(lo)
    for _ in range(10):
        target = float(np.exp((np.log(lo) + np.log(hi)) / 2))
        decoded, row = trial(target)
        if row["cr"] <= ratio and row["cr"] > best[1]["cr"]:
            best = decoded, row
        if row["cr"] < ratio and abs(row["cr"] - ratio) / ratio <= tolerance:
            return decoded, row, trials
        if row["cr"] < ratio:
            lo = target
        else:
            hi = target
    return *best, trials
