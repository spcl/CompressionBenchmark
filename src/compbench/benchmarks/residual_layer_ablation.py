"""Fixed-base, matched-size Gaussian residual experiment."""

import numpy as np

from compbench.compression import compressors, metrics
from compbench.results import save


def spike(cfg, output):
    spec = cfg["residual_layer_ablation"]
    u = np.linspace(-1, 1, spec["size"])
    xx, yy = np.meshgrid(u, u)
    x = np.exp(-(xx**2 + yy**2) / (2 * spec["sigma"] ** 2)).astype(np.float32)
    bound, kappa = spec["bound"], spec["kappa"]
    sample = compressors.roundtrip(x, "ebcc", bound, cfg, sample=True, kappa=kappa)
    base, raw, parts = compressors.ebcc().base(sample["payload"], x.shape)
    trials, best = [], None
    start, stop, count = spec["wavelet_bounds"]
    for target in np.linspace(start, stop, int(count)):
        result = compressors.roundtrip(
            x, "ebcc", float(target), cfg, kappa=kappa * (target / bound) ** 2
        )
        _, other, detail = compressors.ebcc().base(result["payload"], x.shape)
        if other != raw:
            raise ValueError("The spike sweep changed the base reconstruction")
        row = dict(
            target=float(target),
            size=result["size"],
            **metrics.distortion(x, result["decoded"], float(target)),
        )
        trials.append(row)
        score = (abs(result["size"] - sample["size"]), row["rmse"], target)
        if best is None or score < best[0]:
            best = score, result, detail
    _, wavelet, wavelet_parts = best
    np.savez_compressed(
        output / "arrays.npz",
        coordinate=u,
        original=x,
        base=base,
        sample=sample["decoded"],
        wavelet=wavelet["decoded"],
    )
    rows = [
        dict(
            method=name,
            size=result["size"],
            nbytes=x.nbytes,
            **detail,
            **metrics.distortion(x, result["decoded"], bound),
        )
        for name, result, detail in [("sample", sample, parts), ("wavelet", wavelet, wavelet_parts)]
    ]
    save(output / "measurements.json", dict(records=rows, sweep=trials))
    save(
        output / "validation.json",
        dict(
            complete=wavelet["size"] == sample["size"],
            bounds_accepted=all(metrics.accepts_bounds(r) for r in rows),
            bound_violations=sum(not r["verified"] for r in rows),
            outside_tolerance=sum(not metrics.accepts_bounds(r) for r in rows),
            base_identical=True,
            size_matched=wavelet["size"] == sample["size"],
            measurements=len(trials) + 1,
        ),
    )
