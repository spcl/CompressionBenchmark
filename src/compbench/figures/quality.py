"""Render compression-ratio versus quality panels."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

from compbench.compression import metrics
from compbench.figures.io import COLORS, finish


def quality_panel(points, methods, metric, *, hurricane=False):
    """Original 5×4-inch CR/quality panel, with reversed log(1 − SSIM)."""
    fig, ax = plt.subplots(figsize=(5, 4))
    for method in methods:
        group = sorted([r for r in points if r["method"] == method], key=lambda r: r["cr"])
        x = [max(1e-12, 1 - r["ssim"]) if metric == "ssim" else r["psnr"] for r in group]
        ax.plot(
            x,
            [r["cr"] for r in group],
            "o-",
            label=method.upper(),
            markersize=5,
            color=COLORS[method],
        )
    cr = np.array([r["cr"] for r in points])
    ylim = (3, 120) if hurricane else (cr.min() / 1.5, cr.max() * 1.5)
    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ax.grid(which="both")
    if metric == "ssim":
        distortion = np.array([max(1e-12, 1 - r["ssim"]) for r in points])
        threshold = 1 - 0.99919
        left = distortion.max() * 1.1 if hurricane else max(distortion.max(), threshold) * 10**0.5
        right = distortion.min() * 0.9 if hurricane else min(distortion.min(), threshold) / 10**0.5
        if not hurricane:
            ax.axvspan(1e-12, threshold, color="lightblue", alpha=0.3)
            ax.axvline(threshold, color="lightblue", linestyle="-", linewidth=1)
            ax.text(
                threshold * 0.9,
                ylim[1] / 1.6,
                "Baker threshold",
                ha="left",
                va="top",
                fontsize=10,
                color="darkblue",
            )
        ax.set(xlabel="SSIM", xscale="log", xlim=(left, right))
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        candidates = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1] + ([] if hurricane else [0.3])
        ticks = [v for v in candidates if right <= v <= left]
        ax.set_xticks(ticks, [f"{1 - v:g}" for v in ticks])
    else:
        psnr = [r["psnr"] for r in points]
        ax.set(xlabel="PSNR (dB)", xlim=(min(psnr) - 2, max(psnr) + 2))
    ax.set_ylabel("Compression Ratio")
    ax.legend(loc="best" if hurricane else ("lower left" if metric == "ssim" else "upper right"))
    fig.tight_layout()
    return fig


def rd(rows, output, cfg, tier):
    points = [
        dict(method=m, target=t, **p) for (m, t), p in metrics.statistics_summary(rows).items()
    ]
    for metric in ["ssim", "psnr"]:
        finish(
            quality_panel(points, cfg["compressors"], metric),
            output,
            "cr_vs_ssim_lowres" if metric == "ssim" else "cr_vs_psnr",
            tier,
        )
