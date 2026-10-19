"""Render error histograms and spectra with the original panel layouts."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from compbench.figures.io import COLORS, finish
from compbench.results import record_batches


def histogram_panel(curves, rows, variable, ratio, methods, kind, ax=None, zoom=True):
    """Original plot_histogram.py single-variable layout, bins and spectral inset."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5, 3), layout="tight")
    else:
        fig = ax.figure
    inset = None
    if kind == "spectrum" and zoom:
        from matplotlib.patches import Rectangle

        inset = inset_axes(ax, width="30%", height="35%", loc="upper center", borderpad=1.5)
        inset.set(xlim=(340, 360), ylim=(1.2e-2, 5e-2))
        inset.tick_params(labelsize=8)
        formatter = ticker.FuncFormatter(
            lambda v, p: f"{v / 1e-2:.0f}" if 1e-2 <= v < 1e-1 else f"{v:.2g}"
        )
        formatter.set_offset_string(r"$\times 10^{-2}$")
        inset.yaxis.set_major_formatter(formatter)
        inset.yaxis.get_offset_text().set_fontsize(8)
        ax.add_patch(
            Rectangle(
                (340, 1.2e-2), 20, 5e-2 - 1.2e-2, linewidth=1, edgecolor="black", facecolor="none"
            )
        )
    zorders = dict(ebcc=10, hpez=9, sz=7, sz3=8, sperr=6, mgard=3)
    for method in methods:
        y = curves[f"{method}_{ratio}_{kind}"]
        style = dict(label=method.upper(), color=COLORS[method], zorder=zorders[method])
        if kind == "histogram":
            ax.stairs(y, curves[f"edges_{ratio}"], **style)
        else:
            ax.plot(np.arange(len(y)), y, **style)
            if inset is not None:
                inset.plot(np.arange(len(y)), y, color=COLORS[method], zorder=zorders[method])
    if kind == "histogram":
        limit = next(
            r["max_abs_error"] for r in rows if r["method"] == "ebcc" and r["requested_cr"] == ratio
        )
        ax.set_xlim(-limit, limit)
        ymax = ax.get_ylim()[1]
        ax.set_yticks(np.linspace(0, ymax, 5), [str(v) for v in np.linspace(0, 1, 5)])
        ax.set(
            xlabel="Compression Error" + (r" ($\mathrm{K}$)" if variable == "t2m" else ""),
            ylabel="Normalized Frequency",
        )
    else:
        ax.set_yscale("log")
        y = curves["reference_spectrum"]
        for target in [ax] + ([inset] if inset is not None else []):
            target.plot(
                np.arange(len(y)),
                y,
                label="Reference",
                color="black",
                linestyle="--",
                alpha=0.8,
                linewidth=1,
                zorder=100,
            )
        ax.set(xlabel="Sphere Harmonic Degree", ylabel="Power")
    ax.legend(loc="upper right" if kind == "histogram" else "best", fontsize="small")
    title = {"t2m": "2m temperature", "z_500": "Geopotential height"}.get(variable, variable)
    ax.set_title(rf"{title} ({ratio}$\times$)")
    if standalone:
        fig.tight_layout()
    return fig


def histograms(output, cfg, tier):
    fields = {}
    for _, rows in record_batches(output):
        field = rows[0]["field"]
        path = output / "curves" / f"{field['id']}.npz"
        curves = np.load(path)
        if "quantization_bins" not in curves:
            raise ValueError(
                "Histogram results lack quantization-bin metadata; rerun histograms in a fresh output root."
            )
        variable = field["variable"] + (f"_{field['level']}" if field["level"] is not None else "")
        fields[variable] = (curves, rows)
        for kind in ["histogram", "spectrum"]:
            for ratio in cfg["error_statistics"]["ratios"]:
                name = f"{kind}_{variable}_cr_{ratio}" + ("_zoom" if kind == "spectrum" else "")
                finish(
                    histogram_panel(curves, rows, variable, ratio, cfg["compressors"], kind),
                    output,
                    name,
                    tier,
                )

    # Appendix grids use all 28 histogram fields, including all four humidity levels.
    titles = dict(
        t="Temperature",
        z="Geopotential height",
        u="Zonal wind",
        v="Meridional wind",
        w="Vertical velocity",
        q="Specific humidity",
        t2m="2m temperature",
        msl="Mean sea level pressure",
        u10="10m zonal wind",
        v10="10m meridional wind",
    )
    units = dict(
        t="K",
        z="m",
        u=r"m\,s^{-1}",
        v=r"m\,s^{-1}",
        w=r"m\,s^{-1}",
        q=r"kg\,kg^{-1}",
        t2m="K",
        msl="Pa",
        u10=r"m\,s^{-1}",
        v10=r"m\,s^{-1}",
    )
    for group, variables, levels in [
        ("pl", ["t", "z", "u", "v", "w", "q"], cfg["error_statistics"]["pressure_levels"]),
        ("sfc", ["t2m", "msl", "u10", "v10"], [None]),
    ]:
        if not all(
            v + (f"_{level}" if level is not None else "") in fields
            for level in levels
            for v in variables
        ):
            continue
        for kind in ["histogram", "spectrum"]:
            for ratio in cfg["error_statistics"]["ratios"]:
                fig, axes = plt.subplots(
                    len(levels),
                    len(variables),
                    figsize=(5 * len(variables), 3 * len(levels)),
                    layout="tight",
                    squeeze=False,
                )
                for i, level in enumerate(levels):
                    for j, var in enumerate(variables):
                        ax = axes[i, j]
                        name = var + (f"_{level}" if level is not None else "")
                        curves, rows = fields[name]
                        histogram_panel(
                            curves, rows, name, ratio, cfg["compressors"], kind, ax=ax, zoom=False
                        )
                        title = titles[var] + (f" @ {level}" if level is not None else "")
                        ax.set_title(rf"{title} ({ratio}$\times$)")
                        if kind == "histogram":
                            ax.set_xlabel(rf"Compression Error ($\mathrm{{{units[var]}}}$)")
                finish(fig, output, f"{kind}_{group}_cr_{ratio}", tier)
