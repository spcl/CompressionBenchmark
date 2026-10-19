"""Render the hurricane wind and divergence panels."""

import json

import cartopy.crs as ccrs
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.cm import ScalarMappable

from compbench.figures.io import finish

VAR_TO_UNIT = {"u_component_of_wind": r"$\mathrm{m\,s^{-1}}$", "divergence": r"$\mathrm{s^{-1}}$"}
BALANCED_VARS = list(VAR_TO_UNIT)


def hurricane_panel(arrays, rows, methods, ratio, key, plot_delta=False):
    METHODS = methods
    LON, LAT = -84, 14
    crs = ccrs.NearsidePerspective(
        central_longitude=LON, central_latitude=LAT, satellite_height=300000
    )
    ncols = len(methods)
    fig, axs = plt.subplots(
        1,
        ncols,
        figsize=(2.25 * ncols, 3.25),
        subplot_kw={"projection": crs},
        layout="constrained",
        gridspec_kw={"hspace": 0.05, "wspace": 0},
    )
    title_kwargs = dict(fontsize="x-large", pad=18)
    ssim_fontsize = "large"
    tick_fontsize = label_fontsize = "x-large"
    colorbar = True
    variable = "u_component_of_wind" if key == "u" else "divergence"
    axs = axs.flatten()
    extent = [
        float(arrays["longitude"].min()),
        float(arrays["longitude"].max()),
        float(arrays["latitude"].min()),
        float(arrays["latitude"].max()),
    ]
    lat, lon = arrays["latitude"], arrays["longitude"]
    selection = np.ix_((lat >= 4) & (lat <= 24), (lon >= 266) & (lon <= 286))
    original = arrays["original_" + key]
    da_comp_list = []
    for method in methods:
        row = next(r for r in rows if r["method"] == method and r["requested_cr"] == ratio)
        field = arrays[f"{method}_{ratio}_{key}"]
        cr = row["cr"] if key == "u" else row["divergence_cr"]
        da_comp_list.append((field - original if plot_delta else field, cr, 1, row["ssim_" + key]))
    cmap = "viridis" if not plot_delta else "coolwarm"
    if plot_delta:
        vmax = -1
        for da_comp, _, _, _ in da_comp_list:
            da_comp_r = da_comp[selection]
            vmax = max(vmax, abs(da_comp_r.min().item()), abs(da_comp_r.max().item()))
        vmin = -vmax
        norm = colors.AsinhNorm(vmin=vmin, vmax=vmax, linear_width=vmax / 20)
        scale = 10 ** np.floor(np.log10(vmax))
        ticks = [t * scale for t in [-1, -0.2, 0, 0.2, 1]]

        def fmt_label(x, pos):
            x = x / scale
            if abs(x) >= 1 or abs(x) == 0:
                return f"{x:.0f}"
            else:
                return f"{x:.1f}"

        formatter = ticker.FuncFormatter(fmt_label)
        exponent = int(np.log10(scale))
        if exponent != 0:
            formatter.set_offset_string(rf"×10$^{{{exponent} }}$")
    else:
        da_ref_r = original[selection]
        vmin = da_ref_r.min().item()
        vmax = da_ref_r.max().item()
        if variable in BALANCED_VARS:
            abs_max = max(abs(vmin), abs(vmax))
            vmin = -abs_max
            vmax = abs_max
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        ticks = None
        formatter = None
        if max(abs(vmin), abs(vmax)) < 0.1:
            formatter = ticker.ScalarFormatter(useMathText=True, useOffset=True)
            formatter.set_powerlimits((-1, 1))

    for i, method in enumerate(METHODS):
        da_comp, original_size, comp_size, ssim = da_comp_list[i]
        axs[i].coastlines()
        axs[i].imshow(da_comp, extent=extent, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm)
        method_name = method.upper()
        axs[i].set_title(
            rf"{method_name} ({original_size / comp_size:.1f}$\times$)", **title_kwargs
        )
        axs[i].text(
            0.5,
            1.015,
            f"SSIM: {ssim:.3f}",
            transform=axs[i].transAxes,
            ha="center",
            va="bottom",
            fontsize=ssim_fontsize,
        )
    for i in range(len(METHODS), len(axs)):
        axs[i].remove()
    if tick_fontsize is not None:
        for i in range(len(METHODS)):
            axs[i].tick_params(labelsize=tick_fontsize)
    if colorbar:
        # fig.subplots_adjust(bottom=0.12)
        # cbar_ax = fig.add_axes((0.05, 0.08, 0.9, 0.03))
        unit = VAR_TO_UNIT.get(variable, "N/A")
        cbar_var = "Zonal wind" if key == "u" else "Divergence"
        cbar_label = f"{'Compression Error of ' if plot_delta else ''}{cbar_var} ({unit})"
        cbar = fig.colorbar(
            ScalarMappable(norm=norm, cmap=cmap),
            ax=axs[: len(METHODS)],
            orientation="horizontal",
            shrink=0.9,
            aspect=25 * ncols / 3,
            label=cbar_label,
            format=formatter,
            ticks=ticks,
        )
        cbar.ax.xaxis.get_offset_text().set_position((1.05, 0))
        cbar.ax.xaxis.OFFSETTEXTPAD = 0
        if label_fontsize is not None:
            cbar.ax.xaxis.label.set_size(label_fontsize)
        if tick_fontsize is not None:
            cbar.ax.tick_params(labelsize=tick_fontsize)
            cbar.ax.xaxis.get_offset_text().set_size(tick_fontsize)
    return fig


def hurricane(output, cfg, tier):
    from compbench.datasets.era5 import NATIVE_LATITUDE
    from compbench.figures.quality import quality_panel

    arrays = np.load(output / "arrays.npz")
    if len(arrays["latitude"]) != len(NATIVE_LATITUDE):
        raise ValueError(
            "Iota arrays predate the full-globe rendering correction; rerun in a fresh output root."
        )
    rows = json.loads((output / "measurements.json").read_text())["matches"]
    for ratio in cfg["error_statistics"]["ratios"]:
        for key, variable in [("u", "u_component_of_wind"), ("d", "divergence")]:
            for delta in [False, True]:
                name = f"casestudy_{variable}_850_cr{ratio}" + ("_delta" if delta else "")
                finish(
                    hurricane_panel(arrays, rows, cfg["compressors"], ratio, key, delta),
                    output,
                    name,
                    tier,
                )
    for key, variable in [("u", "u_component_of_wind"), ("d", "divergence")]:
        points = [
            dict(
                method=r["method"],
                cr=r["cr"] if key == "u" else r["divergence_cr"],
                ssim=r["ssim_" + key],
            )
            for r in rows
        ]
        finish(
            quality_panel(points, cfg["compressors"], "ssim", hurricane=True),
            output,
            f"casestudy_{variable}_850_cr_vs_ssim",
            tier,
        )
