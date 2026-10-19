"""Render the Gaussian residual experiment from saved arrays."""

import numpy as np

from compbench.figures.io import configure_plot


def spike(arrays, stats):
    """Redraw the spike profiles, surface, and summary table from saved measurements."""
    plt = configure_plot()
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 20,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "legend.frameon": True,
        }
    )
    # Synthetic spike: full profiles, corrections, and gradients.
    a = arrays
    u = a["coordinate"]
    mid = int(np.argmin(np.abs(u)))
    assert u[mid] == 0, "The cross-section must be at y=0"
    fig = plt.figure(figsize=(13.33, 8), layout="constrained")
    columns = fig.subfigures(1, 2, width_ratios=[3, 2])
    axes = columns[0].subplots(3, 1)
    right = columns[1].add_gridspec(2, 1, height_ratios=[1.25, 1])
    original = a["original"][mid].astype(float)
    exact = original - a["base"][mid].astype(float)
    for ax, center in [(axes[0], original), (axes[1], exact)]:
        ax.fill_between(
            u,
            center - stats["bound"],
            center + stats["bound"],
            color=".7",
            alpha=0.35,
            label=r"Error tolerance ($\pm0.05$)",
            zorder=0,
        )
    for name, label, color, ls in [
        ("original", "Original", "black", "-"),
        ("base", "Base only", ".55", ":"),
        ("wavelet", "Wavelet residual", "#0072B2", "-"),
        ("sample", "Sample residual", "#D55E00", "--"),
    ]:
        axes[0].plot(u, a[name][mid], label=label, color=color, ls=ls, lw=1.3)
        if name in ("wavelet", "sample"):
            axes[1].plot(
                u,
                a[name][mid].astype(float) - a["base"][mid],
                label=label,
                color=color,
                ls=ls,
                lw=1.2,
            )
        gradient = np.gradient(a[name][mid].astype(np.float64), u, edge_order=2)
        axes[2].plot(u, gradient, label=label, color=color, ls=ls, lw=1.2)
    axes[1].plot(
        u,
        a["original"][mid].astype(float) - a["base"][mid],
        label="Exact correction",
        color=".4",
        ls=":",
        lw=1.2,
    )
    axes[1].set(xlim=(-0.2, 0.2), ylabel="Added correction", title="(b) Residual corrections")
    axes[1].legend(loc="upper left", fontsize=13)
    axes[0].set(
        xlim=(-0.2, 0.2),
        ylim=(-0.07, 1.07),
        ylabel="Amplitude",
        title="(a) Original and reconstructed profiles",
    )
    axes[0].legend(loc="upper right", fontsize=13)
    axes[2].set(
        xlim=(-0.2, 0.2),
        xlabel=f"Coordinate x (y = {u[mid]:g})",
        ylabel=r"Spatial gradient $\partial f/\partial x$",
        title="(c) Spatial gradients",
    )
    map_grid = right[0].subgridspec(1, 2, width_ratios=[1, 0.05])
    surface = columns[1].add_subplot(map_grid[0])
    colorbar_ax = columns[1].add_subplot(map_grid[1])
    from matplotlib.colors import LinearSegmentedColormap

    spike_colors = plt.get_cmap("Oranges")(np.linspace(0, 1, 256))
    spike_colors[0] = [1, 1, 1, 1]
    spike_cmap = LinearSegmentedColormap.from_list("white_oranges", spike_colors)
    field = surface.pcolormesh(
        u, u, a["original"], shading="nearest", cmap=spike_cmap, vmin=0, vmax=1, rasterized=True
    )
    surface.axhline(u[mid], color="black", ls="--", lw=1.8, label="Cross-section")
    surface.set(
        title="(d) Gaussian spike",
        xlabel="x",
        ylabel="y",
        xlim=(-0.2, 0.2),
        ylim=(-0.2, 0.2),
        aspect="equal",
    )
    surface.set_xticks([-0.2, 0, 0.2])
    surface.set_yticks([-0.2, 0, 0.2])
    surface.legend(loc="upper right", fontsize=13, facecolor="white")
    columns[1].colorbar(field, cax=colorbar_ax, label="Amplitude")
    table_ax = columns[1].add_subplot(right[1])
    table_ax.axis("off")
    table_ax.set_title("(e) Metrics", pad=12)
    records = {r["method"]: r for r in stats["records"]}

    def scientific(value):
        exponent = int(np.floor(np.log10(abs(value))))
        return rf"${value / 10**exponent:.2f}\times10^{{{exponent}}}$"

    rows = []
    for label, key in [
        ("Residual (bytes)", "residual_bytes"),
        ("Total (bytes)", "bytes"),
        ("Compression ratio", "compression_ratio"),
        ("PSNR (dB)", "psnr_db"),
        ("RMSE", "rmse"),
        ("Max. abs. error", "max_abs_error"),
    ]:
        rows.append(
            [label]
            + [
                str(records[m][key])
                if key.endswith("bytes")
                else (
                    f"{records[m][key]:.2f}"
                    if key in ("compression_ratio", "psnr_db")
                    else scientific(records[m][key])
                )
                for m in ("wavelet", "sample")
            ]
        )
    table = table_ax.table(
        cellText=rows,
        colLabels=["Metric", "Wavelet", "Sample-space"],
        colWidths=[0.44, 0.25, 0.31],
        cellLoc="center",
        bbox=[0, 0.18, 1, 0.76],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    for (row, col), cell in table.get_celld().items():
        cell.visible_edges = "B" if row == 0 else ("B" if row == len(rows) else "")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_text_props(weight="bold")
        if col == 0:
            cell.set_text_props(ha="left")
    return fig
