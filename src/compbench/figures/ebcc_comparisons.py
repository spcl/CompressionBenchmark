"""Compare EBCC kappa values and quantization/residual component choices."""

import numpy as np

from compbench.compression import metrics
from compbench.figures.io import configure_plot


def kappa_comparison(rows, cfg):
    plt = configure_plot()
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "legend.frameon": True,
        }
    )
    KAPPA0 = 2.25
    RANGE_RELATIVE_ERROR_TARGETS = np.array(cfg["kappa_study"]["range_relative_error_targets"])
    EXPONENTS = tuple(float(np.log2(k / KAPPA0)) for k in cfg["kappa_study"]["kappa_values"])
    groups = {
        (m, float(t)): [
            dict(
                row,
                bytes=row["size"],
                bits_per_sample=32 * row["size"] / row["nbytes"],
                normalized_rmse=np.sqrt(row["normalized_mse"]),
                residual_percent=100 * row["residual_bytes"] / row["size"],
            )
            for row in rows
            if row["kappa"] == KAPPA0 * 2**m and row["target"] == t
        ]
        for m in EXPONENTS
        for t in RANGE_RELATIVE_ERROR_TARGETS
    }
    # Estimate density in the displayed coordinate, retaining exact zeros.
    # This continuous symmetric-log transform is linear below 0.01 percent.
    threshold = 0.01

    def transform(values):
        values = np.asarray(values, dtype=float)
        return np.where(
            values <= threshold,
            values / threshold,
            1 + np.log10(np.maximum(values, threshold) / threshold),
        )

    fig = plt.figure(figsize=(14, 10), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    absolute_ax = fig.add_subplot(grid[0, 0])
    axes = [fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])]
    selected = RANGE_RELATIVE_ERROR_TARGETS[[0, 1, 5, 9]]
    colors = ["#CC79A7", "#0072B2", "#E69F00", "#009E73"]
    offsets = [-0.3, -0.1, 0.1, 0.3]
    for ax, key, display_transform in [(axes[1], "residual_percent", transform)]:
        for rel, color, offset in zip(selected, colors, offsets):
            for i, m in enumerate(EXPONENTS):
                values = np.array([row[key] for row in groups[(m, float(rel))]])
                display = display_transform(values)
                position = i + offset
                if np.ptp(display) > 0:
                    parts = ax.violinplot(
                        [display],
                        positions=[position],
                        widths=0.18,
                        showextrema=False,
                        bw_method=0.25,
                        points=150,
                    )
                    for body in parts["bodies"]:
                        body.set_facecolor(color)
                        body.set_edgecolor(color)
                        body.set_alpha(0.65)
                    q25, median, q75 = display_transform(np.percentile(values, [25, 50, 75]))
                    ax.plot([position, position], [q25, q75], color=color, lw=2.5, zorder=3)
                    ax.scatter(
                        [position], [median], s=12, facecolor="white", edgecolor=color, zorder=4
                    )
                else:
                    ax.plot(
                        [position - 0.08, position + 0.08],
                        [display[0], display[0]],
                        color=color,
                        lw=3,
                    )
        ax.set_xticks(range(len(EXPONENTS)), [f"{KAPPA0 * 2.0**m:g}" for m in EXPONENTS])
        ax.set(xlabel=r"Allocation coefficient $\kappa$", xlim=(-0.6, len(EXPONENTS) - 0.4))
        ax.xaxis.grid(False)
    from matplotlib.patches import Patch

    axes[1].legend(
        handles=[
            Patch(facecolor=c, alpha=0.65, label=f"{rel * 100:g}%")
            for rel, c in zip(selected, colors)
        ],
        title="Range-relative error target",
        ncol=2,
        loc="upper right",
        frameon=True,
    )
    ticks = np.array([0, 0.01, 0.1, 1, 10, 100])
    axes[1].set_yticks(transform(ticks), [f"{v:g}%" for v in ticks])
    axes[1].set(
        ylabel="Residual payload fraction", ylim=(-0.2, 5.15), title="(c) Residual payload fraction"
    )
    rd = []
    palette = plt.get_cmap("viridis")(np.linspace(0.08, 0.9, len(EXPONENTS)))
    for color, m in zip(palette, EXPONENTS):
        points = []
        for rel in RANGE_RELATIVE_ERROR_TARGETS:
            rows = groups[(m, float(rel))]
            original_bytes = sum(row["bytes"] * 32 / row["bits_per_sample"] for row in rows)
            compressed_bytes = sum(row["bytes"] for row in rows)
            psnr = np.array([-20 * np.log10(row["normalized_rmse"]) for row in rows])
            assert np.isfinite(psnr).all()
            point = dict(
                exponent=m,
                kappa=KAPPA0 * 2.0**m,
                relative_bound=float(rel),
                samples=len(rows),
                original_bytes=original_bytes,
                compressed_bytes=compressed_bytes,
                compression_ratio=original_bytes / compressed_bytes,
                mean_psnr_db=float(psnr.mean()),
            )
            points.append(point)
            rd.append(point)
        points.sort(key=lambda r: r["compression_ratio"])
    reference = sorted([r for r in rd if r["exponent"] == 0], key=lambda r: r["compression_ratio"])
    ref_cr = np.array([r["compression_ratio"] for r in reference])
    ref_psnr = np.array([r["mean_psnr_db"] for r in reference])
    assert np.all(np.diff(ref_cr) > 0)
    for row in rd:
        cr = row["compression_ratio"]
        row["comparison_status"] = "outside_default_range"
        if ref_cr[0] <= cr <= ref_cr[-1]:
            baseline = float(np.interp(np.log(cr), np.log(ref_cr), ref_psnr))
            row.update(
                comparison_status="plotted",
                default_psnr_db=baseline,
                relative_psnr_percent=100 * (row["mean_psnr_db"] / baseline - 1),
            )
    from matplotlib.ticker import PercentFormatter

    for color, m in zip(palette, EXPONENTS):
        absolute_points = sorted(
            [r for r in rd if r["exponent"] == m], key=lambda r: r["compression_ratio"]
        )
        absolute_ax.semilogx(
            [r["compression_ratio"] for r in absolute_points],
            [r["mean_psnr_db"] for r in absolute_points],
            color=color,
            label=f"{KAPPA0 * 2.0**m:g}",
            ls="--" if m == 0 else "-",
            marker="o",
            ms=3,
            lw=2 if m == 0 else 1.3,
        )
        points = sorted(
            [r for r in rd if r["exponent"] == m and r["comparison_status"] == "plotted"],
            key=lambda r: r["compression_ratio"],
        )
        label = f"{KAPPA0 * 2.0**m:g}"
        axes[0].semilogx(
            [r["compression_ratio"] for r in points],
            [r["relative_psnr_percent"] for r in points],
            "--" if m == 0 else "-o",
            color=color,
            label=label,
            ms=3,
            lw=2 if m == 0 else 1.3,
        )
    assert all(abs(r["relative_psnr_percent"]) < 1e-12 for r in rd if r["exponent"] == 0)
    axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=100))
    axes[0].set(
        xlabel="Compression ratio",
        ylabel=r"PSNR change relative to $\kappa = 2.25$",
        title="(b) Relative rate–distortion curves",
    )
    absolute_ax.set(
        xlabel="Compression ratio", ylabel="PSNR (dB)", title="(a) Rate–distortion curves"
    )
    absolute_ax.legend(title=r"$\kappa$", frameon=True, fontsize=12, ncol=2)
    return fig


def component_comparison(rows, cfg):
    plt = configure_plot()
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "legend.frameon": True,
        }
    )
    from matplotlib.ticker import PercentFormatter

    METHODS = ["ebcc", "rate_sample", "fixed_wavelet", "fixed_sample", "sperr"]
    LABELS = {
        "ebcc": "EBCC",
        "rate_sample": "Error-bound-aware allocation + sample residual",
        "fixed_wavelet": "Fixed-step + wavelet residual",
        "fixed_sample": "Fixed-step + sample residual",
        "sperr": "SPERR",
    }
    COLORS = {
        "ebcc": "#0072B2",
        "rate_sample": "#E69F00",
        "fixed_wavelet": "#CC79A7",
        "fixed_sample": "#D55E00",
        "sperr": "#009E73",
    }
    summary = []
    for method in METHODS:
        for target in cfg["quantization_residual"]["range_relative_error_targets"]:
            group = [r for r in rows if r["method"] == method and r["target"] == target]
            p = metrics.aggregate(group, mean_db=True)
            summary.append(
                dict(
                    metric="psnr",
                    method=method,
                    compression_ratio=p["cr"],
                    metric_mean=p["psnr"],
                    comparison_status="outside_reference_range",
                )
            )
    ref = sorted(
        [r for r in summary if r["method"] == "ebcc"], key=lambda r: r["compression_ratio"]
    )
    cr = np.array([r["compression_ratio"] for r in ref])
    values = np.array([r["metric_mean"] for r in ref])
    for r in summary:
        if cr[0] <= r["compression_ratio"] <= cr[-1]:
            baseline = np.interp(np.log(r["compression_ratio"]), np.log(cr), values)
            r.update(
                comparison_status="plotted",
                relative_percent=100 * (r["metric_mean"] / baseline - 1),
            )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), layout="constrained")
    for method in METHODS:
        rows = sorted(
            (r for r in summary if r["metric"] == "psnr" and r["method"] == method),
            key=lambda r: r["compression_ratio"],
        )
        axes[0].semilogx(
            [r["compression_ratio"] for r in rows],
            [r["metric_mean"] for r in rows],
            color=COLORS[method],
            ls="--" if method == "ebcc" else "-",
            marker="o",
            ms=3,
            lw=2 if method == "ebcc" else 1.3,
            label=LABELS[method],
        )
        relative_rows = [r for r in rows if r["comparison_status"] == "plotted"]
        axes[1].semilogx(
            [r["compression_ratio"] for r in relative_rows],
            [r["relative_percent"] for r in relative_rows],
            color=COLORS[method],
            ls="--" if method == "ebcc" else "-",
            marker=None if method == "ebcc" else "o",
            ms=3,
            lw=2 if method == "ebcc" else 1.3,
            label=LABELS[method],
        )
    axes[0].set(xlabel="Compression ratio", ylabel="PSNR (dB)", title="(a) Rate–distortion curves")
    axes[1].set(
        xlabel="Compression ratio",
        ylabel="PSNR change relative to EBCC",
        title="(b) Relative rate–distortion curves",
    )
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=100))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False, fontsize=12)
    return fig
