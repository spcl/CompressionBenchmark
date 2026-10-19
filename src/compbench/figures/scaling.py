"""Combined sequential-throughput and weak-scaling figure."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib import transforms as mtransforms

from compbench.figures.io import finish
from compbench.results import records

METHOD_ORDER = ["ebcc", "sz", "sz3", "hpez", "sperr", "mgard"]
METHOD_LABELS = {m: m.upper() for m in METHOD_ORDER}
METHOD_COLORS = dict(
    ebcc="tab:blue",
    sz="tab:orange",
    sz3="tab:green",
    hpez="tab:brown",
    sperr="tab:red",
    mgard="tab:pink",
)
AXIS_LABEL_FONTSIZE = 12
HIDDEN_RANK_TICK_LABELS = {32, 64, 128}


@dataclass(frozen=True)
class RangeSummary:
    mean: float
    low: float
    high: float


@dataclass(frozen=True)
class AggregatedPoint:
    world_size: int
    speedup: RangeSummary
    throughput: RangeSummary


@dataclass(frozen=True)
class MeanStdSummary:
    mean: float
    std: float


@dataclass(frozen=True)
class LocalThroughputPoint:
    method: str
    compression: MeanStdSummary
    decompression: MeanStdSummary | None


def compressor_sort_key(compressor: str) -> tuple[int, str]:
    try:
        return METHOD_ORDER.index(compressor), compressor
    except ValueError:
        return len(METHOD_ORDER), compressor


def compressor_label(compressor: str) -> str:
    return METHOD_LABELS.get(compressor, compressor.upper())


def compressor_color(compressor: str) -> str:
    return METHOD_COLORS.get(compressor, "tab:gray")


def nudge_ticklabel(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
    label_text: str,
    x_points: float = 0.0,
    y_points: float = 0.0,
) -> None:
    fig.canvas.draw()
    for label in ax.get_xticklabels():
        if label.get_text() == label_text:
            label.set_transform(
                label.get_transform()
                + mtransforms.ScaledTranslation(
                    x_points / 72,
                    y_points / 72,
                    fig.dpi_scale_trans,
                )
            )


def nudge_rank_ticklabels(
    fig: matplotlib.figure.Figure, ax: matplotlib.axes.Axes, ranks: list[int]
) -> None:
    if not ranks:
        return
    nudge_ticklabel(fig, ax, str(min(ranks)), x_points=-2.5)


def format_rank_ticklabels(ranks: list[int], hide_sparse_labels: bool = True) -> list[str]:
    return [
        "" if hide_sparse_labels and rank in HIDDEN_RANK_TICK_LABELS else str(rank)
        for rank in ranks
    ]


def collect_all_ranks(results: dict[str, list[AggregatedPoint]]) -> list[int]:
    return sorted({point.world_size for points in results.values() for point in points})


def format_rel_target(rel_target: float) -> str:
    return f"{rel_target:g}"


def plot_results(
    local_throughput_points: list[LocalThroughputPoint],
    results: dict[str, list[AggregatedPoint]],
    output: Path,
    rel_target: float | None,
    loglog_scale: bool,
    local_throughput_ylabel: str,
    direction: str,
) -> None:
    local_throughput_points = [p for p in local_throughput_points if p.method in METHOD_ORDER]
    results = {method: points for method, points in results.items() if method in METHOD_ORDER}
    if not results:
        raise ValueError("No valid scaling result files found.")

    all_ranks = collect_all_ranks(results)

    if not local_throughput_points:
        fig, ax_throughput = plt.subplots(figsize=(6, 4.5))
        axes = [ax_throughput]
    else:
        shape = (2, 1) if direction == "portrait" else (1, 2)
        size = (6, 7.5) if direction == "portrait" else (10, 4.5)
        fig, axes = plt.subplots(*shape, figsize=size)
        ax_local_throughput, ax_throughput = axes
    for axis in axes:
        axis.set_axisbelow(True)

    if local_throughput_points:
        method_positions = list(range(len(local_throughput_points)))
        has_decompression = any(
            point.decompression is not None for point in local_throughput_points
        )
        bar_width = 0.36 if has_decompression else 0.56
        compression_means = [point.compression.mean for point in local_throughput_points]
        compression_stds = [point.compression.std for point in local_throughput_points]
        compression_positions = (
            [position - bar_width / 2 for position in method_positions]
            if has_decompression
            else method_positions
        )

        compression_bars = ax_local_throughput.bar(
            compression_positions,
            compression_means,
            width=bar_width,
            color="tab:blue",
            label="Compression",
        )

        plotted_bar_series: list[
            tuple[matplotlib.container.BarContainer, list[float], list[float]]
        ] = [(compression_bars, compression_means, compression_stds)]
        if has_decompression:
            decompression_positions = [
                position + bar_width / 2
                for position, point in zip(method_positions, local_throughput_points)
                if point.decompression is not None
            ]
            decompression_means = [
                point.decompression.mean
                for point in local_throughput_points
                if point.decompression is not None
            ]
            decompression_stds = [
                point.decompression.std
                for point in local_throughput_points
                if point.decompression is not None
            ]
            decompression_bars = ax_local_throughput.bar(
                decompression_positions,
                decompression_means,
                width=bar_width,
                color="tab:orange",
                label="Decompression",
            )
            plotted_bar_series.append((decompression_bars, decompression_means, decompression_stds))

        for series_index, (bars, means, stds) in enumerate(plotted_bar_series):
            for bar, mean_value, std_value in zip(bars, means, stds):
                center_x = bar.get_x() + bar.get_width() / 2
                lower_error = min(mean_value, std_value)
                ax_local_throughput.errorbar(
                    center_x,
                    mean_value,
                    yerr=[[lower_error], [std_value]],
                    color="black",
                    capsize=3.5,
                    elinewidth=1.5,
                )
                # Stagger nearby paired values so their labels remain legible.
                label_offset = 0
                if series_index == 1:
                    nearest = min(
                        range(len(method_positions)),
                        key=lambda i: abs(method_positions[i] - center_x),
                    )
                    if abs(mean_value - compression_means[nearest]) < 0.05 * max(
                        compression_means + decompression_means
                    ):
                        label_offset = 10
                ax_local_throughput.annotate(
                    f"{mean_value:.1f}",
                    xy=(center_x, mean_value + 0.5),
                    xytext=(0, label_offset),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    zorder=1000,
                    path_effects=[patheffects.withStroke(linewidth=1.5, foreground="white")],
                )
        ax_local_throughput.set_ylabel(local_throughput_ylabel, fontsize=AXIS_LABEL_FONTSIZE)
        ax_local_throughput.set_xlabel("Method", fontsize=AXIS_LABEL_FONTSIZE)
        ax_local_throughput.set_xticks(method_positions)
        ax_local_throughput.set_xticklabels(
            [compressor_label(point.method) for point in local_throughput_points],
            rotation=35 if direction == "landscape" else 0,
        )
        ax_local_throughput.tick_params(axis="both", labelsize=AXIS_LABEL_FONTSIZE)
        ax_local_throughput.grid(True, axis="y", linestyle="-", alpha=0.4, zorder=0)
        ax_local_throughput.legend()

    for compressor in sorted(results, key=compressor_sort_key):
        points = results[compressor]
        x = [point.world_size for point in points]
        throughput_means = [point.throughput.mean for point in points]
        throughput_lows = [point.throughput.low for point in points]
        throughput_highs = [point.throughput.high for point in points]
        color = compressor_color(compressor)
        label = compressor_label(compressor)
        linestyle = "-"
        ax_throughput.plot(
            x,
            throughput_means,
            marker="o",
            linewidth=2,
            color=color,
            label=label,
            linestyle=linestyle,
        )
        ax_throughput.fill_between(x, throughput_lows, throughput_highs, color=color, alpha=0.18)

    ax_throughput.set_xlabel("Num ranks", fontsize=AXIS_LABEL_FONTSIZE)
    ax_throughput.set_ylabel("Parallel Throughput (MB/s)", fontsize=AXIS_LABEL_FONTSIZE)
    # ax_throughput.set_title("Total Throughput")
    if loglog_scale:
        ax_throughput.set_xscale("log", base=2)
        ax_throughput.set_yscale("log")
    ax_throughput.set_xticks(all_ranks)
    ax_throughput.set_xticklabels(
        format_rank_ticklabels(all_ranks, hide_sparse_labels=not loglog_scale),
        rotation=35 if direction == "landscape" else 0,
        ha="right" if direction == "landscape" else "center",
    )
    ax_throughput.tick_params(axis="both", labelsize=AXIS_LABEL_FONTSIZE)
    ax_throughput.grid(True, linestyle="-", alpha=0.4, zorder=0)
    nudge_rank_ticklabels(fig, ax_throughput, all_ranks)
    ax_throughput.legend(fontsize=8, loc="best")

    if direction == "portrait":
        fig.subplots_adjust(left=0.12, right=0.98, bottom=0.07, top=0.96, hspace=0.3)
    else:
        fig.subplots_adjust(left=0.09, right=0.985, bottom=0.2, top=0.9, wspace=0.3)

    if rel_target is not None:
        fig.suptitle(f"Scaling benchmark (rel_target={format_rel_target(rel_target)})")
    else:
        fig.suptitle("Scaling benchmark (all relative error targets)")
    return fig


def plot_scaling(cfg, root, tier):
    throughput = root / "sequential_throughput"
    scaling = root / "weak_scaling"
    if not (throughput / "validation.json").exists() or not (scaling / "validation.json").exists():
        raise ValueError("Figure 6 needs both completed throughput and scaling workflows.")
    status = json.loads((scaling / "validation.json").read_text())
    if not status["complete"]:
        raise ValueError("Figure 6 needs all configured rank cohorts; scaling is unfinished.")
    rows = records(throughput)
    local = []
    for method in cfg["compressors"]:
        group = [r for r in rows if r["method"] == method]
        values = []
        for key in ["comp_time", "decomp_time"]:
            rates = np.array([r["nbytes"] / r[key] / 1e6 for r in group if r[key] > 0])
            # 226 fields × 7 targets, without repeats.
            values.append(
                MeanStdSummary(float(len(rates) / sum(1 / rates)), float(rates.std(ddof=1)))
            )
        local.append(LocalThroughputPoint(method, *values))
    trials = [r for p in sorted(scaling.glob("trials_*.json")) for r in json.loads(p.read_text())]
    results = {}
    for method in cfg["compressors"]:
        group = [r for r in trials if r["method"] == method]
        points = []
        for n in sorted({r["ranks"] for r in group}):
            rates = np.array([r["throughput"] / 1e6 for r in group if r["ranks"] == n])
            # Each trial is total bytes / elapsed time. Never harmonic-average parallel rates.
            points.append(
                AggregatedPoint(
                    n,
                    RangeSummary(0, 0, 0),
                    RangeSummary(float(rates.mean()), float(rates.min()), float(rates.max())),
                )
            )
        results[method] = points
    # Use display labels independently of internal method names.
    fig = plot_results(
        local, results, None, None, True, "Single-core codec throughput (MB/s)", "landscape"
    )
    finish(fig, scaling, "weak_scaling_landscape", tier)
