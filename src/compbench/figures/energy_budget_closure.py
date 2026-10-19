"""Render energy-flux closure and rate/distortion panels."""

import json

import matplotlib.pyplot as plt
import xarray as xr

from compbench.figures.io import finish

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_dict = {
    "0_0001": colors[5],
    "0_0005": colors[6],
    "0_001": colors[0],
    "0_005": colors[1],
    "0_01": colors[2],
    "0_05": colors[3],
    "0_1": colors[4],
    "ref": "black",
}
COMPRESSOR_COLORS = dict(
    ebcc="tab:blue",
    sz="tab:green",
    sz3="tab:red",
    hpez="tab:orange",
    sperr="tab:purple",
    mgard="tab:pink",
)


def energy_line(delta_dict, cr_dict):
    ref_tag = "ens_member0"
    """Plot the latitudinal energy-budget lines **with** a zoom-in inset.

    Parameters
    ----------
    delta_dict : dict[str, xarray.DataArray]
        Keys are trace names (e.g. ``"ebcc_10"`` or ``"sz3_rel_05"``),
        values are arrays that expose a ``latitude`` coordinate and the
        data to plot on the y-axis.
    cr_dict : dict[str, float]
        Compression ratios for every entry of *delta_dict* except the
        reference curve (``"ref"``).  They are only used to write the
        legend labels.
    """
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    color_dict[ref_tag] = color_dict["ref"]

    # --- Main plot ------------------------------------------------------
    for tag, delta in delta_dict.items():
        rel_target = "0_" + tag.split("_")[-1] if "rel" in tag else ref_tag
        zorder = 100 if tag == ref_tag else 10
        linestyle = "-"

        # Build the label that goes into the legend
        if tag == ref_tag:
            label = "Reference"
            cr = 1
        elif "ens_member" in tag:
            member = tag.removeprefix("ens_member")
            label = f"ERA5 Ensemble\nMember {member}"
            cr = 1
            linestyle = "--"
            color_dict[tag] = "gray"
        else:
            error_percent = float("0." + tag.split("_")[-1]) * 100
            label = f"{error_percent:g}%"
            cr = cr_dict[tag]
        label += rf" ({int(cr)}$\times$)"

        # Plot the series on the main axes
        ax.plot(
            delta.latitude,
            delta,
            color=color_dict[rel_target],
            linestyle=linestyle,
            linewidth=1,
            zorder=zorder,
            label=label,
        )

    # Cosmetics of the main axes
    ax.set_xlabel("Latitude (deg)")
    ax.set_ylabel(r"$R$ ($W/m^2$)")
    ax.set_xlim(-80, 80)
    ax.set_ylim(-110, 80)
    ax.grid(True)

    ax.legend(ncol=2, prop={"size": 8}, title="Relative Error Target", loc="lower right")

    # ------------------------------------------------------------------
    #                       Z O O M   I N S E T
    # ------------------------------------------------------------------
    if "ens" in ref_tag:
        zoom_xlim = (-22, 2)
        zoom_ylim = (-22, 10)
    else:
        zoom_xlim = (-62, -38)
        zoom_ylim = (-5, 30)
    ax_ins = ax.inset_axes([0.14, 0.05, 0.34, 0.29])
    ax_ins.set_xlim(*zoom_xlim)
    ax_ins.set_ylim(*zoom_ylim)

    # Re‑plot the same curves inside the inset
    for tag, delta in delta_dict.items():
        rel_target = "0_" + tag.split("_")[-1] if "rel" in tag else ref_tag
        linestyle = "-"
        if "ens_member" in tag and tag != ref_tag:
            linestyle = "--"
        zorder = 100 if tag == ref_tag else 10
        ax_ins.plot(
            delta.latitude,
            delta,
            color=color_dict[rel_target],
            linestyle=linestyle,
            linewidth=1,
            zorder=zorder,
        )

    ax_ins.tick_params(axis="both", which="major", labelsize=7)
    ax_ins.grid(True, linewidth=0.4)

    # Mark the inset region on the main axes with connecting lines
    ax.indicate_inset_zoom(ax_ins, edgecolor="black", linewidth=1)

    # ------------------------------------------------------------------
    return fig


def energy_rd(records, uncertainty, methods):
    flipped, prefix = True, "ens_"
    method_data = {
        m.upper(): dict(
            cr=[r["cr"] for r in records if r["method"] == m],
            rmse=[r["rmse"] for r in records if r["method"] == m],
        )
        for m in methods
    }
    rmse_dict = {"ens_member1": uncertainty}
    rmse_values = [value for data in method_data.values() for value in data["rmse"]]
    rmse_min = min(rmse_values) / 2
    rmse_max = max(rmse_values) * 2
    ticks_cr = [25, 50, 100, 200, 400]
    ticks_cr_str = [str(t) for t in ticks_cr]
    plt.figure(figsize=(5, 4))
    ens_rmse = None
    if "ens" in prefix:
        ensemble_rmses = [
            float(rmse) for tag, rmse in rmse_dict.items() if tag.startswith("ens_member")
        ]
        if ensemble_rmses:
            ens_rmse = min(ensemble_rmses)
            if not flipped:
                plt.axhspan(rmse_min, ens_rmse, color="lightblue", alpha=0.3)
                plt.axhline(ens_rmse, color="lightblue", linestyle="-", linewidth=1)
                plt.ylim(rmse_min, rmse_max)
            else:
                plt.axvspan(rmse_min, ens_rmse, color="lightblue", alpha=0.5)
                plt.xlim(rmse_min, rmse_max)
                plt.gca().invert_xaxis()
                plt.axvline(ens_rmse, color="lightblue", linestyle="-", linewidth=1)
            # Place the annotation in the clear part of the shaded region, inside the frame.
            plt.gca().text(
                0.97,
                0.73 if flipped else 0.10,
                "ERA5 Uncertainty",
                transform=plt.gca().transAxes,
                ha="right",
                va="top",
                fontsize=10,
                color="darkblue",
                clip_on=True,
            )
    plt.xscale("log")
    plt.yscale("log")
    if not flipped:
        for method, data in method_data.items():
            plt.plot(data["cr"], data["rmse"], "o-", label=method)
        plt.xlabel("Compression Ratio")
        plt.ylabel("RMSE ($W/m^2$)")
        plt.xticks(ticks_cr, ticks_cr_str)
    else:
        for method, data in method_data.items():
            plt.plot(data["rmse"], data["cr"], "o-", label=method)
        plt.xlabel("RMSE ($W/m^2$)")
        plt.ylabel("Compression Ratio")
        plt.yticks(ticks_cr, ticks_cr_str)
    plt.legend(loc="upper right" if flipped else "upper left", ncol=2, fontsize=9)
    plt.grid(True, which="major")
    return plt.gcf()


def plot_energy(cfg, tier, output):
    values = json.loads((output / "measurements.json").read_text())
    series, ratios = {}, {}
    # Draw the compressed curves in decreasing target order, then both members.
    for target in sorted(cfg["range_relative_error_targets"], reverse=True):
        name = f"ebcc_{target:g}"
        tag = "ens_ebcc_3d_rel_" + str(target).replace(".", "_")
        with xr.open_dataset(output / name / "analysis.nc") as ds:
            series[tag] = ds.residual.load()
        ratios[tag] = next(r["cr"] for r in values["records"] if r["name"] == name)
    for i in [0, 1]:
        with xr.open_dataset(output / f"member_{i}" / "analysis.nc") as ds:
            series[f"ens_member{i}"] = ds.residual.load()
    finish(energy_line(series, ratios), output, "ens_energy_budget_delta_ebcc_3d", tier)
    finish(
        energy_rd(values["records"], values["uncertainty"], cfg["compressors"]),
        output,
        "energy_budget_rd_3d",
        tier,
    )
