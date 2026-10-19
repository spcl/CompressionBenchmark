"""Render trajectory rate/distortion, particle paths and density panels."""

import json

import cartopy.crs as ccrs
import cartopy.feature
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

from compbench.figures.io import finish

VAR_TO_UNIT = {"u_component_of_wind": r"$\mathrm{m\,s^{-1}}$", "divergence": r"$\mathrm{s^{-1}}$"}
BALANCED_VARS = list(VAR_TO_UNIT)


COMPRESSOR_COLORS = dict(
    ebcc="tab:blue",
    sz="tab:green",
    sz3="tab:red",
    hpez="tab:orange",
    sperr="tab:purple",
    mgard="tab:pink",
)


def trajectory_rd(records, uncertainty, methods):
    import pandas as pd

    method_data = {
        m: pd.DataFrame(
            [dict(cr=r["cr"], rmse_H=r["rmse"]) for r in records if r["method"] == m]
        ).sort_values("cr")
        for m in methods
    }
    ens_rmse = np.array([uncertainty])
    flipped = True
    alpha = 0.3
    rmse_min = min(0.3, min(data["rmse_H"].min() for data in method_data.values()) / 1.15)
    rmse_max = max(11, max(data["rmse_H"].max() for data in method_data.values()) * 1.15)
    ticks_cr = [25, 50, 100, 200, 400]
    ticks_cr_str = [str(t) for t in ticks_cr]
    ticks_rmse = [0.5, 1, 2, 4, 8]
    ticks_rmse_str = [str(t) for t in ticks_rmse]
    plt.figure(figsize=(5, 4))

    # Shaded region showing ERA5 uncertainty
    if not flipped:
        plt.axhspan(rmse_min, ens_rmse.min(), color="lightblue", alpha=alpha)
        plt.axhline(ens_rmse.min(), color="lightblue", linestyle="-", linewidth=1)
        plt.text(
            30,
            ens_rmse.min() * 0.95,
            "ERA5 Uncertainty",
            ha="center",
            va="top",
            fontsize=12,
            color="darkblue",
        )
        plt.ylim(rmse_min, rmse_max)
    else:
        plt.axvspan(rmse_min, ens_rmse.min(), color="lightblue", alpha=alpha)
        plt.axvline(ens_rmse.min(), color="lightblue", linestyle="-", linewidth=1)
        plt.text(
            1.0, 100, "ERA5 Uncertainty", ha="center", va="center", fontsize=12, color="darkblue"
        )
        plt.xlim(rmse_min, rmse_max)
        plt.gca().invert_xaxis()

    # Plot compression methods
    plt.xscale("log")
    plt.yscale("log")
    if not flipped:
        for method in methods:
            plt.plot(
                method_data[method]["cr"],
                method_data[method]["rmse_H"],
                "o-",
                label=method.upper(),
                linewidth=2,
                markersize=6,
                color=COMPRESSOR_COLORS[method],
            )
        plt.xlabel("Compression Ratio")
        plt.ylabel("RMSE of Density")
        plt.xticks(ticks_cr, ticks_cr_str)
        plt.yticks(ticks_rmse, ticks_rmse_str)
    else:
        for method in methods:
            plt.plot(
                method_data[method]["rmse_H"],
                method_data[method]["cr"],
                "o-",
                label=method.upper(),
                linewidth=2,
                markersize=6,
                color=COMPRESSOR_COLORS[method],
            )
        plt.xlabel("RMSE of Density")
        plt.ylabel("Compression Ratio")
        plt.yticks(ticks_cr, ticks_cr_str)
        plt.xticks(ticks_rmse, ticks_rmse_str)

    loc = "upper right" if flipped else "upper left"
    plt.legend(loc=loc, ncol=2, fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return plt.gcf()


def trajectory_panel(ds, args):
    # Read each array once and avoid xarray indexing inside the particle loop.
    lon = ds["lon"].squeeze().values
    lat = ds["lat"].squeeze().values
    p = ds["p"].squeeze().values

    norm = plt.Normalize(p[0, :].min(), p[0, :].max())
    cmap = plt.get_cmap("viridis")

    plt.figure(figsize=(10, 5))
    central_longitude = 180
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=central_longitude))
    ax.add_feature(
        cartopy.feature.NaturalEarthFeature(
            "physical", "land", "50m", edgecolor=None, facecolor="lightgray"
        ),
        zorder=0,
        rasterized=True,
    )
    gl = ax.gridlines(draw_labels=True)
    if args.tick_fontsize is not None:
        gl.xlabel_style = {"size": args.tick_fontsize}
        gl.ylabel_style = {"size": args.tick_fontsize}
    discard = False
    pbar = range(lon.shape[1])
    for i in pbar:
        if args.sample_rate < 1.0:
            discard = np.random.rand() > args.sample_rate
        if not discard:
            color = cmap(norm(p[0, i].item()))
            ax.plot(
                lon[:, i],
                lat[:, i],
                color=color,
                linewidth=0.1,
                alpha=0.1,
                transform=ccrs.PlateCarree(),
                rasterized=True,
            )
    lonmin = args.lonmin if args.lonmin > 0 else args.lonmin + 360
    lonmax = args.lonmax if args.lonmax > 0 else args.lonmax + 360
    ax.set_extent(
        [lonmin - central_longitude, lonmax - central_longitude, args.latmin, args.latmax],
        crs=ccrs.PlateCarree(central_longitude=central_longitude),
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(
        sm, ax=ax, orientation="horizontal", label="Initial pressure (hPa)", fraction=0.08, pad=0.1
    )
    if args.label_fontsize is not None:
        cbar.ax.xaxis.label.set_size(args.label_fontsize)
    if args.tick_fontsize is not None:
        cbar.ax.tick_params(labelsize=args.tick_fontsize)
        cbar.ax.xaxis.get_offset_text().set_size(args.tick_fontsize)

    if args.title is not None:
        title_kwargs = {}
        if args.title_fontsize is not None:
            title_kwargs["fontsize"] = args.title_fontsize
        ax.set_title(args.title, **title_kwargs)

    return plt.gcf()


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap


def density_panel(ds, args):
    lon, lat = ds.lon.squeeze(), ds.lat.squeeze()
    t = lon.shape[0] - 1
    lonmin, lonmax = args.lonmin, args.lonmax
    central_longitude = 180
    plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=central_longitude))
    ax.coastlines()
    gl = ax.gridlines(draw_labels=True)
    if args.tick_fontsize is not None:
        gl.xlabel_style = {"size": args.tick_fontsize}
        gl.ylabel_style = {"size": args.tick_fontsize}

    # Create 2D histogram
    bins_lon = 600
    bins_lat = int(bins_lon / (lonmax - lonmin) * (args.latmax - args.latmin))
    H, xedges, yedges = np.histogram2d(
        lon[t, :].values.flatten(),
        lat[t, :].values.flatten(),
        bins=[bins_lon, bins_lat],
        range=[[lonmin, lonmax], [args.latmin, args.latmax]],
    )
    H[H == 0] = np.nan
    X, Y = np.meshgrid(xedges, yedges)

    # Plot using contourf
    cmap = plt.get_cmap("Oranges")
    cmap = truncate_colormap(cmap, 0.2, 1.0)
    vmin = 0
    vmax = np.nanmax(H) if args.density_max is None else args.density_max

    im = ax.pcolormesh(
        X, Y, H.T, cmap=cmap, transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax
    )  # norm=colors.Normalize(vmin=vmin, vmax=vmax)

    # Set extent
    ax.set_extent(
        [lonmin - central_longitude, lonmax - central_longitude, args.latmin, args.latmax],
        crs=ccrs.PlateCarree(central_longitude=central_longitude),
    )

    # Add colorbar
    cbar = plt.colorbar(
        im, ax=ax, orientation="horizontal", label="Particle Density", fraction=0.08, pad=0.1
    )
    if args.label_fontsize is not None:
        cbar.ax.xaxis.label.set_size(args.label_fontsize)
    if args.tick_fontsize is not None:
        cbar.ax.tick_params(labelsize=args.tick_fontsize)
        cbar.ax.xaxis.get_offset_text().set_size(args.tick_fontsize)

    if args.title is not None:
        title_kwargs = {}
        if args.title_fontsize is not None:
            title_kwargs["fontsize"] = args.title_fontsize
        ax.set_title(args.title, **title_kwargs)

    return plt.gcf()


def trajectories(output, cfg, tier):
    from types import SimpleNamespace

    import xarray as xr

    names = ["member_0", "member_1", "ebcc_0.01", "ebcc_0.05"]
    for i, name in enumerate(names):
        left, right, bottom, top = cfg["datasets"]["trajectory_simulation"]["density_region"]
        # Publication titles are part of each panel, ready for direct inclusion.
        args = SimpleNamespace(
            lonmin=left,
            lonmax=right,
            latmin=bottom,
            latmax=top,
            sample_rate=1.0,
            title=f"ERA5 uncertainty ensemble member {i}"
            if i < 2
            else f"EBCC, {float(name[5:]):.0%} error target",
            title_fontsize=14 if i < 2 else "xx-large",
            density_max=None if i < 2 else 220,
            label_fontsize=None if i < 2 else "xx-large",
            tick_fontsize=None if i < 2 else "x-large",
        )
        with xr.open_dataset(output / name / "trajectories.nc", decode_times=False) as ds:
            for number, draw in [(13, trajectory_panel), (14, density_panel)]:
                fig = draw(ds, args)
                run = f"ens_member{i}" if i < 2 else f"ens_ebcc_3d_rel_{name[5:].replace('.', '_')}"
                finish(
                    fig, output, f"Fukushima.{run}_{'traj' if number == 13 else 'density'}", tier
                )


def plot_trajectory(cfg, tier, output):
    values = json.loads((output / "measurements.json").read_text())
    finish(
        trajectory_rd(values["records"], values["uncertainty"], cfg["compressors"]),
        output,
        "trajectory_rmse_rd_3d",
        tier,
    )
    trajectories(output, cfg, tier)
