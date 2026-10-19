"""Moist-static-energy closure, using weighted flux decomposition."""

from pathlib import Path

import numpy as np
import xarray as xr

from compbench.compression.reconstruction import reconstruct
from compbench.datasets import ensemble
from compbench.datasets.era5 import filename

LATITUDE_SMOOTHING_DEGREES = 5


def weights(path, template):
    with xr.open_dataset(path) as ds:
        ds = ds.rename({"plev": "level", "lat": "latitude", "lon": "longitude"})
        ds = ds.assign_coords(longitude=ds.longitude % 360).sortby("longitude")
        result = (
            ds.interp(latitude=template.latitude, longitude=template.longitude)
            .sel(level=template.level)
            .load()
        )
    # Preserve the original interpolation; both tiers use the same 0.5° grid.
    return result


def zonal_global_mean(value):
    return (value * np.cos(np.deg2rad(value.latitude))).mean("latitude")


def divergence(flux):
    from metpy.constants import earth_avg_radius

    step = float(abs(flux.latitude[1] - flux.latitude[0]))
    cells = max(1, round(LATITUDE_SMOOTHING_DEGREES / step))
    smooth = (
        (flux * np.cos(np.deg2rad(flux.latitude)))
        .rolling(latitude=cells, center=True)
        .mean()
        .dropna("latitude")
    )
    latitude = np.deg2rad(smooth.latitude.values)
    mid = (latitude[1:] + latitude[:-1]) / 2
    values = np.diff(smooth.values) / (
        np.diff(latitude) * float(earth_avg_radius.magnitude) * np.cos(mid)
    )
    return xr.DataArray(values, dims="latitude", coords={"latitude": np.rad2deg(mid)})


def statistics(dataset):
    from metpy.constants import Cp_d, Lv

    cp, latent = float(Cp_d.magnitude), float(Lv.magnitude)
    h = cp * dataset.t + latent * dataset.q
    mse = h + dataset.z
    ensemble.verify_energy_variables(dataset)
    surface = dataset.rlns + dataset.rsns + dataset.hfss + dataset.hfls
    eia = dataset.rsnt + dataset.rlnt - surface
    eia = eia.mean("longitude")
    return xr.Dataset(
        dict(v=dataset.v, m=mse, vm=dataset.v * mse, h=h, eia=eia - zonal_global_mean(eia))
    )


def close_budget(left, middle, right, w, interval_days):
    from metpy.constants import earth_gravity

    gravity = float(earth_gravity.magnitude)
    vm = middle.v
    mm = middle.m
    vz = (vm * w.ZON_NORM).sum("longitude")
    mz = (mm * w.ZON_NORM).sum("longitude")
    va = vz - (vz * w.ZON_WEIGHT).sum("level") / w.ZON_WEIGHT.sum("level")
    ma = mz - (mz * w.ZON_WEIGHT).sum("level") / w.ZON_WEIGHT.sum("level")
    mean_flux = (va * ma * w.ZON_WEIGHT).sum("level") * 100 / gravity
    stationary = (
        ((w.WEIGHT * (vm - vz) * (mm - mz) / w.WEIGHT_NORM).mean("longitude") * w.WEIGHT_NORM).sum(
            "level"
        )
        * 100
        / gravity
    )
    transient = (
        ((w.WEIGHT * (middle.vm - vm * mm) / w.WEIGHT_NORM).mean("longitude") * w.WEIGHT_NORM).sum(
            "level"
        )
        * 100
        / gravity
    )

    def heat(h):
        column = (
            ((h * w.WEIGHT / w.WEIGHT_NORM).mean("longitude") * w.WEIGHT_NORM).sum("level")
            * 100
            / gravity
        )
        return column - zonal_global_mean(column)

    storage = (heat(right.h) - heat(left.h)) / (interval_days * 86400)
    fields = {
        name: divergence(value)
        for name, value in [("F_MM", mean_flux), ("F_SE", stationary), ("F_TE", transient)]
    }
    latitude = fields["F_MM"].latitude
    net = (middle.eia - storage).interp(latitude=latitude)
    fields["F_NE"] = net
    fields["residual"] = fields["F_MM"] + fields["F_SE"] + fields["F_TE"] - net
    return xr.Dataset(fields)


def energy_member(cfg, opts, tier, member, method=None, target=None):
    spec, chosen, times = ensemble.selection(cfg, "energy_budget_closure", tier)
    accumulators = {}
    counts = {}
    measurements = []
    template = None
    for time in times:
        path = Path(opts["data_root"]) / "energy_budget_closure" / tier / f"member_{member}" / filename(time)
        with xr.open_dataset(path) as handle:
            ds = ensemble.normalize(handle).load()
        ensemble.verify_energy_variables(ds)
        if template is None:
            template = ds
        ds, rows = reconstruct(ds, method, target, cfg)
        measurements.extend(rows)
        instant = statistics(ds.isel(time=0, drop=True)).astype(float)
        if tier == "full":
            group = int(str(time)[5:7]) - 8
        else:
            group = int((time - times[0]) / np.timedelta64(1, "D")) // chosen["window_days"]
        accumulators[group] = (
            instant if group not in accumulators else accumulators[group] + instant
        )
        counts[group] = counts.get(group, 0) + 1
    if set(counts) != {0, 1, 2}:
        raise ValueError("Energy closure needs three complete averaging windows")
    means = [accumulators[i] / counts[i] for i in range(3)]
    w = weights(ensemble.asset_path("weights.nc"), template)
    interval = 60 if tier == "full" else 2 * chosen["window_days"]
    budget = close_budget(*means, w, interval)
    return budget, measurements
