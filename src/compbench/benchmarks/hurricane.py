"""Hurricane wind and divergence measurements at matched ratios."""

import numpy as np
import xarray as xr

from compbench.compression import compressors, metrics
from compbench.compression.ratio_search import match_ratio
from compbench.datasets import era5 as data
from compbench.results import save


def divergence(u, v, lat, lon):
    # Match the original MetPy CF-aware derivative, including projection factors.
    import metpy.calc

    ds = xr.Dataset(coords=dict(latitude=lat, longitude=lon))
    ds.latitude.attrs.update(units="degrees_north", standard_name="latitude")
    ds.longitude.attrs.update(units="degrees_east", standard_name="longitude")
    for name, values in [("u", u), ("v", v)]:
        ds[name] = xr.DataArray(values, dims=("latitude", "longitude"), attrs={"units": "m s**-1"})
    parsed = ds.metpy.parse_cf()
    return metpy.calc.divergence(parsed.u, parsed.v).metpy.dequantify().values


def hurricane(cfg, opts, tier, output):
    path = data.folder(opts["data_root"], "case_study_hurricane", tier) / data.filename(
        data.times(cfg, "case_study_hurricane", tier)[0]
    )
    data.verify_file(
        path, cfg, "case_study_hurricane", data.times(cfg, "case_study_hurricane", tier)[0]
    )
    with xr.open_dataset(path) as ds:
        lat, lon = ds.latitude.values, ds.longitude.values
        u = ds.u850.isel(time=0).values
        v = ds.v850.isel(time=0).values
    left, right, bottom, top = cfg["datasets"]["case_study_hurricane"]["region"]
    selection = np.ix_((lat >= bottom) & (lat <= top), (lon >= left) & (lon <= right))

    def crop(array):
        return array[selection]

    d = divergence(u, v, lat, lon)
    arrays = dict(
        latitude=lat,
        longitude=lon,
        original_u=u,
        original_d=d,
    )
    curves, matches = [], []
    for method in cfg["compressors"]:
        for target in cfg["range_relative_error_targets"]:
            ru = compressors.roundtrip(u, method, float(np.ptp(u)) * target, cfg)
            rv = compressors.roundtrip(v, method, float(np.ptp(v)) * target, cfg)
            du = ru["decoded"]
            dd = divergence(du, rv["decoded"], lat, lon)
            u_error = metrics.distortion(u, du, float(np.ptp(u)) * target)
            v_error = metrics.distortion(v, rv["decoded"], float(np.ptp(v)) * target)
            curves.append(
                dict(
                    method=method,
                    target=target,
                    cr=u.nbytes / ru["size"],
                    divergence_cr=(u.nbytes + v.nbytes) / (ru["size"] + rv["size"]),
                    ssim_u=metrics.dssim(crop(u), crop(du)),
                    ssim_d=metrics.dssim(crop(d), crop(dd)),
                    u_error=u_error,
                    v_error=v_error,
                    verified=u_error["verified"] and v_error["verified"],
                    accepted=metrics.accepts_bounds(u_error) and metrics.accepts_bounds(v_error),
                )
            )
        for ratio in cfg["error_statistics"]["ratios"]:
            du, row, trials = match_ratio(u, method, ratio, cfg)
            dv, other, v_trials = match_ratio(v, method, ratio, cfg)
            dd = divergence(du, dv, lat, lon)
            arrays[f"{method}_{ratio}_u"] = du
            arrays[f"{method}_{ratio}_d"] = dd
            matches.append(
                dict(
                    **row,
                    ssim_u=metrics.dssim(crop(u), crop(du)),
                    ssim_d=metrics.dssim(crop(d), crop(dd)),
                    v_verified=other["verified"],
                    v_error=other,
                    v_accepted=metrics.accepts_bounds(other),
                    divergence_cr=(u.nbytes + v.nbytes) / (row["size"] + other["size"]),
                )
            )
            save(output / "searches" / f"{method}_{ratio}.json", dict(u=trials, v=v_trials))
        print("Measured Iota", method, flush=True)
    np.savez_compressed(output / "arrays.npz", **arrays)
    save(output / "measurements.json", dict(curves=curves, matches=matches))
    save(
        output / "validation.json",
        dict(
            complete=True,
            bounds_accepted=all(r["accepted"] for r in curves)
            and all(metrics.accepts_bounds(r) and r["v_accepted"] for r in matches),
            outside_tolerance=sum(not r["accepted"] for r in curves)
            + sum(not metrics.accepts_bounds(r) for r in matches)
            + sum(not r["v_accepted"] for r in matches),
            measurements=len(curves) + len(matches),
            bound_violations=sum(not r["verified"] for r in curves + matches)
            + sum(not r["v_verified"] for r in matches),
            tier=tier,
        ),
    )
