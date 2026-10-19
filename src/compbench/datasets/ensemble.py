"""CDS ensemble acquisition and canonical member/timestep preparation."""

import glob
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import xarray as xr

from compbench.datasets import (
    LONGITUDE_PERIOD_DEGREES,
    field_bytes,
    global_grid,
)
from compbench.datasets.era5 import filename
from compbench.results import save

# Fixed energy fields use Table 3 short names; CDS request identifiers stay here.
ENERGY_PRESSURE = dict(
    z="geopotential", t="temperature", q="specific_humidity", v="v_component_of_wind"
)
ENERGY_FLUXES = dict(
    rsnt="mean_top_net_short_wave_radiation_flux",
    rlnt="mean_top_net_long_wave_radiation_flux",
    rlns="mean_surface_net_long_wave_radiation_flux",
    rsns="mean_surface_net_short_wave_radiation_flux",
    hfss="mean_surface_sensible_heat_flux",
    hfls="mean_surface_latent_heat_flux",
)
ALIASES = dict(
    mtnswrf="rsnt",
    mtnlwrf="rlnt",
    msnlwrf="rlns",
    msnswrf="rsns",
    msshf="hfss",
    mslhf="hfls",
    avg_tnswrf="rsnt",
    avg_tnlwrf="rlnt",
    avg_snlwrf="rlns",
    avg_snswrf="rsns",
    avg_ishf="hfss",
    avg_slhtf="hfls",
)


def variables(cfg, workflow):
    if workflow == "energy_budget_closure":
        return dict(ENERGY_PRESSURE), dict(ENERGY_FLUXES)
    spec = cfg["datasets"][workflow]
    pressure = {n: cfg["datasets"]["variables"][n] for n in spec["pressure_variables"]}
    surface = {n: {"sp": "surface_pressure"}[n] for n in spec["surface_variables"]}
    return pressure, surface


def verify_energy_variables(ds):
    missing = (ENERGY_PRESSURE.keys() | ENERGY_FLUXES.keys()) - ds.data_vars.keys()
    if missing:
        raise ValueError(
            f"Missing energy variables {sorted(missing)}; fetch/prepare mean-flux inputs; accumulated inputs are unsupported"
        )
    for name in ENERGY_FLUXES:
        if ds[name].attrs.get("units") not in ("W m**-2", "W m-2", "W/m^2", "W m^-2", "W/m²"):
            raise ValueError(f"{name} must contain mean flux in W/m²; fetch mean-flux inputs")


def selection(cfg, workflow, tier):
    spec = cfg["datasets"][workflow]
    chosen = spec[tier]
    start = np.datetime64(chosen.get("start", spec.get("start")), "ns")
    step = chosen.get("step_hours", spec.get("step_hours"))
    stop = (
        np.datetime64(chosen["stop"], "ns")
        if "stop" in chosen
        else start + np.timedelta64(chosen["hours"] + step, "h")
    )
    return spec, chosen, list(np.arange(start, stop, np.timedelta64(step, "h")))


def input_bytes(cfg, workflow, tier):
    spec, chosen, times = selection(cfg, workflow, tier)
    pressure, surface = variables(cfg, workflow)
    fields = len(pressure) * len(cfg["datasets"]["levels"]) + len(surface)
    grid = chosen.get("grid_degrees", spec.get("grid_degrees"))
    return len(chosen["members"]) * len(times) * fields * field_bytes(grid)


def requests(cfg, workflow, tier):
    spec, chosen, times = selection(cfg, workflow, tier)
    days = sorted({np.datetime_as_string(t, unit="D") for t in times})
    for day in days:
        month = day[:7]
        selected = [np.datetime_as_string(t, unit="m") for t in times if str(t).startswith(day)]
        pressure, surface = variables(cfg, workflow)
        for kind, names in [("pressure-levels", pressure), ("single-levels", surface)]:
            request = dict(
                product_type=["ensemble_members"],
                year=[month[:4]],
                month=[month[5:7]],
                day=sorted({t[8:10] for t in selected}),
                time=sorted({t[11:16] for t in selected}),
                variable=list(names.values()),
                number=[str(n) for n in chosen["members"]],
                data_format="netcdf",
                download_format="unarchived",
                grid=[chosen.get("grid_degrees", spec.get("grid_degrees"))] * 2,
            )
            if kind == "pressure-levels":
                request["pressure_level"] = [str(n) for n in cfg["datasets"]["levels"]]
            yield day + "-" + kind, "reanalysis-era5-" + kind, request


def normalize(ds):
    renames = {
        "valid_time": "time",
        "pressure_level": "level",
        "number": "member",
        "lev": "level",
        "lat": "latitude",
        "lon": "longitude",
        "temperature": "t",
        "specific_humidity": "q",
        "geopotential": "z",
        "u_component_of_wind": "u",
        "v_component_of_wind": "v",
        "vertical_velocity": "w",
        "U": "u",
        "V": "v",
        "OMEGA": "w",
        "PS": "sp",
        "surface_pressure": "sp",
        **{v: k for k, v in ENERGY_FLUXES.items()},
        **ALIASES,
    }
    ds = ds.rename({a: b for a, b in renames.items() if a in ds and b not in ds})
    return ds.sortby("latitude").sortby("level") if "level" in ds else ds.sortby("latitude")


def open_inputs(paths):
    datasets = []
    for path in paths:
        path = Path(path)
        if zipfile.is_zipfile(path):
            parts = path.parent / (path.stem + "-parts")
            parts.mkdir(exist_ok=True)
            with zipfile.ZipFile(path) as archive:
                for name in archive.namelist():
                    if name.endswith(".nc"):
                        target = parts / Path(name).name
                        if not target.exists():
                            target.write_bytes(archive.read(name))
                        datasets.append(normalize(xr.open_dataset(target, chunks={})))
        else:
            match = re.fullmatch(r"P(\d{4})-?(\d{2})(\d{2})_(\d{2})", path.name)
            if match:
                ds = xr.open_dataset(path, decode_times=False, chunks={})
                original = ds
                year, month, day, hour = match.groups()
                expected = np.datetime64(f"{year}-{month}-{day}T{hour}:00:00", "ns")
                raw = np.asarray(ds.time.values).ravel()
                units = ds.time.attrs.get("units", "")
                epoch = float((expected - np.datetime64("1970-01-01")) / np.timedelta64(1, "s"))
                if raw.size != 1:
                    raise ValueError(f"Expected one timestep in {path}")
                if float(raw[0]) == epoch:
                    ds.attrs["time_coordinate_source"] = (
                        "P filename; legacy seconds since 1970 mislabeled as hours since 1900"
                    )
                    ds.attrs["original_time_units"] = units
                else:
                    decoded = xr.decode_cf(ds).time.values.ravel()[0]
                    if np.datetime64(str(decoded), "ns") != expected:
                        original.close()
                        raise ValueError(f"Time/filename mismatch in {path}")
                if "time" in ds.dims:
                    ds = ds.assign_coords(time=[expected])
                else:
                    ds = ds.drop_vars("time").expand_dims(time=[expected])
                ds = normalize(ds)
                if "member" not in ds.dims:
                    if not path.parent.name.startswith("member_"):
                        raise ValueError(f"Cannot identify member in {path}")
                    ds = ds.expand_dims(member=[int(path.parent.name.split("_")[-1])])
                ds.set_close(original.close)
                datasets.append(ds)
            else:
                datasets.append(normalize(xr.open_dataset(path, chunks={})))
    return datasets


def asset_path(name):
    source = Path(__file__).resolve().parents[3] / "assets" / name
    if not source.exists():
        source = Path(sys.prefix) / "share/compbench/assets" / name
    return source


def prepare(cfg, opts, workflow, tier, paths):
    spec, chosen, times = selection(cfg, workflow, tier)
    if workflow == "trajectory_simulation":
        target = Path(opts["data_root"]) / spec["initial_particles_file"]
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(asset_path("startf"), target)
    opened = open_inputs(paths)
    try:
        merged = xr.combine_by_coords(opened, combine_attrs="override", join="outer")
        pressure, surface = variables(cfg, workflow)
        required = [*pressure, *surface]
        if workflow == "energy_budget_closure":
            verify_energy_variables(merged)
        missing = set(required) - set(merged.data_vars)
        if missing:
            raise ValueError(f"Missing ensemble variables: {missing}")
        if "member" not in merged.dims:
            raise ValueError("Input is not an ensemble with a member axis")
        grid = chosen.get("grid_degrees", spec.get("grid_degrees"))
        latitude, longitude = global_grid(grid)
        merged = merged.assign_coords(longitude=merged.longitude % LONGITUDE_PERIOD_DEGREES).sortby(
            "longitude"
        )
        # Exact subsetting of the source grid; never interpolate pressure levels.
        merged = merged.sel(
            latitude=latitude, longitude=longitude, level=sorted(cfg["datasets"]["levels"])
        )
        for member in chosen["members"]:
            root = Path(opts["data_root"]) / workflow / tier / f"member_{member}"
            root.mkdir(parents=True, exist_ok=True)
            for time in times:
                path = root / filename(time)
                if path.exists():
                    if workflow != "energy_budget_closure":
                        continue
                    with xr.open_dataset(path) as existing:
                        try:
                            verify_energy_variables(normalize(existing))
                        except ValueError:
                            pass  # Replace legacy inputs from the downloaded mean fields.
                        else:
                            continue
                ds = (
                    merged[required]
                    .sel(member=member, time=[time], drop=True)
                    .astype("float32")
                    .load()
                )
                if any(not np.isfinite(ds[n]).all() for n in required):
                    raise ValueError(f"Nonfinite input at {member}/{time}")
                ds.attrs.update(
                    tier=tier, ensemble_member=member, source="CDS ERA5 ensemble members"
                )
                tmp = path.with_suffix(".partial.nc")
                ds.to_netcdf(tmp, engine="h5netcdf")
                tmp.replace(path)
    finally:
        for ds in opened:
            ds.close()
    return verify(cfg, opts, workflow, tier)


def verify(cfg, opts, workflow, tier, member=None):
    spec, chosen, times = selection(cfg, workflow, tier)
    total = 0
    count = 0
    pressure, surface = variables(cfg, workflow)
    required = [*pressure, *surface]
    grid = chosen.get("grid_degrees", spec.get("grid_degrees"))
    latitude, longitude = global_grid(grid)
    members = chosen["members"] if member is None else [member]
    for member in members:
        for time in times:
            path = Path(opts["data_root"]) / workflow / tier / f"member_{member}" / filename(time)
            with xr.open_dataset(path) as handle:
                ds = normalize(handle)
                if workflow == "energy_budget_closure":
                    verify_energy_variables(ds)
                if (
                    set(required) - set(ds.data_vars)
                    or list(ds.time.values) != [time]
                    or list(ds.level.values) != sorted(cfg["datasets"]["levels"])
                ):
                    raise ValueError(f"Incorrect ensemble variables/time/levels: {path}")
                if ds.sizes["latitude"] != len(latitude) or ds.sizes["longitude"] != len(longitude):
                    raise ValueError(f"Incorrect ensemble grid: {path}")
                if not np.array_equal(ds.latitude, latitude) or not np.array_equal(
                    ds.longitude, longitude
                ):
                    raise ValueError(f"Incorrect ensemble coordinates: {path}")
                if any(ds[n].dtype != np.float32 for n in required):
                    raise ValueError(f"Ensemble fields must be float32: {path}")
                total += sum(ds[n].nbytes for n in required)
                count += 1
    return dict(files=count, members=members, uncompressed_bytes=total, tier=tier)


def ensemble_data(cfg, opts, workflow, tier, action, source=None):
    if action == "verify":
        return verify(cfg, opts, workflow, tier)
    if action == "prepare":
        if not source:
            raise ValueError("Provide the ensemble NetCDF glob with --source")
        paths = sorted(glob.glob(source))
        if not paths:
            raise FileNotFoundError(source)
    else:
        import cdsapi

        client = cdsapi.Client(wait_until_complete=False)
        root = Path(opts["data_root"]) / workflow / tier / "downloads"
        root.mkdir(parents=True, exist_ok=True)
        paths = []
        for name, dataset, request in requests(cfg, workflow, tier):
            path = root / (name + ".nc")
            paths.append(path)
            receipt = root / (name + ".request.json")
            previous = json.loads(receipt.read_text()) if receipt.exists() else {}
            if previous and previous.get("request") != request:
                if path.exists():
                    raise ValueError(f"Download selection changed: remove or move {path}")
                previous = {}
            if path.exists():
                continue
            print(f"CDS {workflow}: {name}", flush=True)
            if previous.get("request_id"):
                remote = client.client.get_remote(previous["request_id"])
                if remote.status in ("failed", "dismissed", "cancelled"):
                    remote = None
            else:
                remote = None
            if remote is None:
                remote = client.retrieve(dataset, request)
                save(receipt, dict(dataset=dataset, request=request, request_id=remote.request_id))
            tmp = path.with_suffix(".partial")
            remote.download(str(tmp))
            tmp.replace(path)
    return prepare(cfg, opts, workflow, tier, paths)
