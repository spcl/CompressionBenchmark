"""Prepare LAGRANTO inputs and measure particle trajectories and densities."""

import datetime as dt
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

from compbench.compression import compressors
from compbench.compression.reconstruction import reconstruct
from compbench.datasets import ensemble
from compbench.datasets.era5 import filename


def write_lagranto(ds, path, time, start):
    import netCDF4

    ds = ds.sortby("latitude").sortby("level")
    with netCDF4.Dataset(path, "w", format="NETCDF4_CLASSIC") as out:
        for name, length in [
            ("time", 1),
            ("lev", ds.sizes["level"]),
            ("nhym", ds.sizes["level"]),
            ("lat", ds.sizes["latitude"]),
            ("lon", ds.sizes["longitude"]),
        ]:
            out.createDimension(name, length)
        for name, values, units in [
            (
                "time",
                [float((time - start) / np.timedelta64(1, "h"))],
                f"hours since {np.datetime_as_string(start, unit='s').replace('T', ' ')}",
            ),
            ("lev", ds.level.values, "hPa"),
            ("lat", ds.latitude.values, "degrees_north"),
            ("lon", ds.longitude.values, "degrees_east"),
        ]:
            variable = out.createVariable(name, "f8" if name == "time" else "f4", (name,))
            variable[:] = values
            variable.units = units
            variable.standard_name = dict(
                time="time", lev="air_pressure", lat="latitude", lon="longitude"
            )[name]
        # The public CDO reader requires hybrid coefficients even on pressure levels.
        for name, values, units in [("hyam", ds.level.values, "hPa"), ("hybm", 0, "1")]:
            variable = out.createVariable(name, "f4", ("nhym",))
            variable[:] = values
            variable.units = units
        for source, name in [("u", "U"), ("v", "V"), ("w", "OMEGA"), ("sp", "PS")]:
            dims = ("time", "lev", "lat", "lon") if source != "sp" else ("time", "lat", "lon")
            variable = out.createVariable(name, "f4", dims)
            values, units = ds[source].values, ds[source].attrs["units"]
            if source == "sp":
                if units == "Pa":
                    values = values * np.float32(0.01)
                elif units != "hPa":
                    raise ValueError(f"Unsupported surface pressure units: {units}")
                units = "hPa"
            variable[:] = values
            variable.units = units


def write_parameters(root, times, particles, step_hours):
    """Write the public caltra parameter file and three-column initial positions."""
    if particles.shape[1] != 4 or np.any(particles[:, 0] != 0):
        raise ValueError("Initial particles must contain zero time offsets, lon, lat, pressure")
    stamps = [np.datetime_as_string(t, unit="h").replace("-", "").replace("T", "_") for t in times]
    start = dt.datetime.strptime(stamps[0], "%Y%m%d_%H")
    interval = step_hours * 60
    duration = int((times[-1] - times[0]) / np.timedelta64(1, "m"))
    np.savetxt(root / "positions", particles[:, 1:], fmt="%.8g")
    # Insertion order follows the public solver's caltra.param reader.
    parameters = dict(
        direction=1,  # forward
        input_file_count=len(stamps),
        input_dates="\n".join(stamps),
        input_step_hours=step_hours,
        longitude_period_degrees=360,
        positions_file="positions",
        particle_count=len(particles),
        trajectory_column_count=4,  # time, longitude, latitude, pressure
        output_file="trajectory.lsl.4",
        integration_step_minutes=interval // 12,
        start_shift_hhmm=0,
        end_shift_hhmm=0,
        output_step_minutes=interval,
        ground_jump=1,  # ground-touching trajectories reenter the atmosphere
        vertical_velocity_factor=1,
        reference_year=start.year,
        reference_month=start.month,
        reference_day=start.day,
        reference_hour=start.hour,
        reference_minute=0,
        duration_minutes=duration,
        check_input_times="yes",
        isentropic_mode="no 0",  # disabled; no potential-temperature field on S files
        model_level_mode="no",
    )
    (root / "caltra.param").write_text("\n".join(map(str, parameters.values())) + "\n")


def trajectory_member(cfg, opts, tier, member, output, method=None, target=None):
    spec, chosen, times = ensemble.selection(cfg, "trajectory_simulation", tier)
    program = compressors.executable("caltra")
    particles = np.loadtxt(Path(opts["data_root"]) / spec["initial_particles_file"])
    if particles.shape != (35298, 4):
        raise ValueError(f"Expected 35,298 particles; found {particles.shape}")
    particles = particles[:: chosen["particle_stride"]]
    measurements = []
    with tempfile.TemporaryDirectory(dir=opts["scratch_root"], prefix="trajectory-") as directory:
        root = Path(directory)
        for time in times:
            path = (
                Path(opts["data_root"])
                / "trajectory_simulation"
                / tier
                / f"member_{member}"
                / filename(time)
            )
            with xr.open_dataset(path) as handle:
                ds = handle.load()
            ds, rows = reconstruct(ds, method, target, cfg)
            measurements.extend(rows)
            stamp = np.datetime_as_string(time, unit="h").replace("-", "").replace("T", "_")
            write_lagranto(ds, root / ("P" + stamp), time, times[0])
        write_parameters(root, times, particles, spec["step_hours"])
        result = subprocess.run([program], cwd=root, capture_output=True, text=True, timeout=86400)
        (output / "caltra.log").write_text(result.stdout + "\n" + result.stderr)
        if result.returncode or not (root / "trajectory.lsl.4").exists():
            raise RuntimeError(f"LAGRANTO failed; see {output}/caltra.log")
        with xr.open_dataset(root / "trajectory.lsl.4", decode_times=False) as handle:
            trajectories = handle.load()
    for name in ("lon", "lat", "p"):
        values = trajectories[name].values
        if not np.isfinite(values).all() or np.any(values == -999):
            raise ValueError(f"Invalid LAGRANTO {name} coordinates")
    lon = np.atleast_2d(trajectories.lon.squeeze().values) % 360
    lat = np.atleast_2d(trajectories.lat.squeeze().values)
    if lon.shape[1] != len(particles) or lon.shape[0] != chosen["hours"] // spec["step_hours"] + 1:
        raise ValueError("LAGRANTO changed particle or output-time coverage")
    left, right, bottom, top = spec["density_region"]
    bins = spec["density_bins"]
    density = np.array(
        [
            np.histogram2d(a, b, bins=bins, range=[[left, right], [bottom, top]])[0]
            for a, b in zip(lon, lat)
        ]
    )
    trajectories.to_netcdf(output / "trajectories.nc")
    return xr.Dataset(dict(density=(("time", "x", "y"), density))), measurements
