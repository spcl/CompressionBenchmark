"""Select ERA5 before downloading; write one timestep at a time."""

from pathlib import Path

import numpy as np
import xarray as xr

from compbench.datasets import field_bytes, global_grid

# Native ARCO ERA5 grid and variable dimensions, independent of benchmark selection.
NATIVE_GRID_DEGREES = 0.25
NATIVE_LATITUDE, NATIVE_LONGITUDE = global_grid(NATIVE_GRID_DEGREES)
NATIVE_LATITUDE = NATIVE_LATITUDE[::-1]
PRESSURE_VARIABLES = frozenset(
    (
        "geopotential",
        "temperature",
        "specific_humidity",
        "u_component_of_wind",
        "v_component_of_wind",
        "vertical_velocity",
    )
)
SURFACE_VARIABLES = frozenset(
    (
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_temperature",
        "mean_sea_level_pressure",
        "surface_pressure",
    )
)
# Method studies exclude humidity at pressures greater than or equal to this cutoff.
METHOD_HUMIDITY_PRESSURE_CUTOFF_HPA = 250


def include_field(cfg, workflow, name, level):
    if (
        workflow not in ("kappa_study", "quantization_residual", "error_statistics")
        or level is None
    ):
        return True
    return level in cfg[workflow]["pressure_levels"] and not (
        workflow != "error_statistics"
        and name == "q"
        and level >= METHOD_HUMIDITY_PRESSURE_CUTOFF_HPA
    )


def configured_fields(cfg, workflow):
    """List variable/level pairs without opening or downloading input files."""
    if workflow == "case_study_hurricane":
        return [
            (name, level)
            for name, levels in cfg["datasets"][workflow]["variables"].items()
            for level in (levels or [None])
        ]
    result = []
    for name, source in cfg["datasets"]["variables"].items():
        if source in PRESSURE_VARIABLES:
            levels = cfg["datasets"]["levels"]
        elif source in SURFACE_VARIABLES:
            levels = [None]
        else:
            raise ValueError(f"Unknown ERA5 variable dimensions: {source}")
        result.extend(
            (name, level) for level in levels if include_field(cfg, workflow, name, level)
        )
    return result


def input_bytes(cfg, dataset, tier):
    # Method studies share the complete rate-distortion input timestep.
    inventory = "case_study_hurricane" if dataset == "case_study_hurricane" else "rate_distortion"
    return (
        len(configured_fields(cfg, inventory))
        * field_bytes(NATIVE_GRID_DEGREES)
        * len(times(cfg, dataset, tier))
    )


def times(cfg, dataset, tier):
    if dataset == "weak_scaling":
        count = max(cfg["weak_scaling"]["sample_ranks" if tier == "sample" else "ranks"])
        start = np.datetime64(cfg["datasets"]["rate_distortion"]["sample"][0], "ns")
        return list(start + np.arange(count) * np.timedelta64(1, "h"))
    if dataset == "case_study_hurricane":
        return [np.datetime64(cfg["datasets"]["case_study_hurricane"]["time"], "ns")]
    if (
        dataset
        in ("error_statistics", "kappa_study", "quantization_residual", "sequential_throughput")
        or tier == "sample"
    ):
        return [np.datetime64(t, "ns") for t in cfg["datasets"]["rate_distortion"]["sample"]]
    spec = cfg["datasets"]["rate_distortion"]["full"]
    return list(
        np.arange(
            np.datetime64(spec["start"], "ns"),
            np.datetime64(spec["stop"], "ns"),
            np.timedelta64(spec["step_hours"], "h"),
        )
    )


def filename(time):
    return np.datetime_as_string(time, unit="h").replace("-", "").replace(":", "") + ".nc"


def folder(root, dataset, tier):
    if dataset in (
        "error_statistics",
        "kappa_study",
        "quantization_residual",
        "sequential_throughput",
    ):
        return Path(root) / "rate_distortion" / tier
    return Path(root) / dataset / tier


def fetch(cfg, root, dataset, tier, source=None):
    if dataset not in (
        "rate_distortion",
        "error_statistics",
        "kappa_study",
        "quantization_residual",
        "sequential_throughput",
        "case_study_hurricane",
        "weak_scaling",
    ):
        raise ValueError(
            "Ensemble inputs require the CDS preparation command or a local mirror; see README."
        )
    source = source or cfg["datasets"]["source"]
    opts = {"storage_options": {"token": "anon"}} if source.startswith("gs://") else {}
    output = folder(root, dataset, tier)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    total = 0
    with xr.open_zarr(source, chunks=None, consolidated=True, **opts) as remote:
        aliases = cfg["datasets"]["variables"]
        selected = (
            cfg["datasets"]["case_study_hurricane"]["variables"]
            if dataset == "case_study_hurricane"
            else aliases
        )
        for time in times(cfg, dataset, tier):
            path = output / filename(time)
            paths.append(str(path))
            if path.exists():
                total += verify_file(path, cfg, dataset, time)
                continue
            fields = {}
            for short in selected:
                print(f"Fetching {time} {short}", flush=True)
                field = remote[aliases[short]].sel(time=[time])
                if "level" in field.dims:
                    levels = (
                        selected[short]
                        if dataset == "case_study_hurricane"
                        else cfg["datasets"]["levels"]
                    )
                    field = field.sel(level=levels)
                field = field.sortby("latitude", ascending=False).load().astype("float32")
                if not np.isfinite(field.values).all():
                    raise ValueError(f"Nonfinite source values for {short} at {time}: {source}")
                if dataset == "case_study_hurricane" and "level" in field.dims:
                    for level in levels:
                        fields[f"{short}{level}"] = field.sel(level=level, drop=True)
                else:
                    fields[short] = field
            ds = xr.Dataset(fields)
            ds.attrs["source"] = source
            ds.attrs["tier"] = tier
            # NetCDF preserves an explicit source, date and coordinate inventory.
            tmp = path.with_suffix(".partial.nc")
            ds.to_netcdf(tmp, engine="h5netcdf")
            tmp.replace(path)
            total += verify_file(path, cfg, dataset, time)
    return dict(dataset=dataset, tier=tier, files=paths, uncompressed_bytes=total)


def verify_file(path, cfg, dataset, time):
    with xr.open_dataset(path) as ds:
        if list(ds.time.values) != [time]:
            raise ValueError(f"Wrong timestamp in {path}")
        if ds.sizes.get("latitude") != len(NATIVE_LATITUDE) or ds.sizes.get("longitude") != len(
            NATIVE_LONGITUDE
        ):
            raise ValueError(
                f"Expected the native {len(NATIVE_LATITUDE)} × {len(NATIVE_LONGITUDE)} grid in {path}"
            )
        if not np.array_equal(ds.latitude, NATIVE_LATITUDE) or not np.array_equal(
            ds.longitude, NATIVE_LONGITUDE
        ):
            raise ValueError(f"Incorrect native grid coordinates in {path}")
        needed = cfg["datasets"]["variables"]
        if dataset == "case_study_hurricane":
            needed = [
                f"{name}{level}"
                for name, levels in cfg["datasets"]["case_study_hurricane"]["variables"].items()
                for level in (levels or [""])
            ]
        if set(needed) - set(ds.data_vars):
            raise ValueError(f"Missing variables in {path}: {set(needed) - set(ds.data_vars)}")
        for name in needed:
            field = ds[name]
            if field.dtype != np.float32:
                raise ValueError(f"{path}/{name} must be float32")
            if (
                "level" in field.dims
                and dataset != "case_study_hurricane"
                and list(field.level.values) != cfg["datasets"]["levels"]
            ):
                raise ValueError(f"Pressure levels differ from the configured selection in {path}")
        return sum(ds[name].nbytes for name in needed)


def fields(cfg, root, workflow, tier, timestamps=None):
    """Each task owns one horizontal field and all its compression settings."""
    result = []
    for time in times(cfg, workflow, tier) if timestamps is None else timestamps:
        path = folder(root, workflow, tier) / filename(time)
        verify_file(path, cfg, workflow, time)
        with xr.open_dataset(path) as ds:
            for name in cfg["datasets"]["variables"]:
                if name not in ds:
                    continue
                levels = list(ds[name].level.values) if "level" in ds[name].dims else [None]
                for level in levels:
                    if not include_field(cfg, workflow, name, level):
                        continue
                    result.append(
                        dict(
                            path=str(path),
                            variable=name,
                            level=None if level is None else int(level),
                            id=f"{path.stem}_{name}"
                            + (f"_{int(level)}" if level is not None else ""),
                        )
                    )
    return result


def load_field(field):
    with xr.open_dataset(field["path"]) as ds:
        data = ds[field["variable"]].isel(time=0)
        if field["level"] is not None:
            data = data.sel(level=field["level"])
        return np.ascontiguousarray(
            data.transpose("latitude", "longitude").values, dtype=np.float32
        )
