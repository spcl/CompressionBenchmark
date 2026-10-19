"""Small, explicit YAML configuration; no executable expressions or hidden overrides."""

import os
from pathlib import Path

import yaml


class UniqueLoader(yaml.SafeLoader):
    pass


def mapping(loader, node):
    result = {}
    for key, value in node.value:
        key = loader.construct_object(key)
        if key in result:
            raise ValueError(f"Duplicate YAML key: {key}")
        result[key] = loader.construct_object(value)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)


def read(path):
    try:
        with Path(path).open() as handle:
            value = yaml.load(handle, Loader=UniqueLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def check_keys(value, allowed, where):
    extra = set(value) - set(allowed)
    if extra:
        raise ValueError(f"Unknown {where} keys: {sorted(extra)}")


def load(path, profile=None):
    cfg = read(path)
    check_keys(
        cfg,
        "schema_version compressors range_relative_error_targets random_seed hpez datasets kappa_study quantization_residual error_statistics weak_scaling residual_layer_ablation".split(),
        "benchmark",
    )
    if cfg.get("schema_version") != 1:
        raise ValueError("Expected schema_version: 1")
    from compbench.compression.compressors import NAMES

    if (
        not cfg["compressors"]
        or len(set(cfg["compressors"])) != len(cfg["compressors"])
        or set(cfg["compressors"]) - set(NAMES)
    ):
        raise ValueError(f"compressors must be unique names from {NAMES}")
    import math

    for name in ("kappa_study", "quantization_residual"):
        allowed = ["pressure_levels", "range_relative_error_targets"]
        if name == "kappa_study":
            allowed.append("kappa_values")
        check_keys(cfg[name], allowed, name)
    for target in (
        cfg["range_relative_error_targets"]
        + cfg["quantization_residual"]["range_relative_error_targets"]
        + cfg["kappa_study"]["range_relative_error_targets"]
    ):
        if not isinstance(target, (int, float)) or not math.isfinite(target) or not 0 < target <= 1:
            raise ValueError("Relative error bounds must be finite fractions in (0, 1]")
    kappas = cfg["kappa_study"]["kappa_values"]
    if (
        not kappas
        or len(set(kappas)) != len(kappas)
        or any(not isinstance(k, (int, float)) or not math.isfinite(k) or k <= 0 for k in kappas)
    ):
        raise ValueError("kappa_values must be unique, finite positive numbers")
    check_keys(cfg["error_statistics"], ["pressure_levels", "ratios"], "error_statistics")
    check_keys(
        cfg["datasets"]["energy_budget_closure"],
        ["full", "sample"],
        "energy_budget_closure",
    )
    options = dict(
        workers=1, data_root="./data/prepared", output_root="./results", scratch_root="./scratch"
    )
    if profile:
        update = read(profile)
        check_keys(update, [*options, "scheduler", "runtime"], "execution profile")
        options.update(update)
    if not isinstance(options["workers"], int) or options["workers"] < 1:
        raise ValueError("workers must be a positive integer")
    for key in ("data_root", "output_root", "scratch_root"):
        options[key] = str(Path(os.path.expandvars(options[key])).expanduser().resolve())
    return cfg, options
