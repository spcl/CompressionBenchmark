"""Distortion metrics and aggregation (float64 error accumulation)."""

import numpy as np

# Acceptance limits approved for this benchmark; exact verification stays separate.
MAX_VIOLATION_FRACTION = 0.001
MAX_BOUND_FRACTION = 1.01


def accepts_bounds(row):
    """Legacy rows without sparsity evidence can only pass exact verification."""
    if row["verified"]:
        return True
    fraction = row.get("bound_violation_fraction")
    ratio = row.get("max_bound_fraction")
    return bool(
        fraction is not None
        and ratio is not None
        and 0 <= fraction <= MAX_VIOLATION_FRACTION
        and 0 <= ratio <= MAX_BOUND_FRACTION
    )


def bound_statistics(error, bounds):
    """Apply both limits independently to each horizontal slab, including 3D data."""
    absolute = np.abs(np.asarray(error, dtype=np.float64))
    bounds = np.asarray(bounds, dtype=np.float64)
    if not np.isfinite(absolute).all() or not np.isfinite(bounds).all() or np.any(bounds < 0):
        raise ValueError("Errors and bounds must be finite, with nonnegative bounds")
    violations = absolute > bounds
    fractions = np.mean(violations, axis=(-2, -1))
    ratios = np.divide(absolute, bounds, out=np.zeros_like(absolute), where=bounds > 0)
    zero_failure = bool(np.any((bounds == 0) & (absolute > 0)))
    row = dict(
        verified=not bool(np.any(violations)),
        bound_violation_count=int(np.count_nonzero(violations)),
        bound_violation_fraction=float(np.max(fractions)),
        max_bound_fraction=None if zero_failure else float(ratios.max()),
    )
    row["accepted"] = accepts_bounds(row)
    return row


def distortion(original, decoded, bound):
    if original.shape != decoded.shape or not np.isfinite(decoded).all():
        raise ValueError("Decoded shape or finite values changed")
    error = decoded.astype(np.float64) - original.astype(np.float64)
    maximum = float(np.abs(error).max())
    mse = float(np.mean(error**2))
    span = float(original.max()) - float(original.min())
    normalized = mse / span**2 if span else mse
    return dict(
        max_abs_error=maximum,
        bound=float(bound),
        **bound_statistics(error, bound),
        rmse=float(np.sqrt(mse)),
        normalized_mse=normalized,
        psnr=None if normalized == 0 else float(-10 * np.log10(normalized)),
    )


def dssim(original, decoded, factor=1):
    import warnings

    import xarray as xr
    from skimage.transform import downscale_local_mean

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import ldcpy
    if factor != 1:
        original = downscale_local_mean(original.astype(float), (factor, factor))
        decoded = downscale_local_mean(decoded.astype(float), (factor, factor))
    if min(original.shape) < 11:
        raise ValueError("Data SSIM needs at least 11 cells on each horizontal axis")
    coords = dict(
        lat=xr.DataArray(
            np.arange(original.shape[0]),
            dims="lat",
            attrs={"units": "degrees_north", "standard_name": "latitude"},
        ),
        lon=xr.DataArray(
            np.arange(original.shape[1]),
            dims="lon",
            attrs={"units": "degrees_east", "standard_name": "longitude"},
        ),
    )
    arrays = [
        xr.DataArray(x.astype(float), dims=("lat", "lon"), coords=coords)
        for x in (original, decoded)
    ]
    value = float(ldcpy.Diffcalcs(*arrays, "cam-fv", ["lat", "lon"]).get_diff_calc("ssim_fp"))
    if not np.isfinite(value):
        if np.array_equal(original, decoded):
            return 1.0
        raise ValueError("Data SSIM is nonfinite")
    return value


def aggregate(rows, *, mean_db=False):
    if not rows:
        raise ValueError("Cannot aggregate empty measurements")
    cr = 1 / np.mean([r["size"] / r["nbytes"] for r in rows])
    nmse = np.mean([r["normalized_mse"] for r in rows])
    psnr = (
        np.mean([r["psnr"] for r in rows])
        if mean_db and all(r["psnr"] is not None for r in rows)
        else (None if nmse == 0 else -10 * np.log10(nmse))
    )
    result = dict(cr=float(cr), psnr=None if psnr is None else float(psnr), count=len(rows))
    if "ssim" in rows[0]:
        result["ssim"] = float(np.mean([r["ssim"] for r in rows]))
    return result


def statistics_summary(rows):
    """Accumulate yearly quality curves without holding every measurement in memory."""
    sums = {}
    for row in rows:
        field = row["field"]
        if field["variable"] == "q" and field["level"] <= 250:
            continue
        key = row["method"], row["target"]
        total = sums.setdefault(key, np.zeros(4))
        total += [1, row["size"] / row["nbytes"], row["normalized_mse"], row["ssim"]]
    return {
        key: dict(
            cr=n / fraction, psnr=None if mse == 0 else -10 * np.log10(mse / n), ssim=ssim / n
        )
        for key, (n, fraction, mse, ssim) in sums.items()
    }
