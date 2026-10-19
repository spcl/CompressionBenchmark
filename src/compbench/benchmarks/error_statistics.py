"""Measure error histograms and spherical spectra at matched ratios."""

import numpy as np

from compbench.compression.ratio_search import match_ratio
from compbench.results import save


def histogram_edges(value_range, reference, target_bins=80):
    """Binary-lattice-aware bins, centered at zero."""
    low, high = value_range
    if high <= low:
        return np.linspace(low - 0.5, high + 0.5, target_bins + 1)
    minimum = (high - low) / target_bins
    distinct = np.unique(reference[np.isfinite(reference)])
    if distinct.size > 1:
        minimum = max(minimum, np.median(np.diff(distinct.astype(np.float64))))
    width = 2.0 ** np.ceil(np.log2(minimum))
    return (np.arange(np.floor(low / width), np.ceil(high / width) + 2) - 0.5) * width


def spectrum(x):
    import pyshtools

    coefficients = pyshtools.expand.SHExpandDH(x[:-1].astype(float), sampling=2)
    return pyshtools.spectralanalysis.spectrum(coefficients, unit="per_l")


def measure_histograms(x, field, cfg, output):
    rows = []
    curves = dict(reference_spectrum=spectrum(x), quantization_bins=True)
    for ratio in cfg["error_statistics"]["ratios"]:
        errors = {}
        for method in cfg["compressors"]:
            decoded, row, trials = match_ratio(x, method, ratio, cfg, tolerance=0.0001)
            rows.append(row)
            errors[method] = x - decoded
            curves[f"{method}_{ratio}_spectrum"] = spectrum(decoded)
            save(output / "searches" / f"{field['id']}_{method}_{ratio}.json", trials)
        limit = float(np.max(np.abs(errors["ebcc"])))
        edges = histogram_edges((-limit, limit), x)
        curves[f"edges_{ratio}"] = edges
        for method, error in errors.items():
            curves[f"{method}_{ratio}_histogram"] = np.histogram(error, bins=edges, density=True)[0]
    (output / "curves").mkdir(exist_ok=True)
    np.savez_compressed(output / "curves" / f"{field['id']}.npz", **curves)
    return rows
