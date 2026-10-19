"""Datasets for the compression benchmark."""

import numpy as np

FLOAT32_BYTES = np.dtype("float32").itemsize
LATITUDE_LIMIT_DEGREES = 90
LONGITUDE_PERIOD_DEGREES = 360


def global_grid(spacing_degrees):
    """Coordinates of a regular global grid, including both poles."""
    latitude = np.arange(
        -LATITUDE_LIMIT_DEGREES,
        LATITUDE_LIMIT_DEGREES + spacing_degrees / 2,
        spacing_degrees,
    )
    longitude = np.arange(0, LONGITUDE_PERIOD_DEGREES, spacing_degrees)
    return latitude, longitude


def field_bytes(spacing_degrees):
    latitude, longitude = global_grid(spacing_degrees)
    return len(latitude) * len(longitude) * FLOAT32_BYTES


def format_bytes(size):
    """Format uncompressed input sizes in decimal units."""
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(size)
    for unit in units:
        if value < 1000 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1000
