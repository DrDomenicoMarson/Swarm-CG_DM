"""Circular-angle utilities shared by scoring and force-field code."""

import numpy as np


def normalize_periodic_degrees(value: float) -> float:
    """Normalize an angle in degrees to the half-open interval [-180, 180).

    Args:
        value: Angle in degrees.

    Returns:
        Equivalent normalized angle.
    """
    return (float(value) + 180.0) % 360.0 - 180.0


def circular_mean_degrees(values: np.ndarray) -> float:
    """Calculate the circular mean of angles in degrees.

    Args:
        values: Angles in degrees.

    Returns:
        Circular mean in the interval ``[-180, 180)``.

    Raises:
        ValueError: If no finite values are supplied or the mean direction is
            undefined.
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Cannot calculate a circular mean without finite angles.")
    vector = np.mean(np.exp(1j * np.deg2rad(finite)))
    if np.isclose(abs(vector), 0.0):
        raise ValueError("Circular mean is undefined for this angle distribution.")
    return normalize_periodic_degrees(np.rad2deg(np.angle(vector)))


def unwrap_degrees_around(values: np.ndarray, center: float) -> np.ndarray:
    """Unwrap angles into the shortest coordinates around a reference center.

    Args:
        values: Angles in degrees.
        center: Reference angle in degrees.

    Returns:
        Unwrapped angles centered near *center*. Values may fall outside the
        conventional serialized interval.
    """
    values_arr = np.asarray(values, dtype=float)
    offsets = (values_arr - center + 180.0) % 360.0 - 180.0
    return center + offsets
