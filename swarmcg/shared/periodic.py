"""Circular-angle utilities shared by scoring and force-field code."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class CircularStatistics:
    """First circular moment of a set of angles.

    Attributes
    ----------
    mean_degrees
        Mean direction in ``[-180, 180)``, or ``None`` when the first moment
        has no defined direction.
    resultant_length
        Magnitude of the mean unit vector in the interval 0--1.
    """

    mean_degrees: Optional[float]
    resultant_length: float


@dataclass(frozen=True, order=True)
class CircularMoment:
    """Ordered circular moment of a set of angles.

    Args:
        order: Strictly positive circular-moment order.
        direction_degrees: Argument of ``<exp(i*order*phi)>`` in
            ``[-180, 180)``, or ``None`` when undefined. The direction is not
            divided by the order.
        resultant_length: Magnitude of the ordered mean vector in ``[0, 1]``.

    Raises:
        ValueError: If the order, direction, or resultant length is invalid.
    """

    order: int
    direction_degrees: Optional[float]
    resultant_length: float

    def __post_init__(self) -> None:
        """Validate the order, optional direction, and resultant length."""
        if (
            isinstance(self.order, bool)
            or not isinstance(self.order, (int, np.integer))
            or self.order <= 0
        ):
            raise ValueError("Circular-moment order must be a positive integer.")
        if self.direction_degrees is not None and not np.isfinite(
            self.direction_degrees
        ):
            raise ValueError("Circular-moment direction must be finite or None.")
        if (
            not np.isfinite(self.resultant_length)
            or not 0.0 <= self.resultant_length <= 1.0 + 1e-12
        ):
            raise ValueError("Circular-moment resultant length must lie in [0, 1].")
        object.__setattr__(self, "order", int(self.order))
        if self.direction_degrees is not None:
            object.__setattr__(
                self,
                "direction_degrees",
                normalize_periodic_degrees(self.direction_degrees),
            )
        object.__setattr__(self, "resultant_length", float(self.resultant_length))


@dataclass(frozen=True)
class PeriodicDihedralParameters:
    """Canonical GROMACS periodic-dihedral parameters.

    Args:
        phase_degrees: Phase in degrees.
        force_constant: Nonnegative force constant in kJ/mol.
        multiplicity: Strictly positive integer multiplicity.

    Raises:
        ValueError: If parameters are non-finite, the force constant is
            negative, or multiplicity is not a positive integer.
    """

    phase_degrees: float
    force_constant: float
    multiplicity: int

    def __post_init__(self) -> None:
        """Validate and normalize the canonical parameters."""
        if not np.isfinite(self.phase_degrees) or not np.isfinite(self.force_constant):
            raise ValueError("Periodic dihedral phase and force constant must be finite.")
        if self.force_constant < 0:
            raise ValueError("Canonical periodic dihedral force constants cannot be negative.")
        if (
            isinstance(self.multiplicity, bool)
            or not isinstance(self.multiplicity, (int, np.integer))
            or self.multiplicity <= 0
        ):
            raise ValueError("Periodic dihedral multiplicity must be a positive integer.")
        object.__setattr__(
            self, "phase_degrees", normalize_periodic_degrees(self.phase_degrees)
        )
        object.__setattr__(self, "force_constant", float(self.force_constant))
        object.__setattr__(self, "multiplicity", int(self.multiplicity))

    @classmethod
    def from_gromacs(
        cls, phase_degrees: float, force_constant: float, multiplicity: int
    ) -> "PeriodicDihedralParameters":
        """Canonicalize a standard GROMACS periodic-dihedral record.

        A negative force constant is made positive while the phase is shifted
        by 180 degrees. This preserves forces and changes only an additive
        energy constant.

        Args:
            phase_degrees: Serialized GROMACS phase in degrees.
            force_constant: Possibly negative GROMACS force constant.
            multiplicity: Positive integer multiplicity.

        Returns:
            Canonical nonnegative-force parameters.

        Raises:
            ValueError: If inputs are non-finite or multiplicity is invalid.
        """
        phase = float(phase_degrees)
        force = float(force_constant)
        if not np.isfinite(phase) or not np.isfinite(force):
            raise ValueError("Periodic dihedral phase and force constant must be finite.")
        if force < 0:
            phase += 180.0
            force = -force
        return cls(phase, force, multiplicity)

    def to_gromacs(self) -> tuple[float, float, int]:
        """Return canonical ``(phase, force_constant, multiplicity)`` values."""
        return self.phase_degrees, self.force_constant, self.multiplicity


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
    statistics = circular_statistics_degrees(values)
    if statistics.mean_degrees is None:
        raise ValueError("Circular mean is undefined for this angle distribution.")
    return statistics.mean_degrees


def circular_statistics_degrees(values: np.ndarray) -> CircularStatistics:
    """Calculate tolerant first-moment circular statistics in degrees.

    Args:
        values: Angles in degrees. Non-finite values are ignored.

    Returns:
        Mean direction and resultant length. The mean is ``None`` when the
        resultant direction is numerically undefined.

    Raises:
        ValueError: If no finite angle is supplied.
    """
    moment = circular_moment_degrees(values, order=1)
    return CircularStatistics(
        mean_degrees=moment.direction_degrees,
        resultant_length=moment.resultant_length,
    )


def circular_moment_degrees(values: np.ndarray, order: int) -> CircularMoment:
    """Calculate a tolerant ordered circular moment in degrees.

    Args:
        values: Angles in degrees. Non-finite values are ignored.
        order: Strictly positive circular-moment order.

    Returns:
        Ordered direction, resultant length, and order. The direction is
        ``None`` when the resultant is numerically undefined.

    Raises:
        ValueError: If ``order`` is invalid or no finite angle is supplied.
    """
    if (
        isinstance(order, bool)
        or not isinstance(order, (int, np.integer))
        or order <= 0
    ):
        raise ValueError("Circular-moment order must be a positive integer.")
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Cannot calculate a circular moment without finite angles.")
    vector = np.mean(np.exp(1j * int(order) * np.deg2rad(finite)))
    resultant_length = float(abs(vector))
    direction = (
        None
        if np.isclose(resultant_length, 0.0, rtol=0.0, atol=1e-8)
        else normalize_periodic_degrees(np.rad2deg(np.angle(vector)))
    )
    return CircularMoment(
        order=int(order),
        direction_degrees=direction,
        resultant_length=resultant_length,
    )


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
