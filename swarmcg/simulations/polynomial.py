"""Canonical parameter representations for polynomial GROMACS dihedrals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import lsq_linear

from swarmcg import config


@dataclass(frozen=True)
class RBParameters:
    """Five force-relevant Ryckaert--Bellemans coefficients.

    Attributes
    ----------
    coefficients
        Independent coefficients ``C1`` through ``C5`` in kJ/mol. ``C0`` is
        deliberately excluded because it only changes the energy origin.
    """

    coefficients: tuple[float, float, float, float, float]

    @classmethod
    def from_gromacs(cls, values: Sequence[float]) -> "RBParameters":
        """Create parameters from a standard six-coefficient GROMACS record.

        Parameters
        ----------
        values
            Coefficients ``C0`` through ``C5``.

        Returns
        -------
        RBParameters
            The five force-relevant coefficients.

        Raises
        ------
        ValueError
            If *values* does not contain exactly six finite numbers.
        """
        array = np.asarray(values, dtype=float)
        if array.shape != (6,) or not np.all(np.isfinite(array)):
            raise ValueError("an RB record must contain exactly six finite coefficients")
        return cls(tuple(float(value) for value in array[1:]))

    def to_gromacs(self) -> tuple[float, float, float, float, float, float]:
        """Return canonical ``C0`` through ``C5`` coefficients.

        Returns
        -------
        tuple of float
            Six coefficients with ``C0 = -sum(C1, ..., C5)``, which fixes the
            potential to zero at the GROMACS RB origin without changing forces.
        """
        c0 = -sum(self.coefficients)
        return (c0, *self.coefficients)


@dataclass(frozen=True)
class CBTParameters:
    """Five identifiable effective combined-bending--torsion coefficients.

    Attributes
    ----------
    effective_coefficients
        Products ``B_i = k_phi * a_i`` in kJ/mol for ``i = 0, ..., 4``.
    """

    effective_coefficients: tuple[float, float, float, float, float]

    @classmethod
    def from_gromacs(cls, values: Sequence[float]) -> "CBTParameters":
        """Create effective coefficients from a six-number GROMACS record.

        Parameters
        ----------
        values
            ``k_phi`` followed by ``a0`` through ``a4``.

        Returns
        -------
        CBTParameters
            The five products ``k_phi * a_i``.

        Raises
        ------
        ValueError
            If *values* does not contain exactly six finite numbers.
        """
        array = np.asarray(values, dtype=float)
        if array.shape != (6,) or not np.all(np.isfinite(array)):
            raise ValueError("a CBT record must contain exactly six finite coefficients")
        return cls(tuple(float(value) for value in array[0] * array[1:]))

    def to_gromacs(self) -> tuple[float, float, float, float, float, float]:
        """Return a deterministic six-number GROMACS factorization.

        Returns
        -------
        tuple of float
            ``k_phi`` and ``a0`` through ``a4``. A zero polynomial is written
            entirely as zeros; otherwise ``k_phi`` is the largest absolute
            effective coefficient and ``a_i = B_i / k_phi``.
        """
        scale = max(abs(value) for value in self.effective_coefficients)
        if scale == 0:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return (scale, *(value / scale for value in self.effective_coefficients))


def adaptive_coefficient_bound(
    probabilities: Iterable[float],
    temperature: float,
) -> float:
    """Derive a conservative polynomial-coefficient bound from a target PMF.

    Parameters
    ----------
    probabilities
        Histogram probabilities. Zero-probability bins are excluded because
        their finite-sampling PMF is undefined.
    temperature
        Absolute temperature in kelvin.

    Returns
    -------
    float
        Absolute coefficient bound in kJ/mol, limited to 25--200 kJ/mol.

    Raises
    ------
    ValueError
        If the temperature is not positive or no finite positive probability
        is available.
    """
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a finite positive value")
    values = np.asarray(tuple(probabilities), dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        raise ValueError("at least one finite nonzero target probability is required")
    thermal_energy = config.kB * temperature
    pmf = -thermal_energy * np.log(values)
    delta_u = float(np.percentile(pmf, 95) - np.min(pmf))
    return float(min(200.0, max(25.0, 5.0 * max(thermal_energy, delta_u))))


def mirrored_total_variation(probabilities: Sequence[float]) -> float:
    """Measure asymmetry between a torsional histogram and its mirror.

    Args:
        probabilities: Normalized, equally spaced torsional probabilities on
            a symmetric ``[-180, 180)`` grid.

    Returns:
        Total-variation distance in the interval 0--1.

    Raises:
        ValueError: If probabilities are invalid or have zero total mass.
    """
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("torsional probabilities must be a finite nonnegative vector")
    total = values.sum()
    if total <= 0:
        raise ValueError("torsional probabilities must have positive total mass")
    normalized = values / total
    return float(0.5 * np.sum(np.abs(normalized - normalized[::-1])))


def fit_rb_coefficients(
    angles_radians: Sequence[float],
    probabilities: Sequence[float],
    temperature: float,
    bound: float,
) -> RBParameters:
    """Fit canonical RB coefficients to a one-dimensional target PMF.

    Parameters
    ----------
    angles_radians
        Dihedral bin centers in radians.
    probabilities
        Normalized probabilities at the same bin centers.
    temperature
        Absolute temperature in kelvin.
    bound
        Positive absolute bound for each independent coefficient.

    Returns
    -------
    RBParameters
        Bounded least-squares estimate of ``C1`` through ``C5``.

    Raises
    ------
    ValueError
        If arrays are incompatible, contain insufficient target support, or
        *temperature*/*bound* is not positive.
    """
    angles = np.asarray(angles_radians, dtype=float)
    probs = np.asarray(probabilities, dtype=float)
    if angles.ndim != 1 or angles.shape != probs.shape:
        raise ValueError("RB fit angles and probabilities must be equal-length vectors")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a finite positive value")
    if not np.isfinite(bound) or bound <= 0:
        raise ValueError("RB coefficient bound must be a finite positive value")
    mask = np.isfinite(angles) & np.isfinite(probs) & (probs > 0)
    if np.count_nonzero(mask) < 5:
        raise ValueError("RB fitting requires at least five nonzero target bins")

    x = np.cos(angles[mask] - np.pi)
    # The PMF has an arbitrary additive origin. Fit a free intercept alongside
    # C1..C5, then discard it and apply the deterministic C0 convention only
    # at serialization time.
    design = np.column_stack(
        [np.ones_like(x), *[x**power for power in range(1, 6)]]
    )
    pmf = -config.kB * temperature * np.log(probs[mask])
    pmf -= np.min(pmf)
    lower = np.array([-np.inf, *([-bound] * 5)])
    upper = np.array([np.inf, *([bound] * 5)])
    result = lsq_linear(design, pmf, bounds=(lower, upper))
    if not result.success or not np.all(np.isfinite(result.x)):
        raise ValueError(f"bounded RB fit failed: {result.message}")
    return RBParameters(tuple(float(value) for value in result.x[1:]))
