"""Count-invariant Boltzmann targets and bounded linear initializers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import lsq_linear

from swarmcg import config
from swarmcg.shared.histograms import observe_histogram


def complete_sample_range(values: Sequence[float]) -> tuple[float, float]:
    """Return the narrowest stable histogram range containing every sample.

    Args:
        values: Finite scalar samples.

    Returns:
        Lower and upper limits expanded outward enough to include both sample
        extrema. A flat distribution receives a small symmetric interval.

    Raises:
        ValueError: If samples are empty or contain ``NaN`` or infinity.
    """
    samples = np.asarray(values, dtype=float).reshape(-1)
    if samples.size == 0 or not np.all(np.isfinite(samples)):
        raise ValueError("Boltzmann range requires at least one finite-only sample vector.")
    minimum = float(np.min(samples))
    maximum = float(np.max(samples))
    if minimum == maximum:
        delta = np.finfo(float).eps * max(1.0, abs(minimum))
        return minimum - delta, maximum + delta
    return float(np.nextafter(minimum, -np.inf)), float(
        np.nextafter(maximum, np.inf)
    )


@dataclass(frozen=True)
class BoltzmannTarget:
    """Normalized one-dimensional target used for Boltzmann initialization.

    Attributes
    ----------
    centers
        Histogram bin centers in the native units used by the fitted
        potential: nanometers for bonds and radians for angular coordinates.
    probabilities
        Normalized probability masses at the corresponding bin centers.
    """

    centers: np.ndarray
    probabilities: np.ndarray

    def __post_init__(self) -> None:
        """Validate, copy, and freeze the target arrays.

        Raises
        ------
        ValueError
            If arrays are incompatible, non-finite, negative, empty, or not
            normalized to unit probability.
        """
        centers = np.array(self.centers, dtype=float, copy=True)
        probabilities = np.array(self.probabilities, dtype=float, copy=True)
        if centers.ndim != 1 or probabilities.ndim != 1:
            raise ValueError("Boltzmann target arrays must be one-dimensional.")
        if centers.shape != probabilities.shape or centers.size == 0:
            raise ValueError(
                "Boltzmann target centers and probabilities must be nonempty equal-length vectors."
            )
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(probabilities)):
            raise ValueError("Boltzmann target values must be finite.")
        if np.any(probabilities < 0):
            raise ValueError("Boltzmann target probabilities cannot be negative.")
        total = float(np.sum(probabilities))
        if not np.isclose(total, 1.0, rtol=1e-10, atol=1e-12):
            raise ValueError(
                f"Boltzmann target probabilities must sum to one, received {total}."
            )
        centers.setflags(write=False)
        probabilities.setflags(write=False)
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "probabilities", probabilities)

    @classmethod
    def from_samples(
        cls,
        values: Sequence[float],
        *,
        bins: int,
        value_range: tuple[float, float],
    ) -> "BoltzmannTarget":
        """Build a normalized target from coordinate samples.

        Parameters
        ----------
        values
            Coordinate samples in the desired native units.
        bins
            Positive number of equal-width histogram bins.
        value_range
            Inclusive lower and upper histogram limits.

        Returns
        -------
        BoltzmannTarget
            Immutable bin centers and probability masses.

        Raises
        ------
        ValueError
            If any sample is non-finite or outside the requested range, or the
            bin definition is invalid.
        """
        samples = np.asarray(values, dtype=float).reshape(-1)
        if (
            isinstance(bins, bool)
            or not isinstance(bins, (int, np.integer))
            or bins <= 0
        ):
            raise ValueError(
                "Boltzmann target bin count must be a positive integer."
            )
        if len(value_range) != 2 or not np.all(np.isfinite(value_range)):
            raise ValueError("Boltzmann target range must contain two finite values.")
        if value_range[1] <= value_range[0]:
            raise ValueError("Boltzmann target upper limit must exceed its lower limit.")
        edges = np.linspace(value_range[0], value_range[1], bins + 1, dtype=float)
        observation = observe_histogram(samples, edges)
        if observation.expected_count == 0 or observation.missing_count:
            raise ValueError(
                "Boltzmann target samples cannot be silently discarded: "
                + observation.coverage_message()
            )
        centers = (edges[:-1] + edges[1:]) / 2.0
        return cls(centers=centers, probabilities=observation.probabilities)

    def pmf_samples(self, temperature: float) -> tuple[np.ndarray, np.ndarray]:
        """Return occupied centers and their zero-referenced marginal PMF.

        Parameters
        ----------
        temperature
            Positive finite temperature in kelvin.

        Returns
        -------
        tuple of numpy.ndarray
            Occupied bin centers and ``-kBT ln(p/pmax)`` values. Unsampled
            zero-probability bins are excluded rather than regularized.

        Raises
        ------
        ValueError
            If the temperature is not finite and positive.
        """
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("Boltzmann temperature must be finite and positive.")
        occupied = self.probabilities > 0
        probabilities = self.probabilities[occupied]
        pmf = -config.kB * temperature * np.log(
            probabilities / np.max(probabilities)
        )
        return self.centers[occupied], pmf


@dataclass(frozen=True)
class LinearBoltzmannFit:
    """Result of fitting one force constant and a free energy intercept.

    Attributes
    ----------
    force_constant
        Fitted coefficient multiplying the supplied potential basis.
    intercept
        Arbitrary additive PMF offset.
    occupied_bins
        Number of nonzero-probability bins used by the fit.
    rank
        Rank of the two-column linear design matrix.
    """

    force_constant: float
    intercept: float
    occupied_bins: int
    rank: int


def fit_bounded_force_constant(
    target: BoltzmannTarget,
    basis: Sequence[float],
    temperature: float,
    lower_bound: float,
    upper_bound: float,
) -> LinearBoltzmannFit:
    """Fit a bounded force constant with an unconstrained additive intercept.

    Parameters
    ----------
    target
        Normalized Boltzmann target.
    basis
        Unit-force potential evaluated at every target bin center.
    temperature
        Positive finite temperature in kelvin.
    lower_bound
        Finite inclusive lower bound for the force constant.
    upper_bound
        Finite inclusive upper bound for the force constant.

    Returns
    -------
    LinearBoltzmannFit
        Bounded least-squares solution and design diagnostics.

    Raises
    ------
    ValueError
        If inputs are incompatible, bounds are invalid, the occupied design
        is rank deficient, or the numerical fit fails.
    """
    basis_values = np.asarray(basis, dtype=float)
    if basis_values.shape != target.centers.shape or not np.all(
        np.isfinite(basis_values)
    ):
        raise ValueError("Potential basis must be finite and match the target centers.")
    if not np.isfinite(lower_bound) or not np.isfinite(upper_bound):
        raise ValueError("Force-constant bounds must be finite.")
    if upper_bound <= lower_bound:
        raise ValueError("Force-constant upper bound must exceed the lower bound.")

    occupied = target.probabilities > 0
    _, pmf = target.pmf_samples(temperature)
    design = np.column_stack(
        (basis_values[occupied], np.ones(np.count_nonzero(occupied), dtype=float))
    )
    rank = int(np.linalg.matrix_rank(design))
    occupied_bins = int(np.count_nonzero(occupied))
    if rank < 2:
        raise ValueError(
            "Boltzmann force fit is underdetermined: "
            f"{occupied_bins} occupied bins give design rank {rank}, expected 2."
        )

    result = lsq_linear(
        design,
        pmf,
        bounds=(np.array([lower_bound, -np.inf]), np.array([upper_bound, np.inf])),
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise ValueError(f"Bounded Boltzmann force fit failed: {result.message}")
    return LinearBoltzmannFit(
        force_constant=float(result.x[0]),
        intercept=float(result.x[1]),
        occupied_bins=occupied_bins,
        rank=rank,
    )
