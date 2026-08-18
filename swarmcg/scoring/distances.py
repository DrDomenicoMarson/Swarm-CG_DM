"""Histogram grids and circular helpers used by bonded-geometry scoring."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=r"pkg_resources is deprecated as an API\..*",
        category=UserWarning,
    )
    from pyemd import emd

from swarmcg.shared.histograms import HistogramObservation, observe_histogram
from swarmcg.shared import exceptions
from swarmcg.shared.periodic import (
    circular_mean_degrees,
    normalize_periodic_degrees,
    unwrap_degrees_around,
)


@dataclass(frozen=True)
class HistogramGrid:
    """Immutable definition of a one-dimensional scoring histogram.

    Args:
        edges: Histogram bin edges.
        centers: Histogram bin centers.
        cost_matrix: Pairwise transport cost between bin centers.
        period: Period of a circular variable, or ``None`` for a linear grid.

    Raises:
        ValueError: If dimensions, edges, costs, or the optional period do not
            define a finite valid transport grid.
    """

    edges: np.ndarray
    centers: np.ndarray
    cost_matrix: np.ndarray
    period: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate, copy, and freeze the histogram-grid arrays."""
        edges = np.array(self.edges, dtype=float, copy=True)
        centers = np.array(self.centers, dtype=float, copy=True)
        cost_matrix = np.array(self.cost_matrix, dtype=float, copy=True)
        if edges.ndim != 1 or centers.ndim != 1:
            raise ValueError("Histogram edges and centers must be one-dimensional.")
        if len(edges) != len(centers) + 1:
            raise ValueError("A histogram grid must have exactly one more edge than center.")
        if (
            not np.all(np.isfinite(edges))
            or not np.all(np.isfinite(centers))
            or np.any(np.diff(edges) <= 0)
        ):
            raise ValueError(
                "Histogram edges and centers must be finite with increasing edges."
            )
        expected_shape = (len(centers), len(centers))
        if cost_matrix.shape != expected_shape:
            raise ValueError(
                f"Cost matrix shape {cost_matrix.shape} does not match histogram shape {expected_shape}."
            )
        if (
            not np.all(np.isfinite(cost_matrix))
            or np.any(cost_matrix < 0)
            or not np.allclose(cost_matrix, cost_matrix.T)
            or not np.allclose(np.diag(cost_matrix), 0.0)
        ):
            raise ValueError(
                "Histogram transport costs must be finite, nonnegative, "
                "symmetric, and zero on the diagonal."
            )
        if self.period is not None and (
            not np.isfinite(self.period) or self.period <= 0
        ):
            raise ValueError("A periodic histogram must have a positive period.")
        edges.setflags(write=False)
        centers.setflags(write=False)
        cost_matrix.setflags(write=False)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "cost_matrix", cost_matrix)


def _uniform_edges(start: float, stop: float, bandwidth: float) -> np.ndarray:
    """Create increasing edges that include both requested endpoints.

    Args:
        start: Lower histogram limit.
        stop: Upper histogram limit.
        bandwidth: Requested maximum bin width.

    Returns:
        Histogram edges whose final value is exactly ``stop``.

    Raises:
        ValueError: If the interval or bandwidth is invalid.
    """
    if not np.all(np.isfinite((start, stop, bandwidth))):
        raise ValueError("Histogram limits and bandwidth must be finite.")
    if bandwidth <= 0:
        raise ValueError("Histogram bandwidth must be greater than zero.")
    if stop <= start:
        raise ValueError("Histogram stop must be greater than start.")

    edges = np.arange(start, stop, bandwidth, dtype=float)
    if edges.size == 0 or not np.isclose(edges[0], start):
        edges = np.insert(edges, 0, start)
    if np.isclose(edges[-1], stop):
        edges[-1] = stop
    else:
        edges = np.append(edges, stop)
    return edges


def create_histogram_grid(
    start: float,
    stop: float,
    bandwidth: float,
    *,
    period: Optional[float] = None,
) -> HistogramGrid:
    """Build a linear or circular histogram transport grid.

    Args:
        start: Lower histogram limit.
        stop: Upper histogram limit.
        bandwidth: Requested maximum bin width.
        period: Circular period. Leave unset for a linear variable.

    Returns:
        A validated immutable histogram grid.

    Raises:
        ValueError: If limits, bandwidth, or period are invalid.
    """
    edges = _uniform_edges(start, stop, bandwidth)
    centers = (edges[:-1] + edges[1:]) / 2.0
    distances = cdist(centers.reshape(-1, 1), centers.reshape(-1, 1))
    if period is not None:
        distances = np.minimum(distances, period - distances)
    return HistogramGrid(edges=edges, centers=centers, cost_matrix=distances, period=period)


def normalized_histogram(values: np.ndarray, grid: HistogramGrid) -> np.ndarray:
    """Return frame-normalized masses for values on a histogram grid.

    Args:
        values: Values to bin.
        grid: Histogram grid defining the bin edges.

    Returns:
        Per-bin counts divided by the number of supplied values. The vector
        sums to the sampled coverage when values are invalid or out of range.
    """
    return observe_histogram(values, grid.edges).probabilities


def require_complete_reference(
    observation: HistogramObservation,
    values: np.ndarray,
    label: str,
    unit: str,
) -> None:
    """Reject an incomplete atomistic reference histogram.

    Args:
        observation: Classified reference histogram.
        values: Original coordinate values used for range diagnostics.
        label: Human-readable geometry group label.
        unit: Unit printed for finite observed values.

    Raises:
        ScientificValidationError: If any sample is non-finite or outside the
            histogram domain, or no sample was supplied.
    """
    if observation.expected_count > 0 and observation.missing_count == 0:
        return
    values_arr = np.asarray(values, dtype=float)
    finite = values_arr[np.isfinite(values_arr)]
    observed = (
        f"min={finite.min():.6g}, max={finite.max():.6g} {unit}"
        if finite.size
        else "no finite values"
    )
    raise exceptions.ScientificValidationError(
        f"Incomplete AA/reference histogram for {label}: "
        f"{observation.coverage_message()}; observed {observed}. "
        "Reference samples cannot be silently discarded."
    )


def earth_movers_distance(
    reference: np.ndarray,
    candidate: np.ndarray,
    grid: HistogramGrid,
) -> float:
    """Calculate EMD while charging missing candidate mass maximum cost.

    Args:
        reference: Complete normalized reference probability masses.
        candidate: Candidate masses whose sum may be less than one.
        grid: Histogram grid and cost matrix shared by both vectors.

    Returns:
        Earth mover's distance including maximum-cost missing mass.

    Raises:
        ValueError: If vectors are incompatible, invalid, or the reference is
            not normalized to unit mass.
    """
    reference_mass = np.ascontiguousarray(reference, dtype=np.float64)
    candidate_mass = np.ascontiguousarray(candidate, dtype=np.float64)
    expected_shape = (len(grid.centers),)
    if reference_mass.shape != expected_shape or candidate_mass.shape != expected_shape:
        raise ValueError(
            f"Histogram vectors must both have shape {expected_shape}; received "
            f"{reference_mass.shape} and {candidate_mass.shape}."
        )
    if (
        not np.all(np.isfinite(reference_mass))
        or not np.all(np.isfinite(candidate_mass))
        or np.any(reference_mass < 0)
        or np.any(candidate_mass < 0)
    ):
        raise ValueError("EMD histogram masses must be finite and nonnegative.")
    reference_total = float(reference_mass.sum())
    candidate_total = float(candidate_mass.sum())
    if not np.isclose(reference_total, 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError(
            f"EMD reference histogram must sum to one; received {reference_total}."
        )
    if candidate_total > 1.0 and not np.isclose(
        candidate_total, 1.0, rtol=1e-10, atol=1e-12
    ):
        raise ValueError(
            f"EMD candidate histogram cannot exceed unit mass; received {candidate_total}."
        )
    maximum_cost = float(np.max(grid.cost_matrix))
    return float(
        emd(
            reference_mass,
            candidate_mass,
            np.ascontiguousarray(grid.cost_matrix, dtype=np.float64),
            extra_mass_penalty=maximum_cost,
        )
    )


def support_neighborhood(
    centers: np.ndarray,
    *histograms: np.ndarray,
    periodic: bool = False,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Select occupied bins plus one neighbor without dropping endpoints.

    Args:
        centers: Histogram bin centers.
        *histograms: One or more mass vectors defined at those centers.
        periodic: Treat the first and last bins as neighbors. When support
            touches the seam, the full ordered grid is retained so plotting
            does not introduce a discontinuous center sequence.

    Returns:
        Selected centers and equally sliced histogram vectors. If every input
        is empty, the complete grid is returned.

    Raises:
        ValueError: If a histogram does not match the centers.
    """
    centers_arr = np.asarray(centers, dtype=float)
    arrays = [np.asarray(histogram, dtype=float) for histogram in histograms]
    if any(array.shape != centers_arr.shape for array in arrays):
        raise ValueError("Plot histograms must match the bin-center vector.")
    occupied = np.zeros(centers_arr.shape, dtype=bool)
    for array in arrays:
        occupied |= array > 0
    indices = np.flatnonzero(occupied)
    if indices.size == 0:
        selected = slice(0, len(centers_arr))
    elif periodic and (indices[0] == 0 or indices[-1] == len(centers_arr) - 1):
        selected = slice(0, len(centers_arr))
    else:
        selected = slice(max(0, int(indices[0]) - 1), min(len(centers_arr), int(indices[-1]) + 2))
    return centers_arr[selected], [array[selected] for array in arrays]


def compose_classwise_l2_score(
    constraints: np.ndarray,
    bonds: np.ndarray,
    angles: np.ndarray,
    dihedrals: np.ndarray,
) -> tuple[float, float, float, float]:
    """Compose the paper's class-wise L2 bonded mismatch score.

    Constraint and bond inputs are expected to contain the configured
    nm-to-degree-equivalent factor (500 by default) already applied.

    Args:
        constraints: Per-group scaled constraint EMD values.
        bonds: Per-group scaled bond EMD values.
        angles: Per-group angle EMD values in degrees.
        dihedrals: Per-group circular dihedral EMD values in degrees.

    Returns:
        ``(total, constraints_bonds, angles, dihedrals)``.
    """
    constraints_bonds = float(
        np.linalg.norm(np.concatenate((np.asarray(constraints), np.asarray(bonds))))
    )
    angles_score = float(np.linalg.norm(np.asarray(angles)))
    dihedrals_score = float(np.linalg.norm(np.asarray(dihedrals)))
    return (
        constraints_bonds + angles_score + dihedrals_score,
        constraints_bonds,
        angles_score,
        dihedrals_score,
    )


def create_bins_and_dist_matrices(ns, constraints: bool = True) -> None:
    """Populate immutable scoring grids and their edge vectors on a context.

    Args:
        ns: Optimization context whose scoring state will be populated.
        constraints: Whether to create the constraints grid.

    Returns:
        ``None``. The context is populated in place.
    """
    opt = ns.config.optimization
    if constraints:
        ns.scoring.constraints_grid = create_histogram_grid(0.0, opt.bonded_max_range, opt.bw_constraints)
        ns.scoring.bins_constraints = ns.scoring.constraints_grid.edges

    ns.scoring.bonds_grid = create_histogram_grid(0.0, opt.bonded_max_range, opt.bw_bonds)
    ns.scoring.angles_grid = create_histogram_grid(0.0, 180.0, opt.bw_angles)
    ns.scoring.dihedrals_grid = create_histogram_grid(-180.0, 180.0, opt.bw_dihedrals, period=360.0)

    ns.scoring.bins_bonds = ns.scoring.bonds_grid.edges
    ns.scoring.bins_angles = ns.scoring.angles_grid.edges
    ns.scoring.bins_dihedrals = ns.scoring.dihedrals_grid.edges
