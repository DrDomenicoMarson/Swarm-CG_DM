"""Histogram grids and circular helpers used by bonded-geometry scoring."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist

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
    """

    edges: np.ndarray
    centers: np.ndarray
    cost_matrix: np.ndarray
    period: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate dimensions and make the underlying arrays read-only."""
        if self.edges.ndim != 1 or self.centers.ndim != 1:
            raise ValueError("Histogram edges and centers must be one-dimensional.")
        if len(self.edges) != len(self.centers) + 1:
            raise ValueError("A histogram grid must have exactly one more edge than center.")
        expected_shape = (len(self.centers), len(self.centers))
        if self.cost_matrix.shape != expected_shape:
            raise ValueError(
                f"Cost matrix shape {self.cost_matrix.shape} does not match histogram shape {expected_shape}."
            )
        if self.period is not None and self.period <= 0:
            raise ValueError("A periodic histogram must have a positive period.")
        self.edges.setflags(write=False)
        self.centers.setflags(write=False)
        self.cost_matrix.setflags(write=False)


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
    """
    edges = _uniform_edges(start, stop, bandwidth)
    centers = (edges[:-1] + edges[1:]) / 2.0
    distances = cdist(centers.reshape(-1, 1), centers.reshape(-1, 1))
    if period is not None:
        distances = np.minimum(distances, period - distances)
    return HistogramGrid(edges=edges, centers=centers, cost_matrix=distances, period=period)


def normalized_histogram(values: np.ndarray, grid: HistogramGrid) -> np.ndarray:
    """Return probability masses for values on a histogram grid.

    Args:
        values: Values to bin.
        grid: Histogram grid defining the bin edges.

    Returns:
        A vector of probability masses. If no values fall inside the grid, a
        zero vector is returned so the caller can provide a domain-specific
        error message.
    """
    counts = np.histogram(np.asarray(values, dtype=float), grid.edges, density=False)[0]
    total = counts.sum()
    if total == 0:
        return np.zeros(len(grid.centers), dtype=float)
    return counts.astype(float) / float(total)


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
    """Populate scoring grids and legacy edge/matrix aliases on a context.

    Args:
        ns: Optimization context whose scoring state will be populated.
        constraints: Whether to create the constraints grid.
    """
    opt = ns.config.optimization
    if constraints:
        ns.scoring.constraints_grid = create_histogram_grid(0.0, opt.bonded_max_range, opt.bw_constraints)
        ns.scoring.bins_constraints = ns.scoring.constraints_grid.edges
        ns.scoring.bins_constraints_dist_matrix = ns.scoring.constraints_grid.cost_matrix

    ns.scoring.bonds_grid = create_histogram_grid(0.0, opt.bonded_max_range, opt.bw_bonds)
    ns.scoring.angles_grid = create_histogram_grid(0.0, 180.0, opt.bw_angles)
    ns.scoring.dihedrals_grid = create_histogram_grid(-180.0, 180.0, opt.bw_dihedrals, period=360.0)

    ns.scoring.bins_bonds = ns.scoring.bonds_grid.edges
    ns.scoring.bins_angles = ns.scoring.angles_grid.edges
    ns.scoring.bins_dihedrals = ns.scoring.dihedrals_grid.edges
    ns.scoring.bins_bonds_dist_matrix = ns.scoring.bonds_grid.cost_matrix
    ns.scoring.bins_angles_dist_matrix = ns.scoring.angles_grid.cost_matrix
    ns.scoring.bins_dihedrals_dist_matrix = ns.scoring.dihedrals_grid.cost_matrix
