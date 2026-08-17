"""Numerical regression tests for histogram and score semantics."""

import numpy as np
from pyemd import emd

from swarmcg.scoring.distances import (
    circular_mean_degrees,
    compose_classwise_l2_score,
    create_histogram_grid,
    normalized_histogram,
    unwrap_degrees_around,
)
from swarmcg.config_types import OptimizationConfig, SwarmConfig
from swarmcg.core.optimization import SwarmOptimizer
from swarmcg.scoring.distances import create_bins_and_dist_matrices


def test_histogram_grid_dimensions_and_normalization():
    grid = create_histogram_grid(0.0, 180.0, 2.5)
    histogram = normalized_histogram(np.array([0.1, 1.0, 179.9]), grid)

    assert histogram.shape == (len(grid.centers),)
    assert grid.cost_matrix.shape == (histogram.size, histogram.size)
    assert np.isclose(histogram.sum(), 1.0)


def test_dihedral_emd_uses_short_distance_across_seam():
    grid = create_histogram_grid(-180.0, 180.0, 2.5, period=360.0)
    left = normalized_histogram(np.array([-179.0]), grid)
    right = normalized_histogram(np.array([179.0]), grid)

    assert np.isclose(emd(left, right, grid.cost_matrix), 2.5)
    assert np.isclose(grid.cost_matrix[0, -1], 2.5)


def test_circular_mean_and_unwrapped_domain_cross_seam():
    values = np.array([179.0, -179.0])
    mean = circular_mean_degrees(values)
    unwrapped = unwrap_degrees_around(values, mean)

    assert np.isclose(abs(mean), 180.0)
    assert np.ptp(unwrapped) == 2.0
    assert np.all(np.abs(unwrapped - mean) <= 1.0)


def test_paper_classwise_l2_composition_is_preserved():
    result = compose_classwise_l2_score(
        constraints=np.array([500.0 * 0.01]),
        bonds=np.array([500.0 * 0.02]),
        angles=np.array([3.0, 4.0]),
        dihedrals=np.array([12.0]),
    )

    constraints_bonds = np.sqrt(5.0**2 + 10.0**2)
    assert np.allclose(result, (constraints_bonds + 5.0 + 12.0, constraints_bonds, 5.0, 12.0))


def test_failure_score_is_next_float_above_active_theoretical_maximum():
    config = SwarmConfig(
        optimization=OptimizationConfig(
            bonded_max_range=1.0,
            bw_constraints=0.1,
            bw_bonds=0.1,
            bw_angles=30.0,
            bw_dihedrals=60.0,
        )
    )
    optimizer = SwarmOptimizer(config)
    optimizer.ns.cg_itp = {
        "nb_constraints": 1,
        "nb_bonds": 2,
        "nb_angles": 1,
        "nb_dihedrals": 1,
    }
    optimizer.ns.opti_cycle = {
        "geoms": ["constraint", "bond", "angle", "dihedral"]
    }
    create_bins_and_dist_matrices(optimizer.ns)

    optimizer._calculate_worst_fit_score()

    theoretical = sum(optimizer.ns.pso.failure_component_scores.values())
    assert optimizer.ns.pso.worst_fit_score == np.nextafter(theoretical, np.inf)
    assert optimizer.ns.pso.worst_fit_score > theoretical
