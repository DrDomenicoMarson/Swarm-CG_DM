"""Numerical regression tests for histogram and score semantics."""

import numpy as np
import pytest
import matplotlib.pyplot as plt

from swarmcg.scoring.distances import (
    circular_mean_degrees,
    compose_classwise_l2_score,
    create_histogram_grid,
    earth_movers_distance,
    normalized_histogram,
    observe_histogram,
    require_complete_reference,
    support_neighborhood,
    unwrap_degrees_around,
)
from swarmcg.config_types import OptimizationConfig, SwarmConfig
from swarmcg.core.optimization import SwarmOptimizer
from swarmcg.scoring.distances import create_bins_and_dist_matrices
from swarmcg.shared.periodic import (
    PeriodicDihedralParameters,
    circular_moment_degrees,
    circular_statistics_degrees,
)
from swarmcg.shared import exceptions
from swarmcg.simulations.potentials import gmx_dihedrals_func_1
from swarmcg.scoring.compare import _annotate_missing_mass
from swarmcg.topology import CGTopology


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

    assert np.isclose(earth_movers_distance(left, right, grid), 2.5)
    assert np.isclose(grid.cost_matrix[0, -1], 2.5)


def test_circular_mean_and_unwrapped_domain_cross_seam():
    values = np.array([179.0, -179.0])
    mean = circular_mean_degrees(values)
    unwrapped = unwrap_degrees_around(values, mean)

    assert np.isclose(abs(mean), 180.0)
    assert np.ptp(unwrapped) == 2.0
    assert np.all(np.abs(unwrapped - mean) <= 1.0)


def test_tolerant_circular_statistics_preserve_strict_mean_api():
    symmetric = np.array([0.0, 180.0])

    statistics = circular_statistics_degrees(symmetric)

    assert statistics.mean_degrees is None
    assert np.isclose(statistics.resultant_length, 0.0)
    with pytest.raises(ValueError, match="Circular mean is undefined"):
        circular_mean_degrees(symmetric)


def test_ordered_circular_moment_recognizes_threefold_target():
    values = np.array([30.0, 150.0, -90.0])

    first = circular_moment_degrees(values, 1)
    third = circular_moment_degrees(values, 3)

    assert first.direction_degrees is None
    assert np.isclose(first.resultant_length, 0.0, atol=1e-15)
    assert np.isclose(third.direction_degrees, 90.0)
    assert np.isclose(third.resultant_length, 1.0)


@pytest.mark.parametrize(
    "order,values,expected_direction",
    [
        (1, [40.0], 40.0),
        (2, [30.0, -150.0], 60.0),
        (3, [30.0, 150.0, -90.0], 90.0),
    ],
)
def test_ordered_circular_moment_supports_multiplicities_one_to_three(
    order, values, expected_direction
):
    moment = circular_moment_degrees(np.asarray(values), order)

    assert moment.order == order
    assert np.isclose(moment.direction_degrees, expected_direction)
    assert np.isclose(moment.resultant_length, 1.0)


def test_negative_periodic_force_canonicalization_preserves_forces():
    canonical = PeriodicDihedralParameters.from_gromacs(35.0, -3.0, 2)
    angles = np.linspace(-np.pi, np.pi, 1001)
    original_energy = gmx_dihedrals_func_1(2)(
        angles, -3.0, np.deg2rad(35.0), 0.0
    )
    canonical_energy = gmx_dihedrals_func_1(2)(
        angles,
        canonical.force_constant,
        np.deg2rad(canonical.phase_degrees),
        0.0,
    )

    assert canonical.force_constant == 3.0
    assert canonical.phase_degrees == -145.0
    assert np.allclose(canonical_energy - original_energy, 6.0)
    assert np.allclose(
        np.gradient(canonical_energy, angles),
        np.gradient(original_energy, angles),
        atol=1e-12,
    )


def test_paper_classwise_l2_composition_is_preserved():
    result = compose_classwise_l2_score(
        constraints=np.array([500.0 * 0.01]),
        bonds=np.array([500.0 * 0.02]),
        angles=np.array([3.0, 4.0]),
        dihedrals=np.array([12.0]),
    )

    constraints_bonds = np.sqrt(5.0**2 + 10.0**2)
    assert np.allclose(result, (constraints_bonds + 5.0 + 12.0, constraints_bonds, 5.0, 12.0))


def test_histogram_observation_preserves_missing_mass_and_freezes_counts():
    grid = create_histogram_grid(0.0, 1.0, 0.5)
    observation = observe_histogram(
        np.array([0.25, np.nan, -1.0, 2.0]), grid.edges
    )

    assert observation.expected_count == 4
    assert observation.binned_count == 1
    assert observation.nonfinite_count == 1
    assert observation.underflow_count == 1
    assert observation.overflow_count == 1
    assert np.isclose(observation.coverage, 0.25)
    assert np.isclose(observation.probabilities.sum(), 0.25)
    with pytest.raises(ValueError):
        observation.probabilities[0] = 1.0


def test_reference_histograms_reject_any_discarded_sample():
    grid = create_histogram_grid(0.0, 1.0, 0.5)
    values = np.array([0.25, 2.0])
    observation = observe_histogram(values, grid.edges)

    with pytest.raises(
        exceptions.ScientificValidationError, match="Reference samples cannot be silently discarded"
    ):
        require_complete_reference(observation, values, "bond group 1", "nm")


def test_missing_cg_mass_receives_maximum_transport_penalty():
    grid = create_histogram_grid(0.0, 1.0, 0.5)
    reference = normalized_histogram(np.array([0.25]), grid)
    partial = normalized_histogram(np.array([0.25, 2.0, 3.0, 4.0]), grid)
    missing = normalized_histogram(np.array([np.nan, 2.0]), grid)

    assert np.isclose(partial.sum(), 0.25)
    assert np.isclose(
        earth_movers_distance(reference, partial, grid),
        0.75 * np.max(grid.cost_matrix),
    )
    assert np.isclose(
        earth_movers_distance(reference, missing, grid),
        np.max(grid.cost_matrix),
    )


def test_plot_support_includes_first_and_last_bins():
    grid = create_histogram_grid(-180.0, 180.0, 60.0, period=360.0)
    left = np.zeros(len(grid.centers))
    right = np.zeros(len(grid.centers))
    left[0] = 1.0
    right[-1] = 1.0

    centers, histograms = support_neighborhood(grid.centers, left, right)

    assert np.array_equal(centers, grid.centers)
    assert histograms[0][0] == 1.0
    assert histograms[1][-1] == 1.0


def test_periodic_plot_support_retains_seam_neighbor_and_order():
    grid = create_histogram_grid(-180.0, 180.0, 60.0, period=360.0)
    seam = np.zeros(len(grid.centers))
    seam[0] = 1.0

    centers, histograms = support_neighborhood(
        grid.centers, seam, periodic=True
    )

    assert np.array_equal(centers, grid.centers)
    assert histograms[0][0] == 1.0


def test_missing_plot_mass_is_annotated():
    figure, axis = plt.subplots()
    try:
        observation = observe_histogram(
            np.array([0.25, np.nan, -1.0, 2.0]),
            np.array([0.0, 0.5, 1.0]),
        )
        _annotate_missing_mass(axis, observation)
        assert [text.get_text() for text in axis.texts] == [
            "CG missing 75.0%\nnonfinite=1, below=1, above=1"
        ]
    finally:
        plt.close(figure)


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
    optimizer.ns.cg_itp = CGTopology(
        constraints=[object()],
        bonds=[object(), object()],
        angles=[object()],
        dihedrals=[object()],
    )
    optimizer.ns.opti_cycle = {
        "geoms": ["constraint", "bond", "angle", "dihedral"]
    }
    create_bins_and_dist_matrices(optimizer.ns)

    optimizer._calculate_worst_fit_score()

    theoretical = sum(optimizer.ns.pso.failure_component_scores.values())
    maximum_evaluable = sum(
        optimizer.ns.pso.failure_component_scores.values()
    )
    assert optimizer.ns.pso.worst_fit_score == np.nextafter(theoretical, np.inf)
    assert optimizer.ns.pso.worst_fit_score > maximum_evaluable
