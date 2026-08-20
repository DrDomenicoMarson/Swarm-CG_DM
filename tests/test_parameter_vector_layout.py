"""Coverage for canonical PSO vector layout construction and application."""

import random
from copy import deepcopy

import pytest

from swarmcg.config_types import SwarmConfig
from swarmcg.optimization_types import OptimizationCycle, ParameterVectorLayout
from swarmcg.particle_initialization import initialize_particles
from swarmcg.shared.periodic import PeriodicDihedralParameters
from swarmcg.simulations.polynomial import CBTParameters, RBParameters
from swarmcg.topology import (
    AngleGroup,
    BondGroup,
    CGTopology,
    ConstraintGroup,
    ConstraintParameters,
    DihedralGroup,
    HarmonicParameters,
)


def _layout_topology():
    constraint = ConstraintParameters(0.2)
    bond = HarmonicParameters(0.3, 1000.0)
    angles = [
        AngleGroup(
            str(function),
            [(0, 1, 2)],
            function,
            HarmonicParameters(90.0 + function, 100.0 + function),
            HarmonicParameters(90.0 + function, 100.0 + function),
        )
        for function in (1, 2, 10)
    ]
    periodic_1 = PeriodicDihedralParameters(30.0, 2.0, 1)
    harmonic = HarmonicParameters(-20.0, -3.0)
    rb = RBParameters((1.0, 2.0, 3.0, 4.0, 5.0))
    periodic_4 = PeriodicDihedralParameters(-45.0, 4.0, 2)
    cbt = CBTParameters((6.0, 7.0, 8.0, 9.0, 10.0))
    return CGTopology(
        constraints=[
            ConstraintGroup("1", [(0, 1)], 1, constraint, constraint)
        ],
        bonds=[BondGroup("1", [(0, 1)], 1, bond, bond)],
        angles=angles,
        dihedrals=[
            DihedralGroup("1", [(0, 1, 2, 3)], 1, periodic_1, periodic_1),
            DihedralGroup("2", [(0, 1, 2, 3)], 2, harmonic, harmonic),
            DihedralGroup(
                "3", [(0, 1, 2, 3)], 3, rb, rb, coefficient_bound=25.0
            ),
            DihedralGroup("4", [(0, 1, 2, 3)], 4, periodic_4, periodic_4),
            DihedralGroup(
                "5", [(0, 1, 2, 3)], 11, cbt, cbt, coefficient_bound=25.0
            ),
        ],
    )


def _domains():
    return {
        "constraint": [[0.1, 0.4]],
        "bond": [[0.1, 0.5]],
        "angle": [[0.0, 180.0]] * 3,
        "dihedral": [[-180.0, 180.0]] * 5,
    }


@pytest.mark.parametrize("execution_mode,dimension", [(1, 25), (2, 17)])
def test_layout_covers_supported_functions_in_both_execution_modes(
    execution_mode, dimension
):
    topology = _layout_topology()
    cycle = OptimizationCycle.from_topology(
        1, ["constraint", "bond", "angle", "dihedral"], topology
    )
    layout = ParameterVectorLayout.build(
        topology, cycle, _domains(), execution_mode, SwarmConfig()
    )

    vector = layout.encode(topology)

    assert layout.dimension == dimension
    assert len(layout.bounds) == dimension
    assert len(vector) == dimension
    layout.apply(topology, vector)
    assert layout.encode(topology) == pytest.approx(vector)


@pytest.mark.parametrize("execution_mode", [1, 2])
def test_layout_rejects_invalid_dimensions_in_both_execution_modes(
    execution_mode,
):
    topology = _layout_topology()
    cycle = OptimizationCycle.from_topology(
        1, ["constraint", "bond", "angle", "dihedral"], topology
    )
    layout = ParameterVectorLayout.build(
        topology, cycle, _domains(), execution_mode, SwarmConfig()
    )

    with pytest.raises(ValueError, match="expected"):
        layout.apply(topology, [0.0] * (layout.dimension - 1))


def test_particle_initialization_matches_pre_layout_numerical_baseline():
    constraint = ConstraintParameters(0.25)
    bond = HarmonicParameters(0.3, 1000.0)
    angle = HarmonicParameters(100.0, 200.0)
    dihedral = PeriodicDihedralParameters(-150.0, 3.0, 2)
    topology = CGTopology(
        constraints=[
            ConstraintGroup("1", [(0, 1)], 1, constraint, constraint)
        ],
        bonds=[BondGroup("1", [(0, 1)], 1, bond, bond)],
        angles=[AngleGroup("1", [(0, 1, 2)], 1, angle, angle)],
        dihedrals=[
            DihedralGroup(
                "1", [(0, 1, 2, 3)], 1, dihedral, dihedral
            )
        ],
    )
    domains = {
        "constraint": [[0.1, 0.4]],
        "bond": [[0.1, 0.5]],
        "angle": [[0.0, 180.0]],
        "dihedral": [[-330.0, 30.0]],
    }
    cycle = OptimizationCycle.from_topology(
        2, ["constraint", "bond", "angle", "dihedral"], topology
    )
    layout = ParameterVectorLayout.build(
        topology, cycle, domains, 1, SwarmConfig()
    )
    best_scores = {
        kind: {0: float("nan")}
        for kind in ("constraints", "bonds", "angles", "dihedrals")
    }
    best_parameters = {kind: {0: {}} for kind in best_scores}
    random.seed(1729)

    particles = initialize_particles(
        3,
        layout,
        topology,
        deepcopy(topology),
        best_scores,
        best_parameters,
        SwarmConfig(),
        equilibrium_guess_factor=0.25,
        force_guess_factor=0.3,
    )

    assert particles == [
        [0.25, 0.3, 1000.0, 100.0, 200.0, -150.0, 3.0],
        [0.256, 0.305, 946.479, 99.789, 233.447, -151.587, 2.584],
        [0.248, 0.305, 824.591, 101.645, 131.206, -148.846, 2.999],
    ]
