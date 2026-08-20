"""Search-space tests for canonical periodic dihedrals."""

from swarmcg.config_types import SwarmConfig
from swarmcg.optimization_types import OptimizationCycle, ParameterVectorLayout
from swarmcg.shared.periodic import PeriodicDihedralParameters
from swarmcg.topology import CGTopology, DihedralGroup


def _periodic_topology():
    """Return one canonical multiplicity-two periodic group."""
    parameters = PeriodicDihedralParameters(-145.0, 3.0, 2)
    return CGTopology(
        dihedrals=[
            DihedralGroup("1", [(0, 1, 2, 3)], 1, parameters, parameters)
        ]
    )


def test_periodic_search_uses_full_phase_domain_and_nonnegative_force():
    topology = _periodic_topology()
    cycle = OptimizationCycle.from_topology(1, ["dihedral"], topology)
    domains = {
        "constraint": [],
        "bond": [],
        "angle": [],
        "dihedral": [[-325.0, 35.0]],
    }

    layout = ParameterVectorLayout.build(
        topology, cycle, domains, 1, SwarmConfig()
    )

    assert layout.bounds == [[-325.0, 35.0], [0.0, 15.0]]


def test_periodic_topology_update_defensively_canonicalizes_negative_force():
    topology = _periodic_topology()
    cycle = OptimizationCycle.from_topology(1, ["dihedral"], topology)
    domains = {
        "constraint": [],
        "bond": [],
        "angle": [],
        "dihedral": [[-325.0, 35.0]],
    }
    layout = ParameterVectorLayout.build(
        topology, cycle, domains, 1, SwarmConfig()
    )

    layout.apply(topology, [35.0, -3.0])

    assert topology.dihedrals[0].equilibrium == -145.0
    assert topology.dihedrals[0].force_constant == 3.0
    assert topology.dihedrals[0].gromacs_parameters == (-145.0, 3.0)


def test_parameter_layout_rejects_wrong_vector_dimension():
    topology = _periodic_topology()
    cycle = OptimizationCycle.from_topology(1, ["dihedral"], topology)
    domains = {
        "constraint": [],
        "bond": [],
        "angle": [],
        "dihedral": [[-325.0, 35.0]],
    }
    layout = ParameterVectorLayout.build(
        topology, cycle, domains, 1, SwarmConfig()
    )

    import pytest

    with pytest.raises(ValueError, match="dimension 1, expected 2"):
        layout.apply(topology, [0.0])
