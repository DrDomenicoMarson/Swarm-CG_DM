"""Search-space tests for canonical periodic dihedrals."""

from swarmcg.config_types import SwarmConfig
from swarmcg.forcefield import get_search_space_boundaries, update_cg_itp_obj
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
    cycle = {
        "nb_geoms": {"constraint": 0, "bond": 0, "angle": 0, "dihedral": 1}
    }
    domains = {
        "constraint": [],
        "bond": [],
        "angle": [],
        "dihedral": [[-325.0, 35.0]],
    }

    boundaries = get_search_space_boundaries(
        topology, cycle, domains, 1, SwarmConfig()
    )

    assert boundaries == [[-325.0, 35.0], [0.0, 15.0]]


def test_periodic_topology_update_defensively_canonicalizes_negative_force():
    topology = _periodic_topology()
    cycle = {
        "nb_geoms": {"constraint": 0, "bond": 0, "angle": 0, "dihedral": 1}
    }

    update_cg_itp_obj(topology, cycle, [35.0, -3.0], exec_mode=1)

    assert topology.dihedrals[0].equilibrium == -145.0
    assert topology.dihedrals[0].force_constant == 3.0
    assert topology.dihedrals[0].gromacs_parameters == (-145.0, 3.0)
