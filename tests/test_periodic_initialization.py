"""Search-space tests for canonical periodic dihedrals."""

from swarmcg.config_types import SwarmConfig
from swarmcg.forcefield import get_search_space_boundaries, update_cg_itp_obj


def _periodic_topology():
    """Return one canonical multiplicity-two periodic group."""
    return {
        "constraint": [],
        "bond": [],
        "angle": [],
        "dihedral": [
            {
                "func": 1,
                "value": -145.0,
                "fct": 3.0,
                "mult": 2,
                "params": [-145.0, 3.0],
            }
        ],
    }


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

    assert topology["dihedral"][0]["value"] == -145.0
    assert topology["dihedral"][0]["fct"] == 3.0
    assert topology["dihedral"][0]["params"] == [-145.0, 3.0]
