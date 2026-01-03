import pytest

from swarmcg.simulations import get_settings


from swarmcg.config_types import SwarmConfig
from swarmcg.context import OptimizationContext

def test_get_settings_fail(ns_opt):
    # when:
    ns = ns_opt(sim_type="NO_VALID")
    config = SwarmConfig.from_namespace(ns)
    context = OptimizationContext(config=config)
    
    # then:
    with pytest.raises(ValueError):
        _ = get_settings(context)


def test_get_settings_optimal(ns_opt):
    # when:
    ns = ns_opt(sim_type="OPTIMAL", cg_itp={"nb_constraints": 2, "nb_bonds": 2, "nb_angles": 2, "nb_dihedrals": 2})
    config = SwarmConfig.from_namespace(ns)
    context = OptimizationContext(config=config)
    context.cg_itp = ns.cg_itp # explicit copy of cg_itp which is not in config but in context root

    # then:
    sim_types, opti_cycles, sim_cycles, particle_setter = get_settings(context)

    # then:
    assert sim_cycles == [0, 1, 2]
    assert particle_setter([1, 2, 3, 4]) == 4
    assert opti_cycles == [["constraint", "bond", "angle"], ["angle", "dihedral"], ["constraint", "bond", "angle", "dihedral"]]


def test_get_settings_test(ns_opt):
    # when:
    ns = ns_opt(sim_type="TEST")
    config = SwarmConfig.from_namespace(ns)
    context = OptimizationContext(config=config)

    # then:
    sim_types, opti_cycles, sim_cycles, particle_setter = get_settings(context)

    # then:
    assert sim_cycles == [0, 1, 2]
    assert particle_setter([1, 2, 3, 4]) == 2
    assert particle_setter(list(range(50))) == 2
    assert opti_cycles == [["constraint", "bond", "angle"], ["dihedral"], ["constraint", "bond", "angle", "dihedral"]]

