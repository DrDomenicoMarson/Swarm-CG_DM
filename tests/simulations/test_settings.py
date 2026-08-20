import pytest

from swarmcg.simulations import get_settings


from swarmcg.config_types import SwarmConfig
from swarmcg.context import OptimizationContext

def test_get_settings_fail(ns_opt):
    # when:
    ns = ns_opt(sim_type="NO_VALID")

    # then:
    with pytest.raises(ValueError):
        config = SwarmConfig.from_namespace(ns)
        context = OptimizationContext(config=config)
        _ = get_settings(context)


def test_get_settings_optimal(ns_opt):
    # when:
    ns = ns_opt(sim_type="OPTIMAL")
    config = SwarmConfig.from_namespace(ns)
    context = OptimizationContext(config=config)

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
