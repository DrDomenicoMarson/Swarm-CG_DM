import pytest
from swarmcg.context import OptimizationContext
from swarmcg.config_types import SwarmConfig

def test_context_initialization():
    config = SwarmConfig()
    context = OptimizationContext(config=config)
    
    assert context.config is config
    assert context.status.nb_eval == 0
    assert context.pso.best_fitness == (float('inf'), None)

def test_context_getattr_fallback():
    # Test that legacy shim is removed
    config = SwarmConfig()
    config.gromacs.gmx_path = "test_gmx"
    
    context = OptimizationContext(config=config)
    
    # Direct access should fail now that shim is removed
    with pytest.raises(AttributeError):
        _ = context.gmx_path

def test_context_state_updates():
    context = OptimizationContext(config=SwarmConfig())
    context.status.nb_eval += 1
    assert context.status.nb_eval == 1
    
    context.results.gyr_aa_mapped = 1.5
    assert context.results.gyr_aa_mapped == 1.5
