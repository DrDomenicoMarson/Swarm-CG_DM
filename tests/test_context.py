import pytest
from swarmcg.context import OptimizationContext
from swarmcg.config_types import SwarmConfig

def test_context_initialization():
    config = SwarmConfig()
    context = OptimizationContext(config=config)
    
    assert context.config is config
    assert context.nb_eval == 0
    assert context.best_fitness == (float('inf'), None)

def test_context_getattr_fallback():
    # Test that we can access config attributes via context (mocking legacy ns behavior)
    config = SwarmConfig()
    config.gromacs.gmx_path = "test_gmx"
    
    context = OptimizationContext(config=config)
    
    # Direct access should fail if not defined in context
    # But __getattr__ should delegate to config
    # Note: OptimizationContext.__getattr__ logic handles specific delegation
    # Let's verify what it actually does. 
    # If it searches recursively through sub-configs, we test that.
    
    # Based on previous view of context.py, it attempts to find attributes in sub-configs
    assert context.gmx_path == "test_gmx"

def test_context_state_updates():
    context = OptimizationContext(config=SwarmConfig())
    context.nb_eval += 1
    assert context.nb_eval == 1
    
    context.gyr_aa_mapped = 1.5
    assert context.gyr_aa_mapped == 1.5
