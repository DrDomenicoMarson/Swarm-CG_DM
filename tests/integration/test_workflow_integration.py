
import os
import pytest
import numpy as np
from types import SimpleNamespace
from swarmcg.scoring.evaluator import SwarmEvaluator
from swarmcg.config_types import SwarmConfig
from swarmcg.context import OptimizationContext
import swarmcg.config as config

TEST_DATA = "tests/data/"

@pytest.fixture
def real_data_config():
    # Construct a config object pointing to real test data
    ns = SimpleNamespace()
    
    # Paths from tests/conftest.py logic but simplified
    ns.aa_tpr_filename = os.path.join(TEST_DATA, "aa_topol.tpr")
    ns.aa_traj_filename = os.path.join(TEST_DATA, "aa_traj.xtc")
    ns.cg_map_filename = os.path.join(TEST_DATA, "cg_map.ndx")
    ns.cg_itp_filename = os.path.join(TEST_DATA, "cg_model.itp")
    ns.mapping_type = "COM"
    
    # Required defaults for objects to function
    ns.bonds_scaling = 1.0
    ns.min_bonds_length = 0.0
    ns.bonds_scaling_str = ""
    ns.bw_constraints = 0.002
    ns.bw_bonds = 0.01
    ns.bw_angles = 2.5
    ns.bw_dihedrals = 2.5
    ns.bonds_max_range = 15
    ns.aa_rg_offset = 0.0
    ns.verbose = False
    
    # Pydantic conversion
    # We populate the SwarmConfig sections
    c = SwarmConfig()
    c.reference.aa_tpr_filename = ns.aa_tpr_filename
    c.reference.aa_traj_filename = ns.aa_traj_filename
    c.reference.cg_map_filename = ns.cg_map_filename
    c.reference.mapping_type = ns.mapping_type
    
    c.cg_model.cg_itp_filename = ns.cg_itp_filename
    
    c.optimization.bonds_scaling = ns.bonds_scaling
    c.optimization.min_bonds_length = ns.min_bonds_length
    c.optimization.bonds_scaling_str = ns.bonds_scaling_str
    
    # Internal bins config (usually set by scores.create_bins...)
    # We will let Evaluator initialize handle this via ns context
    
    return c

def test_evaluator_workflow_real_data(real_data_config):
    # This test runs the actual data loading and mapping process
    # It verifies that we can read the files, map the trajectory, and get distributions
    
    evaluator = SwarmEvaluator(real_data_config)
    
    
    # Mock context - in real app, context is populated with config data + runtime data
    context = OptimizationContext(config=real_data_config)
    
    pass
    
    # Initialize (Heavy lifting: I/O + Mapping)
    # This calls:
    # 1. scores.create_bins_and_dist_matrices(ns)
    # 2. Mapping() -> read_ndx
    # 3. io.read_cg_itp
    # 4. io.read_aa_traj
    # 5. mapping.map_aa2cg_traj (The big loop we optimized)
    
    print("\nStarting Evaluator Initialization with Real Data...")
    evaluator.initialize(context)
    print("Initialization Complete.")
    
    # Verify State
    assert context.scoring.aa_universe is not None
    assert len(context.scoring.aa_universe.trajectory) > 0
    
    assert context.scoring.aa2cg_universe is not None
    assert len(context.scoring.aa2cg_universe.trajectory) == len(context.scoring.aa_universe.trajectory)
    
    assert context.scoring.all_beads is not None
    assert len(context.scoring.all_beads) > 0
    
    # Now run compute_reference_distributions (The vectorized scoring)
    print("Computing Reference Distributions...")
    evaluator.compute_reference_distributions()
    print("Computation Complete.")
    
    # Verify Distributions
    # Check if we have populated distribution arrays in cg_itp (that's where they end up)
    
    # Constraints/Bonds check
    nb_bonds = context.cg_itp["nb_bonds"]
    if nb_bonds > 0:
        first_bond = context.cg_itp["bond"][0]
        assert "hist" in first_bond
        assert "avg" in first_bond
        assert first_bond["avg"] > 0
        assert np.sum(first_bond["hist"]) > 0 # Should have some probability mass
        
    # Angles check
    nb_angles = context.cg_itp["nb_angles"]
    if nb_angles > 0:
        first_angle = context.cg_itp["angle"][0]
        assert "hist" in first_angle
        assert "avg" in first_angle
        # Angles in degrees, usually around 100-120 etc
        assert first_angle["avg"] > 0
        
    print(f"Verified {nb_bonds} bond groups and {nb_angles} angle groups.")
