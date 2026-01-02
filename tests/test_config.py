import pytest
from argparse import Namespace
from swarmcg.config_types import SwarmConfig, GromacsConfig

def test_swarm_config_defaults():
    config = SwarmConfig()
    assert isinstance(config.gromacs, GromacsConfig)
    assert config.gromacs.gmx_path == "gmx"
    assert config.gromacs.nb_threads == 0

def test_swarm_config_from_namespace():
    ns = Namespace(
        gmx_path="gmx_mpi",
        nb_threads=4,
        mpi_tasks=2,
        gpu_id="0",
        gmx_args_str="-v",
        mini_maxwarn=2,
        sim_kill_delay=120,
        
        # Add minimal required fields for other configs to pass if they have validation
        # Assuming defaults handle most missing fields
    )
    
    config = SwarmConfig.from_namespace(ns)
    
    assert config.gromacs.gmx_path == "gmx_mpi"
    assert config.gromacs.nb_threads == 4
    assert config.gromacs.mpi_tasks == 2
    assert config.gromacs.gpu_id == "0"
    assert config.gromacs.gmx_args_str == "-v"
    assert config.gromacs.mini_maxwarn == 2
    assert config.gromacs.sim_kill_delay == 120

def test_config_immutability_partial():
    # Dataclasses are mutable by default, but we check basic integrity
    config = SwarmConfig()
    config.gromacs.gmx_path = "new_path"
    assert config.gromacs.gmx_path == "new_path"
