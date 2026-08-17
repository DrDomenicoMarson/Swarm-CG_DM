import pytest
from argparse import Namespace
from pydantic import ValidationError

from swarmcg.config_types import GromacsConfig, OptimizationConfig, SimulationConfig, SwarmConfig

def test_swarm_config_defaults():
    config = SwarmConfig()
    assert isinstance(config.gromacs, GromacsConfig)
    assert config.gromacs.gmx_path == "gmx"
    assert config.gromacs.nb_threads == 0
    assert config.gromacs.ntomp == 0

def test_swarm_config_from_namespace():
    ns = Namespace(
        gmx_path="gmx_mpi",
        nb_threads=4,
        ntomp=2,
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
    assert config.gromacs.ntomp == 2
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


@pytest.mark.parametrize(
    "factory",
    [
        lambda: GromacsConfig(nb_threads=-1),
        lambda: GromacsConfig(sim_kill_delay=9),
        lambda: SimulationConfig(temp=0),
        lambda: SimulationConfig(sim_duration_short=-1),
        lambda: OptimizationConfig(bw_dihedrals=0),
        lambda: OptimizationConfig(max_abs_rb_coefficient=0),
    ],
)
def test_scientific_numeric_validation(factory):
    with pytest.raises(ValidationError):
        factory()
