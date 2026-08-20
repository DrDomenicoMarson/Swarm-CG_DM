import pytest
from argparse import Namespace
from pathlib import Path
from pydantic import ValidationError

from swarmcg.config_types import (
    GromacsConfig,
    OptimizationConfig,
    OutputConfig,
    ReferenceModelConfig,
    SimulationConfig,
    SwarmConfig,
)
from swarmcg.context import OptimizationContext
from swarmcg.analyze_optimization import _validated_plot_scale
from swarmcg.io.job_args.analyze_config import get_analyze_args
from swarmcg.io.job_args.optimize_config import get_optimize_args
from swarmcg.shared import exceptions
from swarmcg.utils import process_scaling_str
from swarmcg.topology import CGTopology

def test_swarm_config_defaults():
    config = SwarmConfig()
    assert isinstance(config.gromacs, GromacsConfig)
    assert config.gromacs.gmx_path == "gmx"
    assert config.gromacs.nb_threads == 0
    assert config.gromacs.ntomp == 0
    for filename in (
        config.simulation.mdp_minimization_filename,
        config.simulation.mdp_equi_filename,
        config.simulation.mdp_md_filename,
    ):
        path = Path(filename)
        assert path.is_file()
        assert path.parent.name == "data"

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
    assert Path(config.simulation.mdp_minimization_filename).is_file()
    assert Path(config.simulation.mdp_equi_filename).is_file()
    assert Path(config.simulation.mdp_md_filename).is_file()


def test_monitor_supports_standard_long_help_flag():
    """The monitor accepts the same standard help spelling as other CLIs."""
    with pytest.raises(SystemExit) as exit_info:
        get_analyze_args().parse_args(["--help"])

    assert exit_info.value.code == 0

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
        lambda: SimulationConfig(temp=float("nan")),
        lambda: SimulationConfig(sim_duration_long=float("inf")),
        lambda: OptimizationConfig(bw_angles=float("nan")),
        lambda: OptimizationConfig(max_abs_cbt_effective_coefficient=float("inf")),
        lambda: OptimizationConfig(bond_dist_guess_variation=float("nan")),
        lambda: OptimizationConfig(bonds2angles_scoring_factor=float("inf")),
        lambda: OptimizationConfig(bonded_max_range=float("nan")),
        lambda: OptimizationConfig(bonds_scaling=float("inf")),
        lambda: OptimizationConfig(min_bonds_length=float("nan")),
        lambda: ReferenceModelConfig(aa_rg_offset=float("inf")),
        lambda: OutputConfig(plot_scale=float("inf")),
    ],
)
def test_scientific_numeric_validation(factory):
    with pytest.raises(ValidationError):
        factory()


@pytest.mark.parametrize(
    "option,value,field",
    [
        ("-temp", "nan", "temp"),
        ("-cg_time_short", "inf", "sim_duration_short"),
        ("-bw_bonds", "nan", "bw_bonds"),
        ("-b2a_score_fact", "inf", "bonds2angles_scoring_factor"),
        ("-max_rb_coeff", "-inf", "max_abs_rb_coefficient"),
        ("-bonds_max_range", "inf", "bonded_max_range"),
        ("-aa_rg_offset", "nan", "aa_rg_offset"),
    ],
)
def test_cli_rejects_nonfinite_scientific_values(option, value, field):
    arguments = [f"{option}={value}"] if value.startswith("-") else [option, value]
    namespace = get_optimize_args().parse_args(arguments)

    with pytest.raises(ValidationError, match=field):
        SwarmConfig.from_namespace(namespace)


@pytest.mark.parametrize("value", ["0", "nan", "inf", "-inf"])
def test_group_specific_scaling_rejects_invalid_lengths(value):
    config = SwarmConfig(
        optimization=OptimizationConfig(bonds_scaling_str=f"B1 {value}")
    )
    context = OptimizationContext(config=config)
    context.cg_itp = CGTopology(bonds=[object()])

    with pytest.raises(exceptions.InvalidArgument, match="finite and positive"):
        process_scaling_str(context)


@pytest.mark.parametrize("value", ["0", "nan", "inf", "-inf"])
def test_monitor_cli_rejects_invalid_plot_scale(value):
    arguments = [f"-plot_scale={value}"] if value.startswith("-") else ["-plot_scale", value]
    namespace = get_analyze_args().parse_args(arguments)

    with pytest.raises(ValidationError, match="plot_scale"):
        _validated_plot_scale(namespace.plot_scale)
