import os
from dataclasses import replace

import pytest

import swarmcg
from swarmcg import config
from swarmcg.context import SwarmCGArgs, SwarmCGState
from swarmcg.simulations.runner import build_simulation_setup, SimulationStep
from swarmcg.simulations.simulation_steps import Minimisation, Equilibration, Production

TEST_DATA = "tests/data/"
ROOT_DIR = os.path.dirname(swarmcg.__file__)


@pytest.fixture(scope="module")
def opt_ctx():

    args_base = SwarmCGArgs(
        exec_mode=1,
        aa_tpr_filename=f"{TEST_DATA}{config.metavar_aa_tpr}",
        aa_traj_filename=f"{TEST_DATA}{config.metavar_aa_traj}",
        cg_map_filename=f"{TEST_DATA}{config.metavar_cg_map}",
        mapping_type="COM",
        cg_itp_filename=f"{TEST_DATA}{config.metavar_cg_itp}",
        user_input=False,
        gro_input_filename=f"{TEST_DATA}start_conf.gro",
        top_input_filename=f"{TEST_DATA}system.top",
        mdp_minimization_filename=f"{ROOT_DIR}/data/mini.mdp",
        mdp_equi_filename=f"{ROOT_DIR}/data/equi.mdp",
        mdp_md_filename=f"{ROOT_DIR}/data/md.mdp",
        output_folder=".",
        input_folder="",
        gmx_path=config.gmx_path,
        nb_threads=0,
        mpi_tasks=0,
        gpu_id="",
        gmx_args_str="",
        mini_maxwarn=0,
        sim_kill_delay=60,
        default_max_fct_bonds_opti=config.default_max_fct_bonds_opti,
        default_max_fct_angles_opti_f1=config.default_max_fct_angles_opti_f1,
        default_max_fct_angles_opti_f2=config.default_max_fct_angles_opti_f2,
        default_abs_range_fct_dihedrals_opti_func_with_mult=config.default_abs_range_fct_dihedrals_opti_func_with_mult,
        default_abs_range_fct_dihedrals_opti_func_without_mult=config.default_abs_range_fct_dihedrals_opti_func_without_mult,
        sim_duration_short=10,
        sim_duration_long=25,
        bonds2angles_scoring_factor=config.bonds2angles_scoring_factor,
        bw_constraints=config.bw_constraints,
        bw_bonds=config.bw_bonds,
        bw_angles=config.bw_angles,
        bw_dihedrals=config.bw_dihedrals,
        bonded_max_range=config.bonds_max_range,
        aa_rg_offset=0.0,
        bonds_scaling=config.bonds_scaling,
        bonds_scaling_str=config.bonds_scaling_str,
        min_bonds_length=config.min_bonds_length,
        mismatch_order=False,
        row_x_scaling=True,
        row_y_scaling=True,
        ncols_max=0,
        temp=config.sim_temperature,
        keep_all_sims=False,
        verbose=True,
    )

    # add value added in the optimisation process
    state_base = SwarmCGState(
        exec_folder="./MODEL_FOLDER",
        cg_itp_basename=args_base.cg_itp_filename,
        gro_input_basename=args_base.gro_input_filename,
        top_input_basename=args_base.top_input_filename,
        mdp_minimization_basename=args_base.mdp_minimization_filename,
        mdp_equi_basename=args_base.mdp_equi_filename,
        mdp_md_basename=args_base.mdp_md_filename,
        molname_in=None,
        process_alive_time_sleep=10,
        process_alive_nb_cycles_dead=int(args_base.sim_kill_delay / 10),
        bonds_rescaling_performed=False,
    )

    def make(**kwargs):
        args_overrides = {k: v for k, v in kwargs.items() if k in SwarmCGArgs.__dataclass_fields__}
        state_overrides = {k: v for k, v in kwargs.items() if k in SwarmCGState.__dataclass_fields__}
        unknown = set(kwargs) - set(args_overrides) - set(state_overrides)
        if unknown:
            raise KeyError(f"Unknown fields for SwarmCGArgs/SwarmCGState: {sorted(unknown)}")
        return replace(args_base, **args_overrides), replace(state_base, **state_overrides)

    return make


@pytest.fixture(scope="module")
def mini():
    return Minimisation(f"{ROOT_DIR}/data/mini.mdp")


@pytest.fixture(scope="module")
def equi():
    return Equilibration(f"{ROOT_DIR}/data/equi.mdp")


@pytest.fixture(scope="module")
def md():
    return Production(f"{ROOT_DIR}/data/md.mdp")


@pytest.fixture(scope="module")
def simstep_mini(opt_ctx, mini):
    args, state = opt_ctx()
    return build_simulation_setup(args, state, mini, f"{TEST_DATA}start_conf.gro")


@pytest.fixture(scope="module")
def simstep_equi(opt_ctx, equi):
    args, state = opt_ctx()
    return build_simulation_setup(args, state, equi, f"{TEST_DATA}mini.gro")


@pytest.fixture(scope="module")
def simstep_md(opt_ctx, md):
    args, state = opt_ctx()
    return build_simulation_setup(args, state, md, f"{TEST_DATA}equi.gro")
