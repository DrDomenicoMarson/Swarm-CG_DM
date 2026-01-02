import os
from dataclasses import replace

import pytest

import swarmcg
from swarmcg import config
from swarmcg.context import (
    SwarmCGArgs,
    SwarmCGState,
    RuntimeArgs,
    InputArgs,
    PathArgs,
    OptimizationArgs,
    PlotArgs,
    FileState,
    RuntimeState,
    MappingState,
)
from swarmcg.simulations.runner import build_simulation_setup, SimulationStep
from swarmcg.simulations.simulation_steps import Minimisation, Equilibration, Production

TEST_DATA = "tests/data/"
ROOT_DIR = os.path.dirname(swarmcg.__file__)


@pytest.fixture(scope="module")
def opt_ctx():

    args_base = SwarmCGArgs(
        runtime=RuntimeArgs(
            exec_mode=1,
            sim_type="OPTIMAL",
            gmx_path=config.gmx_path,
            nb_threads=0,
            mpi_tasks=0,
            gpu_id="",
            gmx_args_str="",
            mini_maxwarn=0,
            sim_kill_delay=60,
            keep_all_sims=False,
            verbose=True,
        ),
        inputs=InputArgs(
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
        ),
        paths=PathArgs(
            output_folder=".",
            input_folder="",
        ),
        optimization=OptimizationArgs(
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
            temp=config.sim_temperature,
        ),
        plotting=PlotArgs(
            mismatch_order=False,
            row_x_scaling=True,
            row_y_scaling=True,
            ncols_max=0,
        ),
    )

    # add value added in the optimisation process
    state_base = SwarmCGState(
        files=FileState(
            exec_folder="./MODEL_FOLDER",
            cg_itp_basename=args_base.inputs.cg_itp_filename,
            gro_input_basename=args_base.inputs.gro_input_filename,
            top_input_basename=args_base.inputs.top_input_filename,
            mdp_minimization_basename=args_base.inputs.mdp_minimization_filename,
            mdp_equi_basename=args_base.inputs.mdp_equi_filename,
            mdp_md_basename=args_base.inputs.mdp_md_filename,
        ),
        runtime=RuntimeState(
            process_alive_time_sleep=10,
            process_alive_nb_cycles_dead=int(args_base.runtime.sim_kill_delay / 10),
        ),
        mapping=MappingState(
            molname_in=None,
            bonds_rescaling_performed=False,
        ),
    )

    def make(**kwargs):
        args_field_map = {
            "exec_mode": ("runtime", "exec_mode"),
            "sim_type": ("runtime", "sim_type"),
            "gmx_path": ("runtime", "gmx_path"),
            "nb_threads": ("runtime", "nb_threads"),
            "mpi_tasks": ("runtime", "mpi_tasks"),
            "gpu_id": ("runtime", "gpu_id"),
            "gmx_args_str": ("runtime", "gmx_args_str"),
            "mini_maxwarn": ("runtime", "mini_maxwarn"),
            "sim_kill_delay": ("runtime", "sim_kill_delay"),
            "keep_all_sims": ("runtime", "keep_all_sims"),
            "verbose": ("runtime", "verbose"),
            "aa_tpr_filename": ("inputs", "aa_tpr_filename"),
            "aa_traj_filename": ("inputs", "aa_traj_filename"),
            "cg_map_filename": ("inputs", "cg_map_filename"),
            "mapping_type": ("inputs", "mapping_type"),
            "cg_itp_filename": ("inputs", "cg_itp_filename"),
            "user_input": ("inputs", "user_input"),
            "gro_input_filename": ("inputs", "gro_input_filename"),
            "top_input_filename": ("inputs", "top_input_filename"),
            "cg_tpr_filename": ("inputs", "cg_tpr_filename"),
            "cg_traj_filename": ("inputs", "cg_traj_filename"),
            "mdp_minimization_filename": ("inputs", "mdp_minimization_filename"),
            "mdp_equi_filename": ("inputs", "mdp_equi_filename"),
            "mdp_md_filename": ("inputs", "mdp_md_filename"),
            "input_folder": ("paths", "input_folder"),
            "output_folder": ("paths", "output_folder"),
            "opti_dirname": ("paths", "opti_dirname"),
            "plot_filename": ("paths", "plot_filename"),
            "default_max_fct_bonds_opti": ("optimization", "default_max_fct_bonds_opti"),
            "default_max_fct_angles_opti_f1": ("optimization", "default_max_fct_angles_opti_f1"),
            "default_max_fct_angles_opti_f2": ("optimization", "default_max_fct_angles_opti_f2"),
            "default_abs_range_fct_dihedrals_opti_func_with_mult": (
                "optimization",
                "default_abs_range_fct_dihedrals_opti_func_with_mult",
            ),
            "default_abs_range_fct_dihedrals_opti_func_without_mult": (
                "optimization",
                "default_abs_range_fct_dihedrals_opti_func_without_mult",
            ),
            "sim_duration_short": ("optimization", "sim_duration_short"),
            "sim_duration_long": ("optimization", "sim_duration_long"),
            "bonds2angles_scoring_factor": ("optimization", "bonds2angles_scoring_factor"),
            "bw_constraints": ("optimization", "bw_constraints"),
            "bw_bonds": ("optimization", "bw_bonds"),
            "bw_angles": ("optimization", "bw_angles"),
            "bw_dihedrals": ("optimization", "bw_dihedrals"),
            "bonded_max_range": ("optimization", "bonded_max_range"),
            "aa_rg_offset": ("optimization", "aa_rg_offset"),
            "bonds_scaling": ("optimization", "bonds_scaling"),
            "bonds_scaling_str": ("optimization", "bonds_scaling_str"),
            "min_bonds_length": ("optimization", "min_bonds_length"),
            "temp": ("optimization", "temp"),
            "mismatch_order": ("plotting", "mismatch_order"),
            "row_x_scaling": ("plotting", "row_x_scaling"),
            "row_y_scaling": ("plotting", "row_y_scaling"),
            "ncols_max": ("plotting", "ncols_max"),
            "plot_scale": ("plotting", "plot_scale"),
        }
        state_field_map = {
            "cg_itp_basename": ("files", "cg_itp_basename"),
            "gro_input_basename": ("files", "gro_input_basename"),
            "top_input_basename": ("files", "top_input_basename"),
            "mdp_minimization_basename": ("files", "mdp_minimization_basename"),
            "mdp_equi_basename": ("files", "mdp_equi_basename"),
            "mdp_md_basename": ("files", "mdp_md_basename"),
            "exec_folder": ("files", "exec_folder"),
            "process_alive_time_sleep": ("runtime", "process_alive_time_sleep"),
            "process_alive_nb_cycles_dead": ("runtime", "process_alive_nb_cycles_dead"),
            "mda_backend": ("runtime", "mda_backend"),
            "atom_only": ("mapping", "atom_only"),
            "molname_in": ("mapping", "molname_in"),
            "all_atoms": ("mapping", "all_atoms"),
            "all_aa_mols": ("mapping", "all_aa_mols"),
            "all_beads": ("mapping", "all_beads"),
            "atoms_occ_total": ("mapping", "atoms_occ_total"),
            "atom_w": ("mapping", "atom_w"),
            "mda_beads_atom_grps": ("mapping", "mda_beads_atom_grps"),
            "mda_weights_atom_grps": ("mapping", "mda_weights_atom_grps"),
            "bonds_rescaling_performed": ("mapping", "bonds_rescaling_performed"),
            "bonds_scaling_specific": ("mapping", "bonds_scaling_specific"),
            "aa_universe": ("traj", "aa_universe"),
            "aa2cg_universe": ("traj", "aa2cg_universe"),
            "cg_universe": ("traj", "cg_universe"),
            "cg_itp": ("model", "cg_itp"),
        }

        args_updates = {}
        state_updates = {}
        unknown = []
        for key, value in kwargs.items():
            if key in args_field_map:
                group, field_name = args_field_map[key]
                args_updates.setdefault(group, {})[field_name] = value
            elif key in state_field_map:
                group, field_name = state_field_map[key]
                state_updates.setdefault(group, {})[field_name] = value
            else:
                unknown.append(key)
        if unknown:
            raise KeyError(f"Unknown fields for SwarmCGArgs/SwarmCGState: {sorted(unknown)}")

        args_out = args_base
        for group, updates in args_updates.items():
            group_obj = getattr(args_out, group)
            args_out = replace(args_out, **{group: replace(group_obj, **updates)})

        state_out = state_base
        for group, updates in state_updates.items():
            group_obj = getattr(state_out, group)
            state_out = replace(state_out, **{group: replace(group_obj, **updates)})

        return args_out, state_out

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
