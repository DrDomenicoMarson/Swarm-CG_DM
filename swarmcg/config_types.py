
import os
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from swarmcg import config as config_module
from swarmcg.shared.logging_utils import get_logger

logger = get_logger(__name__)

class GromacsConfig(BaseModel):
    gmx_path: str = "gmx"
    nb_threads: int = 0
    mpi_tasks: int = 0
    gpu_id: str = ""
    gmx_args_str: str = ""
    mini_maxwarn: int = 1
    sim_kill_delay: int = 60

    @model_validator(mode='after')
    def check_gmx_args_conflict(self):
        if self.gmx_args_str != "" and (self.nb_threads != 0 or self.gpu_id != ""):
            logger.warning(
                "Argument -gmx_args_str is provided together with -nb_threads or -gpu_id; "
                "only -gmx_args_str will be used."
            )
        return self

class ReferenceModelConfig(BaseModel):
    aa_tpr_filename: str = "aa_topol.tpr"
    aa_traj_filename: str = "aa_traj.xtc"
    cg_map_filename: str = "cg_map.ndx"
    mapping_type: str = "COM"
    aa_rg_offset: float = 0.0

    @field_validator('mapping_type')
    @classmethod
    def check_mapping_type(cls, v: str) -> str:
        if v.upper() not in ('COM', 'COG'):
            raise ValueError(
                "Mapping type provided via argument '-mapping' must be either COM or COG "
                "(Center of Mass or Center of Geometry)."
            )
        return v.upper()

class CGModelConfig(BaseModel):
    cg_itp_filename: str = "cg_model.itp"
    gro_input_filename: str = "start_conf.gro"
    top_input_filename: str = "system.top"
    cg_tpr_filename: str = "cg_topol.tpr"
    cg_traj_filename: str = "cg_traj.xtc"
    user_input: bool = False

class SimulationConfig(BaseModel):
    # MDP files
    mdp_minimization_filename: str = "honeycomb/data/mini.mdp"
    mdp_equi_filename: str = "honeycomb/data/equi.mdp"
    mdp_md_filename: str = "honeycomb/data/md.mdp"
    
    # Simulation Parameters
    sim_duration_short: float = 10.0 # ns
    sim_duration_long: float = 25.0 # ns
    temp: float = 300.0 # Kelvin

class OptimizationConfig(BaseModel):
    exec_mode: int = 1
    sim_type: str = "OPTIMAL"
    
    # Force constants limits
    default_max_fct_bonds_opti: float = 18000.0
    default_max_fct_angles_opti_f1: float = 1700.0
    default_max_fct_angles_opti_f2: float = 1700.0
    default_max_fct_angles_opti_f10: float = 1700.0
    default_abs_range_fct_dihedrals_opti_func_with_mult: float = 15.0
    default_abs_range_fct_dihedrals_opti_func_without_mult: float = 1500.0
    
    # Scoring
    bonds2angles_scoring_factor: float = 500.0
    bw_constraints: float = 0.002
    bw_bonds: float = 0.01
    bw_angles: float = 2.5
    bw_dihedrals: float = 2.5
    bonded_max_range: float = 15.0
    
    # Scaling
    bonds_scaling: float = 1.0
    bonds_scaling_str: str = ""
    min_bonds_length: float = 0.0
    
    # Other
    keep_all_sims: bool = False
    bi_nb_bins: int = 50
    bond_dist_guess_variation: float = config_module.bond_dist_guess_variation
    angle_value_guess_variation: float = config_module.angle_value_guess_variation
    dihedral_value_guess_variation: float = config_module.dihedral_value_guess_variation
    fct_guess_min_flat_diff_bonds: float = config_module.fct_guess_min_flat_diff_bonds
    fct_guess_min_flat_diff_angles: float = config_module.fct_guess_min_flat_diff_angles
    fct_guess_min_flat_diff_dihedrals_without_mult: float = config_module.fct_guess_min_flat_diff_dihedrals_without_mult
    fct_guess_min_flat_diff_dihedrals_with_mult: float = config_module.fct_guess_min_flat_diff_dihedrals_with_mult
    sim_crash_EMD_indep_score: float = config_module.sim_crash_EMD_indep_score

    @field_validator('default_max_fct_bonds_opti', 'default_max_fct_angles_opti_f1',
                     'default_max_fct_angles_opti_f2', 'default_max_fct_angles_opti_f10')
    @classmethod
    def check_positive(cls, v: float, info: Any) -> float:
        if v <= 0:
            raise ValueError(f"Please provide a value > 0 for argument corresponding to {info.field_name}.")
        return v
    
    @model_validator(mode='after')
    def check_bonds_scaling_conflicts(self):
        # We need to know default values to check if they were modified, or rely on logic that inputs were mutually exclusive.
        # Since we are converting from namespace where defaults are populated if missing, detecting "user provided" vs "default" is hard here
        # unless we assume defaults.
        # However, the logic in validation.py checked against 'config' object defaults?
        # Actually validation.py checked: ns.bonds_scaling != config.bonds_scaling
        # Here we only have the model. A robust way is ensuring only one is non-default?
        # For now, let's keep it simple or skip this complex cross-validation if we trust argparse mutual exclusion.
        # But argparse mutual exclusion wasn't strictly enforced in legacy code, validation.py did it.
        # Pydantic validation happens *after* population.
        
        # If we want to strictly follow validation.py:
        # "Only one of arguments -bonds_scaling, -bonds_scaling_str and -min_bonds_length can be provided."
        # This implies checking if more than one is "set".
        # But here they have values.
        # We can implement a simplified check:
        # If bonds_scaling != 1.0, others should be default.
        # If bonds_scaling_str != "", others should be default.
        # If min_bonds_length != 0.0, others should be default.
        
        set_count = 0
        if self.bonds_scaling != 1.0: set_count += 1
        if self.bonds_scaling_str != "": set_count += 1
        if self.min_bonds_length != 0.0: set_count += 1
        
        if set_count > 1:
            raise ValueError(
                "Only one of arguments -bonds_scaling, -bonds_scaling_str and -min_bonds_length "
                "can be provided. Please check your parameters"
            )
        return self

class OutputConfig(BaseModel):
    input_folder: str = ""
    output_folder: str = ""
    opti_dirname: str = ""
    plot_filename: str = "distributions.png"
    mismatch_order: bool = False
    row_x_scaling: bool = True
    row_y_scaling: bool = True
    ncols_max: int = 0
    plot_scale: float = 1.0
    verbose: bool = False

class SwarmConfig(BaseModel):
    gromacs: GromacsConfig = Field(default_factory=GromacsConfig)
    reference: ReferenceModelConfig = Field(default_factory=ReferenceModelConfig)
    cg_model: CGModelConfig = Field(default_factory=CGModelConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    
    # We can add a method to validation files existence if desired, 
    # but Pydantic validators usually validate *data*, not external state (like file existence).
    # However, for a CLI tool, validating file existence at configuration load is good practice.
    
    def validate_files_exist(self):
        """Check for existence of input files."""
        # Reference files
        if not os.path.isfile(self.reference.aa_tpr_filename):
            raise FileNotFoundError(
                f"Cannot find topology file of the atomistic simulation at location: {self.reference.aa_tpr_filename}\n"
                f"(TPR or other portable topology formats supported by MDAnalysis)"
            )
        if not os.path.isfile(self.reference.aa_traj_filename):
            raise FileNotFoundError(
                f"Cannot find trajectory file of the atomistic simulation at location: {self.reference.aa_traj_filename}\n"
                f"(XTC, TRR, or other trajectory formats supported by MDAnalysis)"
            )
        if not os.path.isfile(self.reference.cg_map_filename):
            raise FileNotFoundError(
                f"Cannot find CG beads mapping file at location: {self.reference.cg_map_filename}\n"
                f"(NDX-like file format)"
            )
            
        # CG Model file
        if not os.path.isfile(self.cg_model.cg_itp_filename):
            raise FileNotFoundError(f"Cannot find ITP file of the CG model at location: {self.cg_model.cg_itp_filename}")

    @classmethod
    def from_namespace(cls, ns) -> 'SwarmConfig':
        """
        Convert argparse Namespace (legacy ns object) to SwarmConfig.
        """
        # Gromacs
        gromacs = GromacsConfig(
            gmx_path=getattr(ns, 'gmx_path', 'gmx'),
            nb_threads=getattr(ns, 'nb_threads', 0),
            mpi_tasks=getattr(ns, 'mpi_tasks', 0),
            gpu_id=getattr(ns, 'gpu_id', ""),
            gmx_args_str=getattr(ns, 'gmx_args_str', ""),
            mini_maxwarn=getattr(ns, 'mini_maxwarn', 1),
            sim_kill_delay=getattr(ns, 'sim_kill_delay', 60)
        )

        # Reference Model
        reference = ReferenceModelConfig(
            aa_tpr_filename=getattr(ns, 'aa_tpr_filename', "aa_topol.tpr"),
            aa_traj_filename=getattr(ns, 'aa_traj_filename', "aa_traj.xtc"),
            cg_map_filename=getattr(ns, 'cg_map_filename', "cg_map.ndx"),
            mapping_type=getattr(ns, 'mapping_type', "COM"),
            aa_rg_offset=getattr(ns, 'aa_rg_offset', 0.0)
        )

        # CG Model
        cg_model = CGModelConfig(
            cg_itp_filename=getattr(ns, 'cg_itp_filename', "cg_model.itp"),
            gro_input_filename=getattr(ns, 'gro_input_filename', "start_conf.gro"),
            top_input_filename=getattr(ns, 'top_input_filename', "system.top"),
            cg_tpr_filename=getattr(ns, 'cg_tpr_filename', "cg_topol.tpr"),
            cg_traj_filename=getattr(ns, 'cg_traj_filename', "cg_traj.xtc"),
            user_input=getattr(ns, 'user_input', False)
        )

        # Simulation
        simulation = SimulationConfig(
            mdp_minimization_filename=getattr(ns, 'mdp_minimization_filename', ""),
            mdp_equi_filename=getattr(ns, 'mdp_equi_filename', ""),
            mdp_md_filename=getattr(ns, 'mdp_md_filename', ""),
            sim_duration_short=getattr(ns, 'sim_duration_short', 10.0),
            sim_duration_long=getattr(ns, 'sim_duration_long', 25.0),
            temp=getattr(ns, 'temp', 300.0)
        )

        # Optimization
        optimization = OptimizationConfig(
            exec_mode=getattr(ns, 'exec_mode', 1),
            sim_type=getattr(ns, 'sim_type', "OPTIMAL"),
            default_max_fct_bonds_opti=getattr(ns, 'default_max_fct_bonds_opti', 18000.0),
            default_max_fct_angles_opti_f1=getattr(ns, 'default_max_fct_angles_opti_f1', 1700.0),
            default_max_fct_angles_opti_f2=getattr(ns, 'default_max_fct_angles_opti_f2', 1700.0),
            default_max_fct_angles_opti_f10=getattr(ns, 'default_max_fct_angles_opti_f10', 1700.0),
            default_abs_range_fct_dihedrals_opti_func_with_mult=getattr(ns, 'default_abs_range_fct_dihedrals_opti_func_with_mult', 15.0),
            default_abs_range_fct_dihedrals_opti_func_without_mult=getattr(ns, 'default_abs_range_fct_dihedrals_opti_func_without_mult', 1500.0),
            bonds2angles_scoring_factor=getattr(ns, 'bonds2angles_scoring_factor', 500.0),
            bw_constraints=getattr(ns, 'bw_constraints', 0.002),
            bw_bonds=getattr(ns, 'bw_bonds', 0.01),
            bw_angles=getattr(ns, 'bw_angles', 2.5),
            bw_dihedrals=getattr(ns, 'bw_dihedrals', 2.5),
            bonded_max_range=getattr(ns, 'bonded_max_range', 15.0),
            bonds_scaling=getattr(ns, 'bonds_scaling', 1.0),
            bonds_scaling_str=getattr(ns, 'bonds_scaling_str', ""),
            min_bonds_length=getattr(ns, 'min_bonds_length', 0.0),
            keep_all_sims=getattr(ns, 'keep_all_sims', False),
            bi_nb_bins=getattr(ns, 'bi_nb_bins', 50),
            bond_dist_guess_variation=getattr(ns, 'bond_dist_guess_variation', config_module.bond_dist_guess_variation),
            angle_value_guess_variation=getattr(ns, 'angle_value_guess_variation', config_module.angle_value_guess_variation),
            dihedral_value_guess_variation=getattr(ns, 'dihedral_value_guess_variation', config_module.dihedral_value_guess_variation),
            fct_guess_min_flat_diff_bonds=getattr(ns, 'fct_guess_min_flat_diff_bonds', config_module.fct_guess_min_flat_diff_bonds),
            fct_guess_min_flat_diff_angles=getattr(ns, 'fct_guess_min_flat_diff_angles', config_module.fct_guess_min_flat_diff_angles),
            fct_guess_min_flat_diff_dihedrals_without_mult=getattr(ns, 'fct_guess_min_flat_diff_dihedrals_without_mult', config_module.fct_guess_min_flat_diff_dihedrals_without_mult),
            fct_guess_min_flat_diff_dihedrals_with_mult=getattr(ns, 'fct_guess_min_flat_diff_dihedrals_with_mult', config_module.fct_guess_min_flat_diff_dihedrals_with_mult),
            sim_crash_EMD_indep_score=getattr(ns, 'sim_crash_EMD_indep_score', config_module.sim_crash_EMD_indep_score)
        )

        # Output
        output = OutputConfig(
            input_folder=getattr(ns, 'input_folder', ""),
            output_folder=getattr(ns, 'output_folder', ""),
            opti_dirname=getattr(ns, 'opti_dirname', ""),
            plot_filename=getattr(ns, 'plot_filename', "distributions.png"),
            mismatch_order=getattr(ns, 'mismatch_order', False),
            row_x_scaling=getattr(ns, 'row_x_scaling', True),
            row_y_scaling=getattr(ns, 'row_y_scaling', True),
            ncols_max=getattr(ns, 'ncols_max', 0),
            plot_scale=getattr(ns, 'plot_scale', 1.0),
            verbose=getattr(ns, 'verbose', False)
        )

        config = cls(
            gromacs=gromacs,
            reference=reference,
            cg_model=cg_model,
            simulation=simulation,
            optimization=optimization,
            output=output
        )
        # Run file existence validation immediately after creation, 
        # mimicking the original 'input_parameter_validation' call in main/entry points.
        # However, for unit tests decoupling, we might want to call this explicitly.
        # But 'from_namespace' is main entry point util, so it's safe to validate here if files are expected.
        # NOTE: If unit tests mock files, this will fail if it runs os.path.isfile. 
        # So we should perhaps call validate_files_exist() explicitly in entry points instead of here.
        # I will leave it to be called explicitly.
        
        return config
