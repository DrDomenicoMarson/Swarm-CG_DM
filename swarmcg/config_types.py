import os
from dataclasses import dataclass, field


@dataclass
class GromacsConfig:
    gmx_path: str = "gmx"
    nb_threads: int = 0
    mpi_tasks: int = 0
    gpu_id: str = ""
    gmx_args_str: str = ""
    mini_maxwarn: int = 1
    sim_kill_delay: int = 60

@dataclass
class ReferenceModelConfig:
    aa_tpr_filename: str = "aa_topol.tpr"
    aa_traj_filename: str = "aa_traj.xtc"
    cg_map_filename: str = "cg_map.ndx"
    mapping_type: str = "COM"
    aa_rg_offset: float = 0.0

@dataclass
class CGModelConfig:
    cg_itp_filename: str = "cg_model.itp"
    gro_input_filename: str = "start_conf.gro"
    top_input_filename: str = "system.top"
    cg_tpr_filename: str = "cg_topol.tpr"
    cg_traj_filename: str = "cg_traj.xtc"
    user_input: bool = False
    
@dataclass
class SimulationConfig:
    # MDP files
    mdp_minimization_filename: str = "honeycomb/data/mini.mdp" # Placeholder, needs dynamic path
    mdp_equi_filename: str = "honeycomb/data/equi.mdp" # Placeholder
    mdp_md_filename: str = "honeycomb/data/md.mdp" # Placeholder
    
    # Simulation Parameters
    sim_duration_short: float = 10.0 # ns
    sim_duration_long: float = 25.0 # ns
    temp: float = 300.0 # Kelvin

@dataclass
class OptimizationConfig:
    exec_mode: int = 1
    sim_type: str = "OPTIMAL"
    
    # Force constants limits
    default_max_fct_bonds_opti: float = 18000.0
    default_max_fct_angles_opti_f1: float = 1700.0
    default_max_fct_angles_opti_f2: float = 1700.0
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

@dataclass
class OutputConfig:
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

@dataclass
class SwarmConfig:
    gromacs: GromacsConfig = field(default_factory=GromacsConfig)
    reference: ReferenceModelConfig = field(default_factory=ReferenceModelConfig)
    cg_model: CGModelConfig = field(default_factory=CGModelConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

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
            keep_all_sims=getattr(ns, 'keep_all_sims', False)
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

        return cls(
            gromacs=gromacs,
            reference=reference,
            cg_model=cg_model,
            simulation=simulation,
            optimization=optimization,
            output=output
        )

