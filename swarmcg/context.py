from dataclasses import dataclass, field
from typing import Any

import numpy as np
from swarmcg.config_types import SwarmConfig

@dataclass
class OptimizationContext:
    """
    Context object to hold the state of the optimization process.
    Replaces the legacy 'ns' namespace object.
    """
    config: SwarmConfig
    
    # State variables
    nb_eval: int = 0
    best_fitness: tuple[float, Any] = field(default_factory=lambda: (np.inf, None))
    
    # Data structures (Topology, Mapping, etc.)
    cg_itp: dict = field(default_factory=dict)
    out_itp: dict = field(default_factory=dict) # The ITP being optimized/modified
    opti_cycle: dict = field(default_factory=dict)
    
    # Results of current evaluation
    gyr_aa_mapped: float | None = None
    gyr_aa_mapped_std: float | None = None
    gyr_cg: float | None = None
    gyr_cg_std: float | None = None
    sasa_aa_mapped: float | None = None
    sasa_aa_mapped_std: float | None = None
    sasa_cg: float | None = None
    sasa_cg_std: float | None = None
    
    # Scoring/Optimization state
    all_best_emd_dist_geoms: dict = field(default_factory=dict)
    all_best_params_dist_geoms: dict = field(default_factory=dict)
    worst_fit_score: float = 0.0
    
    # Simulation/Execution state
    total_eval_time: float = 0.0
    total_gmx_time: float = 0.0
    total_model_eval_time: float = 0.0
    start_opti_ts: float = 0.0
    
    # Other legacy attributes compatibility
    # These were in 'ns' and used across functions.
    # We include them here for compatibility during refactoring.
    aa_universe: Any = None
    cg_universe: Any = None
    aa2cg_universe: Any = None
    mda_beads_atom_grps: dict = field(default_factory=dict)
    mda_weights_atom_grps: dict = field(default_factory=dict)
    
    mismatch_order: bool = False
    row_x_scaling: bool = True
    row_y_scaling: bool = True
    atom_only: bool = False
    molname_in: Any = None
    process_alive_time_sleep: int = 10
    process_alive_nb_cycles_dead: int = 0
    bonds_rescaling_performed: bool = False
    
    # Basenames (derived from config filenames)
    cg_itp_basename: str = ""
    gro_input_basename: str = ""
    top_input_basename: str = ""
    mdp_minimization_basename: str = ""
    mdp_equi_basename: str = ""
    mdp_md_basename: str = ""
    
    # Directories
    exec_folder: str = ""
    
    # Optimization specific
    performed_init_BI: dict = field(default_factory=lambda: {"bond": False, "angle": False, "dihedral": False})
    opti_geoms_all: set = field(default_factory=set)
    domains_val: dict = field(default_factory=dict)
    data_BI: dict = field(default_factory=dict)
    
    # Plots
    plot_filename: str = ""
    
    # Bins (calculated in optimize_model)
    bins_constraints: Any = None
    bins_bonds: Any = None
    bins_angles: Any = None
    bins_dihedrals: Any = None
    bw_constraints: Any = None
    bw_bonds: Any = None
    bw_angles: Any = None
    bw_dihedrals: Any = None
    bins_constraints_dist_matrix: Any = None
    bins_bonds_dist_matrix: Any = None
    bins_angles_dist_matrix: Any = None
    # bins_dihedrals_dist_matrix: Any = None # Not always used?

    # MDA backend
    mda_backend: str = "serial"
    
    def __getattr__(self, name):
        """Fallback to config for attributes not found in context (legacy ns behavior support)"""
        # Dictionary mapping legacy flat attribute names to SwarmConfig sections
        # This is a heuristic based on SwarmConfig definition
        if self.config:
            # Check root
            if hasattr(self.config, name):
                return getattr(self.config, name)
            
            # Check sections
            for section in ['gromacs', 'reference', 'cg_model', 'simulation', 'optimization', 'output']:
                if hasattr(self.config, section):
                    c_section = getattr(self.config, section)
                    if hasattr(c_section, name):
                        return getattr(c_section, name)
             
        raise AttributeError(f"'OptimizationContext' object has no attribute '{name}'")
