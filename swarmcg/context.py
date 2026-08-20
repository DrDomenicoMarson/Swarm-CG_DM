from dataclasses import dataclass, field
from typing import Any, Dict, Set, Tuple, Optional

import numpy as np
from swarmcg.config_types import SwarmConfig
from swarmcg.optimization_types import (
    OptimizationCycle,
    ParameterVectorLayout,
    SimulationSetup,
)
from swarmcg.simulations.boltzmann import BoltzmannTarget
from swarmcg.topology import CGTopology, GeometryKind


@dataclass
class SimulationFiles:
    """Stores filenames and basenames for simulation inputs/outputs."""
    cg_itp_basename: str = ""
    gro_input_basename: str = ""
    top_input_basename: str = ""
    mdp_minimization_basename: str = ""
    mdp_equi_basename: str = ""
    mdp_md_basename: str = ""
    
    # Dynamic filenames during optimization steps
    cg_tpr_filename: str = ""
    cg_traj_filename: str = ""
    plot_filename: str = ""
    
    # Directories
    exec_folder: str = ""


@dataclass
class ScoringState:
    """Store trajectory, mapping, plotting, and histogram-scoring state."""
    # Universes
    aa_universe: Any = None
    cg_universe: Any = None
    aa2cg_universe: Any = None

    # MDA Helper data
    mda_backend: str = "serial"
    mda_beads_atom_grps: Dict = field(default_factory=dict)
    mda_weights_atom_grps: Dict = field(default_factory=dict)

    # Bins/Histograms for scoring
    bins_constraints: Any = None
    bins_bonds: Any = None
    bins_angles: Any = None
    bins_dihedrals: Any = None

    constraints_grid: Any = None
    bonds_grid: Any = None
    angles_grid: Any = None
    dihedrals_grid: Any = None

    # Plotting/Scoring configuration
    mismatch_order: bool = False
    row_x_scaling: bool = True
    row_y_scaling: bool = True
    ncols_max: int = 0
    atom_only: bool = False
    molname_in: Any = None
    
    # Dynamic Mapping Data
    all_atoms: Any = None
    all_aa_mols: Any = None
    all_beads: Any = None
    atom_w: Any = None
    bonds_scaling_specific: Optional[Dict] = None
    
    # Reference Metrics
    gyr_aa: Optional[float] = None
    gyr_aa_std: Optional[float] = None
    
    # Results of BI initialization
    performed_init_BI: Dict = field(default_factory=lambda: {"bond": False, "angle": False, "dihedral": False})
    data_BI: Dict[str, list[BoltzmannTarget]] = field(default_factory=dict)
    domains_val: Dict = field(default_factory=dict) # Search domains


@dataclass
class OptimizationStatus:
    """Stores progress indicators and timings."""
    nb_eval: int = 0
    start_opti_ts: float = 0.0

    # Failure tracking
    failed_eval_count: int = 0
    stalled_eval_count: int = 0
    crashed_eval_count: int = 0
    
    # Timings
    total_eval_time: float = 0.0
    total_gmx_time: float = 0.0
    total_model_eval_time: float = 0.0
    
    # Process management
    process_alive_time_sleep: int = 10
    process_alive_nb_cycles_dead: int = 0
    bonds_rescaling_performed: bool = False
    
@dataclass
class OptimizationResults:
    """Stores metrics of the current evaluation."""
    gyr_aa_mapped: Optional[float] = None
    gyr_aa_mapped_std: Optional[float] = None
    gyr_cg: Optional[float] = None
    gyr_cg_std: Optional[float] = None
    
    sasa_aa_mapped: Optional[float] = None
    sasa_aa_mapped_std: Optional[float] = None
    sasa_cg: Optional[float] = None
    sasa_cg_std: Optional[float] = None


@dataclass
class ParticleSwarmState:
    """Store PSO state, independent bests, and failure penalties."""
    best_fitness: Tuple[float, Any] = field(default_factory=lambda: (np.inf, None))
    
    # Tracking best independent parameters
    all_best_emd_dist_geoms: Dict = field(default_factory=dict)
    all_best_params_dist_geoms: Dict = field(default_factory=dict)
    
    worst_fit_score: float = 0.0
    failure_component_scores: Dict[str, float] = field(
        default_factory=lambda: {"constraints_bonds": 0.0, "angles": 0.0, "dihedrals": 0.0}
    )
    
    # Active set of geometries being optimized
    opti_geoms_all: Set[GeometryKind] = field(default_factory=set)


@dataclass
class OptimizationContext:
    """Compose configuration and mutable state for evaluation or optimization."""
    config: SwarmConfig
    
    # Composed State Objects
    files: SimulationFiles = field(default_factory=SimulationFiles)
    scoring: ScoringState = field(default_factory=ScoringState)
    status: OptimizationStatus = field(default_factory=OptimizationStatus)
    results: OptimizationResults = field(default_factory=OptimizationResults)
    pso: ParticleSwarmState = field(default_factory=ParticleSwarmState)
    
    # Core Data Structures (kept at top level or moved?)
    # These are core to the logic flow, could move to 'TopologyState' if desired,
    # but for now let's keep them here or proxy them?
    # Actually, let's keep them as fields here for now as they are central shared state.
    cg_itp: CGTopology | None = None
    opti_itp: CGTopology | None = None
    out_itp: CGTopology | None = None
    opti_cycle: OptimizationCycle | None = None
    simulation_setup: SimulationSetup | None = None
    parameter_layout: ParameterVectorLayout | None = None
    
    # Managers (injected at runtime)
    workspace_manager: Any = None
    evaluator: Any = None
