
import os
import shutil
import numpy as np
from typing import Optional, Tuple, Any

from swarmcg.config_types import SwarmConfig
from swarmcg.context import OptimizationContext
from swarmcg import io
from swarmcg.mapping import Mapping, initialize_cg_traj, make_aa_traj_whole_for_selected_mols
from swarmcg.scoring.compare import compare_models
from swarmcg import scoring as scores
from swarmcg import utils

class SwarmEvaluator:
    """
    Handles the evaluation of CG models against reference AA data.
    Encapsulates Mapping, Trajectory loading, and Scoring logic.
    """
    def __init__(self, config_obj: SwarmConfig):
        self.config = config_obj
        self.mapping: Optional[Mapping] = None
        self.ns: Optional[OptimizationContext] = None # Keeping context for now as compare_models needs it
        
    def initialize(self, context: OptimizationContext):
        """
        Loads reference data and initializes mapping.
        This is a heavy operation and should be done once.
        """
        self.ns = context # Store context to share state with legacy functions if needed
        # But ideally we populate context from here
        
        # 1. Bins
        scores.create_bins_and_dist_matrices(self.ns)
        
        # 2. Mapping
        self.mapping = Mapping(self.config)
        self.mapping.read_ndx_atoms2beads()
        self.mapping.get_atoms_weights_in_beads()
        
        # Expose to context (legacy support)
        self.ns.all_beads = self.mapping.all_beads
        self.ns.atom_w = self.mapping.atom_w
        
        # 3. CG ITP
        self.ns.cg_itp = io.read_cg_itp_file(self.config)
        io.validate_cg_itp(self.ns.cg_itp)
        
        # 4. Scaling
        utils.process_scaling_str(self.ns)
        
        # 5. Load AA Trajectory
        print("\nLoading Reference AA Trajectory...")
        self.ns.aa_universe = io.read_aa_traj(self.config.reference)
        
        self.mapping.load_aa_data(self.ns.aa_universe)
        self.ns.all_atoms = self.mapping.all_atoms
        self.ns.all_aa_mols = self.mapping.all_aa_mols
        
        make_aa_traj_whole_for_selected_mols(self.ns.aa_universe, self.ns.all_aa_mols)
        
        # 6. Create Atom Groups
        self.mapping.get_beads_MDA_atomgroups(self.ns.aa_universe)
        self.ns.mda_beads_atom_grps = self.mapping.mda_beads_atom_grps
        self.ns.mda_weights_atom_grps = self.mapping.mda_weights_atom_grps
        
        # 7. Map AA to CG
        print("\nMapping AA Trajectory to CG representation...")
        self.ns.aa2cg_universe = initialize_cg_traj(self.ns.cg_itp)
        self.mapping.map_aa2cg_traj(self.ns.aa_universe, self.ns.aa2cg_universe, self.ns.cg_itp)
        
    def compute_reference_distributions(self):
        """
        Computes the target distributions from the mapped AA trajectory.
        This replaces the initial distribution calculation loop in optimize_model.py.
        """
        # Checks
        if not self.ns or not self.ns.cg_itp:
            raise RuntimeError("Evaluator not initialized.")
            
        print("Calculating reference distributions...")
        
        # Constraints
        for i, grp in enumerate(self.ns.cg_itp["constraint"]):
             avg, hist, values = scores.get_AA_bonds_distrib(
                 self.ns.aa2cg_universe, 
                 grp["beads"], 
                 "constraint",
                 i,
                 self.config, 
                 self.ns.bins_constraints, 
                 self.ns.bw_constraints
             )
             grp["value"] = avg # Set initial value (exec_mode 1)
             grp["avg"] = avg
             grp["hist"] = hist
             self.ns.domains_val.setdefault("constraint", []).append([round(np.min(values), 3), round(np.max(values), 3)])
             
        # Bonds
        for i, grp in enumerate(self.ns.cg_itp["bond"]):
            avg, hist, values = scores.get_AA_bonds_distrib(
                self.ns.aa2cg_universe,
                grp["beads"],
                "bond",
                i,
                self.config,
                self.ns.bins_bonds,
                self.ns.bw_bonds
            )
            grp["value"] = avg
            grp["avg"] = avg
            grp["hist"] = hist
            
            # BI initialization stats
            xmin, xmax = min(np.inf, self.ns.bins_bonds[np.min(np.nonzero(hist))]), max(-np.inf, self.ns.bins_bonds[np.max(np.nonzero(hist)) + 1])
            xmin, xmax = xmin - self.ns.bw_bonds, xmax + self.ns.bw_bonds
            
            # Helper for hist in BI range
            h, _ = np.histogram(values, range=(xmin, xmax), bins=self.config.optimization.bi_nb_bins) # Using config constant directly if accessible or hardcoded? 
            # Actually bi_nb_bins is not in ConfigType yet? It was in config.py.
            # config object should have it if we migrated everything. 
            # Assuming swarmcg.config.bi_nb_bins exists.
            
            self.ns.data_BI.setdefault("bond", []).append([h, np.std(values), np.mean(values), (xmin, xmax)])
            self.ns.domains_val.setdefault("bond", []).append([round(np.min(values), 3), round(np.max(values), 3)])
            
        # Angles
        for i, grp in enumerate(self.ns.cg_itp["angle"]):
             avg, hist, val_deg, val_rad = scores.get_AA_angles_distrib(
                 self.ns.aa2cg_universe,
                 grp["beads"],
                 self.ns.bins_angles,
                 self.ns.bw_angles
             )
             grp["value"] = avg
             grp["avg"] = avg
             grp["hist"] = hist
             
             xmin, xmax = min(np.inf, self.ns.bins_angles[np.min(np.nonzero(hist))]), max(-np.inf, self.ns.bins_angles[np.max(np.nonzero(hist)) + 1])
             xmin, xmax = xmin + self.ns.bw_angles / 2, xmax - self.ns.bw_angles / 2
             
             h, _ = np.histogram(val_rad, range=(np.deg2rad(xmin), np.deg2rad(xmax)), bins=config.bi_nb_bins)
             
             self.ns.data_BI.setdefault("angle", []).append([h, np.std(val_rad), (xmin, xmax)])
             self.ns.domains_val.setdefault("angle", []).append([round(np.min(val_deg), 2), round(np.max(val_deg), 2)])

        # Dihedrals
        for i, grp in enumerate(self.ns.cg_itp["dihedral"]):
            avg, hist, val_deg, val_rad = scores.get_AA_dihedrals_distrib(
                self.ns.aa2cg_universe,
                grp["beads"],
                self.ns.bins_dihedrals,
                self.ns.bw_dihedrals
            )
            # exec_mode 1 logic handled later/in optimize
            grp["value"] = avg 
            grp["avg"] = avg
            grp["hist"] = hist
            
            xmin, xmax = -180, 180
            h, _ = np.histogram(val_rad, range=(np.deg2rad(xmin), np.deg2rad(xmax)), bins=2 * config.bi_nb_bins)
            
            self.ns.data_BI.setdefault("dihedral", []).append([h, np.std(val_rad), np.mean(val_rad), (xmin, xmax)])
            self.ns.domains_val.setdefault("dihedral", []).append([round(np.min(val_deg), 2), round(np.max(val_deg), 2)])

    def evaluate_model(self, working_dir, manual_mode=False) -> Tuple[float, float, float, float, Any, Any]:
        """
        Runs the scoring logic (compare_models) in the given directory.
        Assumes simulation results (TPR, XTC) are present in working_dir.
        """
        if not self.ns:
             raise RuntimeError("Evaluator not initialized.")
             
        # Temporarily update context to point to current results
        # Assuming filenames are standard or provided in config
        # optimize_model loop might be setting these
        
        # Actually compare_models uses ns attributes:
        # ns.cg_tpr_filename, ns.cg_traj_filename
        
        # We need to ensure these are set correctly in `ns` before calling compare_models
        # For optimization, these are usually "md.tpr" and "md.xtc" in the eval step dir.
        
        current_dir = os.getcwd() # Caller should have chdir'd or we handle it?
        # compare_models looks for files.
        try:
            # We assume we are in the directory or files are relative to CWD
            return compare_models(self.ns, manual_mode=manual_mode, calc_sasa=(not manual_mode))
        except Exception as e:
            # Handle scoring failures
            raise e

