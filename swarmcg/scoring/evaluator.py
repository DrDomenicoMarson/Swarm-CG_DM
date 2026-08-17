
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
from swarmcg.scoring.distances import unwrap_degrees_around
from swarmcg.shared import exceptions
from swarmcg.shared.logging_utils import get_logger
from swarmcg.simulations.polynomial import (
    CBTParameters,
    RBParameters,
    adaptive_coefficient_bound,
    mirrored_total_variation,
)

logger = get_logger(__name__)

class SwarmEvaluator:
    """Evaluate CG models against mapped atomistic reference data.

    Args:
        config_obj: Validated application configuration.
    """
    def __init__(self, config_obj: SwarmConfig):
        self.config = config_obj
        self.mapping: Optional[Mapping] = None
        self.ns: Optional[OptimizationContext] = None # Keeping context for now as compare_models needs it
        
    def initialize(self, context: OptimizationContext) -> None:
        """Load, validate, map, and attach reference state to a context.

        Args:
            context: Runtime context populated in place.

        Raises:
            MissformattedFile: If mapping and topology bead counts differ or
                topology structure is inconsistent.
            FileNotFoundError: If a required input cannot be read.
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
        self.ns.scoring.all_beads = self.mapping.all_beads
        self.ns.scoring.atom_w = self.mapping.atom_w
        
        # 3. CG ITP
        self.ns.cg_itp = io.read_cg_itp_file(self.config)
        if hasattr(self.ns.cg_itp, "validate"):
            self.ns.cg_itp.validate()
        io.validate_cg_itp(self.ns.cg_itp, all_beads=self.mapping.all_beads)
        
        # 4. Scaling
        utils.process_scaling_str(self.ns)
        
        # 5. Load AA Trajectory
        logger.info("")
        logger.info("Loading Reference AA Trajectory...")
        self.ns.scoring.aa_universe = io.read_aa_traj(self.config.reference)
        
        self.mapping.load_aa_data(self.ns.scoring.aa_universe)
        self.ns.scoring.all_atoms = self.mapping.all_atoms
        self.ns.scoring.all_aa_mols = self.mapping.all_aa_mols
        
        make_aa_traj_whole_for_selected_mols(self.ns.scoring.aa_universe, self.ns.scoring.all_aa_mols)
        
        # 6. Create Atom Groups
        self.mapping.get_beads_MDA_atomgroups(self.ns.scoring.aa_universe)
        self.ns.scoring.mda_beads_atom_grps = self.mapping.mda_beads_atom_grps
        self.ns.scoring.mda_weights_atom_grps = self.mapping.mda_weights_atom_grps
        
        # 6b. Initialize Data Containers
        self.ns.scoring.data_BI = {}
        self.ns.scoring.domains_val = {}
        
        # 7. Map AA to CG
        logger.info("")
        logger.info("Mapping AA Trajectory to CG representation...")
        self.ns.scoring.aa2cg_universe = initialize_cg_traj(self.ns.cg_itp)
        self.mapping.map_aa2cg_traj(self.ns.scoring.aa_universe, self.ns.scoring.aa2cg_universe, self.ns.cg_itp)
        
    def compute_reference_distributions(self) -> None:
        """Compute normalized target distributions and search domains.

        Raises:
            RuntimeError: If :meth:`initialize` has not completed.
            ScientificValidationError: If a restricted-bending reference or
                polynomial input lies outside its scientifically valid domain.
        """
        # Checks
        if not self.ns or not self.ns.cg_itp:
            raise RuntimeError("Evaluator not initialized.")
            
        logger.info("Calculating reference distributions...")
        
        # Constraints
        for i, grp in enumerate(self.ns.cg_itp["constraint"]):
             avg, hist, values = scores.get_AA_bonds_distrib(
                 self.ns.scoring.aa2cg_universe, 
                 grp["beads"], 
                 "constraint",
                 i,
                 self.config, 
                 self.ns.scoring.bins_constraints, 
                 self.config.optimization.bw_constraints,
                 bonds_scaling_specific=self.ns.scoring.bonds_scaling_specific
             )
             if self.config.optimization.exec_mode == 1:
                grp["value"] = avg 
             grp["avg"] = avg
             grp["hist"] = hist
             self.ns.scoring.domains_val.setdefault("constraint", []).append([round(np.min(values), 3), round(np.max(values), 3)])
             
        # Bonds
        for i, grp in enumerate(self.ns.cg_itp["bond"]):
            avg, hist, values = scores.get_AA_bonds_distrib(
                self.ns.scoring.aa2cg_universe,
                grp["beads"],
                "bond",
                i,
                self.config,
                self.ns.scoring.bins_bonds,
                self.config.optimization.bw_bonds,
                bonds_scaling_specific=self.ns.scoring.bonds_scaling_specific
            )
            if self.config.optimization.exec_mode == 1:
                grp["value"] = avg
            grp["avg"] = avg
            grp["hist"] = hist
            
            # BI initialization stats
            xmin, xmax = min(np.inf, self.ns.scoring.bins_bonds[np.min(np.nonzero(hist))]), max(-np.inf, self.ns.scoring.bins_bonds[np.max(np.nonzero(hist)) + 1])
            xmin, xmax = xmin - self.config.optimization.bw_bonds, xmax + self.config.optimization.bw_bonds
            
            # Helper for hist in BI range
            h, _ = np.histogram(values, range=(xmin, xmax), bins=self.config.optimization.bi_nb_bins)
            h = h / h.sum()
            
            self.ns.scoring.data_BI.setdefault("bond", []).append([h, np.std(values), np.mean(values), (xmin, xmax)])
            self.ns.scoring.domains_val.setdefault("bond", []).append([round(np.min(values), 3), round(np.max(values), 3)])
            
        # Angles
        for i, grp in enumerate(self.ns.cg_itp["angle"]):
             avg, hist, val_deg, val_rad = scores.get_AA_angles_distrib(
                 self.ns.scoring.aa2cg_universe,
                 grp["beads"],
                 self.ns.scoring.bins_angles,
                 self.config.optimization.bw_angles
             )
             if grp["func"] == 10:
                unsafe_fraction = float(np.mean((val_deg < 10.0) | (val_deg > 170.0)))
                if unsafe_fraction > 0:
                    logger.warning(
                        "Restricted-bending angle group %s has %.2f%% of reference samples outside 10-170 degrees.",
                        i + 1,
                        unsafe_fraction * 100.0,
                    )
                if not 10.0 <= avg <= 170.0:
                    raise exceptions.ScientificValidationError(
                        f"Restricted-bending angle group {i + 1} has reference mean {avg} degrees; "
                        "function 10 requires an equilibrium value between 10 and 170 degrees."
                    )

             if self.config.optimization.exec_mode == 1:
                grp["value"] = avg
             grp["avg"] = avg
             grp["hist"] = hist
             
             xmin, xmax = min(np.inf, self.ns.scoring.bins_angles[np.min(np.nonzero(hist))]), max(-np.inf, self.ns.scoring.bins_angles[np.max(np.nonzero(hist)) + 1])
             xmin, xmax = xmin + self.config.optimization.bw_angles / 2, xmax - self.config.optimization.bw_angles / 2
             
             h, _ = np.histogram(val_rad, range=(np.deg2rad(xmin), np.deg2rad(xmax)), bins=self.config.optimization.bi_nb_bins)
             h = h / h.sum()
             
             self.ns.scoring.data_BI.setdefault("angle", []).append([h, np.std(val_rad), (xmin, xmax)])
             domain_min, domain_max = float(np.min(val_deg)), float(np.max(val_deg))
             if grp["func"] == 10:
                domain_min = max(10.0, domain_min)
                domain_max = min(170.0, domain_max)
                if domain_min > domain_max:
                    raise exceptions.ScientificValidationError(
                        f"Restricted-bending angle group {i + 1} has no target values in the safe 10-170 degree interval."
                    )
             self.ns.scoring.domains_val.setdefault("angle", []).append(
                 [round(domain_min, 2), round(domain_max, 2)]
             )

        # Dihedrals
        for i, grp in enumerate(self.ns.cg_itp["dihedral"]):
            avg, hist, val_deg, val_rad = scores.get_AA_dihedrals_distrib(
                self.ns.scoring.aa2cg_universe,
                grp["beads"],
                self.ns.scoring.bins_dihedrals,
                self.config.optimization.bw_dihedrals
            )
            # exec_mode 1 logic handled later/in optimize
            if self.config.optimization.exec_mode == 1 and grp["func"] not in (3, 11):
                grp["value"] = avg 
            grp["avg"] = avg 
            grp["avg"] = avg
            grp["hist"] = hist
            
            xmin, xmax = -180, 180
            h, _ = np.histogram(val_rad, range=(np.deg2rad(xmin), np.deg2rad(xmax)), bins=2 * self.config.optimization.bi_nb_bins)
            h = h / h.sum()

            unwrapped_deg = unwrap_degrees_around(val_deg, avg)
            unwrapped_rad = np.deg2rad(unwrapped_deg)
            self.ns.scoring.data_BI.setdefault("dihedral", []).append(
                [h, np.std(unwrapped_rad), np.deg2rad(avg), (xmin, xmax)]
            )
            self.ns.scoring.domains_val.setdefault("dihedral", []).append(
                [round(np.min(unwrapped_deg), 2), round(np.max(unwrapped_deg), 2)]
            )

            if grp["func"] == 11:
                total_variation = mirrored_total_variation(hist)
                grp["cbt_symmetry_tv"] = total_variation
                if total_variation > 0.10:
                    logger.warning(
                        "CBT dihedral group %s has mirrored total-variation distance %.3f; "
                        "function 11 cannot reproduce an asymmetric torsional marginal.",
                        i + 1,
                        total_variation,
                    )

            if grp["func"] in (3, 11):
                derived_bound = adaptive_coefficient_bound(
                    hist,
                    self.config.simulation.temp,
                )
                if grp["func"] == 3:
                    override = self.config.optimization.max_abs_rb_coefficient
                    coefficients = RBParameters.from_gromacs(grp["params"]).coefficients
                    option = "-max_rb_coeff"
                else:
                    override = self.config.optimization.max_abs_cbt_effective_coefficient
                    coefficients = CBTParameters.from_gromacs(grp["params"]).effective_coefficients
                    option = "-max_cbt_coeff"
                bound = derived_bound if override is None else override
                grp["coefficient_bound"] = float(bound)
                if any(abs(value) > bound for value in coefficients):
                    if override is None:
                        raise exceptions.ScientificValidationError(
                            f"Dihedral group {i + 1} contains an input coefficient outside "
                            f"the PMF-derived bound of {bound:.3f} kJ/mol. Supply {option} "
                            "explicitly if the larger range is intentional."
                        )
                    raise exceptions.ScientificValidationError(
                        f"Dihedral group {i + 1} contains an input coefficient outside "
                        f"the explicit {option} bound of {bound:.3f} kJ/mol."
                    )

    def evaluate_model(self, working_dir, manual_mode=False) -> Tuple[float, float, float, float, Any, Any]:
        """Run model scoring for simulation outputs in a working directory.

        Args:
            working_dir: Directory containing the configured CG trajectory
                outputs. Retained for API clarity while paths live in context.
            manual_mode: Use evaluation-mode distribution loading and display.

        Returns:
            Fitness total, three class contributions, pairwise-score text, and
            per-geometry EMD values.

        Raises:
            RuntimeError: If :meth:`initialize` has not completed.
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
            return compare_models(
                self.ns,
                manual_mode=manual_mode,
                calc_sasa=self.config.output.calculate_sasa,
            )
        except Exception as e:
            # Handle scoring failures
            raise e
