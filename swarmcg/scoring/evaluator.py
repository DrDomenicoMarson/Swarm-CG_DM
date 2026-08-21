
import numpy as np
from typing import Optional

from swarmcg.config_types import SwarmConfig
from swarmcg.context import OptimizationContext
from swarmcg.optimization_types import EvaluationResult
from swarmcg import io
from swarmcg.mapping import Mapping, initialize_cg_traj, make_aa_traj_whole_for_selected_mols
from swarmcg.scoring.compare import compare_models
from swarmcg import scoring as scores
from swarmcg import utils
from swarmcg.scoring.distances import unwrap_degrees_around
from swarmcg.shared import exceptions
from swarmcg.shared.logging_utils import get_logger
from swarmcg.shared.periodic import (
    circular_moment_degrees,
    normalize_periodic_degrees,
)
from swarmcg.simulations.boltzmann import BoltzmannTarget, complete_sample_range
from swarmcg.simulations.polynomial import (
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
        self.ns: Optional[OptimizationContext] = None
        
    def initialize(
        self,
        context: OptimizationContext,
        *,
        validate_starting_configuration: bool = False,
    ) -> None:
        """Load, validate, map, and attach reference state to a context.

        Args:
            context: Runtime context populated in place.
            validate_starting_configuration: Validate the optimizer starting
                GRO before reading or processing the AA trajectory.

        Raises:
            MissformattedFile: If mapping and topology bead counts differ or
                topology structure is inconsistent.
            FileNotFoundError: If a required input cannot be read.
        """
        self.ns = context
        
        # 1. Bins
        scores.create_bins_and_dist_matrices(self.ns)
        
        # 2. Mapping
        self.mapping = Mapping(self.config)
        self.mapping.read_ndx_atoms2beads()
        self.mapping.get_atoms_weights_in_beads()
        
        # Persist mapping data needed by trajectory construction and scaling.
        self.ns.scoring.all_beads = self.mapping.all_beads
        self.ns.scoring.atom_w = self.mapping.atom_w
        
        # 3. CG ITP
        self.ns.cg_itp = io.read_cg_topology(
            self.config.cg_model.cg_itp_filename
        )
        io.validate_parameter_bounds(self.ns.cg_itp, self.config)
        io.validate_mapping_bead_count(self.ns.cg_itp, self.mapping.all_beads)

        if validate_starting_configuration:
            io.validate_restricted_bending_start(
                self.config.cg_model.gro_input_filename, self.ns.cg_itp
            )
        
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
                polynomial input lies outside its scientifically valid domain,
                or a mode-1 torsion lacks the circular moment required by its
                functional form. Periodic functions 1 and 4 use the moment at
                their multiplicity; function 2 uses the first moment.
        """
        # Checks
        if not self.ns or self.ns.cg_itp is None:
            raise RuntimeError("Evaluator not initialized.")
            
        logger.info("Calculating reference distributions...")
        
        # Constraints
        for i, grp in enumerate(self.ns.cg_itp.constraints):
             avg, hist, values = scores.get_AA_bonds_distrib(
                 self.ns.scoring.aa2cg_universe, 
                 grp.beads,
                 "constraint",
                 i,
                 self.config, 
                 self.ns.scoring.bins_constraints, 
                 self.config.optimization.bw_constraints,
                 bonds_scaling_specific=self.ns.scoring.bonds_scaling_specific
             )
             if self.config.optimization.exec_mode == 1:
                grp.equilibrium = avg
             grp.average = avg
             grp.histogram = hist
             self.ns.scoring.domains_val.setdefault("constraint", []).append([round(np.min(values), 3), round(np.max(values), 3)])
             
        # Bonds
        for i, grp in enumerate(self.ns.cg_itp.bonds):
            avg, hist, values = scores.get_AA_bonds_distrib(
                self.ns.scoring.aa2cg_universe,
                grp.beads,
                "bond",
                i,
                self.config,
                self.ns.scoring.bins_bonds,
                self.config.optimization.bw_bonds,
                bonds_scaling_specific=self.ns.scoring.bonds_scaling_specific
            )
            if self.config.optimization.exec_mode == 1:
                grp.equilibrium = avg
            grp.average = avg
            grp.histogram = hist
            
            target = BoltzmannTarget.from_samples(
                values,
                bins=self.config.optimization.bi_nb_bins,
                value_range=complete_sample_range(values),
            )
            self.ns.scoring.data_BI.setdefault("bond", []).append(target)
            self.ns.scoring.domains_val.setdefault("bond", []).append([round(np.min(values), 3), round(np.max(values), 3)])
            
        # Angles
        for i, grp in enumerate(self.ns.cg_itp.angles):
             avg, hist, val_deg, val_rad = scores.get_AA_angles_distrib(
                 self.ns.scoring.aa2cg_universe,
                 grp.beads,
                 self.ns.scoring.bins_angles,
                 self.config.optimization.bw_angles,
                 group_label=f"angle group {i + 1}",
             )
             if grp.function == 10:
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
                grp.equilibrium = avg
             grp.average = avg
             grp.histogram = hist
             
             target = BoltzmannTarget.from_samples(
                 val_rad,
                 bins=self.config.optimization.bi_nb_bins,
                 value_range=complete_sample_range(val_rad),
             )
             self.ns.scoring.data_BI.setdefault("angle", []).append(target)
             domain_min, domain_max = float(np.min(val_deg)), float(np.max(val_deg))
             if grp.function == 10:
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
        for i, grp in enumerate(self.ns.cg_itp.dihedrals):
            avg, hist, val_deg, val_rad = scores.get_AA_dihedrals_distrib(
                self.ns.scoring.aa2cg_universe,
                grp.beads,
                self.ns.scoring.bins_dihedrals,
                self.config.optimization.bw_dihedrals,
                group_label=f"dihedral group {i + 1}",
            )
            polynomial = grp.function in (3, 11)
            periodic = grp.function in (1, 4)
            phase_center = None
            if self.config.optimization.exec_mode == 1 and periodic:
                moment = circular_moment_degrees(val_deg, grp.multiplicity)
                grp.phase_moment_resultant = moment.resultant_length
                if moment.direction_degrees is None:
                    raise exceptions.ScientificValidationError(
                        f"Dihedral group {i + 1} uses periodic function {grp.function} "
                        f"with multiplicity {grp.multiplicity}, but its order-{grp.multiplicity} "
                        "reference circular moment has no defined direction. Improve "
                        "reference sampling or use execution mode 2 with a fixed ITP phase."
                    )
                phase_center = normalize_periodic_degrees(
                    moment.direction_degrees + 180.0
                )
                grp.equilibrium = phase_center
            elif (
                self.config.optimization.exec_mode == 1
                and not polynomial
                and not np.isfinite(avg)
            ):
                raise exceptions.ScientificValidationError(
                    f"Dihedral group {i + 1} uses phase-based function {grp.function}, "
                    "but its reference first circular moment has no defined direction. "
                    "Improve reference sampling or use execution mode 2 with a fixed ITP phase."
                )
            elif self.config.optimization.exec_mode == 1 and not polynomial:
                grp.equilibrium = avg
            grp.average = avg
            grp.histogram = hist

            target = BoltzmannTarget.from_samples(
                val_rad,
                bins=2 * self.config.optimization.bi_nb_bins,
                value_range=(-np.pi, np.pi),
            )
            self.ns.scoring.data_BI.setdefault("dihedral", []).append(target)

            if self.config.optimization.exec_mode == 1 and periodic:
                domain = [phase_center - 180.0, phase_center + 180.0]
            elif self.config.optimization.exec_mode == 1 and not polynomial:
                unwrapped_deg = unwrap_degrees_around(val_deg, avg)
                domain = [
                    round(float(np.min(unwrapped_deg)), 2),
                    round(float(np.max(unwrapped_deg)), 2),
                ]
            else:
                # Polynomial functions have no phase domain, while execution
                # mode 2 fixes the phase from the input topology.
                domain = None
            self.ns.scoring.domains_val.setdefault("dihedral", []).append(domain)

            if grp.function in (3, 11):
                total_variation = mirrored_total_variation(hist)
                grp.polynomial_symmetry_tv = total_variation
                if total_variation > 0.10:
                    form_name = "RB" if grp.function == 3 else "CBT"
                    logger.warning(
                        "%s dihedral group %s has mirrored total-variation distance %.3f; "
                        "function %s cannot reproduce an asymmetric torsional marginal.",
                        form_name,
                        i + 1,
                        total_variation,
                        grp.function,
                    )

            if grp.function in (3, 11):
                derived_bound = adaptive_coefficient_bound(
                    hist,
                    self.config.simulation.temp,
                )
                if grp.function == 3:
                    override = self.config.optimization.max_abs_rb_coefficient
                    coefficients = grp.parameters.coefficients
                    option = "-max_rb_coeff"
                else:
                    override = self.config.optimization.max_abs_cbt_effective_coefficient
                    coefficients = grp.parameters.effective_coefficients
                    option = "-max_cbt_coeff"
                bound = derived_bound if override is None else override
                grp.coefficient_bound = float(bound)
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

    def evaluate_model(
        self, manual_mode: bool = False
    ) -> EvaluationResult | None:
        """Run model scoring for the trajectories configured in the context.

        Args:
            manual_mode: Use evaluation-mode distribution loading and display.

        Returns:
            Typed scores, observables, and pairwise geometry results, or
            ``None`` for manual display mode.

        Raises:
            RuntimeError: If :meth:`initialize` has not completed.
            Exception: Any scoring or trajectory error from
                :func:`compare_models` is propagated to the caller.
        """
        if not self.ns:
             raise RuntimeError("Evaluator not initialized.")
        return compare_models(
            self.ns,
            manual_mode=manual_mode,
        )
