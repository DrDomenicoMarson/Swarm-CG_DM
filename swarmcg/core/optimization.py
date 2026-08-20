import os
import shutil
import copy
import contextlib
import numpy as np
from datetime import datetime
from fstpso import FuzzyPSO

import swarmcg.shared.styling
from swarmcg.scoring import eval_function
from swarmcg.simulations import SimulationStep, get_settings, WorkspaceManager
from swarmcg.scoring.evaluator import SwarmEvaluator
from swarmcg.scoring.compare import compare_models
from swarmcg import config
from swarmcg import forcefield
from swarmcg.optimization_types import (
    OptimizationCycle,
    ParameterVectorLayout,
    SimulationSetup,
)
from swarmcg.particle_initialization import initialize_particles
from swarmcg.shared import exceptions, catch_warnings
from swarmcg.shared.logging_utils import setup_logging, get_logger
from swarmcg.context import OptimizationContext
from swarmcg.topology import GeometryKind

logger = get_logger(__name__)

class SwarmOptimizer:
    """Coordinate staged fuzzy-PSO bonded-parameter optimization.

    Args:
        config_obj: Validated application configuration.
    """

    def __init__(self, config_obj):
        self.config = config_obj
        self.ns = OptimizationContext(config=config_obj)

    @catch_warnings(ImportWarning)
    @catch_warnings(UserWarning)
    def run(self):
        """Run validation, reference mapping, three PSO cycles, and reporting.

        Raises:
            BaseError: If input validation, mapping, topology, or GROMACS
                execution cannot be completed.
        """
        self._initialize_context()
        self._setup_execution()
        
        logger.info("")
        logger.info(swarmcg.shared.styling.sep_close)
        logger.info("| PRE-PROCESSING AND CONTROLS                                                                 |")
        logger.info(swarmcg.shared.styling.sep_close)
        logger.info("")
        
        self._validate_environment()
        self._initialize_optimization()
        self._create_output_files()
        
        # Reference distributions plot
        self._plot_reference_distributions()

        # Optimization Loops
        self._run_optimization_cycles()

        self._finalize_optimization()

    def _initialize_context(self):
        # Default variable initialization
        self.ns.scoring.mismatch_order = self.config.output.mismatch_order
        self.ns.scoring.row_x_scaling = self.config.output.row_x_scaling
        self.ns.scoring.row_y_scaling = self.config.output.row_y_scaling
        self.ns.scoring.ncols_max = self.config.output.ncols_max
        self.ns.scoring.molname_in = None
        
        self.ns.status.process_alive_time_sleep = 10
        self.ns.status.process_alive_nb_cycles_dead = int(
            self.config.gromacs.sim_kill_delay / self.ns.status.process_alive_time_sleep)
        self.ns.status.bonds_rescaling_performed = False

        # file basenames
        self.ns.files.cg_itp_basename = os.path.basename(self.config.cg_model.cg_itp_filename)
        self.ns.files.gro_input_basename = os.path.basename(self.config.cg_model.gro_input_filename)
        self.ns.files.top_input_basename = os.path.basename(self.config.cg_model.top_input_filename)
        self.ns.files.mdp_minimization_basename = os.path.basename(self.config.simulation.mdp_minimization_filename)
        self.ns.files.mdp_equi_basename = os.path.basename(self.config.simulation.mdp_equi_filename)
        self.ns.files.mdp_md_basename = os.path.basename(self.config.simulation.mdp_md_filename)

        # Initialize Managers
        self.ns.workspace_manager = WorkspaceManager(self.config)
        self.ns.evaluator = SwarmEvaluator(self.config)

    def _setup_execution(self):
        try:
            self.ns.files.exec_folder = self.ns.workspace_manager.setup_execution_folder(self.config.output.output_folder)
            setup_logging(
                module_name="optimize",
                log_dir=self.ns.files.exec_folder,
                verbose=self.config.output.verbose,
            )
        except exceptions.AvoidOverwritingFolder as e:
            raise e

    def _validate_environment(self):
        SimulationStep._validate_exec(self.config.gromacs.gmx_path)
        self.top_includes_filenames = self.ns.workspace_manager.verify_topology_includes()

    def _initialize_optimization(self):
        self.ns.workspace_manager.prepare_simulation_input(self.top_includes_filenames)
        
        self.ns.status.nb_eval = 0
        self.ns.status.start_opti_ts = datetime.now().timestamp()
        
        self.ns.evaluator.initialize(
            self.ns, validate_starting_configuration=True
        )
        self.ns.evaluator.compute_reference_distributions()
        
        logger.info("")

    def _create_output_files(self):
        with open(os.path.join(self.ns.files.exec_folder, config.opti_perf_recap_file), "w") as fp:
            fp.write(f"# nb constraints: {self.ns.cg_itp.constraint_count}\n")
            fp.write(f"# nb bonds: {self.ns.cg_itp.bond_count}\n")
            fp.write(f"# nb angles: {self.ns.cg_itp.angle_count}\n")
            fp.write(f"# nb dihedrals: {self.ns.cg_itp.dihedral_count}\n")
            fp.write("#\n")
            fp.write(
                "# opti_cycle nb_eval fit_score_all fit_score_cstrs_bonds fit_score_angles "
                "fit_score_dihedrals eval_score Rg_AA_mapped Rg_AA_mapped_std Rg_CG "
                "Rg_CG_std SASA_AA_mapped SASA_AA_mapped_std SASA_CG SASA_CG_std "
                "parameters_set eval_time current_total_time\n"
            )
        
        with open(os.path.join(self.ns.files.exec_folder, config.opti_pairwise_distances_file), "w"):
            pass
            
        self.ns.results.gyr_aa_mapped, self.ns.results.gyr_aa_mapped_std = None, None
        self.ns.results.sasa_aa_mapped, self.ns.results.sasa_aa_mapped_std = None, None

    def _plot_reference_distributions(self):
        self.ns.scoring.atom_only = True
        self.ns.files.plot_filename = os.path.join(self.ns.files.exec_folder, config.ref_distrib_plots)
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull):
                compare_models(self.ns, manual_mode=False)
        logger.info("")
        logger.info(
            "Plotted reference AA-mapped distributions (used as target during optimization) at location:\n %s",
            self.ns.files.plot_filename,
        )
        self.ns.scoring.atom_only = False

    def _run_optimization_cycles(self):
        sim_types, opti_cycles, sim_cycles, particle_setter = get_settings(self.ns)
        
        self.ns.opti_itp = copy.deepcopy(self.ns.cg_itp)
        # Handle dihedrals case
        if self.ns.cg_itp.dihedral_count == 0:
            opti_cycles = self._remove_dihedrals_from_cycles(opti_cycles)

        self.ns.scoring.performed_init_BI = {"bond": False, "angle": False, "dihedral": False}
        self.ns.pso.opti_geoms_all = {
            GeometryKind(geometry)
            for cycle_geometries in opti_cycles
            for geometry in cycle_geometries
        }
        self.ns.pso.best_fitness = [np.inf, None]

        # Initialize tracking dictionaries
        group_counts = {
            "constraints": self.ns.cg_itp.constraint_count,
            "bonds": self.ns.cg_itp.bond_count,
            "angles": self.ns.cg_itp.angle_count,
            "dihedrals": self.ns.cg_itp.dihedral_count,
        }
        for geom_type, nb_geom in group_counts.items():
            self.ns.pso.all_best_emd_dist_geoms[geom_type] = {i: np.nan for i in range(nb_geom)}
            self.ns.pso.all_best_params_dist_geoms[geom_type] = {i: {} for i in range(nb_geom)}

        for i, cycle_geoms in enumerate(opti_cycles):
            self._run_single_cycle(i, cycle_geoms, sim_cycles, sim_types, particle_setter)

    def _remove_dihedrals_from_cycles(self, opti_cycles):
        # Logic to remove dihedrals if not present
        new_cycles = []
        for cycle in opti_cycles:
            new_cycle = [g for g in cycle if g != "dihedral"]
            if new_cycle:
                new_cycles.append(new_cycle)
        return new_cycles

    def _run_single_cycle(self, i, cycle_geoms, sim_cycles, sim_types, particle_setter):
        self.ns.opti_cycle = OptimizationCycle.from_topology(
            i + 1, cycle_geoms, self.ns.cg_itp
        )
        self.ns.out_itp = copy.deepcopy(self.ns.opti_itp)
        self.ns.simulation_setup = SimulationSetup.from_mapping(
            sim_types[sim_cycles[i]]
        )
        self.ns.parameter_layout = ParameterVectorLayout.build(
            self.ns.cg_itp,
            self.ns.opti_cycle,
            self.ns.scoring.domains_val,
            self.config.optimization.exec_mode,
            self.config,
        )
        
        geoms_display = self._get_geoms_display_string()

        logger.info("")
        logger.info(swarmcg.shared.styling.sep_close)
        logger.info(
            "| STARTING OPTIMIZATION CYCLE %s                                                              |",
            self.ns.opti_cycle.number,
        )
        logger.info(
            "| Optimizing %s %s|",
            geoms_display,
            " " * (95 - 16 - len(geoms_display)),
        )
        logger.info(swarmcg.shared.styling.sep_close)

        forcefield.perform_BI(
            self.ns.out_itp,
            self.ns.opti_cycle,
            self.ns.scoring.data_BI,
            self.ns.scoring.performed_init_BI,
            self.config.simulation.temp,
            config_obj=self.ns.config,
            exec_mode=self.config.optimization.exec_mode,
        )

        search_space_boundaries = self.ns.parameter_layout.bounds

        max_swarm_iterations = self.ns.simulation_setup.max_swarm_iterations
        if self.config.optimization.sim_type == "OPTIMAL":
            max_swarm_iterations = int(
                round(6 + np.sqrt(len(search_space_boundaries)))
            )

        self._calculate_worst_fit_score()

        nb_particles = particle_setter(search_space_boundaries)
        
        initial_guess_list = initialize_particles(
            nb_particles,
            self.ns.parameter_layout,
            self.ns.cg_itp,
            self.ns.out_itp,
            self.ns.pso.all_best_emd_dist_geoms,
            self.ns.pso.all_best_params_dist_geoms,
            self.ns.config,
            use_input_seed=self.config.cg_model.user_input,
            equilibrium_guess_factor=(
                self.ns.simulation_setup.equilibrium_guess_factor
            ),
            force_guess_factor=self.ns.simulation_setup.force_guess_factor,
        )

        # Optimization
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull):
                FP = FuzzyPSO()
                FP.set_search_space(search_space_boundaries)
                FP.set_swarm_size(nb_particles)
                FP.set_fitness(fitness=eval_function, arguments=self.ns, skip_test=True)
                result = FP.solve_with_fstpso(
                    max_iter=max_swarm_iterations,
                    initial_guess_list=initial_guess_list,
                    max_iter_without_new_global_best=(
                        self.ns.simulation_setup.max_iterations_without_improvement
                    ),
                )

        self.ns.parameter_layout.apply(self.ns.out_itp, result[0].X)
        # Every cycle is a mandatory staged refinement of the cycle optimum.
        # The separately tracked global score still controls the final best-model
        # directory and is intentionally not used as the next-cycle baseline.
        self.ns.opti_itp = copy.deepcopy(self.ns.out_itp)

    def _get_geoms_display_string(self):
        geoms_display = []
        if self.ns.opti_cycle.includes("constraint") or self.ns.opti_cycle.includes("bond"):
            geoms_display.append("constraints/bonds")
        if self.ns.opti_cycle.includes("angle"):
            geoms_display.append("angles")
        if self.ns.opti_cycle.includes("dihedral"):
            geoms_display.append("dihedrals")
        return " & ".join(geoms_display)

    def _calculate_worst_fit_score(self):
        """Set a finite failure objective strictly above any valid cycle score."""
        active = set(self.ns.opti_cycle.geometries)
        factor = self.config.optimization.bonds2angles_scoring_factor

        constraint_max = float(np.max(self.ns.scoring.constraints_grid.cost_matrix)) * factor
        bond_max = float(np.max(self.ns.scoring.bonds_grid.cost_matrix)) * factor
        angle_max = float(np.max(self.ns.scoring.angles_grid.cost_matrix))
        dihedral_max = float(np.max(self.ns.scoring.dihedrals_grid.cost_matrix))

        bonded_class_active = bool({"constraint", "bond"}.intersection(active))
        constraints_bonds = (
            np.sqrt(
                self.ns.cg_itp.constraint_count * constraint_max**2
                + self.ns.cg_itp.bond_count * bond_max**2
            )
            if bonded_class_active
            else 0.0
        )
        angles = (
            np.sqrt(self.ns.cg_itp.angle_count * angle_max**2)
            if "angle" in active
            else 0.0
        )
        dihedrals = (
            np.sqrt(self.ns.cg_itp.dihedral_count * dihedral_max**2)
            if "dihedral" in active
            else 0.0
        )
        self.ns.pso.failure_component_scores = {
            "constraints_bonds": float(constraints_bonds),
            "angles": float(angles),
            "dihedrals": float(dihedrals),
        }
        theoretical_maximum = constraints_bonds + angles + dihedrals
        self.ns.pso.worst_fit_score = float(
            np.nextafter(theoretical_maximum, np.inf)
        )

    def _finalize_optimization(self):
        shutil.rmtree(os.path.join(self.ns.files.exec_folder, config.input_sim_files_dirname))

        total_time_sec = datetime.now().timestamp() - self.ns.status.start_opti_ts
        total_time = round(total_time_sec / (60 * 60), 2)
        init_time = round((total_time_sec - self.ns.status.total_eval_time) / (60 * 60), 2)
        self.ns.status.total_gmx_time = round(self.ns.status.total_gmx_time / (60 * 60), 2)
        self.ns.status.total_model_eval_time = round(self.ns.status.total_model_eval_time / (60 * 60), 2)

        logger.info("")
        logger.info(swarmcg.shared.styling.sep_close)
        logger.info("|  FINISHED PROPERLY                                                                          |")
        logger.info(swarmcg.shared.styling.sep_close)
        logger.info("")
        logger.info("Total nb of evaluation steps: %s", self.ns.status.nb_eval)
        logger.info("Best model obtained at evaluation step number: %s", self.ns.pso.best_fitness[1])
        logger.info("")
        logger.info("Total execution time : %s h", total_time)
        logger.info(
            "Initialization time  : %s h (%s %%)",
            init_time,
            round(init_time / total_time * 100, 2),
        )
        logger.info(
            "Simulations time     : %s h (%s %%)",
            self.ns.status.total_gmx_time,
            round(self.ns.status.total_gmx_time / total_time * 100, 2),
        )
        logger.info(
            "Models scoring time  : %s h (%s %%)",
            self.ns.status.total_model_eval_time,
            round(self.ns.status.total_model_eval_time / total_time * 100, 2),
        )
        logger.info("")
