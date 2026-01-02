import os
import sys
import shutil
import time
import copy
import contextlib
from argparse import ArgumentParser, RawTextHelpFormatter, SUPPRESS
from shlex import quote as cmd_quote
from datetime import datetime

from fstpso import FuzzyPSO
import numpy as np

import swarmcg.shared.styling
import swarmcg.scoring as scores
import swarmcg.io as io
from swarmcg.scoring import eval_function
from swarmcg.simulations import SimulationStep, get_settings
from swarmcg import config
from swarmcg.shared import exceptions, catch_warnings
from swarmcg.scoring.compare import compare_models
from swarmcg import utils
from swarmcg import forcefield
from swarmcg.mapping import Mapping, initialize_cg_traj, make_aa_traj_whole_for_selected_mols
from swarmcg.config_types import SwarmConfig
from swarmcg.context import OptimizationContext


@catch_warnings(ImportWarning)  # filter Matplotlib mpl_toolkits missing __init__ stuff
@catch_warnings(UserWarning)  # filter working when reading scores for each geom at each fitness evaluation/simulation
def run(config_obj: SwarmConfig):
    """
    Main execution logic for model optimization.

    Args:
        config_obj (SwarmConfig): The configuration object containing all runtime parameters.
    """
    # Create context to hold state
    ns = OptimizationContext(config=config_obj)

    # TODO: allow to feed a JSON file for cycles of optimization ?? this is more optional but useful for big stuff possibly
    # TODO: if using SASA through GMX SASA, ensure vdwradii.dat contains the MARTINI radii
    # TODO: give a warning when users specify a bond scaling without specifying an Rg offset !!!

    # TODO: AT OPTI CYCLE 2, FIND ANGLES THAT ARE TOO STEEP (CG) AND WHEN GENERATING THE NEW GUESSES, PUT 10-30-50-70% OF THE CURRENT BEST FORCE CONSTANT IN SEVERAL PARTICLES !!!!!!!!!

    # NOTE: gmx trjconv and sasa may produce bugs when using TPR produced with gromacs v5, only current solution seems to be implementing the SASA calculation using MDTraj

    #####################################
    # ARGUMENTS HANDLING / HELP DISPLAY #
    #####################################

    # namespace variables not directly linked to arguments for plotting or for global package interpretation
    # These are now defaults in OptimizationContext, but we can override if needed
    ns.mismatch_order = False
    ns.row_x_scaling = True
    ns.row_y_scaling = True
    ns.ncols_max = 0  # 0 to display all
    # ns.atom_only = False
    ns.molname_in = None  # if None the first found using TPR atom ordering will be used
    
    # Process alive checks
    ns.process_alive_time_sleep = 10
    # derived from config directly if accessible, but here derived from property
    ns.process_alive_nb_cycles_dead = int(
        ns.sim_kill_delay / ns.process_alive_time_sleep)  # nb of cycles without .log file bytes size changes to determine that the MD run is stuck
    ns.bonds_rescaling_performed = False  # for user information display

    # get basenames for simulation files
    ns.cg_itp_basename = os.path.basename(ns.cg_itp_filename)
    ns.gro_input_basename = os.path.basename(ns.gro_input_filename)
    ns.top_input_basename = os.path.basename(ns.top_input_filename)
    ns.mdp_minimization_basename = os.path.basename(ns.mdp_minimization_filename)
    ns.mdp_equi_basename = os.path.basename(ns.mdp_equi_filename)
    ns.mdp_md_basename = os.path.basename(ns.mdp_md_filename)
    
    # Execution folder setup
    from swarmcg.simulations.workspace import WorkspaceManager
    from swarmcg.scoring.evaluator import SwarmEvaluator
    
    # Initialize Managers
    ns.workspace_manager = WorkspaceManager(config_obj)
    ns.evaluator = SwarmEvaluator(config_obj)
    
    # Setup Workspace
    try:
        ns.exec_folder = ns.workspace_manager.setup_execution_folder(ns.output_folder)
    except exceptions.AvoidOverwritingFolder as e:
        raise e
        
    print()
    print(swarmcg.shared.styling.sep_close)
    print("| PRE-PROCESSING AND CONTROLS                                                                 |")
    print(swarmcg.shared.styling.sep_close)
    print()

    # Verify input files existence (validation handled by WorkspaceManager or Config mostly, 
    # but optimize_model had explicit checks for input/output folder logic).
    # Since config object creation already validated basic existence, 
    # and WorkspaceManager assumes checked paths or checks them during copy.
    # We can rely on Config validation + runtime checks.
    
    # Check executables
    SimulationStep._validate_exec(ns.gmx_path)

    # Verify topology includes
    top_includes_filenames = ns.workspace_manager.verify_topology_includes()

    ##################
    # INITIALIZATION #
    ##################
    
    # Prepare simulation input files
    ns.workspace_manager.prepare_simulation_input(top_includes_filenames)

    # State initialization
    ns.nb_eval = 0
    ns.start_opti_ts = datetime.now().timestamp()
    
    # Initialize Evaluator (this loads TRAJ, MAPPING, calculates BINS, etc.)
    # This replaces ~100 lines of manual setup
    ns.evaluator.initialize(ns)
    
    # Compute reference distributions (populates data_BI, domains_val, etc.)
    ns.evaluator.compute_reference_distributions()
    
    print()

    # touch results files to be appended to later
    with open(ns.exec_folder + "/" + config.opti_perf_recap_file, "w") as fp:
        # TODO: print that file has been generated with Swarm-CG etc -- do this for basically all files
        fp.write(f"# nb constraints: {ns.cg_itp['nb_constraints']}\n")
        fp.write(f"# nb bonds: {ns.cg_itp['nb_bonds']}\n")
        fp.write(f"# nb angles: {ns.cg_itp['nb_angles']}\n")
        fp.write(f"# nb dihedrals: {ns.cg_itp['nb_dihedrals']}\n")
        fp.write("#\n")
        fp.write(
            "# opti_cycle nb_eval fit_score_all fit_score_cstrs_bonds fit_score_angles fit_score_dihedrals eval_score Rg_AA_mapped Rg_CG parameters_set eval_time current_total_time\n")
    with open(ns.exec_folder + "/" + config.opti_pairwise_distances_file, "w"):
        pass

    # set these to None to then check the variables have been filled (is not None), so we will do these calculations
    # one single time in function compare_models that is called at each iteration during optimization
    ns.gyr_aa_mapped, ns.gyr_aa_mapped_std = None, None
    ns.sasa_aa_mapped, ns.sasa_aa_mapped_std = None, None

    # output png with all the reference distributions, so the user can check
    ns.atom_only = True
    ns.plot_filename = ns.exec_folder + "/" + config.ref_distrib_plots
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            compare_models(ns, manual_mode=False)
    print()
    print("Plotted reference AA-mapped distributions (used as target during optimization) at location:\n ",
          ns.exec_folder + "/" + config.ref_distrib_plots)
    ns.atom_only = False

    ##################################
    # ITERATIVE OPTIMIZATION PROCESS #
    ##################################

    sim_types, opti_cycles, sim_cycles, particle_setter = get_settings(ns)

    # NOTE: currently, due to an issue in FST-PSO, number of swarm iterations performed is +2 when compared to the numbers we feed

    ns.opti_itp = copy.deepcopy(
        ns.cg_itp)  # the ITP object that will be optimized stepwise, at the end of each optimization cycle (geom type wise)
    ns.eval_nb_geoms = {"constraint": 0, "bond": 0, "angle": 0, "dihedral": 0}  # geoms to optimize at each step

    # remove dihedrals from cycles if CG ITP file does NOT contain dihedrals
    if ns.cg_itp["nb_dihedrals"] == 0:
        opti_cycles_cp, sim_cycles_cp = [], []
        nb_poped = 0
        for i in range(len(opti_cycles)):
            opti_cycles_cp.extend([[]])
            for j in range(len(opti_cycles[i])):
                if opti_cycles[i][j] != "dihedral":
                    opti_cycles_cp[i - nb_poped].append(opti_cycles[i][j])
                if len(opti_cycles_cp[i - nb_poped]) == 0:
                    opti_cycles_cp.pop()
                    nb_poped += 1
                else:
                    sim_cycles_cp.extend([sim_cycles[i]])
        opti_cycles, sim_cycles = opti_cycles_cp, sim_cycles_cp

    # state variables for the cycles of optimization
    ns.performed_init_BI = {"bond": False, "angle": False, "dihedral": False}
    ns.opti_geoms_all = set(geom for opti_cycle_geoms in opti_cycles for geom in opti_cycle_geoms)
    ns.best_fitness = [np.inf, None]  # fitness_score, eval_step_best_score

    # storage for best independent set of parameters by geom, for initialization of a (few ?) special particle after 1st opti cycle
    ns.all_best_emd_dist_geoms = {"constraints": {}, "bonds": {}, "angles": {}, "dihedrals": {}}
    ns.all_best_params_dist_geoms = {"constraints": {}, "bonds": {}, "angles": {}, "dihedrals": {}}
    for i in range(ns.cg_itp["nb_constraints"]):
        ns.all_best_emd_dist_geoms["constraints"][i] = config.sim_crash_EMD_indep_score
        ns.all_best_params_dist_geoms["constraints"][i] = {}
    for i in range(ns.cg_itp["nb_bonds"]):
        ns.all_best_emd_dist_geoms["bonds"][i] = config.sim_crash_EMD_indep_score
        ns.all_best_params_dist_geoms["bonds"][i] = {}
    for i in range(ns.cg_itp["nb_angles"]):
        ns.all_best_emd_dist_geoms["angles"][i] = config.sim_crash_EMD_indep_score
        ns.all_best_params_dist_geoms["angles"][i] = {}
    for i in range(ns.cg_itp["nb_dihedrals"]):
        ns.all_best_emd_dist_geoms["dihedrals"][i] = config.sim_crash_EMD_indep_score
        ns.all_best_params_dist_geoms["dihedrals"][i] = {}

    #############################
    # START OPTIMIZATION CYCLES #
    #############################

    for i in range(len(opti_cycles)):

        ns.opti_cycle = {"nb_cycle": i + 1, "geoms": opti_cycles[i],
                         "nb_geoms": {"constraint": 0, "bond": 0, "angle": 0, "dihedral": 0}}
        ns.out_itp = copy.deepcopy(
            ns.opti_itp)  # input ITP copy, on which we might perform BI, and that is the object we will modify at each evaluation step to store the values from FST-PSO

        # model selection based on fitness + Rg during last optimization cycle
        # ns.all_rg_last_cycle, ns.all_fitness_last_cycle = np.array([]), np.array([])
        # ns.best_fitness_Rg_combined = 0 # id of the best model based on bonded fitness + Rg selection

        ns.prod_sim_time = sim_types[sim_cycles[i]]["sim_duration"]
        ns.prod_nb_frames = sim_types[sim_cycles[i]]["prod_nb_frames"]

        ns.val_guess_fact = sim_types[sim_cycles[i]]["val_guess_fact"]
        ns.fct_guess_fact = sim_types[sim_cycles[i]]["fct_guess_fact"]
        ns.max_swarm_iter = sim_types[sim_cycles[i]]["max_swarm_iter"]
        ns.max_swarm_iter_without_new_global_best = sim_types[sim_cycles[i]]["max_swarm_iter_without_new_global_best"]

        # adapt number of geoms according to the optimization cycle
        geoms_display = []
        if "constraint" in ns.opti_cycle["geoms"] or "bond" in ns.opti_cycle["geoms"]:
            geoms_display.append("constraints/bonds")
        if "constraint" in ns.opti_cycle["geoms"]:
            ns.opti_cycle["nb_geoms"]["constraint"] = ns.cg_itp["nb_constraints"]
        if "bond" in ns.opti_cycle["geoms"]:
            ns.opti_cycle["nb_geoms"]["bond"] = ns.cg_itp["nb_bonds"]
        if "angle" in ns.opti_cycle["geoms"]:
            ns.opti_cycle["nb_geoms"]["angle"] = ns.cg_itp["nb_angles"]
            geoms_display.append("angles")
        if "dihedral" in ns.opti_cycle["geoms"]:
            ns.opti_cycle["nb_geoms"]["dihedral"] = ns.cg_itp["nb_dihedrals"]
            geoms_display.append("dihedrals")
        geoms_display = " & ".join(geoms_display)

        print()
        print(swarmcg.shared.styling.sep_close)
        print("| STARTING OPTIMIZATION CYCLE", ns.opti_cycle["nb_cycle"],
              "                                                              |")
        print("| Optimizing", geoms_display, " " * (95 - 16 - len(geoms_display)), "|")
        print(swarmcg.shared.styling.sep_close)

        # actual BI to get the initial guesses of force constants, for all selected geoms at this given optimization step
        # BI is performed:
        # -- exec_mode 1: all equilibrium values and force constants
        # -- exec_mode 2: equilibrium values are not touched for bonds, angles and dihedrals, but all their force constants are optimized
        forcefield.perform_BI(ns.out_itp, ns.opti_cycle, ns.data_BI, ns.performed_init_BI, ns.temp, exec_mode=ns.exec_mode)

        # build vector for search space boundaries + create variations around the BI initial guesses
        search_space_boundaries = forcefield.get_search_space_boundaries(ns.cg_itp, ns.opti_cycle, ns.domains_val, ns.exec_mode, optimization_config=ns.config.optimization)

        # ns.worst_fit_score = round(len(search_space_boundaries) * config.sim_crash_EMD_indep_score, 3)
        ns.worst_fit_score = round( \
            np.sqrt((ns.cg_itp["nb_constraints"] + ns.cg_itp["nb_bonds"]) * config.sim_crash_EMD_indep_score) + \
            np.sqrt(ns.cg_itp["nb_angles"] * config.sim_crash_EMD_indep_score) + \
            np.sqrt(ns.cg_itp["nb_dihedrals"] * config.sim_crash_EMD_indep_score) \
            , 3)
        # nb_particles = int(10 + 2*np.sqrt(len(search_space_boundaries)))  # formula used by FST-PSO to choose nb of particles, which defines the number of initial guesses we can use
        nb_particles = particle_setter(
            search_space_boundaries)  # adapted to have less particles and fitted to our problems, which has good initial guesses and error driven initialization
        
        initial_guess_list = forcefield.get_initial_guess_list(nb_particles, ns.opti_cycle, ns.cg_itp, ns.out_itp, ns.domains_val,
                           ns.all_best_emd_dist_geoms, ns.all_best_params_dist_geoms,
                           ns.exec_mode, user_input=ns.user_input,
                           config_obj=ns.config, 
                           val_guess_fact=ns.val_guess_fact, fct_guess_fact=ns.fct_guess_fact)

        # actual optimization
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull):
                FP = FuzzyPSO()
                FP.set_search_space(search_space_boundaries)
                FP.set_swarm_size(nb_particles)
                FP.set_fitness(fitness=eval_function, arguments=ns, skip_test=True)
                result = FP.solve_with_fstpso(max_iter=ns.max_swarm_iter, initial_guess_list=initial_guess_list,
                                              max_iter_without_new_global_best=ns.max_swarm_iter_without_new_global_best)

        # update ITP object with the best solution using geoms considered at this given optimization step
        forcefield.update_cg_itp_obj(ns.out_itp, ns.opti_cycle, parameters_set=result[0].X, exec_mode=ns.exec_mode)

    # clean temporary copied directory with user"s input files
    shutil.rmtree(ns.exec_folder + "/" + config.input_sim_files_dirname)

    # print some stats
    total_time_sec = datetime.now().timestamp() - ns.start_opti_ts
    total_time = round(total_time_sec / (60 * 60), 2)
    fitness_eval_time = round(ns.total_eval_time / (60 * 60), 2)
    init_time = round((total_time_sec - ns.total_eval_time) / (60 * 60), 2)
    ns.total_gmx_time = round(ns.total_gmx_time / (60 * 60), 2)
    ns.total_model_eval_time = round(ns.total_model_eval_time / (60 * 60), 2)
    print()
    print(swarmcg.shared.styling.sep_close)
    print("|  FINISHED PROPERLY                                                                          |")
    print(swarmcg.shared.styling.sep_close)
    print()
    print("Total nb of evaluation steps:", ns.nb_eval)
    print("Best model obtained at evaluation step number:", ns.best_fitness[1])
    print()
    print(f"Total execution time : {total_time} h")
    print(f"Initialization time  : {init_time} h ({round(init_time / total_time * 100, 2)} %)")
    print(f"Simulations time     : {ns.total_gmx_time} h ({round(ns.total_gmx_time / total_time * 100, 2)} %)")
    print(
        f"Models scoring time  : {ns.total_model_eval_time} h ({round(ns.total_model_eval_time / total_time * 100, 2)} %)")
    print()


def main():
    args_parser = io.get_optimize_args()

    # display help if script was called without arguments
    if len(sys.argv) == 1:
        args_parser.print_help()
        sys.exit()

    # arguments handling, display command line if help or no arguments provided
    # argcomplete.autocomplete(parser)
    ns_args = args_parser.parse_args()

    # do NOT display the stack by default
    if not ns_args.verbose:
        sys.tracebacklimit = 0

    input_cmdline = " ".join(map(cmd_quote, sys.argv))

    # Convert to SwarmConfig
    swarm_config = SwarmConfig.from_namespace(ns_args)
    
    # We delay exec_folder creation to run() or handle it here if needed for logging?
    # Original main set ns.exec_folder for print.
    # We can reconstruct it or let run() handle it. 
    # run() logic above sets exec_folder from config.
    # To print it here, we replicate logic
    if swarm_config.output.output_folder != "":
         exec_folder = swarm_config.output.output_folder
    else:
         exec_folder = time.strftime("MODEL_OPTI__STARTED_%d-%m-%Y_%Hh%Mm%Ss")

    print("Working directory:", os.getcwd())
    print("Command line:", input_cmdline)
    print("Results directory:", exec_folder)

    run(swarm_config)


if __name__ == "__main__":
    main()
