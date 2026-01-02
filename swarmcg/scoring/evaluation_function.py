import os
import shutil
import time
from datetime import datetime

from swarmcg import config, io as io, simulations as sim
from swarmcg.context import SwarmCGArgs, SwarmCGState
from swarmcg.swarmCG import update_cg_itp_obj, compare_models
from swarmcg.utils import print_stdout_forced


def eval_function(parameters_set, args: SwarmCGArgs, state: SwarmCGState):
    """Evaluation function to be optimized using FST-PSO.

    args/state requires:
        opti.nb_eval (edited inplace)
        opti.best_fitness (edited inplace)
        model.sasa_cg (edited inplace)
        files.exec_folder
        files.cg_itp_basename
        opti.opti_cycle
        opti.out_itp
        opti.worst_fit_score

    args creates:
        inputs.cg_tpr_filename
        inputs.cg_traj_filename
        paths.plot_filename

    state creates:
        opti.all_emd_dist_geoms
        model.gyr_cg
        model.gyr_cg_std
        model.sasa_cg_std
        opti.total_gmx_time
        opti.total_eval_time
        opti.total_model_eval_time

    pass args/state to:
       compare_models
    """
    state.opti.nb_eval += 1
    start_eval_ts = datetime.now().timestamp()

    print_stdout_forced()
    # TODO: this should use logging
    print_stdout_forced(
        f"Starting iteration {state.opti.nb_eval} at {time.strftime('%H:%M:%S')} on {time.strftime('%d-%m-%Y')}"
    )

    # enter the execution directory
    os.chdir(state.files.exec_folder)

    # create new directory for new parameters evaluation
    current_eval_dir = f"{config.iteration_sim_files_dirname}_eval_step_{state.opti.nb_eval}"
    shutil.copytree(config.input_sim_files_dirname, current_eval_dir)

    # create a modified CG ITP file with parameters according to current evaluation type
    update_cg_itp_obj(args, state, parameters_set=parameters_set, update_type=1)
    out_path_itp = f"{config.iteration_sim_files_dirname}_eval_step_{state.opti.nb_eval}/{state.files.cg_itp_basename}"
    if state.opti.opti_cycle["nb_geoms"]["dihedral"] == 0:
        print_sections = ["constraint", "bond", "angle", "exclusion"]
    else:
        print_sections = ["constraint", "bond", "angle", "dihedral", "exclusion"]
    io.write_cg_itp_file(state.opti.out_itp, out_path_itp, print_sections=print_sections)

    # enter current evaluation directory and stay there until all sims are finished or failed
    os.chdir(current_eval_dir)

    # run simulation with new parameters
    new_best_fit = False
    start_gmx_ts = datetime.now().timestamp()

    for step in sim.generate_steps(args, state):
        step.run(os.getcwd())

    # to verify if MD run finished properly, we check for the .gro file printed in the end
    if os.path.isfile("md.gro"):

        # get distributions and evaluate fitness
        args.inputs.cg_tpr_filename = "md.tpr"
        args.inputs.cg_traj_filename = "md.xtc"
        args.paths.plot_filename = "distributions.png"
        state.opti.total_gmx_time += datetime.now().timestamp() - start_gmx_ts

        start_model_eval_ts = datetime.now().timestamp()
        ignore_dihedrals = False
        if state.opti.opti_cycle["nb_geoms"]["dihedral"] == 0:
            ignore_dihedrals = True
        fit_score_total, fit_score_constraints_bonds, fit_score_angles, fit_score_dihedrals, all_dist_pairwise, all_emd_dist_geoms = compare_models(
            args, state, manual_mode=False, ignore_dihedrals=ignore_dihedrals, calc_sasa=True,
            record_best_indep_params=True)
        state.opti.total_model_eval_time += datetime.now().timestamp() - start_model_eval_ts

        # if gmx sasa failed to compute, it's most likely because there were inconsistent shifts across PBC in the trajectory = failed run
        if state.model.sasa_cg is not None:

            # store the distributions for each evaluation step
            shutil.move("distributions.png",
                        f"../{config.distrib_plots_all_evals_dirname}/distributions_eval_step_{state.opti.nb_eval}.png")

            eval_score = 0
            if "constraint" in state.opti.opti_cycle["geoms"] and "bond" in state.opti.opti_cycle["geoms"]:
                eval_score += fit_score_constraints_bonds
            if "angle" in state.opti.opti_cycle["geoms"]:
                eval_score += fit_score_angles
            if "dihedral" in state.opti.opti_cycle["geoms"]:
                eval_score += fit_score_dihedrals

            global_score = 0
            if "constraint" in state.opti.opti_geoms_all and "bond" in state.opti.opti_geoms_all:
                global_score += fit_score_constraints_bonds
            if "angle" in state.opti.opti_geoms_all:
                global_score += fit_score_angles
            if "dihedral" in state.opti.opti_geoms_all:
                global_score += fit_score_dihedrals

            # model selection based only on bonded parametrization score
            regular_eval = True
            if regular_eval:
                if global_score < state.opti.best_fitness[0]:
                    new_best_fit = True
                    state.opti.best_fitness = global_score, state.opti.nb_eval
                    state.opti.all_emd_dist_geoms = all_emd_dist_geoms

        else:
            eval_score, fit_score_total, fit_score_constraints_bonds, fit_score_angles, fit_score_dihedrals = [
                                                                                                                  state.opti.worst_fit_score] * 5
            state.model.gyr_cg, state.model.gyr_cg_std, state.model.sasa_cg, state.model.sasa_cg_std = None, None, None, None
            state.opti.total_gmx_time += datetime.now().timestamp() - start_gmx_ts

    # exit current eval directory
    os.chdir("..")

    # store all log files
    if os.path.isfile(current_eval_dir + "/md.log"):
        shutil.copy(current_eval_dir + "/md.log",
                    f"{config.log_files_all_evals_dirname}/md_sim_eval_step_{state.opti.nb_eval}.log")
    if os.path.isfile(current_eval_dir + "/equi.log"):
        shutil.copy(current_eval_dir + "/equi.log",
                    f"{config.log_files_all_evals_dirname}/equi_sim_eval_step_{state.opti.nb_eval}.log")
    if os.path.isfile(current_eval_dir + "/mini.log"):
        shutil.copy(current_eval_dir + "/mini.log",
                    f"{config.log_files_all_evals_dirname}/mini_sim_eval_step_{state.opti.nb_eval}.log")

    # update the best results distrib plot in execution directory
    if new_best_fit:
        shutil.copy(f"{config.distrib_plots_all_evals_dirname}/distributions_eval_step_{state.opti.nb_eval}.png",
                    config.best_distrib_plots)

    # keep all sim files if user wants to
    if args.runtime.keep_all_sims:
        shutil.copytree(current_eval_dir, config.sim_files_all_evals_dirname + "/" + current_eval_dir)

    # keep BI files (the very first guess of bonded parameters) only for figures
    # TODO: remove ?? this is redundant because we already produce a directory with output for the best current model
    if state.opti.nb_eval == 1:
        shutil.copytree(current_eval_dir, "boltzmann_inv_CG_model")

    # store sim files for new best fit OR remove eval sim files
    if new_best_fit:
        if os.path.exists(config.best_fitted_model_dirname):
            shutil.rmtree(config.best_fitted_model_dirname)
        shutil.move(current_eval_dir, config.best_fitted_model_dirname)
    else:
        shutil.rmtree(current_eval_dir)

    # when simulation crashes, write the worst possible score considering all geoms
    if eval_score == state.opti.worst_fit_score:
        all_dist_pairwise = ""
        for _ in range(len(state.model.cg_itp["constraint"]) + len(state.model.cg_itp["bond"]) + len(state.model.cg_itp["angle"]) + len(
                state.model.cg_itp["dihedral"])):
            all_dist_pairwise += str(config.sim_crash_EMD_indep_score) + " "
        all_dist_pairwise += "\n"
    else:
        print_stdout_forced("  Total mismatch score:", round(fit_score_total, 3), "(Bonds/Constraints:",
                            fit_score_constraints_bonds, "-- Angles:", fit_score_angles, "-- Dihedrals:",
                            str(fit_score_dihedrals) + ")")
        if new_best_fit:
            print_stdout_forced("    --> Selected as new best bonded parametrization")
        # print_stdout_forced("  Opti context mismatch score:", round(eval_score, 3))
        print_stdout_forced(
            f"  Rg CG:   {round(state.model.gyr_cg, 2)} nm   (Error abs. {round(abs(1 - state.model.gyr_cg / state.model.gyr_aa_mapped) * 100, 1)}% -- Reference Rg AA-mapped: {state.model.gyr_aa_mapped} nm)")
        print_stdout_forced(
            f"  SASA CG: {state.model.sasa_cg} nm2   (Error abs. {round(abs(1 - state.model.sasa_cg / state.model.sasa_aa_mapped) * 100, 1)}% -- Reference SASA AA-mapped: {state.model.sasa_aa_mapped} nm2)")

    current_total_time = round((datetime.now().timestamp() - state.opti.start_opti_ts) / (60 * 60), 2)
    current_eval_time = datetime.now().timestamp() - start_eval_ts
    state.opti.total_eval_time += current_eval_time
    current_eval_time = round(current_eval_time / 60, 2)
    print_stdout_forced(f"  Iteration time: {current_eval_time} min")

    # write all pairwise distances between atom mapped and CG geoms to file for later global optimization perf plotting
    with open(config.opti_pairwise_distances_file, "a") as fp:
        if "dihedral" in state.opti.opti_cycle["geoms"]:
            fp.write("1 " + all_dist_pairwise)
        else:
            fp.write("0 " + all_dist_pairwise)
    with open(config.opti_perf_recap_file, "a") as fp:
        recap_line = " ".join(list(map(str, (
        state.opti.opti_cycle["nb_cycle"], state.opti.nb_eval, fit_score_total, fit_score_constraints_bonds, fit_score_angles,
        fit_score_dihedrals, eval_score, state.model.gyr_aa_mapped, state.model.gyr_aa_mapped_std, state.model.gyr_cg, state.model.gyr_cg_std,
        state.model.sasa_aa_mapped, state.model.sasa_aa_mapped_std, state.model.sasa_cg, state.model.sasa_cg_std)))) + " "
        for i in range(len(state.model.cg_itp["constraint"])):
            recap_line += f"{state.opti.out_itp['constraint'][i]['value']} "
        for i in range(len(state.model.cg_itp["bond"])):
            recap_line += f"{state.opti.out_itp['bond'][i]['value']} {state.opti.out_itp['bond'][i]['fct']} "
        for i in range(len(state.model.cg_itp["angle"])):
            recap_line += f"{state.opti.out_itp['angle'][i]['value']} {state.opti.out_itp['angle'][i]['fct']} "
        for i in range(len(state.model.cg_itp["dihedral"])):
            if state.opti.opti_cycle["nb_geoms"]["dihedral"] == 0:
                recap_line += "0 0 "
            else:
                recap_line += f"{state.opti.out_itp['dihedral'][i]['value']} {state.opti.out_itp['dihedral'][i]['fct']} "
        recap_line += f"{current_eval_time} {current_total_time}"
        fp.write(recap_line + "\n")

    os.chdir("..")  # exit the execution directory

    return eval_score
