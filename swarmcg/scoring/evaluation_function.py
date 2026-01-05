import gzip
import os
import shutil
import time
from datetime import datetime

from swarmcg import config, io as io, simulations as sim
from swarmcg.config_types import SwarmConfig
from swarmcg.context import OptimizationContext
from swarmcg.scoring.compare import compare_models
from swarmcg.forcefield import update_cg_itp_obj
from swarmcg.shared import exceptions, styling
from swarmcg.utils import print_stdout_forced


def eval_function(parameters_set, ns: OptimizationContext):
    """Evaluation function to be optimized using FST-PSO.

    ns requires:
        nb_eval (edited inplace)
        best_fitness (edited inplace)
        sasa_cg (edited inplace)
        exec_folder
        cg_itp_basename
        opti_cycle
        out_itp
        worst_fit_score

    ns creates:
        cg_tpr_filename
        cg_traj_filename
        plot_filename
        all_emd_dist_geoms
        gyr_cg
        gyr_cg_std
        sasa_cg_std
        total_gmx_time
        total_eval_time
        total_model_eval_time

    pass ns to:
       compare_models
    """
    original_dir = os.getcwd()
    exec_dir = os.path.abspath(ns.files.exec_folder)
    all_evals_dir = os.path.join(exec_dir, config.all_evals_files_dirname)
    os.makedirs(all_evals_dir, exist_ok=True)
    ns.status.nb_eval += 1
    start_eval_ts = datetime.now().timestamp()

    print_stdout_forced()
    # TODO: this should use logging
    print_stdout_forced(
        f"Starting iteration {ns.status.nb_eval} at {time.strftime('%H:%M:%S')} on {time.strftime('%d-%m-%Y')}"
    )
    
    eval_score = ns.pso.worst_fit_score
    fit_score_total = ns.pso.worst_fit_score
    fit_score_constraints_bonds = ns.pso.worst_fit_score
    fit_score_angles = ns.pso.worst_fit_score
    fit_score_dihedrals = ns.pso.worst_fit_score
    all_dist_pairwise = ""
    all_emd_dist_geoms = None
    new_best_fit = False
    current_eval_dir = f"{config.iteration_sim_files_dirname}_eval_step_{ns.status.nb_eval}"
    current_eval_path = os.path.join(exec_dir, current_eval_dir)

    try:
        os.chdir(exec_dir)

        if os.path.exists(current_eval_path):
            shutil.rmtree(current_eval_path)

        shutil.copytree(os.path.join(exec_dir, config.input_sim_files_dirname), current_eval_path)

        update_cg_itp_obj(ns.out_itp, ns.opti_cycle, parameters_set, ns.config.optimization.exec_mode)
        out_path_itp = os.path.join(current_eval_path, ns.files.cg_itp_basename)

        if ns.opti_cycle["nb_geoms"]["dihedral"] == 0:
            print_sections = ["constraint", "bond", "angle", "exclusion"]
        else:
            print_sections = ["constraint", "bond", "angle", "dihedral", "exclusion"]
        io.write_cg_itp_file(ns.out_itp, out_path_itp, print_sections=print_sections)
        itp_root, _ = os.path.splitext(ns.files.cg_itp_basename)
        eval_itp_name = f"{itp_root}_eval_step_{ns.status.nb_eval}.itp"
        shutil.copy(out_path_itp, os.path.join(all_evals_dir, eval_itp_name))

        os.chdir(current_eval_path)

        start_gmx_ts = datetime.now().timestamp()
        swarm_config = ns.config

        sim_manager = sim.SimulationManager(swarm_config)
        sim_failed = False
        sim_error = None
        try:
            sim_manager.run_simulation(
                os.getcwd(),
                sim_time=getattr(ns.status, 'prod_sim_time', None),
                nb_frames=getattr(ns.status, 'prod_nb_frames', None)
            )
        except exceptions.ComputationError as exc:
            sim_failed = True
            sim_error = exc

        if not sim_failed and os.path.isfile("md.gro"):

            ns.files.cg_tpr_filename = "md.tpr"
            ns.files.cg_traj_filename = "md.xtc"
            ns.files.plot_filename = "distributions.png"
            ns.status.total_gmx_time += datetime.now().timestamp() - start_gmx_ts

            start_model_eval_ts = datetime.now().timestamp()
            ignore_dihedrals = ns.opti_cycle["nb_geoms"]["dihedral"] == 0
            fit_score_total, fit_score_constraints_bonds, fit_score_angles, fit_score_dihedrals, all_dist_pairwise, all_emd_dist_geoms = compare_models(
                ns, manual_mode=False, ignore_dihedrals=ignore_dihedrals, calc_sasa=True,
                record_best_indep_params=True)
            ns.status.total_model_eval_time += datetime.now().timestamp() - start_model_eval_ts

            if ns.results.sasa_cg is not None:

                shutil.move(
                    "distributions.png",
                    os.path.join(
                        all_evals_dir,
                        f"distributions_eval_step_{ns.status.nb_eval}.png",
                    ),
                )

                eval_score = 0
                if "constraint" in ns.opti_cycle["geoms"] and "bond" in ns.opti_cycle["geoms"]:
                    eval_score += fit_score_constraints_bonds
                if "angle" in ns.opti_cycle["geoms"]:
                    eval_score += fit_score_angles
                if "dihedral" in ns.opti_cycle["geoms"]:
                    eval_score += fit_score_dihedrals

                global_score = 0
                if "constraint" in ns.pso.opti_geoms_all and "bond" in ns.pso.opti_geoms_all:
                    global_score += fit_score_constraints_bonds
                if "angle" in ns.pso.opti_geoms_all:
                    global_score += fit_score_angles
                if "dihedral" in ns.pso.opti_geoms_all:
                    global_score += fit_score_dihedrals

                if global_score < ns.pso.best_fitness[0]:
                    new_best_fit = True
                    ns.pso.best_fitness = global_score, ns.status.nb_eval
                    ns.pso.all_emd_dist_geoms = all_emd_dist_geoms

            else:
                eval_score = ns.pso.worst_fit_score
                fit_score_total = ns.pso.worst_fit_score
                fit_score_constraints_bonds = ns.pso.worst_fit_score
                fit_score_angles = ns.pso.worst_fit_score
                fit_score_dihedrals = ns.pso.worst_fit_score
                ns.results.gyr_cg, ns.results.gyr_cg_std, ns.results.sasa_cg, ns.results.sasa_cg_std = None, None, None, None
                ns.status.total_gmx_time += datetime.now().timestamp() - start_gmx_ts
        else:
            if sim_failed:
                print_stdout_forced(
                    styling.header_warning
                    + "Simulation failed; assigning worst score and continuing.\n"
                    + str(sim_error)
                )
            eval_score = ns.pso.worst_fit_score
            fit_score_total = ns.pso.worst_fit_score
            fit_score_constraints_bonds = ns.pso.worst_fit_score
            fit_score_angles = ns.pso.worst_fit_score
            fit_score_dihedrals = ns.pso.worst_fit_score
            ns.results.gyr_cg, ns.results.gyr_cg_std, ns.results.sasa_cg, ns.results.sasa_cg_std = None, None, None, None
            ns.status.total_gmx_time += datetime.now().timestamp() - start_gmx_ts

        os.chdir(exec_dir)

        log_sources = [
            ("md.log", f"md_sim_eval_step_{ns.status.nb_eval}.log.gz"),
            ("equi.log", f"equi_sim_eval_step_{ns.status.nb_eval}.log.gz"),
            ("mini.log", f"mini_sim_eval_step_{ns.status.nb_eval}.log.gz"),
        ]
        for src_name, dest_name in log_sources:
            src_path = os.path.join(current_eval_path, src_name)
            if os.path.isfile(src_path):
                dest_path = os.path.join(all_evals_dir, dest_name)
                with open(src_path, "rb") as src_fp, gzip.open(dest_path, "wb") as dest_fp:
                    shutil.copyfileobj(src_fp, dest_fp)

        if new_best_fit:
            shutil.copy(
                os.path.join(
                    all_evals_dir,
                    f"distributions_eval_step_{ns.status.nb_eval}.png",
                ),
                os.path.join(exec_dir, config.best_distrib_plots),
            )

        if ns.config.optimization.keep_all_sims:
            shutil.copytree(
                current_eval_path,
                os.path.join(exec_dir, config.sim_files_all_evals_dirname, current_eval_dir),
            )

        if ns.status.nb_eval == 1:
            shutil.copytree(current_eval_path, os.path.join(exec_dir, "boltzmann_inv_CG_model"))

        if new_best_fit:
            best_model_path = os.path.join(exec_dir, config.best_fitted_model_dirname)
            if os.path.exists(best_model_path):
                shutil.rmtree(best_model_path)
            shutil.move(current_eval_path, best_model_path)
        else:
            shutil.rmtree(current_eval_path)

        if eval_score == ns.pso.worst_fit_score:
            all_dist_pairwise = ""
            for _ in range(len(ns.cg_itp["constraint"]) + len(ns.cg_itp["bond"]) + len(ns.cg_itp["angle"]) + len(
                    ns.cg_itp["dihedral"])):
                all_dist_pairwise += str(config.sim_crash_EMD_indep_score) + " "
            all_dist_pairwise += "\n"
        else:
            print_stdout_forced("  Total mismatch score:", round(fit_score_total, 3), "(Bonds/Constraints:",
                                fit_score_constraints_bonds, "-- Angles:", fit_score_angles, "-- Dihedrals:",
                                str(fit_score_dihedrals) + ")")
            if new_best_fit:
                print_stdout_forced("    --> Selected as new best bonded parametrization")
            print_stdout_forced(
                f"  Rg CG:   {round(ns.results.gyr_cg, 2)} nm   (Error abs. {round(abs(1 - ns.results.gyr_cg / ns.results.gyr_aa_mapped) * 100, 1)}% -- Reference Rg AA-mapped: {ns.results.gyr_aa_mapped} nm)")
            print_stdout_forced(
                f"  SASA CG: {ns.results.sasa_cg} nm2   (Error abs. {round(abs(1 - ns.results.sasa_cg / ns.results.sasa_aa_mapped) * 100, 1)}% -- Reference SASA AA-mapped: {ns.results.sasa_aa_mapped} nm2)")

        current_total_time = round((datetime.now().timestamp() - ns.status.start_opti_ts) / (60 * 60), 2)
        current_eval_time = datetime.now().timestamp() - start_eval_ts
        ns.status.total_eval_time += current_eval_time
        current_eval_time = round(current_eval_time / 60, 2)
        print_stdout_forced(f"  Iteration time: {current_eval_time} min")

        with open(os.path.join(exec_dir, config.opti_pairwise_distances_file), "a") as fp:
            if "dihedral" in ns.opti_cycle["geoms"]:
                fp.write("1 " + all_dist_pairwise)
            else:
                fp.write("0 " + all_dist_pairwise)
        with open(os.path.join(exec_dir, config.opti_perf_recap_file), "a") as fp:
            recap_line = " ".join(list(map(str, (
            ns.opti_cycle["nb_cycle"], ns.status.nb_eval, fit_score_total, fit_score_constraints_bonds, fit_score_angles,
            fit_score_dihedrals, eval_score, ns.results.gyr_aa_mapped, ns.results.gyr_aa_mapped_std, ns.results.gyr_cg, ns.results.gyr_cg_std,
            ns.results.sasa_aa_mapped, ns.results.sasa_aa_mapped_std, ns.results.sasa_cg, ns.results.sasa_cg_std)))) + " "
            for i in range(len(ns.cg_itp["constraint"])):
                recap_line += f"{ns.out_itp['constraint'][i]['value']} "
            for i in range(len(ns.cg_itp["bond"])):
                recap_line += f"{ns.out_itp['bond'][i]['value']} {ns.out_itp['bond'][i]['fct']} "
            for i in range(len(ns.cg_itp["angle"])):
                recap_line += f"{ns.out_itp['angle'][i]['value']} {ns.out_itp['angle'][i]['fct']} "
            for i in range(len(ns.cg_itp["dihedral"])):
                func = ns.cg_itp["dihedral"][i]["func"]
                if ns.opti_cycle["nb_geoms"]["dihedral"] == 0:
                    if func in (3, 11):
                        recap_line += "0 0 0 0 0 0 "
                    else:
                        recap_line += "0 0 "
                else:
                    if func in (3, 11):
                        recap_line += " ".join(map(str, ns.out_itp["dihedral"][i]["params"])) + " "
                    else:
                        recap_line += f"{ns.out_itp['dihedral'][i]['value']} {ns.out_itp['dihedral'][i]['fct']} "
            recap_line += f"{current_eval_time} {current_total_time}"
            fp.write(recap_line + "\n")
    finally:
        os.chdir(original_dir)

    return eval_score
