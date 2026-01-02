import collections

import numpy as np
from scipy.optimize import curve_fit

from swarmcg import config
from swarmcg.context import SwarmCGArgs, SwarmCGState
from swarmcg.shared import math_utils, exceptions, catch_warnings
from swarmcg.shared.math_utils import draw_float
from swarmcg.simulations.potentials import (
    gmx_bonds_func_1,
    gmx_angles_func_1,
    gmx_angles_func_2,
    gmx_dihedrals_func_1,
    gmx_dihedrals_func_2,
)


def update_cg_itp_obj(args: SwarmCGArgs, state: SwarmCGState, parameters_set, update_type):
    """Update coarse-grain ITP.

    args requires:
        runtime.exec_mode

    state requires:
        opti.out_itp (edited inplace)
        opti.opti_cycle
    """
    if update_type == 1:  # intermediary
        itp_obj = state.opti.out_itp
    elif update_type == 2:  # cycles optimized
        itp_obj = state.opti.opti_itp
    else:
        msg = (
            f"Code error in function update_cg_itp_obj.\nPlease consider opening an issue on GitHub "
            f"at {config.github_url}."
        )
        raise exceptions.InputArgumentError(msg)

    for i in range(state.opti.opti_cycle["nb_geoms"]["constraint"]):
        if args.runtime.exec_mode == 1:
            itp_obj["constraint"][i]["value"] = round(parameters_set[i], 3)  # constraint - distance

    for i in range(state.opti.opti_cycle["nb_geoms"]["bond"]):
        if args.runtime.exec_mode == 1:
            itp_obj["bond"][i]["value"] = round(parameters_set[state.opti.opti_cycle["nb_geoms"]["constraint"] + i],
                                                3)  # bond - distance
            itp_obj["bond"][i]["fct"] = round(
                parameters_set[state.opti.opti_cycle["nb_geoms"]["constraint"] + state.opti.opti_cycle["nb_geoms"]["bond"] + i],
                3)  # bond - force constant
        else:
            itp_obj["bond"][i]["fct"] = round(parameters_set[i], 3)  # bond - force constant

    for i in range(state.opti.opti_cycle["nb_geoms"]["angle"]):
        if args.runtime.exec_mode == 1:
            itp_obj["angle"][i]["value"] = round(
                parameters_set[state.opti.opti_cycle["nb_geoms"]["constraint"] + 2 * state.opti.opti_cycle["nb_geoms"]["bond"] + i],
                2)  # angle - value
            itp_obj["angle"][i]["fct"] = round(parameters_set[state.opti.opti_cycle["nb_geoms"]["constraint"] + 2 *
                                                              state.opti.opti_cycle["nb_geoms"]["bond"] +
                                                              state.opti.opti_cycle["nb_geoms"]["angle"] + i],
                                               2)  # angle - force constant
        else:
            itp_obj["angle"][i]["fct"] = round(parameters_set[state.opti.opti_cycle["nb_geoms"]["bond"] + i],
                                               2)  # angle - force constant

    for i in range(state.opti.opti_cycle["nb_geoms"]["dihedral"]):
        if args.runtime.exec_mode == 1:
            itp_obj["dihedral"][i]["value"] = round(parameters_set[state.opti.opti_cycle["nb_geoms"]["constraint"] + 2 *
                                                                   state.opti.opti_cycle["nb_geoms"]["bond"] + 2 *
                                                                   state.opti.opti_cycle["nb_geoms"]["angle"] + i],
                                                    2)  # dihedral - value
            itp_obj["dihedral"][i]["fct"] = round(parameters_set[state.opti.opti_cycle["nb_geoms"]["constraint"] + 2 *
                                                                 state.opti.opti_cycle["nb_geoms"]["bond"] + 2 *
                                                                 state.opti.opti_cycle["nb_geoms"]["angle"] +
                                                                 state.opti.opti_cycle["nb_geoms"]["dihedral"] + i],
                                                  2)  # dihedral - force constant
        else:
            itp_obj["dihedral"][i]["fct"] = round(
                parameters_set[state.opti.opti_cycle["nb_geoms"]["bond"] + state.opti.opti_cycle["nb_geoms"]["angle"] + i],
                2)  # dihedral - force constant


def get_search_space_boundaries(args: SwarmCGArgs, state: SwarmCGState):
    """Set dimensions of the search space according to the type of optimization
    (= geom type(s) to optimize).

    args requires:
        runtime.exec_mode
        optimization.default_max_fct_bonds_opti
        optimization.default_max_fct_angles_opti_f1
        optimization.default_max_fct_angles_opti_f2
        optimization.default_abs_range_fct_dihedrals_opti_func_without_mult
        optimization.default_abs_range_fct_dihedrals_opti_func_with_mult

    state requires:
        opti.domains_val
        opti.opti_cycle
        model.cg_itp
    """
    search_space_boundaries = []

    if state.opti.opti_cycle["nb_geoms"]["constraint"] > 0:
        if args.runtime.exec_mode == 1:
            search_space_boundaries.extend(state.opti.domains_val["constraint"])  # constraints equilibrium values

    if state.opti.opti_cycle["nb_geoms"]["bond"] > 0:
        if args.runtime.exec_mode == 1:
            search_space_boundaries.extend(state.opti.domains_val["bond"])  # bonds equilibrium values
        search_space_boundaries.extend(
            [[0, args.optimization.default_max_fct_bonds_opti]] * state.opti.opti_cycle["nb_geoms"]["bond"])  # bonds force constants

    if state.opti.opti_cycle["nb_geoms"]["angle"] > 0:
        if args.runtime.exec_mode == 1:
            search_space_boundaries.extend(state.opti.domains_val["angle"])  # angles equilibrium values

        for grp_angle in range(state.opti.opti_cycle["nb_geoms"]["angle"]):  # angles force constants
            if state.model.cg_itp["angle"][grp_angle]["func"] == 1:
                search_space_boundaries.extend([[0, args.optimization.default_max_fct_angles_opti_f1]])
            elif state.model.cg_itp["angle"][grp_angle]["func"] == 2:
                search_space_boundaries.extend([[0, args.optimization.default_max_fct_angles_opti_f2]])

    if state.opti.opti_cycle["nb_geoms"]["dihedral"] > 0:
        if args.runtime.exec_mode == 1:
            search_space_boundaries.extend(state.opti.domains_val["dihedral"])  # dihedrals equilibrium values

        for grp_dihedral in range(state.opti.opti_cycle["nb_geoms"]["dihedral"]):  # dihedrals force constants
            if state.model.cg_itp["dihedral"][grp_dihedral]["func"] == 2:
                search_space_boundaries.extend([[-args.optimization.default_abs_range_fct_dihedrals_opti_func_without_mult,
                                                 args.optimization.default_abs_range_fct_dihedrals_opti_func_without_mult]])
            elif state.model.cg_itp["dihedral"][grp_dihedral]["func"] in config.dihedral_func_with_mult:
                search_space_boundaries.extend([[-args.optimization.default_abs_range_fct_dihedrals_opti_func_with_mult,
                                                 args.optimization.default_abs_range_fct_dihedrals_opti_func_with_mult]])

    return search_space_boundaries


def get_initial_guess_list(args: SwarmCGArgs, state: SwarmCGState, nb_particles):
    """Build initial guesses for particles initialization, as variations around parameters obtained
    via Boltzmann inversion (BI).

    This is done in an iterative fashion:
        - read atom mapped traj constraints/bonds and perform BI to obtain the 1st set of parameters
        and then find variations in this function
        - read angles from the best constraints/bonds-only optimized model, perform BI and do the
        ratio with BI of the atom mapped traj to add only the required amount of energy and obtain
        1st set of parameters
        - do dihedrals the similarly, using BI ratio

    args requires:
        runtime.exec_mode
        optimization.default_max_fct_bonds_opti
        optimization.default_max_fct_angles_opti_f1
        optimization.default_max_fct_angles_opti_f2
        optimization.default_abs_range_fct_dihedrals_opti_func_without_mult
        optimization.default_abs_range_fct_dihedrals_opti_func_with_mult

    state requires:
        opti.opti_cycle
        opti.domains_val
        opti.out_itp
        opti.all_best_params_dist_geoms
        opti.all_best_emd_dist_geoms
        opti.all_emd_dist_geoms
        opti.val_guess_fact
        opti.fct_guess_fact
    """
    initial_guess_list = []  # array of arrays (inner arrays are the values used for particles initialization)

    # the first particle is initialized as EXACTLY the values of the current CG ITP object (or BI in exec_mode 1)
    # except if force constants are outside of the searchable domain defined for optimization
    # for bonds lengths and angles/dihedrals values, we perform no checks
    input_guess = []

    if args.runtime.exec_mode == 1:
        for i in range(state.opti.opti_cycle["nb_geoms"]["constraint"]):
            input_guess.append(min(max(state.opti.out_itp["constraint"][i]["value"], state.opti.domains_val["constraint"][i][0]),
                                   state.opti.domains_val["constraint"][i][1]))  # constraints equilibrium values

        for i in range(state.opti.opti_cycle["nb_geoms"]["bond"]):
            input_guess.append(min(max(state.opti.out_itp["bond"][i]["value"], state.opti.domains_val["bond"][i][0]),
                                   state.opti.domains_val["bond"][i][1]))  # bonds equilibrium values

    for i in range(state.opti.opti_cycle["nb_geoms"]["bond"]):
        input_guess.append(
            min(max(state.opti.out_itp["bond"][i]["fct"], 0), args.optimization.default_max_fct_bonds_opti))  # bonds force constants

    if args.runtime.exec_mode == 1:
        for i in range(state.opti.opti_cycle["nb_geoms"]["angle"]):
            input_guess.append(min(max(state.opti.out_itp["angle"][i]["value"], state.opti.domains_val["angle"][i][0]),
                                   state.opti.domains_val["angle"][i][1]))  # angles equilibrium values

    for i in range(state.opti.opti_cycle["nb_geoms"]["angle"]):
        if state.model.cg_itp["angle"][i]["func"] == 1:
            input_guess.append(
                min(max(state.opti.out_itp["angle"][i]["fct"], 0), args.optimization.default_max_fct_angles_opti_f1))  # angles force constants
        elif state.model.cg_itp["angle"][i]["func"] == 2:
            input_guess.append(
                min(max(state.opti.out_itp["angle"][i]["fct"], 0), args.optimization.default_max_fct_angles_opti_f2))  # angles force constants

    if args.runtime.exec_mode == 1:
        for i in range(state.opti.opti_cycle["nb_geoms"]["dihedral"]):
            input_guess.append(min(max(state.opti.out_itp["dihedral"][i]["value"], state.opti.domains_val["dihedral"][i][0]),
                                   state.opti.domains_val["dihedral"][i][1]))  # dihedrals equilibrium values

    for i in range(state.opti.opti_cycle["nb_geoms"]["dihedral"]):
        if state.model.cg_itp["dihedral"][i]["func"] == 2:
            input_guess.append(
                min(max(state.opti.out_itp["dihedral"][i]["fct"], -args.optimization.default_abs_range_fct_dihedrals_opti_func_without_mult),
                    args.optimization.default_abs_range_fct_dihedrals_opti_func_without_mult))  # dihedrals force constants
        else:
            input_guess.append(
                min(max(state.opti.out_itp["dihedral"][i]["fct"], -args.optimization.default_abs_range_fct_dihedrals_opti_func_with_mult),
                    args.optimization.default_abs_range_fct_dihedrals_opti_func_with_mult))  # dihedrals force constants

    initial_guess_list.append(input_guess)
    num_particle_random_start = 1  # first particle is DBI

    # The second particle is initialized either:
    # (1) Using best EMD score for each geom and the parameters that yielded these EMD scores. This is independant
    # of exec_mode, because we use only previously selected parameters for this particle. If yet no independant best
    # is recorded for a given geom (dihedrals in fact), values are taken from best optimized model until now.
    # (2) If we are in opti cycle 1 and -user_params is provided, then this particle is instead
    # initialized as the users parameters.
    if state.opti.opti_cycle["nb_cycle"] > 1:

        num_particle_random_start += 1
        input_guess = []

        # constraints equilibrium values
        if args.runtime.exec_mode == 1:
            for i in range(state.opti.opti_cycle["nb_geoms"]["constraint"]):
                if state.opti.all_best_emd_dist_geoms["constraints"][i] != config.sim_crash_EMD_indep_score:
                    input_guess.append(state.opti.all_best_params_dist_geoms["constraints"][i]["params"][0])
                else:
                    input_guess.append(
                        min(max(state.opti.out_itp["constraint"][i]["value"], state.opti.domains_val["constraint"][i][0]),
                            state.opti.domains_val["constraint"][i][1]))

        # bonds equilibrium values
        if args.runtime.exec_mode == 1:
            for i in range(state.opti.opti_cycle["nb_geoms"]["bond"]):
                if state.opti.all_best_emd_dist_geoms["bonds"][i] != config.sim_crash_EMD_indep_score:
                    input_guess.append(state.opti.all_best_params_dist_geoms["bonds"][i]["params"][0])
                else:
                    input_guess.append(min(max(state.opti.out_itp["bond"][i]["value"], state.opti.domains_val["bond"][i][0]),
                                           state.opti.domains_val["bond"][i][1]))
        # bonds force constants
        for i in range(state.opti.opti_cycle["nb_geoms"]["bond"]):
            if state.opti.all_best_emd_dist_geoms["bonds"][i] != config.sim_crash_EMD_indep_score:
                input_guess.append(state.opti.all_best_params_dist_geoms["bonds"][i]["params"][1])
            else:
                input_guess.append(min(max(state.opti.out_itp["bond"][i]["fct"], 0), args.optimization.default_max_fct_bonds_opti))

        # angles equilibrium values
        if args.runtime.exec_mode == 1:
            for i in range(state.opti.opti_cycle["nb_geoms"]["angle"]):
                if state.opti.all_best_emd_dist_geoms["angles"][i] != config.sim_crash_EMD_indep_score:
                    input_guess.append(state.opti.all_best_params_dist_geoms["angles"][i]["params"][0])
                else:
                    input_guess.append(min(max(state.opti.out_itp["angle"][i]["value"], state.opti.domains_val["angle"][i][0]),
                                           state.opti.domains_val["angle"][i][1]))
        # angles force constants
        for i in range(state.opti.opti_cycle["nb_geoms"]["angle"]):
            if state.opti.all_best_emd_dist_geoms["angles"][i] != config.sim_crash_EMD_indep_score:
                input_guess.append(state.opti.all_best_params_dist_geoms["angles"][i]["params"][1])
            else:
                if state.model.cg_itp["angle"][i]["func"] == 1:
                    input_guess.append(min(max(state.opti.out_itp["angle"][i]["fct"], 0), args.optimization.default_max_fct_angles_opti_f1))
                elif state.model.cg_itp["angle"][i]["func"] == 2:
                    input_guess.append(min(max(state.opti.out_itp["angle"][i]["fct"], 0), args.optimization.default_max_fct_angles_opti_f2))

        # dihedrals equilibrium values
        if args.runtime.exec_mode == 1:
            for i in range(state.opti.opti_cycle["nb_geoms"]["dihedral"]):
                if state.opti.all_best_emd_dist_geoms["dihedrals"][i] != config.sim_crash_EMD_indep_score:
                    input_guess.append(state.opti.all_best_params_dist_geoms["dihedrals"][i]["params"][0])
                else:
                    input_guess.append(min(max(state.opti.out_itp["dihedral"][i]["value"], state.opti.domains_val["dihedral"][i][0]),
                                           state.opti.domains_val["dihedral"][i][1]))
        # dihedrals force constants
        for i in range(state.opti.opti_cycle["nb_geoms"]["dihedral"]):
            if state.opti.all_best_emd_dist_geoms["dihedrals"][i] != config.sim_crash_EMD_indep_score:
                input_guess.append(state.opti.all_best_params_dist_geoms["dihedrals"][i]["params"][1])
            else:
                if state.model.cg_itp["dihedral"][i]["func"] == 2:
                    input_guess.append(
                        min(max(state.opti.out_itp["dihedral"][i]["fct"],
                                -args.optimization.default_abs_range_fct_dihedrals_opti_func_without_mult),
                            args.optimization.default_abs_range_fct_dihedrals_opti_func_without_mult))
                else:
                    input_guess.append(min(
                        max(state.opti.out_itp["dihedral"][i]["fct"], -args.optimization.default_abs_range_fct_dihedrals_opti_func_with_mult),
                        args.optimization.default_abs_range_fct_dihedrals_opti_func_with_mult))

        initial_guess_list.append(input_guess)

    # optionally second particle is initialized as input parameters ONLY at start of opti cycle 1
    elif args.inputs.user_input:

        num_particle_random_start += 1
        input_guess = []

        # constraints equilibrium values
        if args.runtime.exec_mode == 1:
            for i in range(state.opti.opti_cycle["nb_geoms"]["constraint"]):
                input_guess.append(
                    min(max(state.opti.out_itp["constraint"][i]["value_user"], state.opti.domains_val["constraint"][i][0]),
                        state.opti.domains_val["constraint"][i][1]))

        # bonds equilibrium values
        if args.runtime.exec_mode == 1:
            for i in range(state.opti.opti_cycle["nb_geoms"]["bond"]):
                input_guess.append(min(max(state.opti.out_itp["bond"][i]["value_user"], state.opti.domains_val["bond"][i][0]),
                                       state.opti.domains_val["bond"][i][1]))

        # bonds force constants
        for i in range(state.opti.opti_cycle["nb_geoms"]["bond"]):
            input_guess.append(min(max(state.opti.out_itp["bond"][i]["fct_user"], 0), args.optimization.default_max_fct_bonds_opti))

        # angles equilibrium values
        if args.runtime.exec_mode == 1:
            for i in range(state.opti.opti_cycle["nb_geoms"]["angle"]):
                input_guess.append(min(max(state.opti.out_itp["angle"][i]["value_user"], state.opti.domains_val["angle"][i][0]),
                                       state.opti.domains_val["angle"][i][1]))

        # angles force constants
        for i in range(state.opti.opti_cycle["nb_geoms"]["angle"]):
            if state.model.cg_itp["angle"][i]["func"] == 1:
                input_guess.append(min(max(state.opti.out_itp["angle"][i]["fct_user"], 0), args.optimization.default_max_fct_angles_opti_f1))
            elif state.model.cg_itp["angle"][i]["func"] == 2:
                input_guess.append(min(max(state.opti.out_itp["angle"][i]["fct_user"], 0), args.optimization.default_max_fct_angles_opti_f2))

        # dihedrals equilibrium values
        if args.runtime.exec_mode == 1:
            for i in range(state.opti.opti_cycle["nb_geoms"]["dihedral"]):
                input_guess.append(min(max(state.opti.out_itp["dihedral"][i]["value_user"], state.opti.domains_val["dihedral"][i][0]),
                                       state.opti.domains_val["dihedral"][i][1]))

        # dihedrals force constants
        for i in range(state.opti.opti_cycle["nb_geoms"]["dihedral"]):
            if state.model.cg_itp["dihedral"][i]["func"] == 2:
                input_guess.append(min(max(state.opti.out_itp["dihedral"][i]["fct_user"],
                                           -args.optimization.default_abs_range_fct_dihedrals_opti_func_without_mult),
                                       args.optimization.default_abs_range_fct_dihedrals_opti_func_without_mult))
            else:
                input_guess.append(min(max(state.opti.out_itp["dihedral"][i]["fct_user"],
                                           -args.optimization.default_abs_range_fct_dihedrals_opti_func_with_mult),
                                       args.optimization.default_abs_range_fct_dihedrals_opti_func_with_mult))

        initial_guess_list.append(input_guess)

    # remaining particles are all random
    for i in range(num_particle_random_start, nb_particles):

        init_guess = []

        # constraints equilibrium values
        if args.runtime.exec_mode == 1:
            for j in range(state.opti.opti_cycle["nb_geoms"]["constraint"]):
                try:
                    emd_err_fact = max(1, state.opti.all_emd_dist_geoms["constraints"][j] / 2)
                except:
                    emd_err_fact = 1

                # initial variations range
                draw_low = max(state.opti.out_itp["constraint"][j][
                                   "value"] - config.bond_dist_guess_variation * state.opti.val_guess_fact * emd_err_fact,
                               state.opti.domains_val["constraint"][j][0])
                draw_high = min(state.opti.out_itp["constraint"][j][
                                    "value"] + config.bond_dist_guess_variation * state.opti.val_guess_fact * emd_err_fact,
                                state.opti.domains_val["constraint"][j][1])

                init_guess.append(draw_float(draw_low, draw_high, 3))
                # print("Particle", i+1, "-- BOND", j+1, "-- VAL RANGE", draw_low, draw_high)

        # bonds equilibrium values
        if args.runtime.exec_mode == 1:
            for j in range(state.opti.opti_cycle["nb_geoms"]["bond"]):
                try:
                    emd_err_fact = max(1, state.opti.all_emd_dist_geoms["bonds"][j] / 2)
                except:
                    emd_err_fact = 1

                # initial variations range
                draw_low = max(state.opti.out_itp["bond"][j][
                                   "value"] - config.bond_dist_guess_variation * state.opti.val_guess_fact * emd_err_fact,
                               state.opti.domains_val["bond"][j][0])
                draw_high = min(state.opti.out_itp["bond"][j][
                                    "value"] + config.bond_dist_guess_variation * state.opti.val_guess_fact * emd_err_fact,
                                state.opti.domains_val["bond"][j][1])

                init_guess.append(draw_float(draw_low, draw_high, 3))
                # print("Particle", i+1, "-- BOND", j+1, "-- VAL RANGE", draw_low, draw_high)

        # bonds force constants
        for j in range(state.opti.opti_cycle["nb_geoms"]["bond"]):
            try:
                emd_err_fact = max(1, state.opti.all_emd_dist_geoms["bonds"][j] / 2)
            except:
                emd_err_fact = 1

            # initial variations range
            draw_low = max(min(state.opti.out_itp["bond"][j]["fct"] * (1 - state.opti.fct_guess_fact * emd_err_fact),
                               state.opti.out_itp["bond"][j]["fct"] - config.fct_guess_min_flat_diff_bonds),
                           0)
            draw_high = min(max(state.opti.out_itp["bond"][j]["fct"] * (1 + state.opti.fct_guess_fact * emd_err_fact),
                                state.opti.out_itp["bond"][j]["fct"] + config.fct_guess_min_flat_diff_bonds),
                            args.optimization.default_max_fct_bonds_opti)

            init_guess.append(draw_float(draw_low, draw_high, 3))
        # print("Particle", i+1, "-- BOND", j+1, "-- FCT RANGE", draw_low, draw_high)

        # angles equilibrium values
        if args.runtime.exec_mode == 1:
            for j in range(state.opti.opti_cycle["nb_geoms"]["angle"]):
                try:
                    emd_err_fact = max(1, state.opti.all_emd_dist_geoms["angles"][j] / 2)
                except:
                    emd_err_fact = 1

                # initial variations range
                draw_low = max(state.opti.out_itp["angle"][j][
                                   "value"] - config.angle_value_guess_variation * state.opti.val_guess_fact * emd_err_fact,
                               state.opti.domains_val["angle"][j][0])
                draw_high = min(state.opti.out_itp["angle"][j][
                                    "value"] + config.angle_value_guess_variation * state.opti.val_guess_fact * emd_err_fact,
                                state.opti.domains_val["angle"][j][1])

                init_guess.append(draw_float(draw_low, draw_high, 3))

        # angles force constants
        for j in range(state.opti.opti_cycle["nb_geoms"]["angle"]):
            try:
                emd_err_fact = max(1, state.opti.all_emd_dist_geoms["angles"][j] / 2)
            except:
                emd_err_fact = 1

            # initial variations range
            draw_low = max(min(state.opti.out_itp["angle"][j]["fct"] * (1 - state.opti.fct_guess_fact * emd_err_fact),
                               state.opti.out_itp["angle"][j]["fct"] - config.fct_guess_min_flat_diff_angles),
                           0)
            if state.model.cg_itp["angle"][j]["func"] == 1:
                draw_high = min(max(state.opti.out_itp["angle"][j]["fct"] * (1 + state.opti.fct_guess_fact * emd_err_fact),
                                    state.opti.out_itp["angle"][j]["fct"] + config.fct_guess_min_flat_diff_angles),
                                args.optimization.default_max_fct_angles_opti_f1)
            elif state.model.cg_itp["angle"][j]["func"] == 2:
                draw_high = min(max(state.opti.out_itp["angle"][j]["fct"] * (1 + state.opti.fct_guess_fact * emd_err_fact),
                                    state.opti.out_itp["angle"][j]["fct"] + config.fct_guess_min_flat_diff_angles),
                                args.optimization.default_max_fct_angles_opti_f2)
            init_guess.append(draw_float(draw_low, draw_high, 3))
        # print("Particle", i+1, "-- ANGLE", j+1, "-- FCT RANGE", draw_low, draw_high)

        # dihedrals equilibrium values
        if args.runtime.exec_mode == 1:
            for j in range(state.opti.opti_cycle["nb_geoms"]["dihedral"]):
                try:
                    emd_err_fact = max(1, state.opti.all_emd_dist_geoms["dihedrals"][j] / 5)
                except:
                    emd_err_fact = 1

                # initial variations range
                draw_low = max(state.opti.out_itp["dihedral"][j][
                                   "value"] - config.dihedral_value_guess_variation * state.opti.val_guess_fact * emd_err_fact,
                               state.opti.domains_val["dihedral"][j][0])
                draw_high = min(state.opti.out_itp["dihedral"][j][
                                    "value"] + config.dihedral_value_guess_variation * state.opti.val_guess_fact * emd_err_fact,
                                state.opti.domains_val["dihedral"][j][1])

                init_guess.append(draw_float(draw_low, draw_high, 3))
        # print("Particle", i+1, "-- DIHEDRAL", j+1, "-- VAL RANGE", draw_low, draw_high)

        # dihedrals force constants
        for j in range(state.opti.opti_cycle["nb_geoms"]["dihedral"]):

            try:
                emd_err_fact = max(1, state.opti.all_emd_dist_geoms["dihedrals"][j] / 5)
            except:
                emd_err_fact = 1

            # here force constants can be negative, proceed accordingly
            if state.opti.out_itp["dihedral"][j]["fct"] > 0:  # if positive
                # initial variations range
                draw_low = state.opti.out_itp["dihedral"][j]["fct"] * (1 - state.opti.fct_guess_fact * emd_err_fact)
                draw_high = state.opti.out_itp["dihedral"][j]["fct"] * (1 + state.opti.fct_guess_fact * emd_err_fact)
            else:
                # initial variations range
                draw_low = state.opti.out_itp["dihedral"][j]["fct"] * (1 + state.opti.fct_guess_fact * emd_err_fact)
                draw_high = state.opti.out_itp["dihedral"][j]["fct"] * (1 - state.opti.fct_guess_fact * emd_err_fact)

            # make sure the minimal variation range is enforced + stay within defined boundaries
            if state.model.cg_itp["dihedral"][j]["func"] == 2:
                draw_low = max(min(draw_low, state.opti.out_itp["dihedral"][j][
                    "fct"] - config.fct_guess_min_flat_diff_dihedrals_without_mult),
                               args.optimization.default_abs_range_fct_dihedrals_opti_func_without_mult)
                draw_high = min(max(draw_high, state.opti.out_itp["dihedral"][j][
                    "fct"] + config.fct_guess_min_flat_diff_dihedrals_without_mult),
                                args.optimization.default_abs_range_fct_dihedrals_opti_func_without_mult)
            else:
                draw_low = max(min(draw_low, state.opti.out_itp["dihedral"][j][
                    "fct"] - config.fct_guess_min_flat_diff_dihedrals_with_mult),
                               -args.optimization.default_abs_range_fct_dihedrals_opti_func_with_mult)
                draw_high = min(max(draw_high, state.opti.out_itp["dihedral"][j][
                    "fct"] + config.fct_guess_min_flat_diff_dihedrals_with_mult),
                                args.optimization.default_abs_range_fct_dihedrals_opti_func_with_mult)
            init_guess.append(draw_float(draw_low, draw_high, 3))
        # print("Particle", i+1, "-- DIHEDRAL", j+1, "-- FCT RANGE", draw_low, draw_high)

        initial_guess_list.append(init_guess)  # register new particle, built during this loop

    return initial_guess_list


@catch_warnings(RuntimeWarning)  # ignore the warning "divide by 0 encountered in true_divide" while calculating sigma
def perform_BI(args: SwarmCGArgs, state: SwarmCGState):
    """Update ITP force constants with Boltzmann inversion for selected geoms at this
    given optimization step.

    args requires:
        optimization.default_max_fct_bonds_opti
        optimization.default_abs_range_fct_dihedrals_opti_func_with_mult
        optimization.default_abs_range_fct_dihedrals_opti_func_without_mult
        optimization.temp
        runtime.verbose

    state requires:
        opti.performed_init_BI (edited inplace)
        opti.out_itp (edited inplace)
        opti.opti_cycle
        opti.data_BI
    """
    # NOTE: currently all of these are just BI, not BI to completion using only required ADDITIONAL amount of energy, which might make a difference when we perform the BI after bonds+angles optimization cycles
    # TODO: refactorize BI in separate function to be used during both model_prep and at start of model_opti
    # TODO: other dihedrals functions
    # TODO: If the first opti run of BI fails, lower force constants by 10% and retry, again and again until it works, or tell the user something is very wrong after 20 tries with 50% of the force constants that all did NOT work

    if not state.opti.performed_init_BI["bond"] and state.opti.opti_cycle["nb_geoms"]["bond"] > 0:

        if args.runtime.verbose:
            print()
            print("Performing Direct Boltzmann Inversion (DBI) to estimate bonds force constants")

        for grp_bond in range(state.opti.opti_cycle["nb_geoms"]["bond"]):

            hists_geoms_bi, std_grp_bond, avg_grp_bond, bi_xrange = state.opti.data_BI["bond"][grp_bond]
            hist_geoms_modif = hists_geoms_bi ** 2 * (max(hists_geoms_bi) / max(hists_geoms_bi ** 2))

            nb_passes = 3
            alpha = 0.55
            for _ in range(nb_passes):
                hist_geoms_modif = math_utils.ewma(hist_geoms_modif, alpha, int(config.bi_nb_bins / 10))

            y = -config.kB * args.optimization.temp * np.log(hist_geoms_modif + 1)
            x = np.linspace(bi_xrange[0], bi_xrange[1], config.bi_nb_bins, endpoint=True)
            k = config.kB * args.optimization.temp / std_grp_bond / std_grp_bond * 100 / 2

            params_guess = [k, avg_grp_bond * 10, min(y)]  # multiply for amgstrom for BI

            # calculate derivative to use as sigma for fitting
            y_forward_shift = collections.deque(y)
            y_forward_shift.rotate(3)
            deriv = abs(y - y_forward_shift)
            deriv = collections.deque(deriv)
            deriv.rotate(-3)

            nb_passes = 5
            for _ in range(nb_passes):
                deriv = math_utils.sma(deriv, int(config.bi_nb_bins / 5))

            deriv *= np.sqrt(y / min(y))
            deriv = 1 / deriv
            sigma = np.where(y < max(y), deriv, np.inf)

            popt, pcov = curve_fit(gmx_bonds_func_1, x * 10, y, p0=params_guess, sigma=sigma, maxfev=99999,
                                   absolute_sigma=False)  # multiply for amgstrom for BI

            # here we just update the force constant, bond length is already set to the average of distribution
            state.opti.out_itp["bond"][grp_bond]["fct"] = popt[0] * 100

            # stay within limits in case user requires low force constants
            if not 0 <= state.opti.out_itp["bond"][grp_bond]["fct"] <= min(config.default_max_fct_bonds_bi,
                                                                         args.optimization.default_max_fct_bonds_opti):
                state.opti.out_itp["bond"][grp_bond]["fct"] = min(
                    config.default_max_fct_bonds_bi, args.optimization.default_max_fct_bonds_opti) / 2

            if args.runtime.verbose:
                print("  Bond group", grp_bond + 1, "estimated force constant:",
                      round(state.opti.out_itp["bond"][grp_bond]["fct"], 2))

        state.opti.performed_init_BI["bond"] = True

    if not state.opti.performed_init_BI["angle"] and state.opti.opti_cycle["nb_geoms"]["angle"] > 0:

        if args.runtime.verbose:
            print()
            print("Performing Direct Boltzmann Inversion (DBI) to estimate angles force constants")

        for grp_angle in range(state.opti.opti_cycle["nb_geoms"]["angle"]):

            hists_geoms_bi, std_grp_angle, avg_grp_angle, bi_xrange = state.opti.data_BI["angle"][grp_angle]
            hist_geoms_modif = hists_geoms_bi ** 2 * (max(hists_geoms_bi) / max(hists_geoms_bi ** 2))

            nb_passes = 3
            alpha = 0.55
            for _ in range(nb_passes):
                hist_geoms_modif = math_utils.ewma(hist_geoms_modif, alpha, int(config.bi_nb_bins / 10))

            y = -config.kB * args.optimization.temp * np.log(hist_geoms_modif + 1)
            x = np.linspace(np.deg2rad(bi_xrange[0]), np.deg2rad(bi_xrange[1]), 2 * config.bi_nb_bins, endpoint=True)
            k = config.kB * args.optimization.temp / std_grp_angle / std_grp_angle * 100 / 2

            params_guess = [k, np.deg2rad(avg_grp_angle), min(y)]

            # use sigma to make the fit more accurate in the trough of the plot (it makes a huge difference)
            sigma = np.where(y < max(y), 0.1, np.inf)
            sigma = 1 / sigma

            if state.model.cg_itp["angle"][grp_angle]["func"] == 1:
                popt, pcov = curve_fit(gmx_angles_func_1, x, y, p0=params_guess, sigma=sigma, maxfev=99999,
                                       absolute_sigma=False)
            elif state.model.cg_itp["angle"][grp_angle]["func"] == 2:
                popt, pcov = curve_fit(gmx_angles_func_2, x, y, p0=params_guess, sigma=sigma, maxfev=99999,
                                       absolute_sigma=False)

            if args.runtime.exec_mode == 1:  # in Mode 1, use the fitted value as equilibrium value (but stay within range)
                # state.opti.out_itp["angle"][grp_angle]["value"] = max(min(np.rad2deg(popt[1]), state.opti.domains_val["angle"][grp_angle][1]), state.opti.domains_val["angle"][grp_angle][0])
                state.opti.out_itp["angle"][grp_angle]["value"] = np.rad2deg(
                    popt[1])  # we will apply limits of equilibrium values later

            if state.model.cg_itp["angle"][grp_angle]["func"] == 1:
                print("  Angle group", grp_angle + 1, "estimated force constant BEFORE MODIFIER:", round(popt[0], 2))
                state.opti.out_itp["angle"][grp_angle]["fct"] = popt[0] * 100
            elif state.model.cg_itp["angle"][grp_angle]["func"] == 2:
                print("  Angle group", grp_angle + 1, "estimated force constant BEFORE MODIFIER:", round(popt[0], 2))
                state.opti.out_itp["angle"][grp_angle]["fct"] = popt[0] * 0.5

            # stay within limits in case user requires low force constants
            if state.model.cg_itp["angle"][grp_angle]["func"] == 1:
                if not 0 <= state.opti.out_itp["angle"][grp_angle]["fct"] <= min(
                        config.default_max_fct_angles_bi, args.optimization.default_max_fct_angles_opti_f1):
                    state.opti.out_itp["angle"][grp_angle]["fct"] = min(
                        config.default_max_fct_angles_bi, args.optimization.default_max_fct_angles_opti_f1) / 2
            elif state.model.cg_itp["angle"][grp_angle]["func"] == 2:
                if not 0 <= state.opti.out_itp["angle"][grp_angle]["fct"] <= min(
                        config.default_max_fct_angles_bi, args.optimization.default_max_fct_angles_opti_f2):
                    state.opti.out_itp["angle"][grp_angle]["fct"] = min(
                        config.default_max_fct_angles_bi, args.optimization.default_max_fct_angles_opti_f2) / 2

            if args.runtime.verbose:
                print("  Angle group", grp_angle + 1, "estimated force constant:",
                      round(state.opti.out_itp["angle"][grp_angle]["fct"], 2))

        state.opti.performed_init_BI["angle"] = True

    if not state.opti.performed_init_BI["dihedral"] and state.opti.opti_cycle["nb_geoms"]["dihedral"] > 0:

        if args.runtime.verbose:
            print()
            print("Performing Direct Boltzmann Inversion (DBI) to estimate dihedrals force constants")

        for grp_dihedral in range(state.opti.opti_cycle["nb_geoms"]["dihedral"]):

            hists_geoms_bi, std_grp_dihedral, avg_grp_dihedral, bi_xrange = state.opti.data_BI["dihedral"][grp_dihedral]
            hist_geoms_modif = hists_geoms_bi ** 2 * (max(hists_geoms_bi) / max(hists_geoms_bi ** 2))

            nb_passes = 3
            alpha = 0.55
            for _ in range(nb_passes):
                hist_geoms_modif = math_utils.ewma(hist_geoms_modif, alpha, int(config.bi_nb_bins / 10))

            y = -config.kB * args.optimization.temp * np.log(hists_geoms_bi + 1)
            x = np.linspace(np.deg2rad(bi_xrange[0]), np.deg2rad(bi_xrange[1]), 2 * config.bi_nb_bins, endpoint=True)
            k = config.kB * args.optimization.temp / std_grp_dihedral / std_grp_dihedral * 100 / 2

            # calculate sigma
            sigma = np.where(y < max(y), 0.1, np.inf)
            sigma = 1 / sigma

            if state.model.cg_itp["dihedral"][grp_dihedral]["func"] in config.dihedral_func_with_mult:
                params_guess = [k, np.deg2rad(avg_grp_dihedral), min(y)]
                if args.inputs.user_input:
                    avg_rad_grp_dihedral = np.deg2rad(state.model.cg_itp["dihedral"][grp_dihedral]["value_user"])
                else:
                    avg_rad_grp_dihedral = params_guess[1]
                multiplicity = state.model.cg_itp["dihedral"][grp_dihedral][
                    "func_mult"]
                popt, pcov = curve_fit(gmx_dihedrals_func_1(mult=multiplicity), x, y, p0=params_guess, sigma=sigma,
                                       maxfev=99999, absolute_sigma=False)

            elif state.model.cg_itp["dihedral"][grp_dihedral]["func"] == 2:
                params_guess = [k, avg_rad_grp_dihedral, min(y)]
                popt, pcov = curve_fit(gmx_dihedrals_func_2, x, y, p0=params_guess, sigma=sigma, maxfev=99999,
                                       absolute_sigma=False)

            if args.runtime.exec_mode == 1:  # in Mode 1, use the fitted value as equilibrium value (but stay within range)
                # state.opti.out_itp["dihedral"][grp_dihedral]["value"] = max(min(np.rad2deg(popt[1]), state.opti.domains_val["dihedral"][grp_dihedral][1]), state.opti.domains_val["dihedral"][grp_dihedral][0])
                state.opti.out_itp["dihedral"][grp_dihedral]["value"] = np.rad2deg(
                    popt[1])  # we will apply limits of equilibrium values later

            print("  Dihedral group", grp_dihedral + 1, "estimated force constant BEFORE MODIFIER:", round(popt[0], 2))
            state.opti.out_itp["dihedral"][grp_dihedral]["fct"] = popt[0]

            # stay within limits in case user requires low force constants
            if state.model.cg_itp["dihedral"][grp_dihedral]["func"] in config.dihedral_func_with_mult:
                if not max(-config.default_abs_range_fct_dihedrals_bi_func_with_mult,
                           -args.optimization.default_abs_range_fct_dihedrals_opti_func_with_mult) <= \
                       state.opti.out_itp["dihedral"][grp_dihedral]["fct"] <= min(
                        -config.default_abs_range_fct_dihedrals_bi_func_with_mult,
                        -args.optimization.default_abs_range_fct_dihedrals_opti_func_with_mult):
                    state.opti.out_itp["dihedral"][grp_dihedral]["fct"] = np.sign(
                        state.opti.out_itp["dihedral"][grp_dihedral]["fct"]) * min(
                        config.default_abs_range_fct_dihedrals_bi_func_with_mult,
                        args.optimization.default_abs_range_fct_dihedrals_opti_func_with_mult) / 2
            else:
                if not max(-config.default_abs_range_fct_dihedrals_bi_func_without_mult,
                           -args.optimization.default_abs_range_fct_dihedrals_opti_func_without_mult) <= \
                       state.opti.out_itp["dihedral"][grp_dihedral]["fct"] <= min(
                        -config.default_abs_range_fct_dihedrals_bi_func_without_mult,
                        -args.optimization.default_abs_range_fct_dihedrals_opti_func_without_mult):
                    state.opti.out_itp["dihedral"][grp_dihedral]["fct"] = np.sign(
                        state.opti.out_itp["dihedral"][grp_dihedral]["fct"]) * min(
                        config.default_abs_range_fct_dihedrals_bi_func_without_mult,
                        args.optimization.default_abs_range_fct_dihedrals_opti_func_without_mult) / 2

            if args.runtime.verbose:
                print("  Dihedral group", grp_dihedral + 1, "estimated force constant:",
                      round(state.opti.out_itp["dihedral"][grp_dihedral]["fct"], 2))

        state.opti.performed_init_BI["dihedral"] = True


def process_scaling_str(args: SwarmCGArgs, state: SwarmCGState):
    """Process specific bonds scaling string, if provided.

    args requires:
        optimization.bonds_scaling_str

    state creates:
        mapping.bonds_scaling_specific
    """
    state.mapping.bonds_scaling_specific = None
    if args.optimization.bonds_scaling_str != config.bonds_scaling_str:
        sp_str = args.optimization.bonds_scaling_str.split()
        if len(sp_str) % 2 != 0:
            raise exceptions.InvalidArgument("bonds_scaling_str", args.optimization.bonds_scaling_str)

        state.mapping.bonds_scaling_specific = dict()
        i = 0
        try:
            while i < len(sp_str):
                geom_id = sp_str[i][1:]
                if sp_str[i][0].upper() == "C":
                    if int(geom_id) > state.model.cg_itp["nb_constraints"]:
                        info = "A constraint group id exceeds the number of constraints groups defined in the input CG ITP file."
                        raise exceptions.InvalidArgument("bonds_scaling_str", args.optimization.bonds_scaling_str, info)
                    if not "C" + geom_id in state.mapping.bonds_scaling_specific:
                        if float(sp_str[i + 1]) < 0:
                            info = "You cannot provide negative values for average distribution length."
                            raise exceptions.InvalidArgument("bonds_scaling_str", args.optimization.bonds_scaling_str, info)
                        state.mapping.bonds_scaling_specific["C" + geom_id] = float(sp_str[i + 1])
                    else:
                        info = f"A constraint group id is provided multiple times (id: {geom_id})"
                        raise exceptions.InvalidArgument("bonds_scaling_str", args.optimization.bonds_scaling_str, info)
                elif sp_str[i][0].upper() == "B":
                    if int(geom_id) > state.model.cg_itp["nb_bonds"]:
                        info = "A bond group id exceeds the number of bonds groups defined in the input CG ITP file."
                        raise exceptions.InvalidArgument("bonds_scaling_str", args.optimization.bonds_scaling_str, info)
                    if not "B" + geom_id in state.mapping.bonds_scaling_specific:
                        if float(sp_str[i + 1]) < 0:
                            info = "You cannot provide negative values for average distribution length."
                            raise exceptions.InvalidArgument("bonds_scaling_str", args.optimization.bonds_scaling_str, info)
                        state.mapping.bonds_scaling_specific["B" + geom_id] = float(sp_str[i + 1])
                    else:
                        info = f"A bond group id is provided multiple times (id: {geom_id})"
                        raise exceptions.InvalidArgument("bonds_scaling_str", args.optimization.bonds_scaling_str, info)
                i += 2
        except ValueError:
            raise exceptions.InvalidArgument("bonds_scaling_str", args.optimization.bonds_scaling_str)
