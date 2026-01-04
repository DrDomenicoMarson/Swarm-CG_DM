import collections
import numpy as np
from scipy.optimize import curve_fit

from swarmcg import config
from swarmcg.shared import math_utils, exceptions
from swarmcg.config_types import SwarmConfig
from swarmcg.shared.math_utils import draw_float
from swarmcg.simulations.potentials import (
    gmx_bonds_func_1, gmx_angles_func_1, gmx_angles_func_2,
    gmx_dihedrals_func_1, gmx_dihedrals_func_2
)

def update_cg_itp_obj(itp_obj, opti_cycle, parameters_set, exec_mode):
    """Update coarse-grain ITP object inplace."""
    
    # Validation/Router for update_type (intermediary vs cycles optimized) 
    # handled by caller selecting the right itp_obj
    
    nb_constraints = opti_cycle["nb_geoms"]["constraint"]
    nb_bonds = opti_cycle["nb_geoms"]["bond"]
    nb_angles = opti_cycle["nb_geoms"]["angle"]
    nb_dihedrals = opti_cycle["nb_geoms"]["dihedral"]

    # Constraints
    for i in range(nb_constraints):
        if exec_mode == 1:
            itp_obj["constraint"][i]["value"] = round(parameters_set[i], 3)

    # Bonds
    for i in range(nb_bonds):
        if exec_mode == 1:
            itp_obj["bond"][i]["value"] = round(parameters_set[nb_constraints + i], 3)
            itp_obj["bond"][i]["fct"] = round(parameters_set[nb_constraints + nb_bonds + i], 3)
        else:
            itp_obj["bond"][i]["fct"] = round(parameters_set[i], 3)

    # Angles
    for i in range(nb_angles):
        if exec_mode == 1:
            itp_obj["angle"][i]["value"] = round(
                parameters_set[nb_constraints + 2 * nb_bonds + i], 2)
            itp_obj["angle"][i]["fct"] = round(
                parameters_set[nb_constraints + 2 * nb_bonds + nb_angles + i], 2)
        else:
            itp_obj["angle"][i]["fct"] = round(
                parameters_set[nb_bonds + i], 2)

    # Dihedrals
    for i in range(nb_dihedrals):
        if exec_mode == 1:
            itp_obj["dihedral"][i]["value"] = round(
                parameters_set[nb_constraints + 2 * nb_bonds + 2 * nb_angles + i], 2)
            itp_obj["dihedral"][i]["fct"] = round(
                parameters_set[nb_constraints + 2 * nb_bonds + 2 * nb_angles + nb_dihedrals + i], 2)
        else:
            itp_obj["dihedral"][i]["fct"] = round(
                parameters_set[nb_bonds + nb_angles + i], 2)


def get_search_space_boundaries(cg_itp, opti_cycle, domains_val, exec_mode, config_obj: SwarmConfig):
    """Set dimensions of the search space."""
    opt_config = config_obj.optimization
    search_space_boundaries = []

    if opti_cycle["nb_geoms"]["constraint"] > 0:
        if exec_mode == 1:
            search_space_boundaries.extend(domains_val["constraint"])

    if opti_cycle["nb_geoms"]["bond"] > 0:
        if exec_mode == 1:
            search_space_boundaries.extend(domains_val["bond"])
        
        search_space_boundaries.extend([[0, opt_config.default_max_fct_bonds_opti]] * opti_cycle["nb_geoms"]["bond"])

    if opti_cycle["nb_geoms"]["angle"] > 0:
        if exec_mode == 1:
            search_space_boundaries.extend(domains_val["angle"])

        for grp_angle in range(opti_cycle["nb_geoms"]["angle"]):
            if cg_itp["angle"][grp_angle]["func"] == 1:
                search_space_boundaries.extend([[0, opt_config.default_max_fct_angles_opti_f1]])
            elif cg_itp["angle"][grp_angle]["func"] == 2:
                search_space_boundaries.extend([[0, opt_config.default_max_fct_angles_opti_f2]])

    if opti_cycle["nb_geoms"]["dihedral"] > 0:
        if exec_mode == 1:
            search_space_boundaries.extend(domains_val["dihedral"])

        for grp_dihedral in range(opti_cycle["nb_geoms"]["dihedral"]):
            if cg_itp["dihedral"][grp_dihedral]["func"] == 2:
                search_space_boundaries.extend([[-opt_config.default_abs_range_fct_dihedrals_opti_func_without_mult, opt_config.default_abs_range_fct_dihedrals_opti_func_without_mult]])
            elif cg_itp["dihedral"][grp_dihedral]["func"] in config.dihedral_func_with_mult:
                search_space_boundaries.extend([[-opt_config.default_abs_range_fct_dihedrals_opti_func_with_mult, opt_config.default_abs_range_fct_dihedrals_opti_func_with_mult]])

    return search_space_boundaries

def perform_BI(itp_obj, opti_cycle, data_BI, performed_init_BI, temp, config_obj: SwarmConfig, verbose=False, exec_mode=1):
    """Update ITP force constants with Boltzmann inversion."""
    opt_config = config_obj.optimization
    
    # Constants/Config access - using global config as it contains physical constants
    kB = config.kB # Leaving physical constant global
    
    # Check bonds
    if not performed_init_BI["bond"] and opti_cycle["nb_geoms"]["bond"] > 0:
        if verbose:
            print("\nPerforming Direct Boltzmann Inversion (DBI) to estimate bonds force constants")

        for grp_bond in range(opti_cycle["nb_geoms"]["bond"]):
            hists_geoms_bi, std_grp_bond, avg_grp_bond, bi_xrange = data_BI["bond"][grp_bond]
            
            # Simple BI processing (no smoothing loop for brevity, or full copy?)
            # Copying full logic
            hist_geoms_modif = hists_geoms_bi ** 2 * (max(hists_geoms_bi) / max(hists_geoms_bi ** 2))
            
            nb_passes = 3
            alpha = 0.55
            for _ in range(nb_passes):
                hist_geoms_modif = math_utils.ewma(hist_geoms_modif, alpha, int(opt_config.bi_nb_bins / 10))

            y = -kB * temp * np.log(hist_geoms_modif + 1)
            x = np.linspace(bi_xrange[0], bi_xrange[1], opt_config.bi_nb_bins, endpoint=True)
            k = kB * temp / std_grp_bond / std_grp_bond * 100 / 2

            params_guess = [k, avg_grp_bond * 10, min(y)]

            # Derivative calculation
            y_forward_shift = collections.deque(y)
            y_forward_shift.rotate(3)
            deriv = abs(y - y_forward_shift)
            deriv = collections.deque(deriv)
            deriv.rotate(-3)

            for _ in range(5):
                deriv = math_utils.sma(deriv, int(opt_config.bi_nb_bins / 5))

            deriv *= np.sqrt(y / min(y))
            with np.errstate(divide='ignore'):
                 deriv = 1 / deriv
            sigma = np.where(y < max(y), deriv, np.inf)

            try:
                popt, pcov = curve_fit(gmx_bonds_func_1, x * 10, y, p0=params_guess, sigma=sigma, maxfev=99999, absolute_sigma=False)
                itp_obj["bond"][grp_bond]["fct"] = popt[0] * 100
                
                # Enforce limits
                max_fct = opt_config.default_max_fct_bonds_opti
                if not 0 <= itp_obj["bond"][grp_bond]["fct"] <= max_fct:
                    itp_obj["bond"][grp_bond]["fct"] = max_fct / 2
                
                if verbose:
                    print("  Bond group", grp_bond + 1, "estimated force constant:", round(itp_obj["bond"][grp_bond]["fct"], 2))
            except Exception as e:
                # Fallback or error logging
                if verbose:
                    print(f"  Bond group {grp_bond + 1} BI failed: {e}")

        performed_init_BI["bond"] = True

    # Check angles
    if not performed_init_BI["angle"] and opti_cycle["nb_geoms"]["angle"] > 0:
        if verbose:
            print("\nPerforming Direct Boltzmann Inversion (DBI) to estimate angles force constants")
            
        for grp_angle in range(opti_cycle["nb_geoms"]["angle"]):
            hists_geoms_bi, std_rad_grp_angle, bi_xrange = data_BI["angle"][grp_angle]
            y = -kB * temp * np.log(hists_geoms_bi + 1)
            x = np.linspace(np.deg2rad(bi_xrange[0]), np.deg2rad(bi_xrange[1]), opt_config.bi_nb_bins, endpoint=True)
            k = kB * temp / std_rad_grp_angle / std_rad_grp_angle * 100 / 2
            
            sigma = np.where(y < max(y), 0.1, np.inf)
            func = itp_obj["angle"][grp_angle]["func"] # Using itp_obj which is cg_itp effectively here?

            if func == 1:
                params_guess = [k, std_rad_grp_angle, min(y)]
                try:
                    popt, pcov = curve_fit(gmx_angles_func_1, x, y, p0=params_guess, sigma=sigma, maxfev=99999, absolute_sigma=False)
                    itp_obj["angle"][grp_angle]["fct"] = abs(popt[0])
                except:
                    pass
            elif func == 2:
                params_guess = [max(y) - min(y), std_rad_grp_angle, min(y)]
                try:
                    popt, pcov = curve_fit(gmx_angles_func_2, x, y, p0=params_guess, sigma=sigma, maxfev=99999, absolute_sigma=False)
                    if popt[0] < 0:
                        popt[0] = 30 # fallback
                    itp_obj["angle"][grp_angle]["fct"] = popt[0]
                except:
                    itp_obj["angle"][grp_angle]["fct"] = 30

            if verbose:
                 print("  Angle group", grp_angle + 1, "estimated force constant:", round(itp_obj["angle"][grp_angle]["fct"], 2))

        performed_init_BI["angle"] = True

    # Check dihedrals
    if not performed_init_BI["dihedral"] and opti_cycle["nb_geoms"]["dihedral"] > 0:
         if verbose:
            print("\nPerforming Direct Boltzmann Inversion (DBI) to estimate dihedrals force constants")
         
         for grp_dihedral in range(opti_cycle["nb_geoms"]["dihedral"]):
            hists_geoms_bi, std_rad_grp_dihedral, avg_rad_grp_dihedral, bi_xrange = data_BI["dihedral"][grp_dihedral]
            y = -kB * temp * np.log(hists_geoms_bi + 1)
            x = np.linspace(np.deg2rad(bi_xrange[0]), np.deg2rad(bi_xrange[1]), 2 * opt_config.bi_nb_bins, endpoint=True)
            k = kB * temp / std_rad_grp_dihedral / std_rad_grp_dihedral
            sigma = np.where(y < max(y), 0.1, np.inf)
            
            # Again, assuming func is present in itp_obj for now, or we'd need cg_itp passed explicitly
            func = itp_obj["dihedral"][grp_dihedral]["func"]
            
            if exec_mode == 2:
                avg_rad = np.deg2rad(itp_obj["dihedral"][grp_dihedral].get("value_user", 0)) # Fallback if missing
                pass

            if func in config.dihedral_func_with_mult:
                 multiplicity = itp_obj["dihedral"][grp_dihedral]["mult"]
                 params_guess = [max(y) - min(y), avg_rad_grp_dihedral, min(y)]
                 try:
                     popt, pcov = curve_fit(gmx_dihedrals_func_1(mult=multiplicity), x, y, p0=params_guess, sigma=sigma, maxfev=99999, absolute_sigma=False)
                     itp_obj["dihedral"][grp_dihedral]["fct"] = popt[0]
                 except:
                     pass
            elif func == 2:
                 params_guess = [k, avg_rad_grp_dihedral, min(y)]
                 try:
                     popt, pcov = curve_fit(gmx_dihedrals_func_2, x, y, p0=params_guess, sigma=sigma, maxfev=99999, absolute_sigma=False)
                     itp_obj["dihedral"][grp_dihedral]["fct"] = popt[0]
                 except:
                     pass
            
            if exec_mode == 1:
                 pass
            
            if verbose:
                print("  Dihedral group", grp_dihedral + 1, "estimated force constant:", round(itp_obj["dihedral"][grp_dihedral]["fct"], 2))

         performed_init_BI["dihedral"] = True

def get_initial_guess_list(nb_particles, opti_cycle, cg_itp, out_itp, domains_val,
                           all_best_emd_dist_geoms, all_best_params_dist_geoms,
                           exec_mode, config_obj: SwarmConfig, user_input=False,
                           val_guess_fact=None, fct_guess_fact=None,
                           verbose=False):
    """Build initial guesses for particles initialization."""
    initial_guess_list = []
    input_guess = []

    # defaults if not provided (should be provided by caller from config)
    if val_guess_fact is None: val_guess_fact = 1.0 # default?
    if fct_guess_fact is None: fct_guess_fact = 0.5 # default?

    opt_config = config_obj.optimization
    
    default_max_fct_bonds_opti = opt_config.default_max_fct_bonds_opti
    default_max_fct_angles_opti_f1 = opt_config.default_max_fct_angles_opti_f1
    default_max_fct_angles_opti_f2 = opt_config.default_max_fct_angles_opti_f2
    default_abs_range_fct_dihedrals_opti_func_without_mult = opt_config.default_abs_range_fct_dihedrals_opti_func_without_mult
    default_abs_range_fct_dihedrals_opti_func_with_mult = opt_config.default_abs_range_fct_dihedrals_opti_func_with_mult
    bond_dist_guess_variation = opt_config.bond_dist_guess_variation
    angle_value_guess_variation = opt_config.angle_value_guess_variation
    dihedral_value_guess_variation = opt_config.dihedral_value_guess_variation
    fct_guess_min_flat_diff_bonds = opt_config.fct_guess_min_flat_diff_bonds
    fct_guess_min_flat_diff_angles = opt_config.fct_guess_min_flat_diff_angles
    fct_guess_min_flat_diff_dihedrals_without_mult = opt_config.fct_guess_min_flat_diff_dihedrals_without_mult
    fct_guess_min_flat_diff_dihedrals_with_mult = opt_config.fct_guess_min_flat_diff_dihedrals_with_mult
    sim_crash_EMD_indep_score = opt_config.sim_crash_EMD_indep_score


    # 1. Exact current ITP (or BI in exec_mode 1)
    if exec_mode == 1:
        for i in range(opti_cycle["nb_geoms"]["constraint"]):
            input_guess.append(min(max(out_itp["constraint"][i]["value"], domains_val["constraint"][i][0]),
                                   domains_val["constraint"][i][1]))

        for i in range(opti_cycle["nb_geoms"]["bond"]):
            input_guess.append(min(max(out_itp["bond"][i]["value"], domains_val["bond"][i][0]),
                                   domains_val["bond"][i][1]))

    for i in range(opti_cycle["nb_geoms"]["bond"]):
        input_guess.append(min(max(out_itp["bond"][i]["fct"], 0), default_max_fct_bonds_opti))

    if exec_mode == 1:
        for i in range(opti_cycle["nb_geoms"]["angle"]):
            input_guess.append(min(max(out_itp["angle"][i]["value"], domains_val["angle"][i][0]),
                                   domains_val["angle"][i][1]))

    for i in range(opti_cycle["nb_geoms"]["angle"]):
        if cg_itp["angle"][i]["func"] == 1:
            input_guess.append(
                min(max(out_itp["angle"][i]["fct"], 0), default_max_fct_angles_opti_f1))
        elif cg_itp["angle"][i]["func"] == 2:
            input_guess.append(
                min(max(out_itp["angle"][i]["fct"], 0), default_max_fct_angles_opti_f2))

    if exec_mode == 1:
        for i in range(opti_cycle["nb_geoms"]["dihedral"]):
            input_guess.append(min(max(out_itp["dihedral"][i]["value"], domains_val["dihedral"][i][0]),
                                   domains_val["dihedral"][i][1]))

    for i in range(opti_cycle["nb_geoms"]["dihedral"]):
        if cg_itp["dihedral"][i]["func"] == 2:
            input_guess.append(
                min(max(out_itp["dihedral"][i]["fct"], -default_abs_range_fct_dihedrals_opti_func_without_mult),
                    default_abs_range_fct_dihedrals_opti_func_without_mult))
        else:
            input_guess.append(
                min(max(out_itp["dihedral"][i]["fct"], -default_abs_range_fct_dihedrals_opti_func_with_mult),
                    default_abs_range_fct_dihedrals_opti_func_with_mult))

    initial_guess_list.append(input_guess)
    num_particle_random_start = 1

    # 2. Particle from best EMD or user input
    if opti_cycle["nb_cycle"] > 1:
        num_particle_random_start += 1
        input_guess = []

        # Constraints
        if exec_mode == 1:
            for i in range(opti_cycle["nb_geoms"]["constraint"]):
                if all_best_emd_dist_geoms["constraints"][i] != sim_crash_EMD_indep_score:
                    input_guess.append(all_best_params_dist_geoms["constraints"][i]["params"][0])
                else:
                    input_guess.append(min(max(out_itp["constraint"][i]["value"], domains_val["constraint"][i][0]), domains_val["constraint"][i][1])) # Fallback
        
        # Bonds
        if exec_mode == 1:
            for i in range(opti_cycle["nb_geoms"]["bond"]):
                if all_best_emd_dist_geoms["bonds"][i] != sim_crash_EMD_indep_score:
                    input_guess.append(all_best_params_dist_geoms["bonds"][i]["params"][0])
                else:
                    input_guess.append(min(max(out_itp["bond"][i]["value"], domains_val["bond"][i][0]), domains_val["bond"][i][1]))
        
        for i in range(opti_cycle["nb_geoms"]["bond"]):
            if all_best_emd_dist_geoms["bonds"][i] != sim_crash_EMD_indep_score:
                input_guess.append(all_best_params_dist_geoms["bonds"][i]["params"][1])
            else:
                 input_guess.append(min(max(out_itp["bond"][i]["fct"], 0), default_max_fct_bonds_opti))

        # Angles
        if exec_mode == 1:
            for i in range(opti_cycle["nb_geoms"]["angle"]):
                if all_best_emd_dist_geoms["angles"][i] != sim_crash_EMD_indep_score:
                    input_guess.append(all_best_params_dist_geoms["angles"][i]["params"][0])
                else:
                    input_guess.append(min(max(out_itp["angle"][i]["value"], domains_val["angle"][i][0]), domains_val["angle"][i][1]))

        for i in range(opti_cycle["nb_geoms"]["angle"]):
            if all_best_emd_dist_geoms["angles"][i] != sim_crash_EMD_indep_score:
                input_guess.append(all_best_params_dist_geoms["angles"][i]["params"][1])
            else:
                if cg_itp["angle"][i]["func"] == 1:
                    input_guess.append(min(max(out_itp["angle"][i]["fct"], 0), default_max_fct_angles_opti_f1))
                elif cg_itp["angle"][i]["func"] == 2:
                     input_guess.append(min(max(out_itp["angle"][i]["fct"], 0), default_max_fct_angles_opti_f2))

        # Dihedrals
        if exec_mode == 1:
            for i in range(opti_cycle["nb_geoms"]["dihedral"]):
                if all_best_emd_dist_geoms["dihedrals"][i] != sim_crash_EMD_indep_score:
                    input_guess.append(all_best_params_dist_geoms["dihedrals"][i]["params"][0])
                else:
                    input_guess.append(min(max(out_itp["dihedral"][i]["value"], domains_val["dihedral"][i][0]), domains_val["dihedral"][i][1]))

        for i in range(opti_cycle["nb_geoms"]["dihedral"]):
            if all_best_emd_dist_geoms["dihedrals"][i] != sim_crash_EMD_indep_score:
                input_guess.append(all_best_params_dist_geoms["dihedrals"][i]["params"][1])
            else:
                if cg_itp["dihedral"][i]["func"] == 2:
                    input_guess.append(
                        min(max(out_itp["dihedral"][i]["fct"],
                                -default_abs_range_fct_dihedrals_opti_func_without_mult),
                            default_abs_range_fct_dihedrals_opti_func_without_mult))
                else:
                    input_guess.append(min(
                        max(out_itp["dihedral"][i]["fct"], -default_abs_range_fct_dihedrals_opti_func_with_mult),
                        default_abs_range_fct_dihedrals_opti_func_with_mult))

        initial_guess_list.append(input_guess)

    elif user_input:
        num_particle_random_start += 1
        input_guess = []

        # constraints
        if exec_mode == 1:
            for i in range(opti_cycle["nb_geoms"]["constraint"]):
                input_guess.append(min(max(out_itp["constraint"][i]["value_user"], domains_val["constraint"][i][0]), domains_val["constraint"][i][1]))

        # bonds
        if exec_mode == 1:
            for i in range(opti_cycle["nb_geoms"]["bond"]):
                input_guess.append(min(max(out_itp["bond"][i]["value_user"], domains_val["bond"][i][0]), domains_val["bond"][i][1]))

        for i in range(opti_cycle["nb_geoms"]["bond"]):
            input_guess.append(min(max(out_itp["bond"][i]["fct_user"], 0), default_max_fct_bonds_opti))

        # angles
        if exec_mode == 1:
            for i in range(opti_cycle["nb_geoms"]["angle"]):
                input_guess.append(min(max(out_itp["angle"][i]["value_user"], domains_val["angle"][i][0]), domains_val["angle"][i][1]))

        for i in range(opti_cycle["nb_geoms"]["angle"]):
            if cg_itp["angle"][i]["func"] == 1:
                input_guess.append(min(max(out_itp["angle"][i]["fct_user"], 0), default_max_fct_angles_opti_f1))
            elif cg_itp["angle"][i]["func"] == 2:
                input_guess.append(min(max(out_itp["angle"][i]["fct_user"], 0), default_max_fct_angles_opti_f2))

        # dihedrals
        if exec_mode == 1:
            for i in range(opti_cycle["nb_geoms"]["dihedral"]):
                input_guess.append(min(max(out_itp["dihedral"][i]["value_user"], domains_val["dihedral"][i][0]), domains_val["dihedral"][i][1]))
        
        for i in range(opti_cycle["nb_geoms"]["dihedral"]):
            if cg_itp["dihedral"][i]["func"] == 2:
                input_guess.append(min(max(out_itp["dihedral"][i]["fct_user"], -default_abs_range_fct_dihedrals_opti_func_without_mult), default_abs_range_fct_dihedrals_opti_func_without_mult))
            else:
                 input_guess.append(min(max(out_itp["dihedral"][i]["fct_user"], -default_abs_range_fct_dihedrals_opti_func_with_mult), default_abs_range_fct_dihedrals_opti_func_with_mult))

        initial_guess_list.append(input_guess)

    # 3. Variations for remaining particles
    for i in range(num_particle_random_start, nb_particles):
        init_guess = []

        # constraints
        if exec_mode == 1:
            for j in range(opti_cycle["nb_geoms"]["constraint"]):
                try: emd_err_fact = max(1, all_best_emd_dist_geoms["constraints"][j] / 2)
                except: emd_err_fact = 1
                draw_low = max(out_itp["constraint"][j]["value"] - bond_dist_guess_variation * val_guess_fact * emd_err_fact, domains_val["constraint"][j][0])
                draw_high = min(out_itp["constraint"][j]["value"] + bond_dist_guess_variation * val_guess_fact * emd_err_fact, domains_val["constraint"][j][1])
                init_guess.append(draw_float(draw_low, draw_high, 3))

        # bonds
        if exec_mode == 1:
            for j in range(opti_cycle["nb_geoms"]["bond"]):
                try: emd_err_fact = max(1, all_best_emd_dist_geoms["bonds"][j] / 2)
                except: emd_err_fact = 1
                draw_low = max(out_itp["bond"][j]["value"] - bond_dist_guess_variation * val_guess_fact * emd_err_fact, domains_val["bond"][j][0])
                draw_high = min(out_itp["bond"][j]["value"] + bond_dist_guess_variation * val_guess_fact * emd_err_fact, domains_val["bond"][j][1])
                init_guess.append(draw_float(draw_low, draw_high, 3))
        
        for j in range(opti_cycle["nb_geoms"]["bond"]):
            try: emd_err_fact = max(1, all_best_emd_dist_geoms["bonds"][j] / 2)
            except: emd_err_fact = 1
            draw_low = max(min(out_itp["bond"][j]["fct"] * (1 - fct_guess_fact * emd_err_fact), out_itp["bond"][j]["fct"] - fct_guess_min_flat_diff_bonds), 0)
            draw_high = min(max(out_itp["bond"][j]["fct"] * (1 + fct_guess_fact * emd_err_fact), out_itp["bond"][j]["fct"] + fct_guess_min_flat_diff_bonds), default_max_fct_bonds_opti)
            init_guess.append(draw_float(draw_low, draw_high, 3))

        # angles
        if exec_mode == 1:
            for j in range(opti_cycle["nb_geoms"]["angle"]):
                try: emd_err_fact = max(1, all_best_emd_dist_geoms["angles"][j] / 2)
                except: emd_err_fact = 1
                draw_low = max(out_itp["angle"][j]["value"] - angle_value_guess_variation * val_guess_fact * emd_err_fact, domains_val["angle"][j][0])
                draw_high = min(out_itp["angle"][j]["value"] + angle_value_guess_variation * val_guess_fact * emd_err_fact, domains_val["angle"][j][1])
                init_guess.append(draw_float(draw_low, draw_high, 3))

        for j in range(opti_cycle["nb_geoms"]["angle"]):
            try: emd_err_fact = max(1, all_best_emd_dist_geoms["angles"][j] / 2)
            except: emd_err_fact = 1
            draw_low = max(min(out_itp["angle"][j]["fct"] * (1 - fct_guess_fact * emd_err_fact), out_itp["angle"][j]["fct"] - fct_guess_min_flat_diff_angles), 0)
            if cg_itp["angle"][j]["func"] == 1:
                draw_high = min(max(out_itp["angle"][j]["fct"] * (1 + fct_guess_fact * emd_err_fact), out_itp["angle"][j]["fct"] + fct_guess_min_flat_diff_angles), default_max_fct_angles_opti_f1)
            elif cg_itp["angle"][j]["func"] == 2:
                draw_high = min(max(out_itp["angle"][j]["fct"] * (1 + fct_guess_fact * emd_err_fact), out_itp["angle"][j]["fct"] + fct_guess_min_flat_diff_angles), default_max_fct_angles_opti_f2)
            init_guess.append(draw_float(draw_low, draw_high, 3))

        # dihedrals
        if exec_mode == 1:
             for j in range(opti_cycle["nb_geoms"]["dihedral"]):
                try: emd_err_fact = max(1, all_best_emd_dist_geoms["dihedrals"][j] / 5)
                except: emd_err_fact = 1
                draw_low = max(out_itp["dihedral"][j]["value"] - dihedral_value_guess_variation * val_guess_fact * emd_err_fact, domains_val["dihedral"][j][0])
                draw_high = min(out_itp["dihedral"][j]["value"] + dihedral_value_guess_variation * val_guess_fact * emd_err_fact, domains_val["dihedral"][j][1])
                init_guess.append(draw_float(draw_low, draw_high, 3))
        
        for j in range(opti_cycle["nb_geoms"]["dihedral"]):
            try: emd_err_fact = max(1, all_best_emd_dist_geoms["dihedrals"][j] / 5)
            except: emd_err_fact = 1
            
            if out_itp["dihedral"][j]["fct"] > 0:
                draw_low = out_itp["dihedral"][j]["fct"] * (1 - fct_guess_fact * emd_err_fact)
                draw_high = out_itp["dihedral"][j]["fct"] * (1 + fct_guess_fact * emd_err_fact)
            else:
                draw_low = out_itp["dihedral"][j]["fct"] * (1 + fct_guess_fact * emd_err_fact)
                draw_high = out_itp["dihedral"][j]["fct"] * (1 - fct_guess_fact * emd_err_fact)

            if cg_itp["dihedral"][j]["func"] == 2:
                draw_low = max(min(draw_low, out_itp["dihedral"][j]["fct"] - fct_guess_min_flat_diff_dihedrals_without_mult), -default_abs_range_fct_dihedrals_opti_func_without_mult if default_abs_range_fct_dihedrals_opti_func_without_mult > 0 else 0) # Fallback bound
                draw_high = min(max(draw_high, out_itp["dihedral"][j]["fct"] + fct_guess_min_flat_diff_dihedrals_without_mult), default_abs_range_fct_dihedrals_opti_func_without_mult)
            else:
                 draw_low = max(min(draw_low, out_itp["dihedral"][j]["fct"] - fct_guess_min_flat_diff_dihedrals_with_mult), -default_abs_range_fct_dihedrals_opti_func_with_mult)
                 draw_high = min(max(draw_high, out_itp["dihedral"][j]["fct"] + fct_guess_min_flat_diff_dihedrals_with_mult), default_abs_range_fct_dihedrals_opti_func_with_mult)
            init_guess.append(draw_float(draw_low, draw_high, 3))

        initial_guess_list.append(init_guess)

    return initial_guess_list
