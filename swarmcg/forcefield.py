import collections
import numpy as np
from scipy.optimize import curve_fit

from swarmcg import config
from swarmcg.shared import math_utils, exceptions
from swarmcg.config_types import SwarmConfig
from swarmcg.shared.logging_utils import get_logger
from swarmcg.shared.math_utils import draw_float
from swarmcg.shared.periodic import normalize_periodic_degrees, unwrap_degrees_around
from swarmcg.simulations.polynomial import CBTParameters, RBParameters, fit_rb_coefficients
from swarmcg.simulations.potentials import (
    gmx_bonds_func_1, gmx_angles_func_1, gmx_angles_func_2, gmx_angles_func_10,
    gmx_dihedrals_func_1, gmx_dihedrals_func_2, gmx_dihedrals_func_3, gmx_dihedrals_func_11
)

logger = get_logger(__name__)

def update_cg_itp_obj(itp_obj, opti_cycle, parameters_set, exec_mode):
    """Update a coarse-grained ITP object in place.

    Args:
        itp_obj: Parsed topology object to update.
        opti_cycle: Active optimization-cycle geometry description.
        parameters_set: Flat vector of free PSO parameters.
        exec_mode: ``1`` to update equilibrium values and force terms, or ``2``
            to update force terms only.

    Raises:
        ValueError: If the parameter vector has the wrong dimension.
    """
    
    # Validation/Router for update_type (intermediary vs cycles optimized) 
    # handled by caller selecting the right itp_obj
    
    nb_constraints = opti_cycle["nb_geoms"]["constraint"]
    nb_bonds = opti_cycle["nb_geoms"]["bond"]
    nb_angles = opti_cycle["nb_geoms"]["angle"]
    nb_dihedrals = opti_cycle["nb_geoms"]["dihedral"]

    idx = 0

    # Constraints
    if exec_mode == 1:
        for i in range(nb_constraints):
            itp_obj["constraint"][i]["value"] = round(parameters_set[idx], 3)
            idx += 1

    # Bonds
    if exec_mode == 1:
        for i in range(nb_bonds):
            itp_obj["bond"][i]["value"] = round(parameters_set[idx], 3)
            idx += 1

    for i in range(nb_bonds):
        itp_obj["bond"][i]["fct"] = round(parameters_set[idx], 3)
        idx += 1

    # Angles
    if exec_mode == 1:
        for i in range(nb_angles):
            itp_obj["angle"][i]["value"] = round(parameters_set[idx], 2)
            idx += 1

    for i in range(nb_angles):
        itp_obj["angle"][i]["fct"] = round(parameters_set[idx], 2)
        idx += 1

    # Dihedrals
    for i in range(nb_dihedrals):
        func = itp_obj["dihedral"][i]["func"]
        if func in (3, 11):
            coeffs = tuple(float(param) for param in parameters_set[idx:idx + 5])
            if len(coeffs) != 5:
                raise ValueError("polynomial dihedrals require five free coefficients")
            idx += 5
            if func == 3:
                canonical = RBParameters(coeffs).to_gromacs()
            else:
                canonical = CBTParameters(coeffs).to_gromacs()
            itp_obj["dihedral"][i]["params"] = [round(param, 8) for param in canonical]
        else:
            if exec_mode == 1:
                itp_obj["dihedral"][i]["value"] = round(
                    normalize_periodic_degrees(parameters_set[idx]), 2
                )
                itp_obj["dihedral"][i]["fct"] = round(parameters_set[idx + 1], 2)
                idx += 2
            else:
                itp_obj["dihedral"][i]["fct"] = round(parameters_set[idx], 2)
                idx += 1
            itp_obj["dihedral"][i]["params"] = [
                itp_obj["dihedral"][i]["value"],
                itp_obj["dihedral"][i]["fct"],
            ]

    if idx != len(parameters_set):
        raise ValueError(
            f"parameter vector has dimension {len(parameters_set)}, expected {idx}"
        )


def get_search_space_boundaries(cg_itp, opti_cycle, domains_val, exec_mode, config_obj: SwarmConfig):
    """Build bounds for every free parameter in the active optimization cycle.

    Args:
        cg_itp: Parsed coarse-grained topology.
        opti_cycle: Active optimization-cycle geometry description.
        domains_val: Reference-derived equilibrium-value domains.
        exec_mode: Optimization execution mode.
        config_obj: Validated application configuration.

    Returns:
        A list of ``[lower, upper]`` pairs in PSO-vector order.
    """
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
            elif cg_itp["angle"][grp_angle]["func"] == 10:
                search_space_boundaries.extend([[0, opt_config.default_max_fct_angles_opti_f10]])

    if opti_cycle["nb_geoms"]["dihedral"] > 0:
        for grp_dihedral in range(opti_cycle["nb_geoms"]["dihedral"]):
            func = cg_itp["dihedral"][grp_dihedral]["func"]
            if func in (3, 11):
                max_abs = cg_itp["dihedral"][grp_dihedral]["coefficient_bound"]
                search_space_boundaries.extend([[-max_abs, max_abs]] * 5)
                continue

            if exec_mode == 1:
                search_space_boundaries.append(domains_val["dihedral"][grp_dihedral])

            if func == 2:
                max_abs = opt_config.default_abs_range_fct_dihedrals_opti_func_without_mult
            else:
                max_abs = opt_config.default_abs_range_fct_dihedrals_opti_func_with_mult
            search_space_boundaries.append([-max_abs, max_abs])

    return search_space_boundaries

def perform_BI(itp_obj, opti_cycle, data_BI, performed_init_BI, temp, config_obj: SwarmConfig, verbose=False, exec_mode=1):
    """Initialize active force terms from reference distributions.

    Harmonic terms use the legacy direct-Boltzmann-inversion fits. RB terms use
    bounded linear least squares against the target PMF. CBT terms retain their
    canonicalized input seed because their angle-coupled coefficients are not
    identifiable from a one-dimensional torsional marginal.

    Args:
        itp_obj: Working topology modified in place.
        opti_cycle: Active optimization-cycle description.
        data_BI: Reference histogram and moment data by geometry class.
        performed_init_BI: Mutable flags preventing repeated initialization.
        temp: Simulation temperature in kelvin.
        config_obj: Validated application configuration.
        verbose: Whether to report fitted parameters.
        exec_mode: Optimization execution mode.
    """
    opt_config = config_obj.optimization
    
    # Constants/Config access - using global config as it contains physical constants
    kB = config.kB # Leaving physical constant global
    
    # Check bonds
    if not performed_init_BI["bond"] and opti_cycle["nb_geoms"]["bond"] > 0:
        if verbose:
            logger.info("")
            logger.info("Performing Direct Boltzmann Inversion (DBI) to estimate bonds force constants")

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
                    logger.info(
                        "  Bond group %s estimated force constant: %s",
                        grp_bond + 1,
                        round(itp_obj["bond"][grp_bond]["fct"], 2),
                    )
            except Exception as e:
                # Fallback or error logging
                if verbose:
                    logger.warning("  Bond group %s BI failed: %s", grp_bond + 1, e)

        performed_init_BI["bond"] = True

    # Check angles
    if not performed_init_BI["angle"] and opti_cycle["nb_geoms"]["angle"] > 0:
        if verbose:
            logger.info("")
            logger.info("Performing Direct Boltzmann Inversion (DBI) to estimate angles force constants")
            
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
            elif func == 10:
                params_guess = [max(y) - min(y), std_rad_grp_angle, min(y)]
                try:
                    popt, pcov = curve_fit(gmx_angles_func_10, x, y, p0=params_guess, sigma=sigma, maxfev=99999, absolute_sigma=False)
                    if popt[0] < 0:
                        popt[0] = 30 # fallback
                    itp_obj["angle"][grp_angle]["fct"] = popt[0]
                except:
                    itp_obj["angle"][grp_angle]["fct"] = 30

            if verbose:
                logger.info(
                    "  Angle group %s estimated force constant: %s",
                    grp_angle + 1,
                    round(itp_obj["angle"][grp_angle]["fct"], 2),
                )

        performed_init_BI["angle"] = True

    # Check dihedrals
    if not performed_init_BI["dihedral"] and opti_cycle["nb_geoms"]["dihedral"] > 0:
        if verbose:
            logger.info("")
            logger.info("Initializing dihedral force parameters from reference distributions")

        for grp_dihedral in range(opti_cycle["nb_geoms"]["dihedral"]):
            hists_geoms_bi, std_rad_grp_dihedral, avg_rad_grp_dihedral, bi_xrange = data_BI["dihedral"][grp_dihedral]
            y = -kB * temp * np.log(hists_geoms_bi + 1)
            x = np.linspace(
                np.deg2rad(bi_xrange[0]),
                np.deg2rad(bi_xrange[1]),
                2 * opt_config.bi_nb_bins,
                endpoint=True,
            )
            k = kB * temp / std_rad_grp_dihedral / std_rad_grp_dihedral
            sigma = np.where(y < max(y), 0.1, np.inf)

            # Again, assuming func is present in itp_obj for now, or we'd need cg_itp passed explicitly
            func = itp_obj["dihedral"][grp_dihedral]["func"]

            if exec_mode == 2:
                value_user = itp_obj["dihedral"][grp_dihedral].get("value_user")
                avg_rad = np.deg2rad(value_user) if value_user is not None else 0

            if func in config.dihedral_func_with_mult:
                multiplicity = itp_obj["dihedral"][grp_dihedral]["mult"]
                params_guess = [max(y) - min(y), avg_rad_grp_dihedral, min(y)]
                try:
                    popt, pcov = curve_fit(
                        gmx_dihedrals_func_1(mult=multiplicity),
                        x,
                        y,
                        p0=params_guess,
                        sigma=sigma,
                        maxfev=99999,
                        absolute_sigma=False,
                    )
                    itp_obj["dihedral"][grp_dihedral]["fct"] = popt[0]
                except:
                    pass
            elif func == 2:
                params_guess = [k, avg_rad_grp_dihedral, min(y)]
                try:
                    popt, pcov = curve_fit(
                        gmx_dihedrals_func_2,
                        x,
                        y,
                        p0=params_guess,
                        sigma=sigma,
                        maxfev=99999,
                        absolute_sigma=False,
                    )
                    itp_obj["dihedral"][grp_dihedral]["fct"] = popt[0]
                except:
                    pass
            elif func == 3:
                centers = np.linspace(
                    np.deg2rad(bi_xrange[0]),
                    np.deg2rad(bi_xrange[1]),
                    len(hists_geoms_bi),
                    endpoint=False,
                )
                centers += np.deg2rad(bi_xrange[1] - bi_xrange[0]) / (2 * len(hists_geoms_bi))
                fitted = fit_rb_coefficients(
                    centers,
                    hists_geoms_bi,
                    temp,
                    itp_obj["dihedral"][grp_dihedral]["coefficient_bound"],
                )
                itp_obj["dihedral"][grp_dihedral]["params"] = list(fitted.to_gromacs())
            elif func == 11 and verbose:
                logger.info(
                    "  CBT group %s retains its canonical input seed; its angular "
                    "coupling is not identifiable by one-dimensional DBI.",
                    grp_dihedral + 1,
                )

            if func not in (3, 11):
                itp_obj["dihedral"][grp_dihedral]["params"] = [
                    itp_obj["dihedral"][grp_dihedral]["value"],
                    itp_obj["dihedral"][grp_dihedral]["fct"],
                ]

            if verbose:
                if func in (3, 11):
                    coeffs = itp_obj["dihedral"][grp_dihedral]["params"]
                    logger.info(
                        "  Dihedral group %s coefficients: %s",
                        grp_dihedral + 1,
                        [round(coeff, 2) for coeff in coeffs],
                    )
                else:
                    logger.info(
                        "  Dihedral group %s estimated force constant: %s",
                        grp_dihedral + 1,
                        round(itp_obj["dihedral"][grp_dihedral]["fct"], 2),
                    )

        performed_init_BI["dihedral"] = True

def get_initial_guess_list(nb_particles, opti_cycle, cg_itp, out_itp, domains_val,
                           all_best_emd_dist_geoms, all_best_params_dist_geoms,
                           exec_mode, config_obj: SwarmConfig, user_input=False,
                           val_guess_fact=None, fct_guess_fact=None,
                           verbose=False):
    """Build deterministic and exploratory particle initial positions.

    Args:
        nb_particles: Number of particles to initialize.
        opti_cycle: Active optimization-cycle description.
        cg_itp: Parsed baseline topology.
        out_itp: Working topology containing staged parameters.
        domains_val: Reference-derived equilibrium-value domains.
        all_best_emd_dist_geoms: Best known per-geometry EMD values.
        all_best_params_dist_geoms: Parameters associated with those EMDs.
        exec_mode: Optimization execution mode.
        config_obj: Validated application configuration.
        user_input: Whether to add the explicit input parameters as a seed.
        val_guess_fact: Equilibrium-value exploration multiplier.
        fct_guess_fact: Force-parameter exploration multiplier.
        verbose: Reserved verbosity flag.

    Returns:
        One flat parameter vector per particle.
    """
    initial_guess_list = []
    input_guess = []

    # defaults if not provided (should be provided by caller from config)
    if val_guess_fact is None: val_guess_fact = 1.0 # default?
    if fct_guess_fact is None: fct_guess_fact = 0.5 # default?

    opt_config = config_obj.optimization
    
    default_max_fct_bonds_opti = opt_config.default_max_fct_bonds_opti
    default_max_fct_angles_opti_f1 = opt_config.default_max_fct_angles_opti_f1
    default_max_fct_angles_opti_f2 = opt_config.default_max_fct_angles_opti_f2
    default_max_fct_angles_opti_f10 = opt_config.default_max_fct_angles_opti_f10
    default_abs_range_fct_dihedrals_opti_func_without_mult = opt_config.default_abs_range_fct_dihedrals_opti_func_without_mult
    default_abs_range_fct_dihedrals_opti_func_with_mult = opt_config.default_abs_range_fct_dihedrals_opti_func_with_mult
    bond_dist_guess_variation = opt_config.bond_dist_guess_variation
    angle_value_guess_variation = opt_config.angle_value_guess_variation
    dihedral_value_guess_variation = opt_config.dihedral_value_guess_variation
    fct_guess_min_flat_diff_bonds = opt_config.fct_guess_min_flat_diff_bonds
    fct_guess_min_flat_diff_angles = opt_config.fct_guess_min_flat_diff_angles
    fct_guess_min_flat_diff_dihedrals_without_mult = opt_config.fct_guess_min_flat_diff_dihedrals_without_mult
    fct_guess_min_flat_diff_dihedrals_with_mult = opt_config.fct_guess_min_flat_diff_dihedrals_with_mult

    def polynomial_coefficients(group):
        """Return the five free coefficients from a topology group."""
        if group["func"] == 3:
            return RBParameters.from_gromacs(group["params"]).coefficients
        return CBTParameters.from_gromacs(group["params"]).effective_coefficients

    def has_valid_best(geometry, index):
        """Return whether a per-geometry best score is available."""
        return np.isfinite(all_best_emd_dist_geoms[geometry][index])

    def unwrapped_phase(index, value):
        """Express a serialized phase in the reference mean's local branch."""
        center = cg_itp["dihedral"][index]["avg"]
        return float(unwrap_degrees_around(np.array([value]), center)[0])


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
        elif cg_itp["angle"][i]["func"] == 10:
            input_guess.append(
                min(max(out_itp["angle"][i]["fct"], 0), default_max_fct_angles_opti_f10))

    for i in range(opti_cycle["nb_geoms"]["dihedral"]):
        func = cg_itp["dihedral"][i]["func"]
        if func in (3, 11):
            input_guess.extend(polynomial_coefficients(out_itp["dihedral"][i]))
            continue

        if exec_mode == 1:
            phase = unwrapped_phase(i, out_itp["dihedral"][i]["value"])
            input_guess.append(min(max(phase, domains_val["dihedral"][i][0]),
                                   domains_val["dihedral"][i][1]))

        if func == 2:
            max_abs = default_abs_range_fct_dihedrals_opti_func_without_mult
        else:
            max_abs = default_abs_range_fct_dihedrals_opti_func_with_mult
        input_guess.append(min(max(out_itp["dihedral"][i]["fct"], -max_abs), max_abs))

    initial_guess_list.append(input_guess)
    num_particle_random_start = 1

    # 2. Particle from best EMD or user input
    if opti_cycle["nb_cycle"] > 1:
        num_particle_random_start += 1
        input_guess = []

        # Constraints
        if exec_mode == 1:
            for i in range(opti_cycle["nb_geoms"]["constraint"]):
                if has_valid_best("constraints", i):
                    input_guess.append(all_best_params_dist_geoms["constraints"][i]["params"][0])
                else:
                    input_guess.append(min(max(out_itp["constraint"][i]["value"], domains_val["constraint"][i][0]), domains_val["constraint"][i][1])) # Fallback
        
        # Bonds
        if exec_mode == 1:
            for i in range(opti_cycle["nb_geoms"]["bond"]):
                if has_valid_best("bonds", i):
                    input_guess.append(all_best_params_dist_geoms["bonds"][i]["params"][0])
                else:
                    input_guess.append(min(max(out_itp["bond"][i]["value"], domains_val["bond"][i][0]), domains_val["bond"][i][1]))
        
        for i in range(opti_cycle["nb_geoms"]["bond"]):
            if has_valid_best("bonds", i):
                input_guess.append(all_best_params_dist_geoms["bonds"][i]["params"][1])
            else:
                 input_guess.append(min(max(out_itp["bond"][i]["fct"], 0), default_max_fct_bonds_opti))

        # Angles
        if exec_mode == 1:
            for i in range(opti_cycle["nb_geoms"]["angle"]):
                if has_valid_best("angles", i):
                    input_guess.append(all_best_params_dist_geoms["angles"][i]["params"][0])
                else:
                    input_guess.append(min(max(out_itp["angle"][i]["value"], domains_val["angle"][i][0]), domains_val["angle"][i][1]))

        for i in range(opti_cycle["nb_geoms"]["angle"]):
            if has_valid_best("angles", i):
                input_guess.append(all_best_params_dist_geoms["angles"][i]["params"][1])
            else:
                if cg_itp["angle"][i]["func"] == 1:
                    input_guess.append(min(max(out_itp["angle"][i]["fct"], 0), default_max_fct_angles_opti_f1))
                elif cg_itp["angle"][i]["func"] == 2:
                     input_guess.append(min(max(out_itp["angle"][i]["fct"], 0), default_max_fct_angles_opti_f2))
                elif cg_itp["angle"][i]["func"] == 10:
                     input_guess.append(min(max(out_itp["angle"][i]["fct"], 0), default_max_fct_angles_opti_f10))

        # Dihedrals
        for i in range(opti_cycle["nb_geoms"]["dihedral"]):
            func = cg_itp["dihedral"][i]["func"]
            best_params = None
            if has_valid_best("dihedrals", i):
                best_params = all_best_params_dist_geoms["dihedrals"][i]["params"]

            if func in (3, 11):
                params = best_params if best_params is not None else out_itp["dihedral"][i]["params"]
                group = dict(out_itp["dihedral"][i])
                group["params"] = params
                input_guess.extend(polynomial_coefficients(group))
                continue

            if exec_mode == 1:
                if best_params is not None:
                    input_guess.append(unwrapped_phase(i, best_params[0]))
                else:
                    phase = unwrapped_phase(i, out_itp["dihedral"][i]["value"])
                    input_guess.append(min(max(phase, domains_val["dihedral"][i][0]),
                                           domains_val["dihedral"][i][1]))

            if best_params is not None:
                fct_val = best_params[1]
            else:
                fct_val = out_itp["dihedral"][i]["fct"]

            if func == 2:
                max_abs = default_abs_range_fct_dihedrals_opti_func_without_mult
            else:
                max_abs = default_abs_range_fct_dihedrals_opti_func_with_mult
            input_guess.append(min(max(fct_val, -max_abs), max_abs))

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
            elif cg_itp["angle"][i]["func"] == 10:
                input_guess.append(min(max(out_itp["angle"][i]["fct_user"], 0), default_max_fct_angles_opti_f10))

        # dihedrals
        for i in range(opti_cycle["nb_geoms"]["dihedral"]):
            func = cg_itp["dihedral"][i]["func"]
            if func in (3, 11):
                group = dict(out_itp["dihedral"][i])
                group["params"] = out_itp["dihedral"][i]["params_user"]
                input_guess.extend(polynomial_coefficients(group))
                continue

            if exec_mode == 1:
                phase = unwrapped_phase(i, out_itp["dihedral"][i]["value_user"])
                input_guess.append(min(max(phase, domains_val["dihedral"][i][0]),
                                       domains_val["dihedral"][i][1]))

            if func == 2:
                max_abs = default_abs_range_fct_dihedrals_opti_func_without_mult
            else:
                max_abs = default_abs_range_fct_dihedrals_opti_func_with_mult
            input_guess.append(min(max(out_itp["dihedral"][i]["fct_user"], -max_abs), max_abs))

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
            elif cg_itp["angle"][j]["func"] == 10:
                draw_high = min(max(out_itp["angle"][j]["fct"] * (1 + fct_guess_fact * emd_err_fact), out_itp["angle"][j]["fct"] + fct_guess_min_flat_diff_angles), default_max_fct_angles_opti_f10)
            init_guess.append(draw_float(draw_low, draw_high, 3))

        # dihedrals
        for j in range(opti_cycle["nb_geoms"]["dihedral"]):
            try:
                emd_err_fact = max(1, all_best_emd_dist_geoms["dihedrals"][j] / 5)
            except:
                emd_err_fact = 1

            func = cg_itp["dihedral"][j]["func"]
            if func in (3, 11):
                max_abs = cg_itp["dihedral"][j]["coefficient_bound"]
                for coeff in polynomial_coefficients(out_itp["dihedral"][j]):
                    if coeff > 0:
                        draw_low = coeff * (1 - fct_guess_fact * emd_err_fact)
                        draw_high = coeff * (1 + fct_guess_fact * emd_err_fact)
                    else:
                        draw_low = coeff * (1 + fct_guess_fact * emd_err_fact)
                        draw_high = coeff * (1 - fct_guess_fact * emd_err_fact)
                    draw_low = max(min(draw_low, coeff - fct_guess_min_flat_diff_dihedrals_without_mult), -max_abs)
                    draw_high = min(max(draw_high, coeff + fct_guess_min_flat_diff_dihedrals_without_mult), max_abs)
                    init_guess.append(draw_float(draw_low, draw_high, 3))
                continue

            if exec_mode == 1:
                phase = unwrapped_phase(j, out_itp["dihedral"][j]["value"])
                draw_low = max(phase - dihedral_value_guess_variation * val_guess_fact * emd_err_fact, domains_val["dihedral"][j][0])
                draw_high = min(phase + dihedral_value_guess_variation * val_guess_fact * emd_err_fact, domains_val["dihedral"][j][1])
                init_guess.append(draw_float(draw_low, draw_high, 3))

            if out_itp["dihedral"][j]["fct"] > 0:
                draw_low = out_itp["dihedral"][j]["fct"] * (1 - fct_guess_fact * emd_err_fact)
                draw_high = out_itp["dihedral"][j]["fct"] * (1 + fct_guess_fact * emd_err_fact)
            else:
                draw_low = out_itp["dihedral"][j]["fct"] * (1 + fct_guess_fact * emd_err_fact)
                draw_high = out_itp["dihedral"][j]["fct"] * (1 - fct_guess_fact * emd_err_fact)

            if func == 2:
                draw_low = max(min(draw_low, out_itp["dihedral"][j]["fct"] - fct_guess_min_flat_diff_dihedrals_without_mult), -default_abs_range_fct_dihedrals_opti_func_without_mult if default_abs_range_fct_dihedrals_opti_func_without_mult > 0 else 0) # Fallback bound
                draw_high = min(max(draw_high, out_itp["dihedral"][j]["fct"] + fct_guess_min_flat_diff_dihedrals_without_mult), default_abs_range_fct_dihedrals_opti_func_without_mult)
            else:
                draw_low = max(min(draw_low, out_itp["dihedral"][j]["fct"] - fct_guess_min_flat_diff_dihedrals_with_mult), -default_abs_range_fct_dihedrals_opti_func_with_mult)
                draw_high = min(max(draw_high, out_itp["dihedral"][j]["fct"] + fct_guess_min_flat_diff_dihedrals_with_mult), default_abs_range_fct_dihedrals_opti_func_with_mult)
            init_guess.append(draw_float(draw_low, draw_high, 3))

        initial_guess_list.append(init_guess)

    return initial_guess_list
