import numpy as np

from swarmcg import config
from swarmcg.config_types import SwarmConfig
from swarmcg.shared.logging_utils import get_logger
from swarmcg.shared.math_utils import draw_float
from swarmcg.shared.periodic import (
    PeriodicDihedralParameters,
    normalize_periodic_degrees,
    unwrap_degrees_around,
)
from swarmcg.simulations.boltzmann import (
    BoltzmannTarget,
    fit_bounded_force_constant,
)
from swarmcg.simulations.polynomial import CBTParameters, RBParameters, fit_rb_coefficients

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
            if func in (1, 4):
                canonical = PeriodicDihedralParameters.from_gromacs(
                    itp_obj["dihedral"][i]["value"],
                    itp_obj["dihedral"][i]["fct"],
                    itp_obj["dihedral"][i]["mult"],
                )
                itp_obj["dihedral"][i]["value"] = round(
                    canonical.phase_degrees, 2
                )
                itp_obj["dihedral"][i]["fct"] = round(
                    canonical.force_constant, 2
                )
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
            search_space_boundaries.append(
                [0.0, max_abs] if func in (1, 4) else [-max_abs, max_abs]
            )

    return search_space_boundaries

def perform_BI(
    itp_obj,
    opti_cycle,
    data_BI,
    performed_init_BI,
    temp,
    config_obj: SwarmConfig,
    verbose=False,
    exec_mode=1,
):
    """Initialize active force terms from normalized reference probabilities.

    Ordinary bonded functions fit one bounded force constant plus an arbitrary
    PMF intercept while holding their current equilibrium value fixed. RB fits
    its five force-relevant coefficients only when the polynomial design is
    identifiable. CBT retains its canonical input seed because its angle-
    coupled coefficients cannot be initialized from a torsional marginal.

    Args:
        itp_obj: Working topology modified in place.
        opti_cycle: Active optimization-cycle description.
        data_BI: ``BoltzmannTarget`` instances by geometry class and group.
        performed_init_BI: Mutable flags preventing repeated initialization.
        temp: Simulation temperature in kelvin.
        config_obj: Validated application configuration.
        verbose: Whether to report successful fitted parameters.
        exec_mode: ``1`` for reference-derived equilibria or ``2`` for fixed
            input equilibria.

    Returns:
        ``None``. Active topology groups are updated in place. Failed
        underdetermined fits retain their existing parameters and emit a
        warning.
    """
    opt_config = config_obj.optimization

    def fit_group_force(group, target, basis, lower, upper, label, index):
        """Fit one group force constant, retaining its seed on failure."""
        try:
            fitted = fit_bounded_force_constant(
                target,
                basis,
                temp,
                lower_bound=lower,
                upper_bound=upper,
            )
        except ValueError as exc:
            logger.warning(
                "%s group %s normalized-PMF initialization failed; retaining "
                "input force constant %s: %s",
                label,
                index + 1,
                group["fct"],
                exc,
            )
            return
        group["fct"] = fitted.force_constant
        if verbose:
            logger.info(
                "  %s group %s estimated force constant: %s",
                label,
                index + 1,
                round(group["fct"], 3),
            )

    if not performed_init_BI["bond"] and opti_cycle["nb_geoms"]["bond"] > 0:
        if verbose:
            logger.info("")
            logger.info("Initializing bond force constants from normalized marginal PMFs")
        for index in range(opti_cycle["nb_geoms"]["bond"]):
            group = itp_obj["bond"][index]
            target: BoltzmannTarget = data_BI["bond"][index]
            basis = 0.5 * (target.centers - float(group["value"])) ** 2
            fit_group_force(
                group,
                target,
                basis,
                0.0,
                opt_config.default_max_fct_bonds_opti,
                "Bond",
                index,
            )
        performed_init_BI["bond"] = True

    if not performed_init_BI["angle"] and opti_cycle["nb_geoms"]["angle"] > 0:
        if verbose:
            logger.info("")
            logger.info("Initializing angle force constants from normalized marginal PMFs")
        for index in range(opti_cycle["nb_geoms"]["angle"]):
            group = itp_obj["angle"][index]
            target = data_BI["angle"][index]
            equilibrium = np.deg2rad(float(group["value"]))
            if group["func"] == 1:
                basis = 0.5 * (target.centers - equilibrium) ** 2
                upper = opt_config.default_max_fct_angles_opti_f1
            elif group["func"] == 2:
                basis = 0.5 * (
                    np.cos(target.centers) - np.cos(equilibrium)
                ) ** 2
                upper = opt_config.default_max_fct_angles_opti_f2
            elif group["func"] == 10:
                sin_sq = np.sin(target.centers) ** 2
                basis = (
                    0.5
                    * (np.cos(target.centers) - np.cos(equilibrium)) ** 2
                    / sin_sq
                )
                upper = opt_config.default_max_fct_angles_opti_f10
            else:
                continue
            fit_group_force(
                group,
                target,
                basis,
                0.0,
                upper,
                "Angle",
                index,
            )
        performed_init_BI["angle"] = True

    if (
        not performed_init_BI["dihedral"]
        and opti_cycle["nb_geoms"]["dihedral"] > 0
    ):
        if verbose:
            logger.info("")
            logger.info("Initializing dihedral parameters from normalized marginal PMFs")
        for index in range(opti_cycle["nb_geoms"]["dihedral"]):
            group = itp_obj["dihedral"][index]
            target = data_BI["dihedral"][index]
            func = group["func"]

            if func in config.dihedral_func_with_mult:
                phase = np.deg2rad(float(group["value"]))
                basis = 1.0 + np.cos(group["mult"] * target.centers - phase)
                max_abs = opt_config.default_abs_range_fct_dihedrals_opti_func_with_mult
                fit_group_force(
                    group,
                    target,
                    basis,
                    0.0,
                    max_abs,
                    "Dihedral",
                    index,
                )
            elif func == 2:
                equilibrium = np.deg2rad(float(group["value"]))
                offset = (target.centers - equilibrium + np.pi) % (2 * np.pi) - np.pi
                basis = 0.5 * offset**2
                max_abs = opt_config.default_abs_range_fct_dihedrals_opti_func_without_mult
                fit_group_force(
                    group,
                    target,
                    basis,
                    -max_abs,
                    max_abs,
                    "Dihedral",
                    index,
                )
            elif func == 3:
                try:
                    fitted = fit_rb_coefficients(
                        target.centers,
                        target.probabilities,
                        temp,
                        group["coefficient_bound"],
                    )
                except ValueError as exc:
                    logger.warning(
                        "RB dihedral group %s normalized-PMF initialization is "
                        "not identifiable; retaining canonical input coefficients "
                        "and using broad first-activation exploration: %s",
                        index + 1,
                        exc,
                    )
                else:
                    group["params"] = list(fitted.to_gromacs())
            elif func == 11 and verbose:
                logger.info(
                    "  CBT group %s retains its canonical input seed; its angular "
                    "coupling is not identifiable from a torsional marginal.",
                    index + 1,
                )

            if func not in (3, 11):
                group["params"] = [group["value"], group["fct"]]
            elif verbose:
                logger.info(
                    "  Dihedral group %s coefficients: %s",
                    index + 1,
                    [round(coeff, 3) for coeff in group["params"]],
                )
        performed_init_BI["dihedral"] = True

def get_initial_guess_list(nb_particles, opti_cycle, cg_itp, out_itp, domains_val,
                           all_best_emd_dist_geoms, all_best_params_dist_geoms,
                           exec_mode, config_obj: SwarmConfig, user_input=False,
                           val_guess_fact=None, fct_guess_fact=None,
                           verbose=False):
    """Build deterministic and exploratory particle initial positions.

    On a polynomial group's first active cycle, one fitted/canonical seed is
    retained and every other RB/CBT coefficient is drawn independently over
    the complete coefficient bound. Once a finite group optimum exists,
    subsequent cycles use clipped local perturbations around the staged
    optimum. Equivalent deterministic seeds are not duplicated.

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
        try:
            return bool(np.isfinite(all_best_emd_dist_geoms[geometry][index]))
        except (KeyError, TypeError):
            return False

    def append_unique_guess(candidate):
        """Append a seed unless an equivalent particle is already present."""
        candidate_array = np.asarray(candidate, dtype=float)
        if not any(
            candidate_array.shape == np.asarray(existing).shape
            and np.allclose(candidate_array, existing, rtol=0.0, atol=1e-12)
            for existing in initial_guess_list
        ):
            initial_guess_list.append(list(candidate))

    def unwrapped_phase(index, value):
        """Express a serialized phase in its search domain's local branch."""
        domain = domains_val["dihedral"][index]
        center = (
            (domain[0] + domain[1]) / 2.0
            if domain is not None
            else float(value)
        )
        return float(unwrap_degrees_around(np.array([value]), center)[0])

    def force_bounds(func, maximum):
        """Return canonical force bounds for an ordinary dihedral form."""
        return (0.0, maximum) if func in (1, 4) else (-maximum, maximum)


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
        lower, upper = force_bounds(func, max_abs)
        input_guess.append(min(max(out_itp["dihedral"][i]["fct"], lower), upper))

    append_unique_guess(input_guess)

    # 2. Particle from best EMD or user input
    if opti_cycle["nb_cycle"] > 1:
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
            lower, upper = force_bounds(func, max_abs)
            input_guess.append(min(max(fct_val, lower), upper))

        append_unique_guess(input_guess)

    elif user_input:
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
            lower, upper = force_bounds(func, max_abs)
            input_guess.append(
                min(max(out_itp["dihedral"][i]["fct_user"], lower), upper)
            )

        append_unique_guess(input_guess)

    # 3. Variations for remaining particles
    for i in range(len(initial_guess_list), nb_particles):
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
                if not has_valid_best("dihedrals", j):
                    # A polynomial group has no equilibrium phase and CBT has
                    # no identifiable one-dimensional DBI seed. Its first
                    # active cycle therefore starts with one canonical/fitted
                    # particle and full-bound exploration for all others.
                    init_guess.extend(
                        draw_float(-max_abs, max_abs, 3) for _ in range(5)
                    )
                    continue
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
                draw_low = max(min(draw_low, out_itp["dihedral"][j]["fct"] - fct_guess_min_flat_diff_dihedrals_with_mult), 0.0)
                draw_high = min(max(draw_high, out_itp["dihedral"][j]["fct"] + fct_guess_min_flat_diff_dihedrals_with_mult), default_abs_range_fct_dihedrals_opti_func_with_mult)
            init_guess.append(draw_float(draw_low, draw_high, 3))

        initial_guess_list.append(init_guess)

    return initial_guess_list
