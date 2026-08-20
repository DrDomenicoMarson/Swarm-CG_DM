"""Force-field initialization helpers."""

import numpy as np

from swarmcg import config
from swarmcg.config_types import SwarmConfig
from swarmcg.optimization_types import OptimizationCycle
from swarmcg.shared.logging_utils import get_logger
from swarmcg.simulations.boltzmann import (
    BoltzmannTarget,
    fit_bounded_force_constant,
)
from swarmcg.simulations.polynomial import fit_rb_coefficients
from swarmcg.topology import CGTopology

logger = get_logger(__name__)


def perform_BI(
    itp_obj: CGTopology,
    opti_cycle: OptimizationCycle,
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
    if not performed_init_BI["bond"] and opti_cycle.counts.bonds > 0:
        _initialize_bonds(
            itp_obj,
            opti_cycle.counts.bonds,
            data_BI["bond"],
            temp,
            config_obj,
            verbose,
        )
        performed_init_BI["bond"] = True
    if not performed_init_BI["angle"] and opti_cycle.counts.angles > 0:
        _initialize_angles(
            itp_obj,
            opti_cycle.counts.angles,
            data_BI["angle"],
            temp,
            config_obj,
            verbose,
        )
        performed_init_BI["angle"] = True
    if (
        not performed_init_BI["dihedral"]
        and opti_cycle.counts.dihedrals > 0
    ):
        _initialize_dihedrals(
            itp_obj,
            opti_cycle.counts.dihedrals,
            data_BI["dihedral"],
            temp,
            config_obj,
            verbose,
        )
        performed_init_BI["dihedral"] = True


def _fit_group_force(
    group, target, basis, temperature, lower, upper, label, index, verbose
) -> None:
    try:
        fitted = fit_bounded_force_constant(
            target,
            basis,
            temperature,
            lower_bound=lower,
            upper_bound=upper,
        )
    except ValueError as exc:
        logger.warning(
            "%s group %s normalized-PMF initialization failed; retaining "
            "input force constant %s: %s",
            label,
            index + 1,
            group.force_constant,
            exc,
        )
        return
    group.force_constant = fitted.force_constant
    if verbose:
        logger.info(
            "  %s group %s estimated force constant: %s",
            label,
            index + 1,
            round(group.force_constant, 3),
        )


def _initialize_bonds(
    topology, count, targets, temperature, config_obj, verbose
) -> None:
    if verbose:
        logger.info("")
        logger.info("Initializing bond force constants from normalized marginal PMFs")
    for index in range(count):
        group = topology.bonds[index]
        target: BoltzmannTarget = targets[index]
        basis = 0.5 * (target.centers - float(group.equilibrium)) ** 2
        _fit_group_force(
            group,
            target,
            basis,
            temperature,
            0.0,
            config_obj.optimization.default_max_fct_bonds_opti,
            "Bond",
            index,
            verbose,
        )


def _initialize_angles(
    topology, count, targets, temperature, config_obj, verbose
) -> None:
    if verbose:
        logger.info("")
        logger.info("Initializing angle force constants from normalized marginal PMFs")
    maxima = {
        1: config_obj.optimization.default_max_fct_angles_opti_f1,
        2: config_obj.optimization.default_max_fct_angles_opti_f2,
        10: config_obj.optimization.default_max_fct_angles_opti_f10,
    }
    for index in range(count):
        group = topology.angles[index]
        target = targets[index]
        equilibrium = np.deg2rad(float(group.equilibrium))
        if group.function == 1:
            basis = 0.5 * (target.centers - equilibrium) ** 2
        elif group.function == 2:
            basis = 0.5 * (
                np.cos(target.centers) - np.cos(equilibrium)
            ) ** 2
        elif group.function == 10:
            basis = (
                0.5
                * (np.cos(target.centers) - np.cos(equilibrium)) ** 2
                / np.sin(target.centers) ** 2
            )
        else:
            continue
        _fit_group_force(
            group,
            target,
            basis,
            temperature,
            0.0,
            maxima[group.function],
            "Angle",
            index,
            verbose,
        )


def _initialize_dihedrals(
    topology, count, targets, temperature, config_obj, verbose
) -> None:
    if verbose:
        logger.info("")
        logger.info(
            "Initializing dihedral parameters from normalized marginal PMFs"
        )
    for index in range(count):
        group = topology.dihedrals[index]
        target = targets[index]
        if group.function in config.dihedral_func_with_mult:
            phase = np.deg2rad(float(group.equilibrium))
            basis = 1.0 + np.cos(group.multiplicity * target.centers - phase)
            maximum = (
                config_obj.optimization.default_abs_range_fct_dihedrals_opti_func_with_mult
            )
            _fit_group_force(
                group, target, basis, temperature, 0.0, maximum,
                "Dihedral", index, verbose
            )
        elif group.function == 2:
            equilibrium = np.deg2rad(float(group.equilibrium))
            offset = (target.centers - equilibrium + np.pi) % (2 * np.pi) - np.pi
            basis = 0.5 * offset**2
            maximum = (
                config_obj.optimization.default_abs_range_fct_dihedrals_opti_func_without_mult
            )
            _fit_group_force(
                group, target, basis, temperature, -maximum, maximum,
                "Dihedral", index, verbose
            )
        elif group.function == 3:
            _initialize_rb(group, target, temperature, index)
        elif group.function == 11 and verbose:
            logger.info(
                "  CBT group %s retains its canonical input seed; its angular "
                "coupling is not identifiable from a torsional marginal.",
                index + 1,
            )
        if group.function in (3, 11) and verbose:
            logger.info(
                "  Dihedral group %s coefficients: %s",
                index + 1,
                [round(coeff, 3) for coeff in group.gromacs_parameters],
            )


def _initialize_rb(group, target, temperature, index) -> None:
    try:
        fitted = fit_rb_coefficients(
            target.centers,
            target.probabilities,
            temperature,
            group.coefficient_bound,
        )
    except ValueError as exc:
        logger.warning(
            "RB dihedral group %s normalized-PMF initialization is not "
            "identifiable; retaining canonical input coefficients and using "
            "broad first-activation exploration: %s",
            index + 1,
            exc,
        )
    else:
        group.parameters = fitted
