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

    if not performed_init_BI["bond"] and opti_cycle.counts.bonds > 0:
        if verbose:
            logger.info("")
            logger.info(
                "Initializing bond force constants from normalized marginal PMFs"
            )
        for index in range(opti_cycle.counts.bonds):
            group = itp_obj.bonds[index]
            target: BoltzmannTarget = data_BI["bond"][index]
            basis = 0.5 * (target.centers - float(group.equilibrium)) ** 2
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

    if not performed_init_BI["angle"] and opti_cycle.counts.angles > 0:
        if verbose:
            logger.info("")
            logger.info(
                "Initializing angle force constants from normalized marginal PMFs"
            )
        for index in range(opti_cycle.counts.angles):
            group = itp_obj.angles[index]
            target = data_BI["angle"][index]
            equilibrium = np.deg2rad(float(group.equilibrium))
            if group.function == 1:
                basis = 0.5 * (target.centers - equilibrium) ** 2
                upper = opt_config.default_max_fct_angles_opti_f1
            elif group.function == 2:
                basis = 0.5 * (
                    np.cos(target.centers) - np.cos(equilibrium)
                ) ** 2
                upper = opt_config.default_max_fct_angles_opti_f2
            elif group.function == 10:
                basis = (
                    0.5
                    * (np.cos(target.centers) - np.cos(equilibrium)) ** 2
                    / np.sin(target.centers) ** 2
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
        and opti_cycle.counts.dihedrals > 0
    ):
        if verbose:
            logger.info("")
            logger.info(
                "Initializing dihedral parameters from normalized marginal PMFs"
            )
        for index in range(opti_cycle.counts.dihedrals):
            group = itp_obj.dihedrals[index]
            target = data_BI["dihedral"][index]
            function = group.function

            if function in config.dihedral_func_with_mult:
                phase = np.deg2rad(float(group.equilibrium))
                basis = 1.0 + np.cos(
                    group.multiplicity * target.centers - phase
                )
                maximum = (
                    opt_config.default_abs_range_fct_dihedrals_opti_func_with_mult
                )
                fit_group_force(
                    group, target, basis, 0.0, maximum, "Dihedral", index
                )
            elif function == 2:
                equilibrium = np.deg2rad(float(group.equilibrium))
                offset = (
                    target.centers - equilibrium + np.pi
                ) % (2 * np.pi) - np.pi
                basis = 0.5 * offset**2
                maximum = (
                    opt_config.default_abs_range_fct_dihedrals_opti_func_without_mult
                )
                fit_group_force(
                    group,
                    target,
                    basis,
                    -maximum,
                    maximum,
                    "Dihedral",
                    index,
                )
            elif function == 3:
                try:
                    fitted = fit_rb_coefficients(
                        target.centers,
                        target.probabilities,
                        temp,
                        group.coefficient_bound,
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
                    group.parameters = fitted
            elif function == 11 and verbose:
                logger.info(
                    "  CBT group %s retains its canonical input seed; its angular "
                    "coupling is not identifiable from a torsional marginal.",
                    index + 1,
                )

            if function in (3, 11) and verbose:
                logger.info(
                    "  Dihedral group %s coefficients: %s",
                    index + 1,
                    [round(coeff, 3) for coeff in group.gromacs_parameters],
                )
        performed_init_BI["dihedral"] = True
