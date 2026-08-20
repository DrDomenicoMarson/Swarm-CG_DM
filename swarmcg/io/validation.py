"""Scientific preflight validation for simulation input coordinates."""

from __future__ import annotations

import MDAnalysis as mda
import numpy as np

from swarmcg.config_types import SwarmConfig
from swarmcg.shared import exceptions
from swarmcg.shared.periodic import PeriodicDihedralParameters
from swarmcg.simulations.polynomial import CBTParameters, RBParameters
from swarmcg.topology import CGTopology


def validate_mapping_bead_count(topology: CGTopology, all_beads) -> None:
    """Validate that a mapping defines every real topology bead.

    Args:
        topology: Typed coarse-grained topology.
        all_beads: Mapping records keyed by real bead identifier.

    Returns:
        ``None``.

    Raises:
        MissformattedFile: If mapping and topology real-bead counts differ.
    """
    if len(all_beads) != len(topology.real_bead_ids):
        raise exceptions.MissformattedFile(
            "The CG beads mapping (NDX) file does not include as many CG beads "
            "as the ITP file. Please check the NDX and ITP inputs."
        )


def validate_parameter_bounds(topology: CGTopology, config: SwarmConfig) -> None:
    """Validate user-supplied parameters against configured optimization bounds.

    Args:
        topology: Typed coarse-grained topology.
        config: Validated application configuration.

    Returns:
        ``None``. Bounds are skipped unless ``user_input`` is enabled.

    Raises:
        MissformattedFile: If an input force or polynomial coefficient lies
            outside its configured bound.
    """
    if not config.cg_model.user_input:
        return

    def require(value: float, maximum: float, label: str) -> None:
        """Require a symmetric or nonnegative force-parameter bound."""
        lower = 0.0 if label in {"bond", "angle", "periodic dihedral"} else -maximum
        if not lower <= value <= maximum:
            raise exceptions.MissformattedFile(
                f"Input {label} parameter {value} lies outside [{lower}, {maximum}]."
            )

    for group in topology.bonds:
        require(
            group.input_force_constant,
            config.optimization.default_max_fct_bonds_opti,
            "bond",
        )
    angle_bounds = {
        1: config.optimization.default_max_fct_angles_opti_f1,
        2: config.optimization.default_max_fct_angles_opti_f2,
        10: config.optimization.default_max_fct_angles_opti_f10,
    }
    for group in topology.angles:
        require(group.input_force_constant, angle_bounds[group.function], "angle")
    for group in topology.dihedrals:
        parameters = group.input_parameters
        if isinstance(parameters, PeriodicDihedralParameters):
            require(
                parameters.force_constant,
                config.optimization.default_abs_range_fct_dihedrals_opti_func_with_mult,
                "periodic dihedral",
            )
        elif group.function == 2:
            require(
                parameters.force_constant,
                config.optimization.default_abs_range_fct_dihedrals_opti_func_without_mult,
                "dihedral",
            )
        elif isinstance(parameters, RBParameters):
            maximum = config.optimization.max_abs_rb_coefficient
            if maximum is not None and any(
                abs(value) > maximum for value in parameters.coefficients
            ):
                raise exceptions.MissformattedFile(
                    f"Input RB coefficient lies outside [-{maximum}, {maximum}]."
                )
        elif isinstance(parameters, CBTParameters):
            maximum = config.optimization.max_abs_cbt_effective_coefficient
            if maximum is not None and any(
                abs(value) > maximum for value in parameters.effective_coefficients
            ):
                raise exceptions.MissformattedFile(
                    f"Input CBT coefficient lies outside [-{maximum}, {maximum}]."
                )


def validate_restricted_bending_start(
    gro_filename: str, topology: CGTopology
) -> None:
    """Validate the modeled molecule and every starting ReB angle.

    The optimizer and scoring pipeline both operate on the first
    ``len(topology.atoms)`` atoms of the CG system. This preflight applies the
    same ordering contract to the starting GRO and evaluates function-10
    angles using minimum-image vectors when a valid periodic box is present.

    Args:
        gro_filename: Starting GROMACS coordinate path.
        topology: Typed coarse-grained topology.

    Raises:
        ScientificValidationError: If the GRO contains too few atoms,
            non-finite modeled coordinates, or a function-10 angle outside
            the inclusive 10--170 degree safety interval.
        OSError: If the coordinate file cannot be opened.
    """
    universe = mda.Universe(gro_filename, guess_bonds=False)
    modeled_atom_count = len(topology.atoms)
    if len(universe.atoms) < modeled_atom_count:
        raise exceptions.ScientificValidationError(
            f"Starting GRO contains {len(universe.atoms)} atoms but the modeled "
            f"molecule requires its first {modeled_atom_count} atoms."
        )
    positions = np.asarray(
        universe.atoms[:modeled_atom_count].positions, dtype=float
    )
    if not np.all(np.isfinite(positions)):
        bad_atoms = np.flatnonzero(~np.all(np.isfinite(positions), axis=1)) + 1
        raise exceptions.ScientificValidationError(
            "Starting GRO modeled-molecule coordinates must be finite; "
            f"offending one-based atom ids: {bad_atoms.tolist()}."
        )

    dimensions = universe.dimensions
    box = None
    if dimensions is not None:
        dimensions = np.asarray(dimensions, dtype=np.float32)
        if dimensions.shape == (6,) and np.all(np.isfinite(dimensions)) and np.all(
            dimensions[:3] > 0
        ):
            box = dimensions

    for group_index, group in enumerate(topology.angles, start=1):
        if group.function != 10:
            continue
        bead_tuples = np.asarray(group.beads, dtype=int)
        angles = np.rad2deg(
            mda.lib.distances.calc_angles(
                positions[bead_tuples[:, 0]],
                positions[bead_tuples[:, 1]],
                positions[bead_tuples[:, 2]],
                box=box,
                backend="serial",
            )
        )
        unsafe = ~np.isfinite(angles) | (angles < 10.0) | (angles > 170.0)
        if np.any(unsafe):
            first = int(np.flatnonzero(unsafe)[0])
            finite = angles[np.isfinite(angles)]
            observed_range = (
                f"{finite.min():.3f}--{finite.max():.3f} degrees"
                if finite.size
                else "no finite angles"
            )
            offending_angle = (
                f"{angles[first]:.6g} degrees"
                if np.isfinite(angles[first])
                else "non-finite"
            )
            beads = (bead_tuples[first] + 1).tolist()
            raise exceptions.ScientificValidationError(
                f"Starting GRO restricted-bending group {group_index} has "
                f"{int(np.count_nonzero(unsafe))}/{len(angles)} unsafe angles; "
                f"first offending bead tuple {beads} is {offending_angle}. "
                f"Observed group range: {observed_range}; function 10 requires "
                "every starting angle to be within 10--170 degrees."
            )
