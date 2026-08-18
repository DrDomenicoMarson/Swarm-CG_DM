"""Scientific preflight validation for simulation input coordinates."""

from __future__ import annotations

import MDAnalysis as mda
import numpy as np

from swarmcg.shared import exceptions


def validate_restricted_bending_start(gro_filename: str, cg_itp) -> None:
    """Validate the modeled molecule and every starting ReB angle.

    The optimizer and scoring pipeline both operate on the first
    ``len(cg_itp["atoms"])`` atoms of the CG system. This preflight applies the
    same ordering contract to the starting GRO and evaluates function-10
    angles using minimum-image vectors when a valid periodic box is present.

    Args:
        gro_filename: Starting GROMACS coordinate path.
        cg_itp: Parsed coarse-grained topology.

    Raises:
        ScientificValidationError: If the GRO contains too few atoms,
            non-finite modeled coordinates, or a function-10 angle outside
            the inclusive 10--170 degree safety interval.
        OSError: If the coordinate file cannot be opened.
    """
    universe = mda.Universe(gro_filename, guess_bonds=False)
    modeled_atom_count = len(cg_itp["atoms"])
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

    for group_index, group in enumerate(cg_itp["angle"], start=1):
        if group["func"] != 10:
            continue
        bead_tuples = np.asarray(group["beads"], dtype=int)
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
