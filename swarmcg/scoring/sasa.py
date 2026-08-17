"""Optional solvent-accessible surface-area diagnostics."""

from __future__ import annotations

import os

import MDAnalysis as mda
import numpy as np

from swarmcg.context import OptimizationContext
from swarmcg.io import read_xvg_col
from swarmcg.shared import exceptions
from swarmcg.simulations.runner import exec_gmx


PROBE_RADIUS = 0.26  # nm


def _write_real_bead_index(path: str, real_bead_ids: list[int]) -> None:
    """Write a GROMACS index containing the actual real-bead atom IDs.

    Args:
        path: Output index path.
        real_bead_ids: Zero-based atom IDs from the parsed CG topology.
    """
    with open(path, "w") as handle:
        handle.write("[ real_beads ]\n")
        handle.write(" ".join(str(bead_id + 1) for bead_id in real_bead_ids))
        handle.write("\n")


def _write_selected_trajectory(
    universe: mda.Universe,
    atom_ids: list[int],
    output_path: str,
    dimensions_source: mda.Universe | None = None,
) -> None:
    """Write selected coordinates from an existing mapped trajectory.

    Args:
        universe: Source trajectory universe.
        atom_ids: Zero-based atom IDs to write in the requested order.
        output_path: Output XTC path.
        dimensions_source: Optional aligned trajectory supplying periodic box
            dimensions when the in-memory mapped universe has none.
    """
    selection = universe.atoms[atom_ids]
    with mda.Writer(output_path, n_atoms=len(selection)) as writer:
        for frame_index, _ in enumerate(universe.trajectory):
            if dimensions_source is not None:
                dimensions_source.trajectory[frame_index]
                universe.trajectory.ts.dimensions = dimensions_source.trajectory.ts.dimensions
            writer.write(selection)


def _require_success(command: list[str], *, stdin_text: str, cwd: str, label: str) -> None:
    """Run a GROMACS diagnostic command and raise on failure.

    Args:
        command: GROMACS argument list.
        stdin_text: Interactive selections supplied through standard input.
        cwd: Command working directory.
        label: Human-readable operation name.

    Raises:
        ComputationError: If the command exits unsuccessfully.
    """
    if exec_gmx(command, stdin_text=stdin_text, cwd=cwd) != 0:
        raise exceptions.ComputationError(f"GROMACS failed while {label} for optional SASA analysis.")


def compute_SASA(context: OptimizationContext, traj_type: str):
    """Compute an optional real-bead SASA diagnostic.

    AA-mapped coordinates are written from Swarm-CG's already mapped
    trajectory. This preserves the requested COM/COG convention and split-atom
    weights instead of asking GROMACS to remap the atomistic trajectory.

    Args:
        context: Initialized optimization/evaluation context.
        traj_type: ``"AA_mapped"`` or ``"CG"``.

    Returns:
        Pair ``(mean, standard_deviation)`` in nm squared.

    Raises:
        InvalidArgument: If *traj_type* is unsupported.
        ComputationError: If CG topology data is absent, no real beads exist,
            a GROMACS command fails, or no SASA data is produced.
    """
    ns = context
    if traj_type not in {"AA_mapped", "CG"}:
        raise exceptions.InvalidArgument(
            "SASA is available only for AA_mapped and CG trajectories."
        )

    cg_tpr = os.path.abspath(ns.files.cg_tpr_filename)
    if not os.path.isfile(cg_tpr):
        raise exceptions.ComputationError(
            "Optional SASA requires a CG TPR topology; no readable CG topology data was found."
        )
    real_bead_ids = list(ns.cg_itp["real_beads_ids"])
    if not real_bead_ids:
        raise exceptions.ComputationError("Optional SASA requires at least one real CG bead.")

    working_dir = os.path.dirname(os.path.abspath(ns.files.plot_filename))
    os.makedirs(working_dir, exist_ok=True)
    prefix = "aa_mapped" if traj_type == "AA_mapped" else "cg"
    index_path = os.path.join(working_dir, f"{prefix}_sasa_real_beads.ndx")
    topology_path = os.path.join(working_dir, f"{prefix}_sasa_real_beads.tpr")
    trajectory_path = os.path.join(working_dir, f"{prefix}_sasa_real_beads.xtc")
    output_path = os.path.join(working_dir, f"{prefix}_sasa.xvg")
    _write_real_bead_index(index_path, real_bead_ids)

    gmx = ns.config.gromacs.gmx_path
    _require_success(
        [gmx, "convert-tpr", "-s", cg_tpr, "-n", index_path, "-o", topology_path],
        stdin_text="0\n",
        cwd=working_dir,
        label="creating the real-bead SASA topology",
    )

    source = ns.scoring.aa2cg_universe if traj_type == "AA_mapped" else ns.scoring.cg_universe
    dimensions_source = ns.scoring.aa_universe if traj_type == "AA_mapped" else None
    _write_selected_trajectory(
        source,
        real_bead_ids,
        trajectory_path,
        dimensions_source=dimensions_source,
    )

    _require_success(
        [
            gmx,
            "sasa",
            "-s",
            topology_path,
            "-f",
            trajectory_path,
            "-surface",
            "0",
            "-output",
            "0",
            "-o",
            output_path,
            "-probe",
            str(PROBE_RADIUS),
        ],
        stdin_text="0\n",
        cwd=working_dir,
        label=f"calculating {traj_type} SASA",
    )
    if not os.path.isfile(output_path):
        raise exceptions.ComputationError(
            f"GROMACS did not create the expected optional SASA output: {output_path}"
        )
    per_frame = np.asarray(read_xvg_col(output_path, 1), dtype=float)
    if per_frame.size == 0 or not np.any(np.isfinite(per_frame)):
        raise exceptions.ComputationError("Optional SASA output contains no finite values.")
    return round(float(np.nanmean(per_frame)), 2), round(float(np.nanstd(per_frame)), 2)
