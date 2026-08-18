import MDAnalysis as mda
import numpy as np

from swarmcg.shared.periodic import circular_statistics_degrees
from swarmcg.scoring.distances import observe_histogram, require_complete_reference
from swarmcg.shared.logging_utils import get_logger

logger = get_logger(__name__)


def get_AA_dihedrals_distrib(universe, beads_ids, bins=None, bandwidth=None, group_label="dihedral group"):
    """Calculate an AA-mapped circular dihedral distribution.

    Args:
        universe: MDAnalysis universe containing the mapped AA trajectory.
        beads_ids: Quadruplets of zero-based bead indices.
        bins: Optional histogram edges in degrees.
        bandwidth: Retained for API compatibility; normalization uses counts.
        group_label: Human-readable group label for validation diagnostics.

    Returns:
        Circular mean, probability masses, degree values, and radian values.
        The mean is ``NaN`` when the first circular moment is undefined.

    Raises:
        ValueError: If the trajectory contains no finite dihedral angle.
    """
    dihedral_values_rad = np.empty(len(universe.trajectory) * len(beads_ids))
    frame_values = np.empty(len(beads_ids))
    bead_pos_1 = np.empty((len(beads_ids), 3), dtype=np.float32)
    bead_pos_2 = np.empty((len(beads_ids), 3), dtype=np.float32)
    bead_pos_3 = np.empty((len(beads_ids), 3), dtype=np.float32)
    bead_pos_4 = np.empty((len(beads_ids), 3), dtype=np.float32)

    # Indices
    indices = np.array(beads_ids)
    idx1 = indices[:, 0]
    idx2 = indices[:, 1]
    idx3 = indices[:, 2]
    idx4 = indices[:, 3]
    
    ag1 = universe.atoms[idx1]
    ag2 = universe.atoms[idx2]
    ag3 = universe.atoms[idx3]
    ag4 = universe.atoms[idx4]

    for ts in universe.trajectory:
        mda.lib.distances.calc_dihedrals(ag1.positions, ag2.positions, ag3.positions, ag4.positions,
                                         backend='serial', box=None, result=frame_values)
        dihedral_values_rad[
        len(beads_ids) * ts.frame:len(beads_ids) * (ts.frame + 1)] = frame_values

    dihedral_values_deg = np.rad2deg(dihedral_values_rad)
    dihedral_hist = None
    if bins is not None and bandwidth is not None:
        observation = observe_histogram(dihedral_values_deg, bins)
        require_complete_reference(
            observation, dihedral_values_deg, group_label, "degrees"
        )
        dihedral_hist = observation.probabilities

    circular_statistics = circular_statistics_degrees(dihedral_values_deg)
    dihedral_avg = (
        round(circular_statistics.mean_degrees, 3)
        if circular_statistics.mean_degrees is not None
        else float("nan")
    )

    return dihedral_avg, dihedral_hist, dihedral_values_deg, dihedral_values_rad


def get_CG_dihedrals_distrib(universe, beads_ids, bins=None, bandwidth=None, group_label="dihedral group"):
    """Calculate a CG circular dihedral distribution.

    Args:
        universe: MDAnalysis universe containing the CG trajectory.
        beads_ids: Quadruplets of zero-based bead indices.
        bins: Optional histogram edges in degrees.
        bandwidth: Retained for API compatibility; normalization uses counts.
        group_label: Human-readable group label for coverage diagnostics.

    Returns:
        Circular mean, probability masses, degree values, and radian values.
        The mean is ``NaN`` when the first circular moment is undefined.

    Raises:
        ValueError: If the trajectory contains no finite dihedral angle.
    """
    dihedral_values_rad = np.empty(len(universe.trajectory) * len(beads_ids))
    frame_values = np.empty(len(beads_ids))
    bead_pos_1 = np.empty((len(beads_ids), 3), dtype=np.float32)
    bead_pos_2 = np.empty((len(beads_ids), 3), dtype=np.float32)
    bead_pos_3 = np.empty((len(beads_ids), 3), dtype=np.float32)
    bead_pos_4 = np.empty((len(beads_ids), 3), dtype=np.float32)

    # Indices
    indices = np.array(beads_ids)
    idx1 = indices[:, 0]
    idx2 = indices[:, 1]
    idx3 = indices[:, 2]
    idx4 = indices[:, 3]
    
    ag1 = universe.atoms[idx1]
    ag2 = universe.atoms[idx2]
    ag3 = universe.atoms[idx3]
    ag4 = universe.atoms[idx4]

    for ts in universe.trajectory:  # no need for PBC handling
        mda.lib.distances.calc_dihedrals(ag1.positions, ag2.positions, ag3.positions, ag4.positions,
                                         backend='serial', box=None, result=frame_values)
        dihedral_values_rad[
        len(beads_ids) * ts.frame:len(beads_ids) * (ts.frame + 1)] = frame_values

    dihedral_values_deg = np.rad2deg(dihedral_values_rad)

    # get group average and histogram non-null values for comparison and display
    dihedral_hist = None
    if bins is not None and bandwidth is not None:
        observation = observe_histogram(dihedral_values_deg, bins)
        dihedral_hist = observation.probabilities
        if observation.missing_count:
            logger.warning(
                "CG %s distribution has missing mass charged at maximum EMD cost: %s",
                group_label,
                observation.coverage_message(),
            )

    try:
        circular_statistics = circular_statistics_degrees(dihedral_values_deg)
    except ValueError:
        dihedral_avg = float("nan")
    else:
        dihedral_avg = (
            round(circular_statistics.mean_degrees, 3)
            if circular_statistics.mean_degrees is not None
            else float("nan")
        )

    return dihedral_avg, dihedral_hist, dihedral_values_deg, dihedral_values_rad
