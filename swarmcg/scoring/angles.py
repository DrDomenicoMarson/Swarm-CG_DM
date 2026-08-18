import MDAnalysis as mda
import numpy as np

from swarmcg.scoring.distances import observe_histogram, require_complete_reference
from swarmcg.shared.logging_utils import get_logger

logger = get_logger(__name__)


def get_AA_angles_distrib(universe, beads_ids, bins=None, bandwidth=None, group_label="angle group"):
    """Calculate an AA-mapped angle mean and normalized distribution.

    Args:
        universe: MDAnalysis universe containing the mapped AA trajectory.
        beads_ids: Triplets of zero-based bead indices.
        bins: Optional histogram edges in degrees.
        bandwidth: Retained for API compatibility; normalization uses counts.
        group_label: Human-readable group label for validation diagnostics.

    Returns:
        Mean angle, probability masses, degree values, and radian values.
    """
    angle_values_rad = np.empty(len(universe.trajectory) * len(beads_ids))
    frame_values = np.empty(len(beads_ids))
    bead_pos_1 = np.empty((len(beads_ids), 3), dtype=np.float32)
    bead_pos_2 = np.empty((len(beads_ids), 3), dtype=np.float32)
    bead_pos_3 = np.empty((len(beads_ids), 3), dtype=np.float32)

    # Indices
    indices = np.array(beads_ids)
    idx1 = indices[:, 0]
    idx2 = indices[:, 1]
    idx3 = indices[:, 2]
    
    ag1 = universe.atoms[idx1]
    ag2 = universe.atoms[idx2]
    ag3 = universe.atoms[idx3]

    for ts in universe.trajectory:
        mda.lib.distances.calc_angles(ag1.positions, ag2.positions, ag3.positions, backend='serial', box=None,
                                      result=frame_values)
        angle_values_rad[len(beads_ids) * ts.frame:len(beads_ids) * (ts.frame + 1)] = frame_values

    angle_values_deg = np.rad2deg(angle_values_rad)
    angle_avg = round(np.mean(angle_values_deg), 3)
    
    angle_hist = None
    if bins is not None and bandwidth is not None:
        observation = observe_histogram(angle_values_deg, bins)
        require_complete_reference(
            observation, angle_values_deg, group_label, "degrees"
        )
        angle_hist = observation.probabilities

    return angle_avg, angle_hist, angle_values_deg, angle_values_rad


def get_CG_angles_distrib(universe, beads_ids, bins=None, bandwidth=None, group_label="angle group"):
    """Calculate a CG angle mean and normalized distribution.

    Args:
        universe: MDAnalysis universe containing the CG trajectory.
        beads_ids: Triplets of zero-based bead indices.
        bins: Optional histogram edges in degrees.
        bandwidth: Retained for API compatibility; normalization uses counts.
        group_label: Human-readable group label for coverage diagnostics.

    Returns:
        Mean angle, probability masses, degree values, and radian values.
    """
    angle_values_rad = np.empty(len(universe.trajectory) * len(beads_ids))
    frame_values = np.empty(len(beads_ids))
    bead_pos_1 = np.empty((len(beads_ids), 3), dtype=np.float32)
    bead_pos_2 = np.empty((len(beads_ids), 3), dtype=np.float32)
    bead_pos_3 = np.empty((len(beads_ids), 3), dtype=np.float32)

    # Indices
    indices = np.array(beads_ids)
    idx1 = indices[:, 0]
    idx2 = indices[:, 1]
    idx3 = indices[:, 2]
    
    ag1 = universe.atoms[idx1]
    ag2 = universe.atoms[idx2]
    ag3 = universe.atoms[idx3]

    for ts in universe.trajectory:  # no need for PBC handling
        mda.lib.distances.calc_angles(ag1.positions, ag2.positions, ag3.positions, backend='serial', box=None,
                                      result=frame_values)
        angle_values_rad[len(beads_ids) * ts.frame:len(beads_ids) * (ts.frame + 1)] = frame_values

    angle_values_deg = np.rad2deg(angle_values_rad)

    # get group average and histogram non-null values for comparison and display
    finite_values = angle_values_deg[np.isfinite(angle_values_deg)]
    angle_avg = round(float(np.mean(finite_values)), 3) if finite_values.size else float("nan")
    
    angle_hist = None
    if bins is not None and bandwidth is not None:
        observation = observe_histogram(angle_values_deg, bins)
        angle_hist = observation.probabilities
        if observation.missing_count:
            logger.warning(
                "CG %s distribution has missing mass charged at maximum EMD cost: %s",
                group_label,
                observation.coverage_message(),
            )

    return angle_avg, angle_hist, angle_values_deg, angle_values_rad
