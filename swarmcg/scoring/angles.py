import MDAnalysis as mda
import numpy as np


def get_AA_angles_distrib(universe, beads_ids, bins=None, bandwidth=None):
    """Calculate angles distribution from AA trajectory."""
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
        angle_hist = np.histogram(angle_values_deg, bins, density=True)[0] * bandwidth

    return angle_avg, angle_hist, angle_values_deg, angle_values_rad


def get_CG_angles_distrib(universe, beads_ids, bins=None, bandwidth=None):
    """Calculate angles using MDAnalysis."""
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
    angle_avg = round(np.mean(angle_values_deg), 3)
    
    angle_hist = None
    if bins is not None and bandwidth is not None:
        angle_hist = np.histogram(angle_values_deg, bins, density=True)[0] * bandwidth

    return angle_avg, angle_hist, angle_values_deg, angle_values_rad
