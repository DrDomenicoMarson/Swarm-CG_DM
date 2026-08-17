import MDAnalysis as mda
import numpy as np

from swarmcg.scoring.distances import circular_mean_degrees


def get_AA_dihedrals_distrib(universe, beads_ids, bins=None, bandwidth=None):
    """Calculate an AA-mapped circular dihedral distribution.

    Args:
        universe: MDAnalysis universe containing the mapped AA trajectory.
        beads_ids: Quadruplets of zero-based bead indices.
        bins: Optional histogram edges in degrees.
        bandwidth: Retained for API compatibility; normalization uses counts.

    Returns:
        Circular mean, probability masses, degree values, and radian values.
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
    dihedral_avg = round(circular_mean_degrees(dihedral_values_deg), 3)
    
    dihedral_hist = None
    if bins is not None and bandwidth is not None:
        counts = np.histogram(dihedral_values_deg, bins, density=False)[0]
        dihedral_hist = counts.astype(float) / counts.sum() if counts.sum() else np.zeros_like(counts, dtype=float)

    return dihedral_avg, dihedral_hist, dihedral_values_deg, dihedral_values_rad


def get_CG_dihedrals_distrib(universe, beads_ids, bins=None, bandwidth=None):
    """Calculate a CG circular dihedral distribution.

    Args:
        universe: MDAnalysis universe containing the CG trajectory.
        beads_ids: Quadruplets of zero-based bead indices.
        bins: Optional histogram edges in degrees.
        bandwidth: Retained for API compatibility; normalization uses counts.

    Returns:
        Circular mean, probability masses, degree values, and radian values.
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
    dihedral_avg = round(circular_mean_degrees(dihedral_values_deg), 3)
    
    dihedral_hist = None
    if bins is not None and bandwidth is not None:
        counts = np.histogram(dihedral_values_deg, bins, density=False)[0]
        dihedral_hist = counts.astype(float) / counts.sum() if counts.sum() else np.zeros_like(counts, dtype=float)

    return dihedral_avg, dihedral_hist, dihedral_values_deg, dihedral_values_rad
