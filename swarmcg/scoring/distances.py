import numpy as np
from scipy.spatial.distance import cdist


def create_bins_and_dist_matrices(ns, constraints=True):
    """Get bins and distance matrix for pairwise distributions comparison using Earth Mover's
    Distance (EMD).

    ns requires:
        bw_bonds
        bw_angles
        bw_constraints
        bw_dihedrals
        bins_constraints
        bonded_max_range

    ns creates:
        bins_bonds
        bins_angles
        bins_dihedrals
        bins_constraints
        bins_bonds_dist_matrix
        bins_angles_dist_matrix
        bins_dihedrals_dist_matrix
        bins_constraints_dist_matrix
    """
    if constraints:
        ns.scoring.bins_constraints = np.arange(0, ns.config.optimization.bonded_max_range + ns.config.optimization.bw_constraints, ns.config.optimization.bw_constraints)
    ns.scoring.bins_bonds = np.arange(0, ns.config.optimization.bonded_max_range + ns.config.optimization.bw_bonds, ns.config.optimization.bw_bonds)
    ns.scoring.bins_angles = np.arange(0, 180 + 2 * ns.config.optimization.bw_angles,
                               ns.config.optimization.bw_angles)  # one more bin for angle/dihedral because we are later using a strict inferior for bins definitions
    ns.scoring.bins_dihedrals = np.arange(-180, 180 + 2 * ns.config.optimization.bw_dihedrals, ns.config.optimization.bw_dihedrals)

    # bins distance for Earth Mover's Distance (EMD) to calculate histograms similarity
    if constraints:
        bins_constraints_reshape = np.array(ns.scoring.bins_constraints).reshape(-1, 1)
        ns.scoring.bins_constraints_dist_matrix = cdist(bins_constraints_reshape, bins_constraints_reshape)
    bins_bonds_reshape = np.array(ns.scoring.bins_bonds).reshape(-1, 1)
    ns.scoring.bins_bonds_dist_matrix = cdist(bins_bonds_reshape, bins_bonds_reshape)
    bins_angles_reshape = np.array(ns.scoring.bins_angles).reshape(-1, 1)
    ns.scoring.bins_angles_dist_matrix = cdist(bins_angles_reshape, bins_angles_reshape)
    bins_dihedrals_reshape = np.array(ns.scoring.bins_dihedrals).reshape(-1, 1)
    bins_dihedrals_dist_matrix = cdist(bins_dihedrals_reshape, bins_dihedrals_reshape)  # 'classical' distance matrix
    ns.scoring.bins_dihedrals_dist_matrix = np.where(bins_dihedrals_dist_matrix > max(bins_dihedrals_dist_matrix[0]) / 2,
                                             max(bins_dihedrals_dist_matrix[0]) - bins_dihedrals_dist_matrix,
                                             bins_dihedrals_dist_matrix)  # periodic distance matrix
