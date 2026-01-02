import numpy as np
from scipy.spatial.distance import cdist

from swarmcg.context import SwarmCGArgs, SwarmCGState


def create_bins_and_dist_matrices(args: SwarmCGArgs, state: SwarmCGState, constraints: bool = True):
    """Get bins and distance matrix for pairwise distributions comparison using Earth Mover's
    Distance (EMD).

    args/state requires:
        bw_bonds
        bw_angles
        bw_constraints
        bw_dihedrals
        bins_constraints
        bonded_max_range

    state creates:
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
        state.bins_constraints = np.arange(0, args.bonded_max_range + args.bw_constraints, args.bw_constraints)
    state.bins_bonds = np.arange(0, args.bonded_max_range + args.bw_bonds, args.bw_bonds)
    state.bins_angles = np.arange(0, 180 + 2 * args.bw_angles,
                               args.bw_angles)  # one more bin for angle/dihedral because we are later using a strict inferior for bins definitions
    state.bins_dihedrals = np.arange(-180, 180 + 2 * args.bw_dihedrals, args.bw_dihedrals)

    # bins distance for Earth Mover's Distance (EMD) to calculate histograms similarity
    if constraints:
        bins_constraints_reshape = np.array(state.bins_constraints).reshape(-1, 1)
        state.bins_constraints_dist_matrix = cdist(bins_constraints_reshape, bins_constraints_reshape)
    bins_bonds_reshape = np.array(state.bins_bonds).reshape(-1, 1)
    state.bins_bonds_dist_matrix = cdist(bins_bonds_reshape, bins_bonds_reshape)
    bins_angles_reshape = np.array(state.bins_angles).reshape(-1, 1)
    state.bins_angles_dist_matrix = cdist(bins_angles_reshape, bins_angles_reshape)
    bins_dihedrals_reshape = np.array(state.bins_dihedrals).reshape(-1, 1)
    bins_dihedrals_dist_matrix = cdist(bins_dihedrals_reshape, bins_dihedrals_reshape)  # 'classical' distance matrix
    state.bins_dihedrals_dist_matrix = np.where(bins_dihedrals_dist_matrix > max(bins_dihedrals_dist_matrix[0]) / 2,
                                             max(bins_dihedrals_dist_matrix[0]) - bins_dihedrals_dist_matrix,
                                             bins_dihedrals_dist_matrix)  # periodic distance matrix
