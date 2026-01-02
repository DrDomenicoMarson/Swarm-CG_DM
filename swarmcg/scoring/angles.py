import MDAnalysis as mda
import numpy as np

from swarmcg.context import SwarmCGArgs, SwarmCGState


def get_AA_angles_distrib(args: SwarmCGArgs, state: SwarmCGState, beads_ids):
    """Calculate angles distribution from AA trajectory.

    args/state requires:
        aa2cg_universe
        mda_backend
        bw_angles
    """
    angle_values_rad = np.empty(len(state.traj.aa2cg_universe.trajectory) * len(beads_ids))
    frame_values = np.empty(len(beads_ids))
    bead_pos_1 = np.empty((len(beads_ids), 3), dtype=np.float32)
    bead_pos_2 = np.empty((len(beads_ids), 3), dtype=np.float32)
    bead_pos_3 = np.empty((len(beads_ids), 3), dtype=np.float32)

    for ts in state.traj.aa2cg_universe.trajectory:
        for i in range(len(beads_ids)):
            bead_id_1, bead_id_2, bead_id_3 = beads_ids[i]
            bead_pos_1[i] = state.traj.aa2cg_universe.atoms[bead_id_1].position
            bead_pos_2[i] = state.traj.aa2cg_universe.atoms[bead_id_2].position
            bead_pos_3[i] = state.traj.aa2cg_universe.atoms[bead_id_3].position

        mda.lib.distances.calc_angles(bead_pos_1, bead_pos_2, bead_pos_3, backend=state.runtime.mda_backend, box=None,
                                      result=frame_values)
        angle_values_rad[len(beads_ids) * ts.frame:len(beads_ids) * (ts.frame + 1)] = frame_values

    angle_values_deg = np.rad2deg(angle_values_rad)
    angle_avg = round(np.mean(angle_values_deg), 3)
    angle_hist = np.histogram(angle_values_deg, state.bins.bins_angles, density=True)[
                     0] * args.optimization.bw_angles  # retrieve 1-sum densities

    return angle_avg, angle_hist, angle_values_deg, angle_values_rad


def get_CG_angles_distrib(args: SwarmCGArgs, state: SwarmCGState, beads_ids):
    """Calculate angles using MDAnalysis.

    args/state requires:
        cg_universe
        mda_backend
        bw_angles
    """
    angle_values_rad = np.empty(len(state.traj.cg_universe.trajectory) * len(beads_ids))
    frame_values = np.empty(len(beads_ids))
    bead_pos_1 = np.empty((len(beads_ids), 3), dtype=np.float32)
    bead_pos_2 = np.empty((len(beads_ids), 3), dtype=np.float32)
    bead_pos_3 = np.empty((len(beads_ids), 3), dtype=np.float32)

    for ts in state.traj.cg_universe.trajectory:  # no need for PBC handling, trajectories were made wholes for the molecule
        for i in range(len(beads_ids)):
            bead_id_1, bead_id_2, bead_id_3 = beads_ids[i]
            bead_pos_1[i] = state.traj.cg_universe.atoms[bead_id_1].position
            bead_pos_2[i] = state.traj.cg_universe.atoms[bead_id_2].position
            bead_pos_3[i] = state.traj.cg_universe.atoms[bead_id_3].position

        mda.lib.distances.calc_angles(bead_pos_1, bead_pos_2, bead_pos_3, backend=state.runtime.mda_backend, box=None,
                                      result=frame_values)
        angle_values_rad[len(beads_ids) * ts.frame:len(beads_ids) * (ts.frame + 1)] = frame_values

    angle_values_deg = np.rad2deg(angle_values_rad)

    # get group average and histogram non-null values for comparison and display
    angle_avg = round(np.mean(angle_values_deg), 3)
    angle_hist = np.histogram(angle_values_deg, state.bins.bins_angles, density=True)[
                     0] * args.optimization.bw_angles  # retrieve 1-sum densities

    return angle_avg, angle_hist, angle_values_deg, angle_values_rad
