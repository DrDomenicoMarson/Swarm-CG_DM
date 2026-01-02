import numpy as np

from swarmcg.context import SwarmCGArgs, SwarmCGState


def compute_Rg(args: SwarmCGArgs, state: SwarmCGState, traj_type):
    """Compute average radius of gyration.

    args/state requires:
        aa_universe
        aa2cg_universe
        cg_universe
        mda_backend

    state creates:
        gyr_aa
        gyr_aa_std
        gyr_aa_mapped
        gyr_aa_mapped_std
        gyr_cg
        gyr_cg_std
    """
    if traj_type == "AA":

        gyr_aa = np.empty(len(state.traj.aa_universe.trajectory))
        for ts in state.traj.aa_universe.trajectory:
            gyr_aa[ts.frame] = state.traj.aa_universe.atoms[:len(state.mapping.all_atoms)].radius_of_gyration(pbc=None,
                                                                                           backend=state.runtime.mda_backend)
        state.model.gyr_aa = round(np.average(gyr_aa) / 10, 3)  # retrieve nm
        state.model.gyr_aa_std = round(np.std(gyr_aa) / 10, 3)  # retrieve nm

    elif traj_type == "AA_mapped":

        gyr_aa_mapped = np.empty(len(state.traj.aa_universe.trajectory))
        for ts in state.traj.aa2cg_universe.trajectory:
            gyr_aa_mapped[ts.frame] = state.traj.aa2cg_universe.atoms[:len(state.model.cg_itp["atoms"])].radius_of_gyration(pbc=None,
                                                                                                           backend=state.runtime.mda_backend)
        state.model.gyr_aa_mapped = round(np.average(gyr_aa_mapped) / 10 + args.optimization.aa_rg_offset, 3)  # retrieve nm
        state.model.gyr_aa_mapped_std = round(np.std(gyr_aa_mapped) / 10, 3)  # retrieve nm

    elif traj_type == "CG":

        gyr_cg = np.empty(len(state.traj.cg_universe.trajectory))
        for ts in state.traj.cg_universe.trajectory:
            gyr_cg[ts.frame] = state.traj.cg_universe.atoms[:len(state.model.cg_itp["atoms"])].radius_of_gyration(pbc=None,
                                                                                                 backend=state.runtime.mda_backend)
        state.model.gyr_cg = round(np.average(gyr_cg) / 10, 3)  # retrieve nm
        state.model.gyr_cg_std = round(np.std(gyr_cg) / 10, 3)  # retrieve nm

    else:
        raise RuntimeError("Unexpected error in function: compute_Rg")
