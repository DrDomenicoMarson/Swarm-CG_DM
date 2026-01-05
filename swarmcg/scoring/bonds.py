import MDAnalysis as mda
import numpy as np
from swarmcg.config_types import SwarmConfig
from swarmcg.shared.logging_utils import get_logger

logger = get_logger(__name__)



# Re-writing the function with BETTER signature
def get_AA_bonds_distrib(universe, beads_ids, grp_type, grp_nb, config: SwarmConfig, bins=None, bandwidth=None, bonds_scaling_specific=None):
    """
    Calculate bonds distribution from AA trajectory.
    Returns: avg, hist, values
    """
    bond_values = np.empty(len(universe.trajectory) * len(beads_ids))
    frame_values = np.empty(len(beads_ids))
    bead_pos_1 = np.empty((len(beads_ids), 3), dtype=np.float32)
    bead_pos_2 = np.empty((len(beads_ids), 3), dtype=np.float32)

    # Pre-calculate indices
    beads_ids_arr = np.array(beads_ids)
    idx1 = beads_ids_arr[:, 0]
    idx2 = beads_ids_arr[:, 1]
    
    # Pre-fetch AtomGroups to avoid repeated indexing if allowed, but universe.atoms[idx] is fast enough
    # Actually, retrieving atoms by index repeatedly in a loop is okay-ish, but pre-selecting might be better.
    # ag1 = universe.atoms[idx1] 
    # ag2 = universe.atoms[idx2]
    # But positions update when TS updates.
    
    ag1 = universe.atoms[idx1]
    ag2 = universe.atoms[idx2]

    for ts in universe.trajectory:
        # Direct vectorized access to positions
        # calc_bonds expects (N, 3) arrays
        mda.lib.distances.calc_bonds(ag1.positions, ag2.positions, backend='serial', box=None, result=frame_values)
        bond_values[len(beads_ids) * ts.frame:len(beads_ids) * (ts.frame + 1)] = frame_values / 10

    bond_avg_init = round(np.average(bond_values), 3)
    bond_avg_final = bond_avg_init
    
    opt_config = config.optimization

    # Rescaling
    if opt_config.bonds_scaling != 1.0:
        bond_values = [x * opt_config.bonds_scaling for x in bond_values]
        bond_avg_final = round(np.average(bond_values), 3)
        logger.info(
            "  Ref. AA-mapped distrib. rescaled to avg %s nm for %s %s",
            bond_avg_final,
            grp_type,
            grp_nb + 1,
        )
    elif bond_avg_init < opt_config.min_bonds_length:
        factor = opt_config.min_bonds_length / bond_avg_init
        bond_values = [x * factor for x in bond_values]
        bond_avg_final = round(np.average(bond_values), 3)
        logger.info(
            "  Ref. AA-mapped distrib. rescaled to avg %s nm for %s %s",
            bond_avg_final,
            grp_type,
            grp_nb + 1,
        )
    elif bonds_scaling_specific is not None:
        geom_id_full = f"C{grp_nb + 1}" if grp_type.startswith("constraint") else f"B{grp_nb + 1}"
        if geom_id_full in bonds_scaling_specific:
            bond_rescale_factor = bonds_scaling_specific[geom_id_full] / bond_avg_init
            bond_values = [x * bond_rescale_factor for x in bond_values]
            bond_avg_final = round(np.average(bond_values), 3)
            logger.info(
                "  Ref. AA-mapped distrib. rescaled to avg %s nm for %s %s",
                bond_avg_final,
                grp_type,
                grp_nb + 1,
            )
    
    # Binning
    bond_hist = None
    if bins is not None and bandwidth is not None:
         bond_hist = np.histogram(bond_values, bins, density=True)[0] * bandwidth

    return bond_avg_final, bond_hist, bond_values


def get_CG_bonds_distrib(universe, beads_ids, grp_type, bins=None, bandwidth=None):
    """Calculate bonds distribution from CG trajectory."""
    bond_values = np.empty(len(universe.trajectory) * len(beads_ids))
    frame_values = np.empty(len(beads_ids))
    bead_pos_1 = np.empty((len(beads_ids), 3), dtype=np.float32)
    bead_pos_2 = np.empty((len(beads_ids), 3), dtype=np.float32)

    # Pre-calculate indices
    beads_ids_arr = np.array(beads_ids)
    idx1 = beads_ids_arr[:, 0]
    idx2 = beads_ids_arr[:, 1]
    
    ag1 = universe.atoms[idx1]
    ag2 = universe.atoms[idx2]

    for ts in universe.trajectory:
        mda.lib.distances.calc_bonds(ag1.positions, ag2.positions, backend='serial', box=None, result=frame_values)
        bond_values[len(beads_ids) * ts.frame:len(beads_ids) * (ts.frame + 1)] = frame_values / 10

    bond_avg = round(np.mean(bond_values), 3)
    
    bond_hist = None
    if bins is not None and bandwidth is not None:
        bond_hist = np.histogram(bond_values, bins, density=True)[0] * bandwidth

    return bond_avg, bond_hist, bond_values
