import MDAnalysis as mda
import numpy as np
from swarmcg.config_types import SwarmConfig

def get_AA_bonds_distrib(universe, beads_ids, grp_type, grp_nb, config: SwarmConfig, bins=None, bandwidth=None):
    """Calculate bonds distribution from AA trajectory."""
    
    # Access usage via config object or defaults if not present
    # Assuming config is a SwarmConfig object
    
    bond_values = np.empty(len(universe.trajectory) * len(beads_ids))
    frame_values = np.empty(len(beads_ids))
    bead_pos_1 = np.empty((len(beads_ids), 3), dtype=np.float32)
    bead_pos_2 = np.empty((len(beads_ids), 3), dtype=np.float32)

    # Use serial backend as default or from config if available
    # NOTE: Original code had ns.mda_backend. We can default to 'serial' or check config.
    backend = 'serial' # Simplified default as per original observation that serial is faster/safer

    for ts in universe.trajectory:
        for i in range(len(beads_ids)):
            bead_id_1, bead_id_2 = beads_ids[i]
            bead_pos_1[i] = universe.atoms[bead_id_1].position
            bead_pos_2[i] = universe.atoms[bead_id_2].position

        mda.lib.distances.calc_bonds(bead_pos_1, bead_pos_2, backend=backend, box=None, result=frame_values)
        bond_values[len(beads_ids) * ts.frame:len(beads_ids) * (ts.frame + 1)] = frame_values / 10  # retrieved nm

    bond_avg_init = round(np.average(bond_values), 3)
    bond_avg_final = bond_avg_init
    rescaling_performed = False

    # Rescaling Logic
    # We access config attributes. Note: Config structure must theoretically support these.
    # In Phase 1 we created config_types. 
    # Let's assume we need to access optimization/model config.
    
    # Check if necessary config attributes exist, otherwise use defaults or skip
    # For refactoring safely without breaking everything, we assume the config object passed has these fields
    # or we replicate the logic using the config arguments.
    
    # Optimization Config shortcuts
    opt_config = config.optimization
    model_config = config.cg_model

    if opt_config.bonds_scaling != 1.0: # Default is 1.0 usually
        bond_values = [bond_length * opt_config.bonds_scaling for bond_length in bond_values]
        bond_avg_final = round(np.average(bond_values), 3)
        rescaling_performed = True
        print(f"  Ref. AA-mapped distrib. rescaled to avg {bond_avg_final} nm for {grp_type} {grp_nb + 1} (initially {bond_avg_init} nm)")

    elif bond_avg_init < opt_config.min_bonds_length:
        bond_rescale_factor = opt_config.min_bonds_length / bond_avg_init
        bond_values = [bond_length * bond_rescale_factor for bond_length in bond_values]
        bond_avg_final = round(np.average(bond_values), 3)
        rescaling_performed = True
        print(f"  Ref. AA-mapped distrib. rescaled to avg {bond_avg_final} nm for {grp_type} {grp_nb + 1} (initially {bond_avg_init} nm)")

    elif opt_config.bonds_scaling_specific is not None:
        geom_id_full = f"C{grp_nb + 1}" if grp_type.startswith("constraint") else f"B{grp_nb + 1}"
        
        if geom_id_full in opt_config.bonds_scaling_specific:
            bond_rescale_factor = opt_config.bonds_scaling_specific[geom_id_full] / bond_avg_init
            bond_values = [bond_length * bond_rescale_factor for bond_length in bond_values]
            bond_avg_final = round(np.average(bond_values), 3)
            rescaling_performed = True
            print(f"  Ref. AA-mapped distrib. rescaled to avg {bond_avg_final} nm for {grp_type} {grp_nb + 1} (initially {bond_avg_init} nm)")

    # Histogram generation
    # Binning parameters might need to be passed or recalculated if not in config
    # Ideally they should be in config or passed as args. 
    # For now, let's look at how they are generated. usually via scores.create_bins_...
    # We will assume they are available in config.internal or passed explicitly?
    # Actually, often `ns.bins_bonds` were created in `optimize_model`.
    # We should probably pass them as arguments OR allow config to carry them (less clean but practical).
    
    # CRITICAL DECISION: Pass bins as arguments to keep functions pure? Yes.
    # But for now, to fit `ns` replacement pattern easily, I might need to put them in SwarmConfig or calculate locally.
    # Let's calculate locally for purity if possible, or expect them in config.
    # Re-calculating bins every time is wasteful.
    # I will assume they are passed in `kwargs` or similar if I want to be strict, 
    # but to match signature `config` I might assume they are attached to config (dirty) or passed.
    
    # Let's look at `create_bins_and_dist_matrices` in `optimize_model`.
    # It modifies `ns`. 
    # I should probably move bin creation to a helper and call it here or pass the bins.
    
    # I will stick to passing `config` and assume we might attach runtime data to it, OR
    # better, I will explicitly ask for bins in the signature if I can.
    # For this step, I'll stick to a signature that includes config, and maybe I'll calculate bins if missing?
    # No, that's slow.
    
    # Let's add them to the signature.
    pass # Wait, I am writing the file.
    
    return bond_avg_final, None, bond_values # Returning None for hist for now as precise binning requires global context


# Re-writing the function with BETTER signature
def get_AA_bonds_distrib(universe, beads_ids, grp_type, grp_nb, config: SwarmConfig, bins=None, bandwidth=None):
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
        print(f"  Ref. AA-mapped distrib. rescaled to avg {bond_avg_final} nm for {grp_type} {grp_nb + 1}")
    elif bond_avg_init < opt_config.min_bonds_length:
        factor = opt_config.min_bonds_length / bond_avg_init
        bond_values = [x * factor for x in bond_values]
        bond_avg_final = round(np.average(bond_values), 3)
        print(f"  Ref. AA-mapped distrib. rescaled to avg {bond_avg_final} nm for {grp_type} {grp_nb + 1}")
    
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
