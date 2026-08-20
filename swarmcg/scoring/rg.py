import numpy as np

def compute_Rg(universe, atom_selection, backend='serial', offset=0.0):
    """Compute average radius of gyration.
    Returns: (avg_rg, std_rg) in nm
    """
    gyr_values = np.empty(len(universe.trajectory))
    
    # Pre-calculate selection indices to avoid repeated parsing if possible, 
    # but here we pass the AtomGroup or indices directly?
    # The original code used specific slicing:
    # AA: ns.aa_universe.atoms[:len(ns.all_atoms)]
    
    # So the caller should pass the *AtomGroup* or sliced atoms object.
    
    for ts in universe.trajectory:
        gyr_values[ts.frame] = atom_selection.radius_of_gyration(pbc=None, backend=backend)
        
    avg_rg = round(np.average(gyr_values) / 10 + offset, 3) # retrieve nm
    std_rg = round(np.std(gyr_values) / 10, 3)
    
    return avg_rg, std_rg
