import numpy as np


def _defatul_particle_setter(search_space_size):
    """Return the dimension-aware production swarm size.

    Args:
        search_space_size: Search-space bounds, one pair per free parameter.

    Returns:
        ``max(3, round(2 + sqrt(D)))`` where ``D`` is the free dimension.
    """
    return max(int(round(2 + np.sqrt(len(search_space_size)))), 3)


def get_settings(ns):
    """Return simulation-cycle settings for a configured strategy.

    Args:
        ns: Optimization context containing the normalized strategy name and
            modern simulation durations.

    Returns:
        Simulation types, geometry cycles, simulation-cycle indices, and a
        dimension-aware particle-count function.

    Raises:
        ValueError: If the strategy is not ``OPTIMAL``, ``FAST``, or ``TEST``.
    """
    sim_type = ns.config.optimization.sim_type
    
    if sim_type == "OPTIMAL":
        return _optimal(ns)
    elif sim_type == "FAST":
        return _fast(ns)
    elif sim_type == "TEST":
        return _test(ns)
    else:
        msg = f"Simulation type {sim_type} is not valid. (OPTIMAL, FAST, TEST)"
        raise ValueError(msg)


def _optimal(ns):
    """OPTIMAL Simulation strategy.

    Should be fine with any type of molecule, big or small,
    as long as the BI keeps yielding close enough results, which should be the case

    ns requires:
        cg_itp
        config.simulation.sim_duration_short
        config.simulation.sim_duration_long
    """
    sim_short = ns.config.simulation.sim_duration_short
    sim_long = ns.config.simulation.sim_duration_long
    
    sim_types = {
        0: {"sim_duration": sim_short,
            "prod_nb_frames": 15000,
            "max_swarm_iter": None,
            "max_swarm_iter_without_new_global_best": 6,
            "val_guess_fact": 1,
            "fct_guess_fact": 0.40},
        1: {"sim_duration": sim_short,
            "prod_nb_frames": 15000,
            "max_swarm_iter": None,
            "max_swarm_iter_without_new_global_best": 6,
            "val_guess_fact": 0.25,
            "fct_guess_fact": 0.30},
        2: {"sim_duration": sim_long,
            "prod_nb_frames": 15000,
            "max_swarm_iter": None,
            "max_swarm_iter_without_new_global_best": 6,
            "val_guess_fact": 0.25,
            "fct_guess_fact": 0.20}
    }
    opti_cycles = [["constraint", "bond", "angle"], ["angle", "dihedral"], ["constraint", "bond", "angle", "dihedral"]]
    sim_cycles = [0, 1, 2]  # simulations types
    return sim_types, opti_cycles, sim_cycles, _defatul_particle_setter


def _fast(ns):
    """ Simulation strategy FAST - Suited for small molecules or rapid optimization"""
    sim_types = {
        0: {"sim_duration": 10, "prod_nb_frames": 5000, "max_swarm_iter": 10,
            "max_swarm_iter_without_new_global_best": 5, "val_guess_fact": 1, "fct_guess_fact": 0.40},
        1: {"sim_duration": 10, "prod_nb_frames": 5000, "max_swarm_iter": 10,
            "max_swarm_iter_without_new_global_best": 5, "val_guess_fact": 0.25, "fct_guess_fact": 0.30},
        2: {"sim_duration": 15, "prod_nb_frames": 5000, "max_swarm_iter": 15,
            "max_swarm_iter_without_new_global_best": 5, "val_guess_fact": 0.25, "fct_guess_fact": 0.25}
    }
    opti_cycles = [["constraint", "bond", "angle"], ["dihedral"], ["constraint", "bond", "angle", "dihedral"]]
    sim_cycles = [0, 1, 2]  # simulations types
    return sim_types, opti_cycles, sim_cycles, _defatul_particle_setter


def _test(ns):
    """ Simulation strategy TEST - Suited for test"""
    sim_types = {
        0: {"sim_duration": 0.5, "prod_nb_frames": 500, "max_swarm_iter": 1,
            "max_swarm_iter_without_new_global_best": 1, "val_guess_fact": 1, "fct_guess_fact": 0.40},
        1: {"sim_duration": 0.5, "prod_nb_frames": 500, "max_swarm_iter": 1,
            "max_swarm_iter_without_new_global_best": 1, "val_guess_fact": 0.25, "fct_guess_fact": 0.30},
        2: {"sim_duration": 0.5, "prod_nb_frames": 500, "max_swarm_iter": 1,
            "max_swarm_iter_without_new_global_best": 1, "val_guess_fact": 0.25, "fct_guess_fact": 0.25}
    }
    opti_cycles = [["constraint", "bond", "angle"], ["dihedral"], ["constraint", "bond", "angle", "dihedral"]]
    sim_cycles = [0, 1, 2]  # simulations types
    return sim_types, opti_cycles, sim_cycles, lambda search_space_size: 2
