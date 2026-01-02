import numpy as np

from swarmcg.context import SwarmCGArgs, SwarmCGState


def _defatul_particle_setter(search_space_size):
    """Function to determined the number of particles"""
    return max(int(round(2 + np.sqrt(len(search_space_size)))), 3)


def get_settings(args: SwarmCGArgs, state: SwarmCGState):
    """Get simulation and optimzation settings.

    args requires:
        runtime.sim_type

    pass args/state to:
        _optimal
        _fast
        _test
    """
    if args.runtime.sim_type == "OPTIMAL":
        return _optimal(args, state)
    elif args.runtime.sim_type == "FAST":
        return _fast(args, state)
    elif args.runtime.sim_type == "TEST":
        return _test(args, state)
    else:
        msg = f"Simulation type {args.runtime.sim_type} is not valid. (OPTIMAL, FAST, TEST)"
        raise ValueError(msg)


def _optimal(args: SwarmCGArgs, state: SwarmCGState):
    """OPTIMAL Simulation strategy.

    Should be fine with any type of molecule, big or small,
    as long as the BI keeps yielding close enough results, which should be the case

    args requires:
        optimization.sim_duration_short
        optimization.sim_duration_long

    state requires:
        model.cg_itp
    """
    sim_types = {
        0: {"sim_duration": args.optimization.sim_duration_short,
            "prod_nb_frames": 15000,
            "max_swarm_iter": int(
                round(6 + np.sqrt(state.model.cg_itp["nb_constraints"] + state.model.cg_itp["nb_bonds"] + state.model.cg_itp["nb_angles"]))),
            "max_swarm_iter_without_new_global_best": 6,
            "val_guess_fact": 1,
            "fct_guess_fact": 0.40},
        1: {"sim_duration": args.optimization.sim_duration_short,
            "prod_nb_frames": 15000,
            "max_swarm_iter": int(round(6 + np.sqrt(state.model.cg_itp["nb_angles"] + state.model.cg_itp["nb_dihedrals"]))),
            "max_swarm_iter_without_new_global_best": 6,
            "val_guess_fact": 0.25,
            "fct_guess_fact": 0.30},
        2: {"sim_duration": args.optimization.sim_duration_long,
            "prod_nb_frames": 15000,
            "max_swarm_iter": int(round(6 + np.sqrt(
                state.model.cg_itp["nb_constraints"] + state.model.cg_itp["nb_bonds"] + state.model.cg_itp["nb_angles"] + state.model.cg_itp[
                    "nb_dihedrals"]))),
            "max_swarm_iter_without_new_global_best": 6,
            "val_guess_fact": 0.25,
            "fct_guess_fact": 0.20}
    }
    opti_cycles = [["constraint", "bond", "angle"], ["angle", "dihedral"], ["constraint", "bond", "angle", "dihedral"]]
    sim_cycles = [0, 1, 2]  # simulations types
    return sim_types, opti_cycles, sim_cycles, _defatul_particle_setter


def _fast(args: SwarmCGArgs, state: SwarmCGState):
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


def _test(args: SwarmCGArgs, state: SwarmCGState):
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
