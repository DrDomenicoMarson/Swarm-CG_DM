"""Regression test for mandatory staged PSO cycle carry-over."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from swarmcg.config_types import OptimizationConfig, SwarmConfig
from swarmcg.core.optimization import SwarmOptimizer


def _single_bond_topology():
    return {
        "nb_constraints": 0,
        "nb_bonds": 1,
        "nb_angles": 0,
        "nb_dihedrals": 0,
        "constraint": [],
        "bond": [{"value": 0.3, "fct": 0.0, "func": 1}],
        "angle": [],
        "dihedral": [],
    }


def test_each_cycle_starts_from_previous_cycle_optimum():
    config = SwarmConfig(optimization=OptimizationConfig(exec_mode=2, sim_type="TEST"))
    optimizer = SwarmOptimizer(config)
    optimizer.ns.cg_itp = _single_bond_topology()
    optimizer.ns.opti_itp = _single_bond_topology()
    optimizer.ns.scoring.domains_val = {"constraint": [], "bond": [], "angle": [], "dihedral": []}
    optimizer.ns.scoring.data_BI = {}
    optimizer.ns.scoring.performed_init_BI = {"bond": False, "angle": False, "dihedral": False}
    optimizer.ns.pso.all_best_emd_dist_geoms = {
        "constraints": {}, "bonds": {0: float("nan")}, "angles": {}, "dihedrals": {}
    }
    optimizer.ns.pso.all_best_params_dist_geoms = {
        "constraints": {}, "bonds": {0: {}}, "angles": {}, "dihedrals": {}
    }
    optimizer._calculate_worst_fit_score = MagicMock()
    baselines = []

    def initial_guesses(*args, **kwargs):
        baselines.append(args[3]["bond"][0]["fct"])
        return [[baselines[-1]]]

    results = [
        (SimpleNamespace(X=[1.0]),),
        (SimpleNamespace(X=[2.0]),),
        (SimpleNamespace(X=[3.0]),),
    ]
    pso = MagicMock()
    pso.solve_with_fstpso.side_effect = results
    sim_types = {
        0: {
            "sim_duration": 1.0,
            "prod_nb_frames": 10,
            "val_guess_fact": 1.0,
            "fct_guess_fact": 0.4,
            "max_swarm_iter": 1,
            "max_swarm_iter_without_new_global_best": 1,
        }
    }

    with (
        patch("swarmcg.core.optimization.forcefield.perform_BI"),
        patch(
            "swarmcg.core.optimization.forcefield.get_search_space_boundaries",
            return_value=[[0.0, 10.0]],
        ),
        patch(
            "swarmcg.core.optimization.forcefield.get_initial_guess_list",
            side_effect=initial_guesses,
        ),
        patch("swarmcg.core.optimization.FuzzyPSO", return_value=pso),
    ):
        for cycle_index in range(3):
            optimizer._run_single_cycle(
                cycle_index,
                ["bond"],
                [0, 0, 0],
                sim_types,
                lambda _: 1,
            )

    assert baselines == [0.0, 1.0, 2.0]
    assert optimizer.ns.opti_itp["bond"][0]["fct"] == 3.0
