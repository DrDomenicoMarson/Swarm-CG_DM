"""Tests for broad-then-local polynomial particle initialization."""

from copy import deepcopy

import numpy as np
import pytest

from swarmcg.config_types import SwarmConfig
from swarmcg.forcefield import get_initial_guess_list
from swarmcg.simulations.polynomial import CBTParameters, RBParameters


def _polynomial_group(func, coefficients, bound=25.0):
    """Build a minimal RB or CBT topology group."""
    if func == 3:
        params = RBParameters(tuple(coefficients)).to_gromacs()
    else:
        params = CBTParameters(tuple(coefficients)).to_gromacs()
    return {
        "func": func,
        "params": list(params),
        "params_user": list(params),
        "value": None,
        "value_user": None,
        "fct": None,
        "fct_user": None,
        "mult": None,
        "avg": float("nan"),
        "coefficient_bound": bound,
    }


def _tracking_state(score=np.nan, params=None):
    """Build independent-best dictionaries for one polynomial group."""
    return (
        {
            "constraints": {},
            "bonds": {},
            "angles": {},
            "dihedrals": {0: score},
        },
        {
            "constraints": {},
            "bonds": {},
            "angles": {},
            "dihedrals": {0: {} if params is None else {"params": list(params)}},
        },
    )


@pytest.mark.parametrize("func", [3, 11])
def test_first_polynomial_activation_uses_one_seed_and_full_bounds(
    monkeypatch, func
):
    group = _polynomial_group(func, (0.0,) * 5)
    topology = {"constraint": [], "bond": [], "angle": [], "dihedral": [group]}
    cycle = {
        "nb_cycle": 2,
        "nb_geoms": {"constraint": 0, "bond": 0, "angle": 0, "dihedral": 1},
    }
    best_scores, best_params = _tracking_state()
    ranges = []

    def draw_high(low, high, digits):
        ranges.append((low, high, digits))
        return round(high, digits)

    monkeypatch.setattr("swarmcg.forcefield.draw_float", draw_high)

    particles = get_initial_guess_list(
        4,
        cycle,
        topology,
        deepcopy(topology),
        {"constraint": [], "bond": [], "angle": [], "dihedral": [None]},
        best_scores,
        best_params,
        1,
        SwarmConfig(),
        fct_guess_fact=0.2,
    )

    assert len(particles) == 4
    assert particles[0] == [0.0] * 5
    assert particles[1:] == [[25.0] * 5] * 3
    assert ranges == [(-25.0, 25.0, 3)] * 15


def test_later_cbt_activation_refines_locally_around_prior_optimum(monkeypatch):
    effective = (10.0,) * 5
    group = _polynomial_group(11, effective)
    topology = {"constraint": [], "bond": [], "angle": [], "dihedral": [group]}
    cycle = {
        "nb_cycle": 3,
        "nb_geoms": {"constraint": 0, "bond": 0, "angle": 0, "dihedral": 1},
    }
    best_scores, best_params = _tracking_state(score=1.0, params=group["params"])
    ranges = []

    def draw_midpoint(low, high, digits):
        ranges.append((low, high, digits))
        return round((low + high) / 2.0, digits)

    monkeypatch.setattr("swarmcg.forcefield.draw_float", draw_midpoint)

    particles = get_initial_guess_list(
        4,
        cycle,
        topology,
        deepcopy(topology),
        {"constraint": [], "bond": [], "angle": [], "dihedral": [None]},
        best_scores,
        best_params,
        1,
        SwarmConfig(),
        fct_guess_fact=0.2,
    )

    assert len(particles) == 4
    assert particles[0] == [10.0] * 5
    assert all(particle == [10.0] * 5 for particle in particles)
    assert ranges == [(5.0, 15.0, 3)] * 15
