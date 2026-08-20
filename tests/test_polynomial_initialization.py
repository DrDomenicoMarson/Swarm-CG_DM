"""Tests for broad-then-local polynomial particle initialization."""

from copy import deepcopy

import numpy as np
import pytest

from swarmcg.config_types import SwarmConfig
from swarmcg.optimization_types import OptimizationCycle, ParameterVectorLayout
from swarmcg.particle_initialization import initialize_particles
from swarmcg.simulations.polynomial import CBTParameters, RBParameters
from swarmcg.topology import CGTopology, DihedralGroup


def _polynomial_group(func, coefficients, bound=25.0):
    """Build a minimal RB or CBT topology group."""
    parameters = (
        RBParameters(tuple(coefficients))
        if func == 3
        else CBTParameters(tuple(coefficients))
    )
    return DihedralGroup(
        "1",
        [(0, 1, 2, 3)],
        func,
        parameters,
        parameters,
        average=float("nan"),
        coefficient_bound=bound,
    )


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
    topology = CGTopology(dihedrals=[group])
    cycle = OptimizationCycle.from_topology(2, ["dihedral"], topology)
    best_scores, best_params = _tracking_state()
    ranges = []

    def draw_high(low, high, digits):
        ranges.append((low, high, digits))
        return round(high, digits)

    monkeypatch.setattr("swarmcg.particle_initialization.draw_float", draw_high)

    domains = {"constraint": [], "bond": [], "angle": [], "dihedral": [None]}
    layout = ParameterVectorLayout.build(
        topology, cycle, domains, 1, SwarmConfig()
    )
    particles = initialize_particles(
        4,
        layout,
        topology,
        deepcopy(topology),
        best_scores,
        best_params,
        SwarmConfig(),
        force_guess_factor=0.2,
    )

    assert len(particles) == 4
    assert particles[0] == [0.0] * 5
    assert particles[1:] == [[25.0] * 5] * 3
    assert ranges == [(-25.0, 25.0, 3)] * 15


def test_later_cbt_activation_refines_locally_around_prior_optimum(monkeypatch):
    effective = (10.0,) * 5
    group = _polynomial_group(11, effective)
    topology = CGTopology(dihedrals=[group])
    cycle = OptimizationCycle.from_topology(3, ["dihedral"], topology)
    best_scores, best_params = _tracking_state(
        score=1.0, params=group.gromacs_parameters
    )
    ranges = []

    def draw_midpoint(low, high, digits):
        ranges.append((low, high, digits))
        return round((low + high) / 2.0, digits)

    monkeypatch.setattr(
        "swarmcg.particle_initialization.draw_float", draw_midpoint
    )

    domains = {"constraint": [], "bond": [], "angle": [], "dihedral": [None]}
    layout = ParameterVectorLayout.build(
        topology, cycle, domains, 1, SwarmConfig()
    )
    particles = initialize_particles(
        4,
        layout,
        topology,
        deepcopy(topology),
        best_scores,
        best_params,
        SwarmConfig(),
        force_guess_factor=0.2,
    )

    assert len(particles) == 4
    assert particles[0] == [10.0] * 5
    assert all(particle == [10.0] * 5 for particle in particles)
    assert ranges == [(5.0, 15.0, 3)] * 15
