"""Regression tests for count-invariant Boltzmann initialization."""

import numpy as np
import pytest

from swarmcg import config
from swarmcg.config_types import SwarmConfig
from swarmcg.forcefield import perform_BI
from swarmcg.simulations.boltzmann import (
    BoltzmannTarget,
    fit_bounded_force_constant,
)
from swarmcg.simulations.polynomial import RBParameters
from swarmcg.simulations.potentials import gmx_dihedrals_func_3


def _target_from_basis(centers, basis, force_constant, temperature=300.0):
    """Return an exact normalized Boltzmann target for a linear basis."""
    probabilities = np.exp(
        -force_constant * np.asarray(basis) / (config.kB * temperature)
    )
    probabilities /= probabilities.sum()
    return BoltzmannTarget(np.asarray(centers), probabilities)


def test_target_is_count_invariant_and_excludes_zero_bins():
    values = np.array([0.1, 0.2, 0.2, 0.3, 0.3, 0.3])
    target = BoltzmannTarget.from_samples(values, bins=5, value_range=(0.0, 0.5))
    repeated = BoltzmannTarget.from_samples(
        np.tile(values, 11), bins=5, value_range=(0.0, 0.5)
    )

    assert np.array_equal(target.centers, repeated.centers)
    assert np.allclose(target.probabilities, repeated.probabilities)
    occupied_centers, pmf = target.pmf_samples(300.0)
    assert occupied_centers.size == np.count_nonzero(target.probabilities)
    assert pmf.size == occupied_centers.size
    assert np.min(pmf) == 0.0


@pytest.mark.parametrize(
    "values",
    [np.array([0.1, np.nan]), np.array([0.1, np.inf]), np.array([0.1, 0.6])],
)
def test_boltzmann_target_rejects_discarded_samples(values):
    with pytest.raises(ValueError, match="cannot be silently discarded"):
        BoltzmannTarget.from_samples(values, bins=5, value_range=(0.0, 0.5))


def test_bounded_linear_fit_recovers_force_constant_with_zero_bins():
    centers = np.linspace(-1.0, 1.0, 21)
    basis = 0.5 * centers**2
    target = _target_from_basis(centers, basis, 12.5)
    probabilities = target.probabilities.copy()
    probabilities[[0, -1]] = 0.0
    probabilities /= probabilities.sum()
    sparse_target = BoltzmannTarget(centers, probabilities)

    result = fit_bounded_force_constant(
        sparse_target, basis, 300.0, lower_bound=0.0, upper_bound=25.0
    )

    assert np.isclose(result.force_constant, 12.5, atol=1e-10)
    assert result.rank == 2


def test_boltzmann_force_fit_is_invariant_to_duplicated_samples():
    values = np.repeat(np.linspace(0.35, 0.65, 31), np.arange(1, 32))
    original = BoltzmannTarget.from_samples(
        values, bins=25, value_range=(0.35, 0.65)
    )
    duplicated = BoltzmannTarget.from_samples(
        np.tile(values, 9), bins=25, value_range=(0.35, 0.65)
    )
    basis = 0.5 * (original.centers - 0.5) ** 2

    first = fit_bounded_force_constant(original, basis, 300.0, 0.0, 18000.0)
    second = fit_bounded_force_constant(
        duplicated, basis, 300.0, 0.0, 18000.0
    )

    assert np.isclose(first.force_constant, second.force_constant)
    assert np.isclose(first.intercept, second.intercept)


def test_boltzmann_force_fit_respects_coefficient_bounds():
    centers = np.linspace(-1.0, 1.0, 41)
    basis = 0.5 * centers**2
    target = _target_from_basis(centers, basis, 50.0)

    result = fit_bounded_force_constant(
        target, basis, 300.0, lower_bound=0.0, upper_bound=25.0
    )

    assert np.isclose(result.force_constant, 25.0)


@pytest.mark.parametrize("exec_mode,equilibrium", [(1, 0.45), (2, 0.55)])
def test_perform_bi_fits_around_the_current_fixed_equilibrium(
    exec_mode, equilibrium
):
    centers = np.linspace(0.3, 0.7, 81)
    basis = 0.5 * (centers - equilibrium) ** 2
    topology = {
        "bond": [{"func": 1, "value": equilibrium, "fct": 10.0}],
        "angle": [],
        "dihedral": [],
    }

    perform_BI(
        topology,
        {"nb_geoms": {"constraint": 0, "bond": 1, "angle": 0, "dihedral": 0}},
        {
            "bond": [_target_from_basis(centers, basis, 900.0)],
            "angle": [],
            "dihedral": [],
        },
        {"bond": False, "angle": False, "dihedral": False},
        300.0,
        SwarmConfig(),
        exec_mode=exec_mode,
    )

    assert topology["bond"][0]["value"] == equilibrium
    assert np.isclose(topology["bond"][0]["fct"], 900.0, atol=1e-8)


def test_perform_bi_recovers_all_supported_linear_forms_and_rb():
    temperature = 300.0
    bond_centers = np.linspace(0.35, 0.65, 61)
    bond_equilibrium = 0.5
    bond_basis = 0.5 * (bond_centers - bond_equilibrium) ** 2

    angle_centers = np.deg2rad(np.linspace(70.0, 170.0, 101))
    angle_equilibrium = np.deg2rad(120.0)
    angle_bases = [
        0.5 * (angle_centers - angle_equilibrium) ** 2,
        0.5 * (np.cos(angle_centers) - np.cos(angle_equilibrium)) ** 2,
        0.5
        * (np.cos(angle_centers) - np.cos(angle_equilibrium)) ** 2
        / np.sin(angle_centers) ** 2,
    ]
    angle_constants = [320.0, 85.0, 200.0]

    dihedral_centers = np.linspace(-np.pi, np.pi, 144, endpoint=False)
    dihedral_centers += np.pi / 144
    periodic_phase = np.deg2rad(35.0)
    periodic_basis = 1.0 + np.cos(2 * dihedral_centers - periodic_phase)
    canonical_shifted_phase = periodic_phase + np.pi
    canonical_shifted_basis = 1.0 + np.cos(
        2 * dihedral_centers - canonical_shifted_phase
    )
    improper_phase = np.deg2rad(-45.0)
    improper_offset = (
        dihedral_centers - improper_phase + np.pi
    ) % (2 * np.pi) - np.pi
    improper_basis = 0.5 * improper_offset**2

    rb_expected = RBParameters((2.0, -1.0, 0.5, 1.5, -0.25))
    rb_energy = gmx_dihedrals_func_3(
        dihedral_centers, *rb_expected.to_gromacs()
    )
    rb_probabilities = np.exp(
        -(rb_energy - rb_energy.min()) / (config.kB * temperature)
    )
    rb_probabilities /= rb_probabilities.sum()

    topology = {
        "bond": [{"func": 1, "value": bond_equilibrium, "fct": 10.0}],
        "angle": [
            {"func": func, "value": 120.0, "fct": 10.0}
            for func in (1, 2, 10)
        ],
        "dihedral": [
            {"func": 1, "value": 35.0, "fct": 1.0, "mult": 2, "params": []},
            {"func": 4, "value": -145.0, "fct": 1.0, "mult": 2, "params": []},
            {"func": 2, "value": -45.0, "fct": 1.0, "mult": None, "params": []},
            {
                "func": 3,
                "value": None,
                "fct": None,
                "mult": None,
                "params": [0.0] * 6,
                "coefficient_bound": 25.0,
            },
        ],
    }
    original_values = {
        "bond": topology["bond"][0]["value"],
        "angles": [group["value"] for group in topology["angle"]],
        "dihedrals": [group["value"] for group in topology["dihedral"]],
    }
    targets = {
        "bond": [_target_from_basis(bond_centers, bond_basis, 1500.0)],
        "angle": [
            _target_from_basis(angle_centers, basis, force_constant)
            for basis, force_constant in zip(angle_bases, angle_constants)
        ],
        "dihedral": [
            _target_from_basis(dihedral_centers, periodic_basis, 4.0),
            _target_from_basis(dihedral_centers, canonical_shifted_basis, 3.0),
            _target_from_basis(dihedral_centers, improper_basis, 175.0),
            BoltzmannTarget(dihedral_centers, rb_probabilities),
        ],
    }
    cycle = {
        "nb_geoms": {"constraint": 0, "bond": 1, "angle": 3, "dihedral": 4}
    }
    flags = {"bond": False, "angle": False, "dihedral": False}

    perform_BI(
        topology,
        cycle,
        targets,
        flags,
        temperature,
        SwarmConfig(),
        exec_mode=1,
    )

    assert np.isclose(topology["bond"][0]["fct"], 1500.0, atol=1e-7)
    assert np.allclose(
        [group["fct"] for group in topology["angle"]],
        angle_constants,
        atol=1e-7,
    )
    assert np.isclose(topology["dihedral"][0]["fct"], 4.0, atol=1e-8)
    assert np.isclose(topology["dihedral"][1]["fct"], 3.0, atol=1e-8)
    assert np.isclose(topology["dihedral"][2]["fct"], 175.0, atol=1e-7)
    assert np.allclose(
        RBParameters.from_gromacs(topology["dihedral"][3]["params"]).coefficients,
        rb_expected.coefficients,
        atol=1e-8,
    )
    assert topology["bond"][0]["value"] == original_values["bond"]
    assert [group["value"] for group in topology["angle"]] == original_values["angles"]
    assert [group["value"] for group in topology["dihedral"]] == original_values["dihedrals"]
    assert all(flags.values())


def test_underdetermined_force_and_rb_fits_retain_input_seeds(caplog):
    centers = np.linspace(-np.pi, np.pi, 5, endpoint=False)
    one_bin = BoltzmannTarget(centers, np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
    original_rb = RBParameters((1.0, -2.0, 3.0, -4.0, 5.0)).to_gromacs()
    topology = {
        "bond": [{"func": 1, "value": 0.5, "fct": 321.0}],
        "angle": [],
        "dihedral": [
            {
                "func": 3,
                "value": None,
                "fct": None,
                "mult": None,
                "params": list(original_rb),
                "coefficient_bound": 25.0,
            }
        ],
    }
    cycle = {
        "nb_geoms": {"constraint": 0, "bond": 1, "angle": 0, "dihedral": 1}
    }
    flags = {"bond": False, "angle": False, "dihedral": False}

    perform_BI(
        topology,
        cycle,
        {"bond": [one_bin], "angle": [], "dihedral": [one_bin]},
        flags,
        300.0,
        SwarmConfig(),
    )

    assert topology["bond"][0]["fct"] == 321.0
    assert topology["dihedral"][0]["params"] == list(original_rb)
    assert "design rank 1, expected 2" in caplog.text
    assert "design rank" in caplog.text
    assert "broad first-activation exploration" in caplog.text
