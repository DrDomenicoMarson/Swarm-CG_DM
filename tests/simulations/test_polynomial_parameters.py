"""Scientific tests for RB and CBT parameter representations."""

import numpy as np

from swarmcg.simulations.polynomial import (
    CBTParameters,
    RBParameters,
    adaptive_coefficient_bound,
    fit_rb_coefficients,
    mirrored_total_variation,
)
from swarmcg.simulations.potentials import gmx_dihedrals_func_3, gmx_dihedrals_func_11


def test_rb_canonicalization_preserves_forces():
    original = np.array([41.0, 1.0, -2.0, 3.0, -4.0, 5.0])
    canonical = np.array(RBParameters.from_gromacs(original).to_gromacs())
    angles = np.linspace(-np.pi, np.pi, 1001)
    original_energy = gmx_dihedrals_func_3(angles, *original)
    canonical_energy = gmx_dihedrals_func_3(angles, *canonical)

    assert np.allclose(np.gradient(original_energy, angles), np.gradient(canonical_energy, angles))
    assert np.allclose(original_energy - canonical_energy, original_energy[0] - canonical_energy[0])
    assert np.isclose(canonical.sum(), 0.0)


def test_rb_bounded_fit_recovers_synthetic_coefficients():
    expected = RBParameters((2.0, -1.0, 0.5, 1.5, -0.25))
    angles = np.linspace(-np.pi, np.pi, 720, endpoint=False)
    energy = gmx_dihedrals_func_3(angles, *expected.to_gromacs())
    probabilities = np.exp(-(energy - energy.min()) / (0.008314462618 * 300.0))
    probabilities /= probabilities.sum()

    fitted = fit_rb_coefficients(angles, probabilities, 300.0, bound=25.0)
    assert np.allclose(fitted.coefficients, expected.coefficients, atol=1e-8)


def test_cbt_canonicalization_preserves_energy_and_zero_case():
    original = (7.0, 0.2, -0.4, 0.6, -0.8, 1.0)
    effective = CBTParameters.from_gromacs(original)
    canonical = effective.to_gromacs()
    phi = np.linspace(-np.pi, np.pi, 101)
    theta_prev = np.full_like(phi, 1.1)
    theta_curr = np.full_like(phi, 2.0)

    assert np.allclose(
        gmx_dihedrals_func_11(theta_prev, theta_curr, phi, *original),
        gmx_dihedrals_func_11(theta_prev, theta_curr, phi, *canonical),
    )
    assert CBTParameters((0.0,) * 5).to_gromacs() == (0.0,) * 6


def test_adaptive_bounds_and_cbt_symmetry_metric():
    assert adaptive_coefficient_bound([0.5, 0.5], 300.0) == 25.0
    assert 25.0 <= adaptive_coefficient_bound([0.999, 0.001], 300.0) <= 200.0
    assert mirrored_total_variation([0.1, 0.4, 0.4, 0.1]) == 0.0
    assert mirrored_total_variation([1.0, 0.0, 0.0, 0.0]) == 1.0
