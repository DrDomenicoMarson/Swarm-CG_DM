"""Tests for circular and polynomial torsional target limitations."""

import logging

import numpy as np
import pytest

from swarmcg.config_types import OptimizationConfig, SwarmConfig
from swarmcg.context import OptimizationContext
from swarmcg.scoring import create_bins_and_dist_matrices
from swarmcg.scoring.compare import _format_circular_mean
from swarmcg.scoring.evaluator import SwarmEvaluator
from swarmcg.shared import exceptions
from swarmcg.simulations.polynomial import mirrored_total_variation
from swarmcg.simulations.polynomial import CBTParameters, RBParameters
from swarmcg.shared.periodic import PeriodicDihedralParameters
from swarmcg.topology import CGTopology, DihedralGroup


def _evaluator_with_dihedrals(groups, exec_mode):
    """Return an initialized lightweight evaluator for distribution tests."""
    config = SwarmConfig(optimization=OptimizationConfig(exec_mode=exec_mode))
    context = OptimizationContext(config=config)
    context.cg_itp = CGTopology(dihedrals=groups)
    create_bins_and_dist_matrices(context)
    evaluator = SwarmEvaluator(config)
    evaluator.ns = context
    return evaluator


def _undefined_mean_distribution(*args, **kwargs):
    """Return a symmetric target whose first circular moment is undefined."""
    values_degrees = np.array([0.0, 180.0])
    values_radians = np.deg2rad(values_degrees)
    histogram = np.zeros(144)
    histogram[71] = 0.5
    histogram[72] = 0.5
    return float("nan"), histogram, values_degrees, values_radians


def _polynomial_group(function):
    """Return a zero-coefficient RB or CBT dihedral group."""
    parameters = (
        RBParameters((0.0,) * 5)
        if function == 3
        else CBTParameters((0.0,) * 5)
    )
    return DihedralGroup(
        "1", [(0, 1, 2, 3)], function, parameters, parameters
    )


def _periodic_group(function, phase, force, multiplicity):
    """Return one typed periodic dihedral group."""
    parameters = PeriodicDihedralParameters(phase, force, multiplicity)
    return DihedralGroup(
        "1", [(0, 1, 2, 3)], function, parameters, parameters
    )


def test_symmetric_rb_and_cbt_targets_do_not_require_phase(monkeypatch):
    groups = [_polynomial_group(3), _polynomial_group(11)]
    evaluator = _evaluator_with_dihedrals(groups, exec_mode=1)
    monkeypatch.setattr(
        "swarmcg.scoring.evaluator.scores.get_AA_dihedrals_distrib",
        _undefined_mean_distribution,
    )

    evaluator.compute_reference_distributions()

    assert all(np.isnan(group.average) for group in groups)
    assert evaluator.ns.scoring.domains_val["dihedral"] == [None, None]
    assert len(evaluator.ns.scoring.data_BI["dihedral"]) == 2
    assert all(group.coefficient_bound == 25.0 for group in groups)
    assert all(group.polynomial_symmetry_tv <= 0.10 for group in groups)
    assert _format_circular_mean(float("nan")) == "unavailable"


def test_phase_function_rejects_undefined_mean_only_in_mode_one(monkeypatch):
    def group():
        return _periodic_group(1, 30.0, 2.0, 1)

    monkeypatch.setattr(
        "swarmcg.scoring.evaluator.scores.get_AA_dihedrals_distrib",
        _undefined_mean_distribution,
    )
    mode_one = _evaluator_with_dihedrals([group()], exec_mode=1)
    with pytest.raises(
        exceptions.ScientificValidationError,
        match="use execution mode 2 with a fixed ITP phase",
    ):
        mode_one.compute_reference_distributions()

    mode_two_group = group()
    mode_two = _evaluator_with_dihedrals([mode_two_group], exec_mode=2)
    mode_two.compute_reference_distributions()

    assert np.isnan(mode_two_group.average)
    assert mode_two_group.equilibrium == 30.0
    assert mode_two.ns.scoring.domains_val["dihedral"] == [None]


def test_threefold_periodic_target_uses_third_moment_not_first(monkeypatch):
    values_degrees = np.array([30.0, 150.0, -90.0])
    values_radians = np.deg2rad(values_degrees)
    histogram = np.histogram(
        values_degrees, bins=np.linspace(-180.0, 180.0, 145)
    )[0].astype(float)
    histogram /= histogram.sum()

    def distribution(*args, **kwargs):
        return float("nan"), histogram, values_degrees, values_radians

    group = _periodic_group(1, 0.0, 2.0, 3)
    evaluator = _evaluator_with_dihedrals([group], exec_mode=1)
    monkeypatch.setattr(
        "swarmcg.scoring.evaluator.scores.get_AA_dihedrals_distrib",
        distribution,
    )

    evaluator.compute_reference_distributions()

    assert np.isnan(group.average)
    assert np.isclose(group.phase_moment_resultant, 1.0)
    assert np.isclose(group.equilibrium, -90.0)
    assert evaluator.ns.scoring.domains_val["dihedral"] == [[-270.0, 90.0]]


def test_mode_one_rejects_undefined_multiplicity_order_moment(monkeypatch):
    values_degrees = np.array([0.0, 90.0])
    values_radians = np.deg2rad(values_degrees)
    histogram = np.histogram(
        values_degrees, bins=np.linspace(-180.0, 180.0, 145)
    )[0].astype(float)
    histogram /= histogram.sum()

    def distribution(*args, **kwargs):
        return 45.0, histogram, values_degrees, values_radians

    group = _periodic_group(4, 20.0, 2.0, 2)
    evaluator = _evaluator_with_dihedrals([group], exec_mode=1)
    monkeypatch.setattr(
        "swarmcg.scoring.evaluator.scores.get_AA_dihedrals_distrib",
        distribution,
    )

    with pytest.raises(exceptions.ScientificValidationError, match="order-2"):
        evaluator.compute_reference_distributions()


@pytest.mark.parametrize(
    "first_mass,expected_tv",
    [(0.545, 0.09), (0.55, 0.10), (0.555, 0.11)],
)
def test_polynomial_symmetry_threshold_values(first_mass, expected_tv):
    probabilities = np.array([first_mass, 1.0 - first_mass])

    assert np.isclose(
        mirrored_total_variation(probabilities), expected_tv
    )


@pytest.mark.parametrize("func,form_name", [(3, "RB"), (11, "CBT")])
def test_polynomial_forms_warn_for_asymmetric_target(
    monkeypatch, caplog, func, form_name
):
    histogram = np.zeros(144)
    histogram[12] = 1.0
    values_degrees = np.array([-148.0, -147.0])
    values_radians = np.deg2rad(values_degrees)

    def distribution(*args, **kwargs):
        return -147.5, histogram, values_degrees, values_radians

    group = _polynomial_group(func)
    evaluator = _evaluator_with_dihedrals([group], exec_mode=1)
    monkeypatch.setattr(
        "swarmcg.scoring.evaluator.scores.get_AA_dihedrals_distrib",
        distribution,
    )
    caplog.set_level(logging.WARNING)

    evaluator.compute_reference_distributions()

    assert group.polynomial_symmetry_tv > 0.10
    assert f"{form_name} dihedral group 1" in caplog.text
    assert "cannot reproduce an asymmetric torsional marginal" in caplog.text
