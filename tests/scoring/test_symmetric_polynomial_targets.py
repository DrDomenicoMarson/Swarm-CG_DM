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


def _evaluator_with_dihedrals(groups, exec_mode):
    """Return an initialized lightweight evaluator for distribution tests."""
    config = SwarmConfig(optimization=OptimizationConfig(exec_mode=exec_mode))
    context = OptimizationContext(config=config)
    context.cg_itp = {
        "constraint": [],
        "bond": [],
        "angle": [],
        "dihedral": groups,
    }
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


def test_symmetric_rb_and_cbt_targets_do_not_require_phase(monkeypatch):
    groups = [
        {
            "func": 3,
            "params": [0.0] * 6,
            "value": None,
            "value_user": None,
            "fct": None,
            "fct_user": None,
            "mult": None,
            "beads": [[0, 1, 2, 3]],
        },
        {
            "func": 11,
            "params": [0.0] * 6,
            "value": None,
            "value_user": None,
            "fct": None,
            "fct_user": None,
            "mult": None,
            "beads": [[0, 1, 2, 3]],
        },
    ]
    evaluator = _evaluator_with_dihedrals(groups, exec_mode=1)
    monkeypatch.setattr(
        "swarmcg.scoring.evaluator.scores.get_AA_dihedrals_distrib",
        _undefined_mean_distribution,
    )

    evaluator.compute_reference_distributions()

    assert all(np.isnan(group["avg"]) for group in groups)
    assert evaluator.ns.scoring.domains_val["dihedral"] == [None, None]
    assert len(evaluator.ns.scoring.data_BI["dihedral"]) == 2
    assert all(group["coefficient_bound"] == 25.0 for group in groups)
    assert all(group["polynomial_symmetry_tv"] <= 0.10 for group in groups)
    assert _format_circular_mean(float("nan")) == "unavailable"


def test_phase_function_rejects_undefined_mean_only_in_mode_one(monkeypatch):
    def group():
        return {
            "func": 1,
            "params": [30.0, 2.0],
            "value": 30.0,
            "value_user": 30.0,
            "fct": 2.0,
            "fct_user": 2.0,
            "mult": 1,
            "beads": [[0, 1, 2, 3]],
        }

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

    assert np.isnan(mode_two_group["avg"])
    assert mode_two_group["value"] == 30.0
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

    group = {
        "func": 1,
        "params": [0.0, 2.0],
        "value": 0.0,
        "value_user": 0.0,
        "fct": 2.0,
        "fct_user": 2.0,
        "mult": 3,
        "beads": [[0, 1, 2, 3]],
    }
    evaluator = _evaluator_with_dihedrals([group], exec_mode=1)
    monkeypatch.setattr(
        "swarmcg.scoring.evaluator.scores.get_AA_dihedrals_distrib",
        distribution,
    )

    evaluator.compute_reference_distributions()

    assert np.isnan(group["avg"])
    assert np.isclose(group["phase_moment_resultant"], 1.0)
    assert np.isclose(group["value"], -90.0)
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

    group = {
        "func": 4,
        "params": [20.0, 2.0],
        "value": 20.0,
        "value_user": 20.0,
        "fct": 2.0,
        "fct_user": 2.0,
        "mult": 2,
        "beads": [[0, 1, 2, 3]],
    }
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

    group = {
        "func": func,
        "params": [0.0] * 6,
        "value": None,
        "value_user": None,
        "fct": None,
        "fct_user": None,
        "mult": None,
        "beads": [[0, 1, 2, 3]],
    }
    evaluator = _evaluator_with_dihedrals([group], exec_mode=1)
    monkeypatch.setattr(
        "swarmcg.scoring.evaluator.scores.get_AA_dihedrals_distrib",
        distribution,
    )
    caplog.set_level(logging.WARNING)

    evaluator.compute_reference_distributions()

    assert group["polynomial_symmetry_tv"] > 0.10
    assert f"{form_name} dihedral group 1" in caplog.text
    assert "cannot reproduce an asymmetric torsional marginal" in caplog.text
