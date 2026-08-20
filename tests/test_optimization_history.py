"""Tests for strict JSONL history persistence, analysis, and rendering."""

import json

import pytest

from swarmcg.analyze_optimization import render_optimization_history
from swarmcg.history import (
    HISTORY_SCHEMA_VERSION,
    analyze_optimization_history,
    append_history_record,
    load_optimization_history,
)
from swarmcg.shared import exceptions


def _record(
    evaluation_id,
    *,
    cycle_id=1,
    status="success",
    total=5.0,
    failure_kind=None,
):
    """Return one complete schema-version-1 history dictionary."""
    return {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "evaluation_id": evaluation_id,
        "cycle_id": cycle_id,
        "status": status,
        "active_geometries": ["bond", "dihedral"],
        "scores": {
            "total": total,
            "constraints_bonds": total,
            "angles": 0.0,
            "dihedrals": 0.0,
            "objective": total,
        },
        "observables": {
            "radius_of_gyration": {
                "aa_mapped": {"mean": 1.0, "standard_deviation": 0.1},
                "cg": {
                    "mean": 0.9 if status == "success" else None,
                    "standard_deviation": (
                        0.1 if status == "success" else None
                    ),
                },
            },
            "sasa": {
                "aa_mapped": {"mean": None, "standard_deviation": None},
                "cg": {"mean": None, "standard_deviation": None},
            },
        },
        "pairwise_scores": {
            "constraints": [],
            "bonds": [total if status == "success" else None],
            "angles": [],
            "dihedrals": [0.0 if status == "success" else None],
        },
        "parameters": {
            "constraints": [],
            "bonds": [
                {"function": 1, "equilibrium": 0.3, "force_constant": 1000.0}
            ],
            "angles": [],
            "dihedrals": [
                {
                    "function": 3,
                    "coefficients": [-15.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                }
            ],
        },
        "timings": {
            "evaluation_minutes": 1.0,
            "total_hours": evaluation_id / 60,
            "gromacs_seconds": 50.0,
            "scoring_seconds": 10.0,
        },
        "failure": (
            None
            if status == "success"
            else {"kind": failure_kind, "message": f"synthetic {failure_kind}"}
        ),
    }


def test_history_supports_success_and_all_failure_kinds(tmp_path):
    history = tmp_path / "optimization_history.jsonl"
    append_history_record(history, _record(1, total=4.0))
    append_history_record(
        history,
        _record(2, status="failure", total=0.1, failure_kind="stalled"),
    )
    append_history_record(
        history,
        _record(3, status="failure", total=0.1, failure_kind="crashed"),
    )
    append_history_record(
        history,
        _record(
            4,
            cycle_id=2,
            status="failure",
            total=0.1,
            failure_kind="scoring_failed",
        ),
    )
    append_history_record(history, _record(5, cycle_id=2, total=3.0))

    records = load_optimization_history(history)
    analysis = analyze_optimization_history(records)

    assert [record.failure and record.failure["kind"] for record in records] == [
        None,
        "stalled",
        "crashed",
        "scoring_failed",
        None,
    ]
    assert analysis.best_record.evaluation_id == 5
    assert analysis.cycle_separators == (3.5,)


def test_writer_serializes_nonfinite_values_as_json_null(tmp_path):
    history = tmp_path / "optimization_history.jsonl"
    record = _record(1)
    record["observables"]["sasa"]["cg"]["mean"] = float("nan")

    append_history_record(history, record)

    text = history.read_text()
    assert "NaN" not in text
    assert json.loads(text)["observables"]["sasa"]["cg"]["mean"] is None


def test_loader_ignores_only_a_truncated_final_line(tmp_path):
    history = tmp_path / "optimization_history.jsonl"
    append_history_record(history, _record(1))
    with history.open("a") as handle:
        handle.write('{"schema_version":1,"evaluation_id":')

    records = load_optimization_history(history)

    assert [record.evaluation_id for record in records] == [1]


@pytest.mark.parametrize(
    "first_line",
    ["{broken json}", '{"schema_version":1,"evaluation_id":NaN}'],
)
def test_loader_rejects_malformed_earlier_records(tmp_path, first_line):
    history = tmp_path / "optimization_history.jsonl"
    history.write_text(first_line + "\n" + json.dumps(_record(2)) + "\n")

    with pytest.raises(exceptions.OptimisationResultsError, match="line 1"):
        load_optimization_history(history)


def test_loader_does_not_treat_nonstandard_nan_as_truncation(tmp_path):
    history = tmp_path / "optimization_history.jsonl"
    history.write_text('{"schema_version":1,"evaluation_id":NaN}')

    with pytest.raises(exceptions.OptimisationResultsError, match="line 1"):
        load_optimization_history(history)


def test_monitor_renders_polynomial_history_without_sasa(tmp_path):
    history = tmp_path / "optimization_history.jsonl"
    append_history_record(history, _record(1, total=4.0))
    append_history_record(history, _record(2, cycle_id=2, total=3.0))
    analysis = analyze_optimization_history(load_optimization_history(history))
    output = tmp_path / "monitor.png"

    render_optimization_history(analysis, output, plot_scale=1.0)

    assert output.is_file()
    assert output.stat().st_size > 0
