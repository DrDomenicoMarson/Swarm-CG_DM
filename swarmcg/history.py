"""Strict versioned JSONL persistence for optimization evaluations."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from swarmcg.shared import exceptions

HISTORY_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class HistoryScoreBreakdown:
    """Score fields stored for one optimization evaluation.

    Args:
        total: Full bonded mismatch score.
        constraints_bonds: Combined constraint/bond contribution.
        angles: Angle contribution.
        dihedrals: Dihedral contribution.
        objective: Active-cycle objective returned to FST-PSO.
    """

    total: float | None
    constraints_bonds: float | None
    angles: float | None
    dihedrals: float | None
    objective: float | None


@dataclass(frozen=True)
class OptimizationHistoryRecord:
    """Validated schema-version-2 optimization history record.

    Args:
        evaluation_id: One-based evaluation identifier.
        cycle_id: One-based optimization-cycle identifier.
        status: ``"success"`` or ``"failure"``.
        active_geometries: Geometry names active in the cycle.
        scores: Aggregate and active-objective score breakdown.
        observables: Radius-of-gyration and SASA measurements.
        pairwise_scores: Per-group EMD scores by plural geometry name.
        parameters: Canonical topology parameters by geometry name.
        timings: Evaluation, total, GROMACS, and scoring timings.
        failure: Optional structured failure details.
    """

    evaluation_id: int
    cycle_id: int
    status: str
    active_geometries: tuple[str, ...]
    scores: HistoryScoreBreakdown
    observables: Mapping[str, Any]
    pairwise_scores: Mapping[str, tuple[float | None, ...]]
    parameters: Mapping[str, tuple[Mapping[str, Any], ...]]
    timings: Mapping[str, float | None]
    failure: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class OptimizationHistoryAnalysis:
    """Monitor-ready optimization records and best-selection metadata.

    Args:
        records: Validated evaluation records in file order.
        best_index: Zero-based index of the best successful total score.
        cycle_separators: X positions between changes of optimization cycle.
    """

    records: tuple[OptimizationHistoryRecord, ...]
    best_index: int
    cycle_separators: tuple[float, ...]

    @property
    def best_record(self) -> OptimizationHistoryRecord:
        """Return the best successful evaluation record."""
        return self.records[self.best_index]


def append_history_record(path: str | Path, record: Mapping[str, Any]) -> None:
    """Append one complete strict-JSON line and flush it immediately.

    Args:
        path: Destination JSONL file.
        record: Schema-version-2 record to validate and serialize.

    Raises:
        ValueError: If the record is invalid or contains non-finite numbers.
        OSError: If the destination cannot be created or written.
    """
    normalized = _normalize_record(record)
    _parse_record(normalized)
    line = json.dumps(
        normalized,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ) + "\n"
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()


def load_optimization_history(
    path: str | Path,
) -> list[OptimizationHistoryRecord]:
    """Load strict JSONL records, tolerating only a truncated final line.

    Args:
        path: Versioned optimization-history JSONL file.

    Returns:
        Validated records in file order. A syntactically incomplete final line
        without a trailing newline is ignored so active runs can be monitored.

    Raises:
        IncompleteOptimisationFile: If the file has no complete records.
        OptimisationResultsError: If a complete record is malformed, uses an
            unsupported schema, or contains non-standard JSON values.
        OSError: If the file cannot be read.
    """
    source = Path(path)
    content = source.read_text(encoding="utf-8")
    lines = content.splitlines(keepends=True)
    records = []
    for index, line in enumerate(lines):
        complete_line = line.endswith(("\n", "\r"))
        payload = line.strip()
        if not payload:
            if index == len(lines) - 1 and not complete_line:
                continue
            raise _history_error(index, "empty JSONL record")
        try:
            raw = json.loads(payload, parse_constant=_reject_json_constant)
        except json.JSONDecodeError as exc:
            if index == len(lines) - 1 and not complete_line:
                continue
            raise _history_error(index, str(exc)) from exc
        except ValueError as exc:
            raise _history_error(index, str(exc)) from exc
        try:
            records.append(_parse_record(raw))
        except (KeyError, TypeError, ValueError) as exc:
            raise _history_error(index, str(exc)) from exc
    if not records:
        raise exceptions.IncompleteOptimisationFile(
            "The optimization history contains no complete evaluation records."
        )
    return records


def analyze_optimization_history(
    records: Sequence[OptimizationHistoryRecord],
) -> OptimizationHistoryAnalysis:
    """Select the best successful record and locate cycle boundaries.

    Args:
        records: Validated history records in evaluation order.

    Returns:
        Monitor-ready analysis containing best and cycle metadata.

    Raises:
        IncompleteOptimisationFile: If no records are supplied.
        OptimisationResultsError: If no successful finite score is selectable.
    """
    if not records:
        raise exceptions.IncompleteOptimisationFile(
            "The optimization history contains no evaluations."
        )
    candidates = [
        (index, record.scores.total)
        for index, record in enumerate(records)
        if record.status == "success"
        and record.scores.total is not None
        and math.isfinite(record.scores.total)
    ]
    if not candidates:
        raise exceptions.OptimisationResultsError(
            "No successful finite evaluation is available for model selection."
        )
    best_index = min(candidates, key=lambda item: item[1])[0]
    separators = tuple(
        index + 0.5
        for index in range(1, len(records))
        if records[index].cycle_id != records[index - 1].cycle_id
    )
    return OptimizationHistoryAnalysis(
        tuple(records), best_index, separators
    )


def _normalize_record(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize_record(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_record(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"history value {value!r} is not JSON serializable")
    return numeric if math.isfinite(numeric) else None


def _parse_record(raw: Mapping[str, Any]) -> OptimizationHistoryRecord:
    if not isinstance(raw, Mapping):
        raise TypeError("history record must be a JSON object")
    if raw.get("schema_version") != HISTORY_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported history schema_version {raw.get('schema_version')!r}"
        )
    evaluation_id = _positive_int(raw["evaluation_id"], "evaluation_id")
    cycle_id = _positive_int(raw["cycle_id"], "cycle_id")
    status = raw["status"]
    if status not in ("success", "failure"):
        raise ValueError("status must be 'success' or 'failure'")
    active = tuple(str(kind) for kind in raw["active_geometries"])
    valid_kinds = {"constraint", "bond", "angle", "dihedral"}
    if not active or any(kind not in valid_kinds for kind in active):
        raise ValueError("active_geometries contains an unsupported geometry")
    score_data = raw["scores"]
    scores = HistoryScoreBreakdown(
        total=_optional_number(score_data["total"], "scores.total"),
        constraints_bonds=_optional_number(
            score_data["constraints_bonds"], "scores.constraints_bonds"
        ),
        angles=_optional_number(score_data["angles"], "scores.angles"),
        dihedrals=_optional_number(
            score_data["dihedrals"], "scores.dihedrals"
        ),
        objective=_optional_number(
            score_data["objective"], "scores.objective"
        ),
    )
    pairwise_raw = raw["pairwise_scores"]
    pairwise = {
        kind: tuple(
            _optional_number(value, f"pairwise_scores.{kind}")
            for value in pairwise_raw[kind]
        )
        for kind in ("constraints", "bonds", "angles", "dihedrals")
    }
    parameters = {}
    for kind in ("constraints", "bonds", "angles", "dihedrals"):
        values = tuple(raw["parameters"][kind])
        if any(not isinstance(value, Mapping) for value in values):
            raise TypeError(f"parameters.{kind} must contain JSON objects")
        if len(values) != len(pairwise[kind]):
            raise ValueError(
                f"parameters.{kind} and pairwise_scores.{kind} lengths differ"
            )
        parameters[kind] = values
    timings = {
        key: _optional_number(raw["timings"][key], f"timings.{key}")
        for key in (
            "evaluation_minutes",
            "total_hours",
            "gromacs_seconds",
            "scoring_seconds",
        )
    }
    observables = _parse_observables(raw["observables"])
    failure = raw.get("failure")
    if status == "failure" and failure is not None and not isinstance(failure, Mapping):
        raise TypeError("failure details must be a JSON object or null")
    return OptimizationHistoryRecord(
        evaluation_id,
        cycle_id,
        status,
        active,
        scores,
        observables,
        pairwise,
        parameters,
        timings,
        failure,
    )


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _parse_observables(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    rg = {}
    for representation in ("aa_mapped", "cg"):
        values = raw["radius_of_gyration"][representation]
        rg[representation] = {
            "mean": _optional_number(
                values["mean"],
                f"observables.radius_of_gyration.{representation}.mean",
            ),
            "standard_deviation": _optional_number(
                values["standard_deviation"],
                "observables.radius_of_gyration."
                f"{representation}.standard_deviation",
            ),
        }
    sasa = {
        representation: _parse_sasa_diagnostic(
            raw["sasa"][representation], representation
        )
        for representation in ("aa", "aa_mapped", "cg")
    }
    return {"radius_of_gyration": rg, "sasa": sasa}


def _parse_sasa_diagnostic(
    raw: Mapping[str, Any], representation: str
) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise TypeError(
            f"observables.sasa.{representation} must be a JSON object"
        )
    status = raw["status"]
    if status not in ("not_scheduled", "success", "failed"):
        raise ValueError(
            f"observables.sasa.{representation}.status is unsupported"
        )
    measurement = raw.get("measurement")
    error = raw.get("error")
    if status == "not_scheduled":
        if measurement is not None or error is not None:
            raise ValueError(
                f"not_scheduled SASA {representation} cannot contain results"
            )
        return {"status": status, "measurement": None, "error": None}
    if status == "failed":
        if measurement is not None or not isinstance(error, str) or not error:
            raise ValueError(
                f"failed SASA {representation} requires only non-empty error details"
            )
        return {"status": status, "measurement": None, "error": error}
    if not isinstance(measurement, Mapping) or error is not None:
        raise ValueError(
            f"successful SASA {representation} requires one measurement and no error"
        )
    if measurement["representation"] != representation:
        raise ValueError(
            f"SASA measurement representation does not match {representation}"
        )
    protocol = measurement["protocol"]
    if not isinstance(protocol, Mapping):
        raise TypeError(
            f"observables.sasa.{representation}.measurement.protocol must be an object"
        )
    frame_count = _positive_int(
        measurement["frame_count"],
        f"observables.sasa.{representation}.measurement.frame_count",
    )
    probe = _required_number(
        protocol["probe_radius_nm"],
        f"observables.sasa.{representation}.measurement.protocol.probe_radius_nm",
    )
    sphere_points = _positive_int(
        protocol["sphere_points"],
        f"observables.sasa.{representation}.measurement.protocol.sphere_points",
    )
    source = protocol["radii_source"]
    digest = protocol["radii_sha256"]
    if not isinstance(source, str) or not source:
        raise ValueError("SASA radii_source must be a non-empty string")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("SASA radii_sha256 must be a lowercase SHA-256 digest")
    if probe <= 0:
        raise ValueError("SASA probe_radius_nm must be positive")
    parsed_measurement = {
        "representation": representation,
        "mean": _required_number(
            measurement["mean"],
            f"observables.sasa.{representation}.measurement.mean",
        ),
        "standard_deviation": _required_number(
            measurement["standard_deviation"],
            f"observables.sasa.{representation}.measurement.standard_deviation",
        ),
        "frame_count": frame_count,
        "protocol": {
            "probe_radius_nm": probe,
            "sphere_points": sphere_points,
            "radii_source": source,
            "radii_sha256": digest,
        },
    }
    if parsed_measurement["standard_deviation"] < 0:
        raise ValueError("SASA standard_deviation must be nonnegative")
    return {"status": status, "measurement": parsed_measurement, "error": None}


def _required_number(value: Any, label: str) -> float:
    numeric = _optional_number(value, label)
    if numeric is None:
        raise ValueError(f"{label} must be a finite number")
    return numeric


def _optional_number(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a number or null")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite or null")
    return numeric


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value}")


def _history_error(index: int, detail: str) -> exceptions.OptimisationResultsError:
    return exceptions.OptimisationResultsError(
        f"Malformed optimization history record on line {index + 1}: {detail}"
    )
