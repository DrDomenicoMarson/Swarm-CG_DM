"""FST-PSO evaluation callback and its execution helpers."""

from __future__ import annotations

import gzip
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

from swarmcg import config, io, simulations as sim
from swarmcg.context import OptimizationContext
from swarmcg.history import HISTORY_SCHEMA_VERSION, append_history_record
from swarmcg.optimization_types import EvaluationResult
from swarmcg.scoring.compare import compare_models
from swarmcg.shared import exceptions, styling
from swarmcg.topology import GeometryKind
from swarmcg.utils import print_stdout_forced


@dataclass(frozen=True)
class _EvaluationPaths:
    execution: Path
    archive: Path
    workspace: Path
    topology: Path
    plot: Path

    @classmethod
    def build(cls, ns: OptimizationContext) -> "_EvaluationPaths":
        execution = Path(ns.files.exec_folder).resolve()
        archive = execution / config.all_evals_files_dirname
        workspace = execution / (
            f"{config.iteration_sim_files_dirname}_eval_step_{ns.status.nb_eval}"
        )
        return cls(
            execution=execution,
            archive=archive,
            workspace=workspace,
            topology=workspace / ns.files.cg_itp_basename,
            plot=workspace / "distributions.png",
        )


@dataclass
class _EvaluationOutcome:
    comparison: EvaluationResult
    objective: float
    new_global_best: bool = False
    failure_kind: str | None = None
    failure_message: str | None = None
    gromacs_seconds: float = 0.0
    scoring_seconds: float = 0.0


def eval_function(parameters_set, ns: OptimizationContext) -> float:
    """Evaluate one FST-PSO particle through simulation and bonded scoring.

    Args:
        parameters_set: Flat particle vector in the active cycle's canonical
            parameter order.
        ns: Mutable optimization context containing topology, typed cycle and
            vector layout, paths, counters, and scoring state.

    Returns:
        Finite active-cycle objective. Failed evaluations return the next
        representable float above the active theoretical score maximum.

    Raises:
        RuntimeError: If the callback is invoked without its typed cycle,
            simulation setup, parameter layout, or working topology.
        OSError: If mandatory workspace files cannot be staged or recorded.
    """
    _validate_context(ns)
    ns.status.nb_eval += 1
    started_at = datetime.now().timestamp()
    paths = _EvaluationPaths.build(ns)
    _announce_evaluation(ns)
    _stage_workspace(paths)
    ns.parameter_layout.apply(ns.out_itp, parameters_set)
    _write_evaluation_topology(ns, paths)

    outcome = _run_and_score(ns, paths)
    _archive_artifacts(ns, paths, outcome)
    _report_outcome(ns, outcome)
    elapsed_minutes, elapsed_hours = _record_timing(ns, started_at)
    _write_history_record(
        ns,
        paths.execution / config.optimization_history_file,
        outcome,
        elapsed_minutes,
        elapsed_hours,
    )
    return outcome.objective


def _validate_context(ns: OptimizationContext) -> None:
    missing = []
    for name in (
        "cg_itp",
        "out_itp",
        "opti_cycle",
        "simulation_setup",
        "parameter_layout",
    ):
        if getattr(ns, name) is None:
            missing.append(name)
    if missing:
        raise RuntimeError(
            "evaluation context is missing: " + ", ".join(missing)
        )


def _announce_evaluation(ns: OptimizationContext) -> None:
    print_stdout_forced()
    failures = (
        f"(failed {ns.status.failed_eval_count}; "
        f"stalled {ns.status.stalled_eval_count}; "
        f"crashed {ns.status.crashed_eval_count})"
    )
    print_stdout_forced(
        f"Starting iteration {ns.status.nb_eval} at {time.strftime('%H:%M:%S')} "
        f"on {time.strftime('%d-%m-%Y')} {failures}"
    )


def _stage_workspace(paths: _EvaluationPaths) -> None:
    paths.archive.mkdir(parents=True, exist_ok=True)
    if paths.workspace.exists():
        shutil.rmtree(paths.workspace)
    shutil.copytree(
        paths.execution / config.input_sim_files_dirname,
        paths.workspace,
    )


def _write_evaluation_topology(
    ns: OptimizationContext, paths: _EvaluationPaths
) -> None:
    sections = ["constraint", "bond", "angle"]
    if ns.opti_cycle.counts.dihedrals:
        sections.append("dihedral")
    sections.append("exclusion")
    io.write_cg_topology(ns.out_itp, paths.topology, sections=sections)
    archived_name = (
        f"{paths.topology.stem}_eval_step_{ns.status.nb_eval}.itp"
    )
    shutil.copy(paths.topology, paths.archive / archived_name)


def _failure_comparison(ns: OptimizationContext) -> EvaluationResult:
    return EvaluationResult(
        total_score=ns.pso.worst_fit_score,
        constraints_bonds_score=ns.pso.failure_component_scores[
            "constraints_bonds"
        ],
        angles_score=ns.pso.failure_component_scores["angles"],
        dihedrals_score=ns.pso.failure_component_scores["dihedrals"],
        pairwise_scores={
            GeometryKind.CONSTRAINT: (float("nan"),)
            * ns.cg_itp.constraint_count,
            GeometryKind.BOND: (float("nan"),) * ns.cg_itp.bond_count,
            GeometryKind.ANGLE: (float("nan"),) * ns.cg_itp.angle_count,
            GeometryKind.DIHEDRAL: (float("nan"),)
            * ns.cg_itp.dihedral_count,
        },
    )


def _run_and_score(
    ns: OptimizationContext, paths: _EvaluationPaths
) -> _EvaluationOutcome:
    failure = _failure_comparison(ns)
    simulation_started = datetime.now().timestamp()
    simulation_error = None
    try:
        sim.SimulationManager(ns.config).run_simulation(
            str(paths.workspace),
            sim_time=ns.simulation_setup.duration_ns,
            nb_frames=ns.simulation_setup.frame_count,
        )
    except exceptions.ComputationError as exc:
        simulation_error = exc
    gromacs_seconds = datetime.now().timestamp() - simulation_started
    ns.status.total_gmx_time += gromacs_seconds

    if simulation_error is not None:
        kind = (
            "stalled"
            if "unstable simulation was killed" in str(simulation_error).lower()
            else "crashed"
        )
        print_stdout_forced(
            styling.header_warning
            + "Simulation failed; assigning worst score and continuing.\n"
            + str(simulation_error)
        )
        _record_failure(ns, kind)
        _clear_cg_observables(ns)
        return _EvaluationOutcome(
            failure,
            ns.pso.worst_fit_score,
            failure_kind=kind,
            failure_message=str(simulation_error),
            gromacs_seconds=gromacs_seconds,
        )

    if not (paths.workspace / "md.gro").is_file():
        print_stdout_forced(
            styling.header_warning
            + "Simulation output missing; assigning worst score and continuing."
        )
        _record_failure(ns, "crashed")
        _clear_cg_observables(ns)
        return _EvaluationOutcome(
            failure,
            ns.pso.worst_fit_score,
            failure_kind="crashed",
            failure_message="Simulation output md.gro is missing.",
            gromacs_seconds=gromacs_seconds,
        )

    ns.files.cg_tpr_filename = str(paths.workspace / "md.tpr")
    ns.files.cg_traj_filename = str(paths.workspace / "md.xtc")
    ns.files.plot_filename = str(paths.plot)
    scoring_started = datetime.now().timestamp()
    scoring_error = None
    try:
        comparison = _compare_model(ns)
    except Exception as exc:
        scoring_error = exc
        print_stdout_forced(
            styling.header_warning
            + "Model scoring failed; assigning worst score and continuing.\n"
            + str(exc)
        )
        _record_failure(ns, "crashed")
        _clear_cg_observables(ns)
    finally:
        scoring_seconds = datetime.now().timestamp() - scoring_started
        ns.status.total_model_eval_time += scoring_seconds

    if scoring_error is not None:
        return _EvaluationOutcome(
            failure,
            ns.pso.worst_fit_score,
            failure_kind="scoring_failed",
            failure_message=str(scoring_error),
            gromacs_seconds=gromacs_seconds,
            scoring_seconds=scoring_seconds,
        )

    objective = _score_for_geometries(comparison, ns.opti_cycle.geometries)
    global_score = _score_for_geometries(comparison, ns.pso.opti_geoms_all)
    new_best = global_score < ns.pso.best_fitness[0]
    if new_best:
        ns.pso.best_fitness = global_score, ns.status.nb_eval
    return _EvaluationOutcome(
        comparison,
        objective,
        new_global_best=new_best,
        gromacs_seconds=gromacs_seconds,
        scoring_seconds=scoring_seconds,
    )


def _compare_model(ns: OptimizationContext) -> EvaluationResult:
    result = compare_models(
        ns,
        manual_mode=False,
        ignore_dihedrals=ns.opti_cycle.counts.dihedrals == 0,
        calc_sasa=ns.config.output.calculate_sasa,
        record_best_indep_params=True,
    )
    if result is None:
        raise RuntimeError("optimization scoring returned no evaluation result")
    return result


def _score_for_geometries(
    comparison: EvaluationResult,
    geometries: Sequence[str | GeometryKind],
) -> float:
    active = {GeometryKind(kind) for kind in geometries}
    score = 0.0
    if {GeometryKind.CONSTRAINT, GeometryKind.BOND}.intersection(active):
        score += comparison.constraints_bonds_score
    if GeometryKind.ANGLE in active:
        score += comparison.angles_score
    if GeometryKind.DIHEDRAL in active:
        score += comparison.dihedrals_score
    return float(score)


def _record_failure(ns: OptimizationContext, kind: str) -> None:
    ns.status.failed_eval_count += 1
    if kind == "stalled":
        ns.status.stalled_eval_count += 1
    else:
        ns.status.crashed_eval_count += 1


def _clear_cg_observables(ns: OptimizationContext) -> None:
    ns.results.gyr_cg = None
    ns.results.gyr_cg_std = None
    ns.results.sasa_cg = None
    ns.results.sasa_cg_std = None


def _archive_artifacts(
    ns: OptimizationContext,
    paths: _EvaluationPaths,
    outcome: _EvaluationOutcome,
) -> None:
    if paths.plot.is_file():
        archived_plot = paths.archive / (
            f"distributions_eval_step_{ns.status.nb_eval}.png"
        )
        shutil.move(paths.plot, archived_plot)
        if outcome.new_global_best:
            shutil.copy(archived_plot, paths.execution / config.best_distrib_plots)

    for source_name, destination_name in (
        ("md.log", f"md_sim_eval_step_{ns.status.nb_eval}.log.gz"),
        ("equi.log", f"equi_sim_eval_step_{ns.status.nb_eval}.log.gz"),
        ("mini.log", f"mini_sim_eval_step_{ns.status.nb_eval}.log.gz"),
    ):
        source = paths.workspace / source_name
        if source.is_file():
            with source.open("rb") as source_file, gzip.open(
                paths.archive / destination_name, "wb"
            ) as destination_file:
                shutil.copyfileobj(source_file, destination_file)

    if ns.config.optimization.keep_all_sims:
        shutil.copytree(
            paths.workspace,
            paths.execution / config.sim_files_all_evals_dirname / paths.workspace.name,
        )
    if ns.status.nb_eval == 1:
        shutil.copytree(
            paths.workspace, paths.execution / "boltzmann_inv_CG_model"
        )
    if outcome.new_global_best:
        best_model = paths.execution / config.best_fitted_model_dirname
        if best_model.exists():
            shutil.rmtree(best_model)
        shutil.move(paths.workspace, best_model)
    else:
        shutil.rmtree(paths.workspace)


def _report_outcome(ns: OptimizationContext, outcome: _EvaluationOutcome) -> None:
    if outcome.failure_kind is not None:
        print_stdout_forced(
            f"  Evaluation failed; finite penalty objective: {outcome.objective}"
        )
        return

    result = outcome.comparison
    print_stdout_forced(
        "  Total mismatch score:",
        round(result.total_score, 3),
        "(Bonds/Constraints:",
        result.constraints_bonds_score,
        "-- Angles:",
        result.angles_score,
        "-- Dihedrals:",
        str(result.dihedrals_score) + ")",
    )
    if outcome.new_global_best:
        print_stdout_forced("    --> Selected as new best bonded parametrization")
    print_stdout_forced(
        f"  Rg CG:   {round(ns.results.gyr_cg, 2)} nm   "
        f"(Error abs. {round(abs(1 - ns.results.gyr_cg / ns.results.gyr_aa_mapped) * 100, 1)}% "
        f"-- Reference Rg AA-mapped: {ns.results.gyr_aa_mapped} nm)"
    )
    if ns.config.output.calculate_sasa and ns.results.sasa_cg is not None:
        print_stdout_forced(
            f"  SASA CG: {ns.results.sasa_cg} nm2   "
            f"(Error abs. {round(abs(1 - ns.results.sasa_cg / ns.results.sasa_aa_mapped) * 100, 1)}% "
            f"-- Reference SASA AA-mapped: {ns.results.sasa_aa_mapped} nm2)"
        )
    return None


def _record_timing(
    ns: OptimizationContext, started_at: float
) -> tuple[float, float]:
    now = datetime.now().timestamp()
    current_total_hours = round((now - ns.status.start_opti_ts) / 3600, 2)
    elapsed_seconds = now - started_at
    ns.status.total_eval_time += elapsed_seconds
    elapsed_minutes = round(elapsed_seconds / 60, 2)
    print_stdout_forced(f"  Iteration time: {elapsed_minutes} min")
    return elapsed_minutes, current_total_hours


def _write_history_record(
    ns: OptimizationContext,
    history_path: Path,
    outcome: _EvaluationOutcome,
    elapsed_minutes: float,
    elapsed_hours: float,
) -> None:
    result = outcome.comparison
    record = {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "evaluation_id": ns.status.nb_eval,
        "cycle_id": ns.opti_cycle.number,
        "status": "success" if outcome.failure_kind is None else "failure",
        "active_geometries": [kind.value for kind in ns.opti_cycle.geometries],
        "scores": {
            "total": result.total_score,
            "constraints_bonds": result.constraints_bonds_score,
            "angles": result.angles_score,
            "dihedrals": result.dihedrals_score,
            "objective": outcome.objective,
        },
        "observables": _serialize_observables(ns),
        "pairwise_scores": {
            kind.plural: result.pairwise_scores[kind] for kind in GeometryKind
        },
        "parameters": _serialize_topology_parameters(ns),
        "timings": {
            "evaluation_minutes": elapsed_minutes,
            "total_hours": elapsed_hours,
            "gromacs_seconds": outcome.gromacs_seconds,
            "scoring_seconds": outcome.scoring_seconds,
        },
        "failure": (
            None
            if outcome.failure_kind is None
            else {
                "kind": outcome.failure_kind,
                "message": outcome.failure_message,
            }
        ),
    }
    append_history_record(history_path, record)


def _serialize_observables(ns: OptimizationContext) -> dict:
    return {
        "radius_of_gyration": {
            "aa_mapped": {
                "mean": ns.results.gyr_aa_mapped,
                "standard_deviation": ns.results.gyr_aa_mapped_std,
            },
            "cg": {
                "mean": ns.results.gyr_cg,
                "standard_deviation": ns.results.gyr_cg_std,
            },
        },
        "sasa": {
            "aa_mapped": {
                "mean": ns.results.sasa_aa_mapped,
                "standard_deviation": ns.results.sasa_aa_mapped_std,
            },
            "cg": {
                "mean": ns.results.sasa_cg,
                "standard_deviation": ns.results.sasa_cg_std,
            },
        },
    }


def _serialize_topology_parameters(ns: OptimizationContext) -> dict:
    constraints = [
        {"function": group.function, "equilibrium": group.equilibrium}
        for group in ns.out_itp.constraints
    ]
    bonds = [
        {
            "function": group.function,
            "equilibrium": group.equilibrium,
            "force_constant": group.force_constant,
        }
        for group in ns.out_itp.bonds
    ]
    angles = [
        {
            "function": group.function,
            "equilibrium": group.equilibrium,
            "force_constant": group.force_constant,
        }
        for group in ns.out_itp.angles
    ]
    dihedrals = []
    for group in ns.out_itp.dihedrals:
        parameters = {"function": group.function}
        if group.function in (3, 11):
            parameters["coefficients"] = group.gromacs_parameters
        else:
            parameters["equilibrium"] = group.equilibrium
            parameters["force_constant"] = group.force_constant
            if group.multiplicity is not None:
                parameters["multiplicity"] = group.multiplicity
        dihedrals.append(parameters)
    return {
        "constraints": constraints,
        "bonds": bonds,
        "angles": angles,
        "dihedrals": dihedrals,
    }
