"""Load, analyze, and render versioned optimization JSONL histories."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from shlex import quote as cmd_quote
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

import swarmcg.io as io
import swarmcg.shared.styling
from swarmcg import config
from swarmcg.config_types import OutputConfig
from swarmcg.history import (
    OptimizationHistoryAnalysis,
    OptimizationHistoryRecord,
    analyze_optimization_history,
    load_optimization_history,
)
from swarmcg.shared import catch_warnings
from swarmcg.shared.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


def _validated_plot_scale(value):
    """Return a finite positive monitor plot scale.

    Args:
        value: Raw scale parsed from the monitor command line.

    Returns:
        Validated finite positive scale.

    Raises:
        ValidationError: If *value* is zero, negative, ``NaN``, or infinite.
    """
    return OutputConfig(plot_scale=value).plot_scale


@catch_warnings(DeprecationWarning)
@catch_warnings(ImportWarning)
@catch_warnings(UserWarning)
def run(ns) -> None:
    """Load a JSONL optimization history and render its monitoring plot.

    Args:
        ns: Parsed monitor arguments containing the optimization directory,
            output plot path, and plot scale.

    Raises:
        ValidationError: If the plot scale is invalid.
        IncompleteOptimisationFile: If no complete JSONL record exists.
        OptimisationResultsError: If records are malformed or no successful
            evaluation can be selected.
    """
    scale = _validated_plot_scale(ns.plot_scale)
    optimization_dir = Path(ns.opti_dirname)
    records = load_optimization_history(
        optimization_dir / config.optimization_history_file
    )
    analysis = analyze_optimization_history(records)
    output = optimization_dir / ns.plot_filename
    render_optimization_history(analysis, output, scale)
    _log_best_evaluation(analysis)
    logger.info("")
    logger.info("Wrote visual optimization summary file at location:\n %s", output)
    logger.info("")


def render_optimization_history(
    analysis: OptimizationHistoryAnalysis,
    output_path: str | Path,
    plot_scale: float = 1.0,
) -> None:
    """Render scores, observables, parameters, and pairwise EMD histories.

    Args:
        analysis: Validated records plus best-selection metadata.
        output_path: Destination image filename.
        plot_scale: Positive multiplier applied to the figure dimensions.

    Returns:
        ``None``. The monitor image is written to ``output_path``.
    """
    scale = _validated_plot_scale(plot_scale)
    records = analysis.records
    counts = _geometry_counts(records)
    populated = [kind for kind, count in counts.items() if count]
    columns = max(9, *(counts.values() or [0]))
    rows = 1 + 2 * len(populated)
    figure, axes = plt.subplots(
        rows,
        columns,
        squeeze=False,
        figsize=(columns * 4 * scale, rows * 3 * scale),
    )
    x_values = np.arange(1, len(records) + 1)
    _render_summary_row(axes[0], analysis, x_values)
    row = 1
    for kind in populated:
        _render_parameter_row(
            axes[row], records, kind, counts[kind], x_values, analysis
        )
        _render_pairwise_row(
            axes[row + 1], records, kind, counts[kind], x_values, analysis
        )
        row += 2
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def _geometry_counts(
    records: tuple[OptimizationHistoryRecord, ...],
) -> dict[str, int]:
    kinds = ("constraints", "bonds", "angles", "dihedrals")
    counts = {kind: len(records[0].parameters[kind]) for kind in kinds}
    for record in records[1:]:
        for kind in kinds:
            if len(record.parameters[kind]) != counts[kind]:
                raise ValueError(
                    f"history parameter count changes for {kind} at "
                    f"evaluation {record.evaluation_id}"
                )
    return counts


def _render_summary_row(
    axes,
    analysis: OptimizationHistoryAnalysis,
    x_values: np.ndarray,
) -> None:
    records = analysis.records
    failures = np.array(
        [index + 1 for index, record in enumerate(records) if record.status == "failure"]
    )
    score_series = (
        ("Total fitness score", lambda record: record.scores.total),
        ("Evaluation objective", lambda record: record.scores.objective),
        ("Constraints/Bonds score", lambda record: record.scores.constraints_bonds),
        ("Angles score", lambda record: record.scores.angles),
        ("Dihedrals score", lambda record: record.scores.dihedrals),
    )
    for column, (title, accessor) in enumerate(score_series):
        values = _numeric_series(records, accessor)
        _plot_summary_series(
            axes[column],
            x_values,
            values,
            title,
            analysis,
            failures,
            color="darkgreen" if column < 2 else "mediumseagreen",
        )
    _plot_observable(
        axes[5],
        x_values,
        analysis,
        "radius_of_gyration",
        "Radius of gyration",
        failures,
    )
    sasa_cg = _sasa_measurement_series(records, "cg", "mean")
    if np.any(np.isfinite(sasa_cg)):
        _plot_sasa(axes[6], x_values, analysis)
    else:
        axes[6].set_visible(False)
    _plot_summary_series(
        axes[7],
        x_values,
        _timing_series(records, "total_hours"),
        "Total time (hours)",
        analysis,
        failures,
        color="purple",
    )
    _plot_summary_series(
        axes[8],
        x_values,
        _timing_series(records, "evaluation_minutes"),
        "All evaluation times (min)",
        analysis,
        failures,
        color="mediumorchid",
        mark_best=False,
    )
    for axis in axes[9:]:
        axis.set_visible(False)


def _plot_summary_series(
    axis,
    x_values,
    values,
    title,
    analysis,
    failures,
    *,
    color,
    mark_best=True,
) -> None:
    axis.set_title(title)
    axis.grid(zorder=0.5)
    axis.plot(x_values, values, color=color)
    if failures.size:
        axis.scatter(
            failures,
            values[failures - 1],
            marker="x",
            color="black",
            zorder=2,
        )
    _draw_cycle_separators(axis, analysis)
    if mark_best and np.isfinite(values[analysis.best_index]):
        axis.plot(
            analysis.best_index + 1,
            values[analysis.best_index],
            marker="D",
            color="white",
            markerfacecolor="gold",
            markersize=10,
            markeredgewidth=1.5,
            markeredgecolor="black",
            zorder=3,
        )


def _plot_observable(
    axis,
    x_values,
    analysis,
    observable,
    title,
    failures,
) -> None:
    records = analysis.records
    reference = _observable_series(records, observable, "aa_mapped", "mean")
    reference_std = _observable_series(
        records, observable, "aa_mapped", "standard_deviation"
    )
    cg = _observable_series(records, observable, "cg", "mean")
    cg_std = _observable_series(
        records, observable, "cg", "standard_deviation"
    )
    axis.set_title(title)
    axis.grid(zorder=0.5)
    axis.plot(x_values, reference, color=config.atom_color, label="AA-mapped", lw=2.5)
    axis.fill_between(
        x_values,
        reference - reference_std,
        reference + reference_std,
        color=config.atom_color,
        alpha=0.1,
    )
    axis.plot(x_values, cg, color=config.cg_color, label="CG estimation")
    axis.fill_between(
        x_values,
        cg - cg_std,
        cg + cg_std,
        color=config.cg_color,
        alpha=0.2,
    )
    if failures.size:
        axis.scatter(
            failures, cg[failures - 1], marker="x", color="black", zorder=2
        )
    _draw_cycle_separators(axis, analysis)
    if np.isfinite(cg[analysis.best_index]):
        axis.plot(
            analysis.best_index + 1,
            cg[analysis.best_index],
            marker="D",
            color="white",
            markerfacecolor="gold",
            markersize=10,
            markeredgewidth=1.5,
            markeredgecolor="black",
            zorder=3,
        )
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    axis.legend(loc="lower right")


def _plot_sasa(axis, x_values, analysis) -> None:
    """Plot sparse new-best CG SASA against both reference definitions.

    Args:
        axis: Matplotlib axis receiving the plot.
        x_values: One-based evaluation positions.
        analysis: Validated schema-version-2 history analysis.

    Returns:
        ``None``. The axis is populated in place.
    """
    records = analysis.records
    aa = _sasa_measurement_series(records, "aa", "mean")
    aa_std = _sasa_measurement_series(records, "aa", "standard_deviation")
    mapped = _sasa_measurement_series(records, "aa_mapped", "mean")
    mapped_std = _sasa_measurement_series(
        records, "aa_mapped", "standard_deviation"
    )
    cg = _sasa_measurement_series(records, "cg", "mean")
    cg_std = _sasa_measurement_series(records, "cg", "standard_deviation")
    axis.set_title("SASA (new global bests only)")
    axis.grid(zorder=0.5)
    axis.plot(x_values, aa, color=config.atom_color, label="Full AA (primary)", lw=2.5)
    axis.fill_between(
        x_values, aa - aa_std, aa + aa_std, color=config.atom_color, alpha=0.1
    )
    axis.plot(
        x_values,
        mapped,
        color="darkorange",
        label="AA mapped to CG centres (secondary)",
        lw=2,
    )
    axis.fill_between(
        x_values,
        mapped - mapped_std,
        mapped + mapped_std,
        color="darkorange",
        alpha=0.1,
    )
    axis.errorbar(
        x_values,
        cg,
        yerr=cg_std,
        color=config.cg_color,
        marker="o",
        linestyle="none",
        label="CG new-best diagnostic",
    )
    _draw_cycle_separators(axis, analysis)
    if np.isfinite(cg[analysis.best_index]):
        axis.plot(
            analysis.best_index + 1,
            cg[analysis.best_index],
            marker="D",
            color="white",
            markerfacecolor="gold",
            markersize=10,
            markeredgewidth=1.5,
            markeredgecolor="black",
            zorder=3,
        )
    axis.legend(loc="lower right")


def _render_parameter_row(
    axes,
    records,
    kind,
    count,
    x_values,
    analysis,
) -> None:
    label = kind[:-1].capitalize() if kind != "dihedrals" else "Dihedral"
    for index, axis in enumerate(axes):
        if index >= count:
            axis.set_visible(False)
            continue
        parameters = [record.parameters[kind][index] for record in records]
        coefficients = [parameter.get("coefficients") for parameter in parameters]
        if any(value is not None for value in coefficients):
            axis.set_title(f"{label} {index + 1} - Coefficients")
            width = max(len(value or ()) for value in coefficients)
            for coefficient_index in range(width):
                values = np.array(
                    [
                        np.nan
                        if value is None or coefficient_index >= len(value)
                        else value[coefficient_index]
                        for value in coefficients
                    ],
                    dtype=float,
                )
                axis.plot(x_values, values, label=f"C{coefficient_index}")
            axis.legend(loc="best", fontsize="small")
        else:
            equilibrium = np.array(
                [parameter.get("equilibrium", np.nan) for parameter in parameters],
                dtype=float,
            )
            axis.set_title(f"{label} {index + 1} - Parameters")
            axis.plot(x_values, equilibrium, color="tab:blue")
            if any("force_constant" in parameter for parameter in parameters):
                force = np.array(
                    [parameter.get("force_constant", np.nan) for parameter in parameters],
                    dtype=float,
                )
                twin = axis.twinx()
                twin.plot(x_values, force, color="tab:red")
                if np.isfinite(force[analysis.best_index]):
                    twin.plot(
                        analysis.best_index + 1,
                        force[analysis.best_index],
                        marker="D",
                        color="salmon",
                        markeredgecolor="black",
                        zorder=3,
                    )
            if np.isfinite(equilibrium[analysis.best_index]):
                axis.plot(
                    analysis.best_index + 1,
                    equilibrium[analysis.best_index],
                    marker="D",
                    color="lightskyblue",
                    markeredgecolor="black",
                    zorder=3,
                )
        axis.grid(zorder=0.5)
        _draw_cycle_separators(axis, analysis)


def _render_pairwise_row(
    axes,
    records,
    kind,
    count,
    x_values,
    analysis,
) -> None:
    label = kind[:-1].capitalize() if kind != "dihedrals" else "Dihedral"
    for index, axis in enumerate(axes):
        if index >= count:
            axis.set_visible(False)
            continue
        values = np.array(
            [
                np.nan
                if record.pairwise_scores[kind][index] is None
                else record.pairwise_scores[kind][index]
                for record in records
            ],
            dtype=float,
        )
        axis.set_title(f"{label} {index + 1} - Score")
        axis.grid(zorder=0.5)
        axis.plot(x_values, values, color="mediumseagreen")
        _draw_cycle_separators(axis, analysis)
        if np.isfinite(values[analysis.best_index]):
            axis.plot(
                analysis.best_index + 1,
                values[analysis.best_index],
                marker="D",
                color="palegreen",
                markersize=10,
                markeredgewidth=1.5,
                markeredgecolor="black",
                zorder=3,
            )


def _draw_cycle_separators(axis, analysis) -> None:
    for separator in analysis.cycle_separators:
        axis.axvline(x=separator, color="black")


def _numeric_series(
    records: tuple[OptimizationHistoryRecord, ...],
    accessor: Callable[[OptimizationHistoryRecord], float | None],
) -> np.ndarray:
    return np.array(
        [np.nan if accessor(record) is None else accessor(record) for record in records],
        dtype=float,
    )


def _timing_series(records, key) -> np.ndarray:
    return _numeric_series(records, lambda record: record.timings[key])


def _observable_series(records, observable, representation, statistic):
    return _numeric_series(
        records,
        lambda record: record.observables[observable][representation][statistic],
    )


def _sasa_measurement_series(records, representation, statistic):
    return _numeric_series(
        records,
        lambda record: (
            None
            if record.observables["sasa"][representation]["measurement"] is None
            else record.observables["sasa"][representation]["measurement"][statistic]
        ),
    )


def _log_best_evaluation(analysis: OptimizationHistoryAnalysis) -> None:
    best = analysis.best_record
    rg = best.observables["radius_of_gyration"]
    logger.info(
        "Best bonded terms found at step %s with estimated Rg %s nm",
        best.evaluation_id,
        rg["cg"]["mean"],
    )
    if rg["cg"]["mean"] is not None and rg["aa_mapped"]["mean"] is not None:
        error = abs(1 - rg["cg"]["mean"] / rg["aa_mapped"]["mean"]) * 100
        logger.info(
            "  Rg CG: %s nm (Error abs. %s%% -- Reference Rg AA-mapped: %s nm)",
            round(rg["cg"]["mean"], 3),
            round(error, 1),
            rg["aa_mapped"]["mean"],
        )
    sasa = best.observables["sasa"]
    cg = sasa["cg"]["measurement"]
    aa = sasa["aa"]["measurement"]
    mapped = sasa["aa_mapped"]["measurement"]
    if cg is not None and aa is not None:
        primary_error = abs(1 - cg["mean"] / aa["mean"]) * 100
        logger.info(
            "  SASA CG: %s nm2 (primary error vs full AA: %s%%; "
            "full-AA reference: %s nm2)",
            round(cg["mean"], 3),
            round(primary_error, 1),
            aa["mean"],
        )
        if mapped is not None:
            secondary_error = abs(1 - cg["mean"] / mapped["mean"]) * 100
            logger.info(
                "  Secondary CG-vs-mapped SASA error: %s%% "
                "(mapped reference: %s nm2)",
                round(secondary_error, 1),
                mapped["mean"],
            )
    elif sasa["cg"]["status"] == "failed":
        logger.warning("  Best-evaluation CG SASA failed: %s", sasa["cg"]["error"])


def main() -> None:
    """Run the ``scg_monitor`` command-line entry point."""
    module_name = "monitor"
    setup_logging(
        module_name=module_name,
        verbose=("-v" in sys.argv or "--verbose" in sys.argv),
    )
    if "--nobanner" in sys.argv or "-nobanner" in sys.argv:
        logger.info(swarmcg.shared.styling.header_simple(module_name))
    else:
        logger.info(
            swarmcg.shared.styling.header_package(
                "Module: Optimization run analysis\n"
            )
        )
    parser = io.get_analyze_args()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    ns = parser.parse_args()
    setup_logging(
        module_name=module_name,
        log_dir=ns.opti_dirname,
        verbose=getattr(ns, "verbose", False),
    )
    logger.info("Working directory: %s", os.getcwd())
    logger.info("Command line: %s", " ".join(map(cmd_quote, sys.argv)))
    logger.info("")
    run(ns)
