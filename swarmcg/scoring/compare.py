"""Typed bonded-distribution comparison, aggregation, and rendering."""

from __future__ import annotations

from dataclasses import dataclass

import MDAnalysis as mda
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from swarmcg import config
from swarmcg import scoring as scores
from swarmcg.context import OptimizationContext
from swarmcg.optimization_types import EvaluationResult, ObservableStatistics
from swarmcg.shared import styling
from swarmcg.shared.histograms import HistogramObservation
from swarmcg.shared.logging_utils import get_logger
from swarmcg.topology import GeometryKind

# Use the non-interactive Anti-Grain Geometry backend for scripted PNG output.
matplotlib.use("AGG")

logger = get_logger(__name__)


@dataclass
class DistributionStatistics:
    """Histogram and descriptive statistics for one molecular representation.

    Args:
        average: Linear or circular mean of the sampled geometry.
        histogram: Normalized mass on the geometry's fixed scoring grid.
        x: Histogram support selected for plotting.
        y: Histogram values selected for plotting.
        observation: Missing-mass classification for a CG distribution.
    """

    average: float
    histogram: np.ndarray
    x: np.ndarray | None = None
    y: np.ndarray | None = None
    observation: HistogramObservation | None = None


@dataclass
class GroupComparison:
    """Typed AA/CG distribution comparison for one topology group.

    Args:
        kind: Bonded geometry kind.
        group_index: Zero-based topology group index.
        aa: Mapped atomistic distribution and statistics.
        cg: Coarse-grained distribution and statistics, if available.
        emd_score: Pairwise EMD after the configured class scaling.
        x_range: Inclusive plotting range selected from occupied support.
    """

    kind: GeometryKind
    group_index: int
    aa: DistributionStatistics
    cg: DistributionStatistics | None
    emd_score: float
    x_range: tuple[float, float]


@dataclass
class GeometryComparisons:
    """All group comparisons and plot ordering for one geometry kind.

    Args:
        kind: Bonded geometry kind.
        groups: Comparisons in topology order.
        ordered_indices: Topology indices in display order.
        max_x_range: Largest occupied plotting range in the row.
    """

    kind: GeometryKind
    groups: list[GroupComparison]
    ordered_indices: list[int]
    max_x_range: float


def compare_models(
    context: OptimizationContext,
    manual_mode: bool = True,
    ignore_dihedrals: bool = False,
    calc_sasa: bool = False,
    record_best_indep_params: bool = False,
) -> EvaluationResult | None:
    """Compare mapped atomistic and CG bonded distributions and plot them.

    Args:
        context: Optimization context containing topology, trajectories,
            histogram grids, output paths, and mutable optimization state.
        manual_mode: Recalculate atomistic references for ``scg_evaluate``.
        ignore_dihedrals: Mark dihedrals as excluded from the cycle objective.
        calc_sasa: Calculate SASA as a nonfatal diagnostic after scoring.
        record_best_indep_params: Record improving per-group parameters.

    Returns:
        A typed evaluation result during optimization. Manual and AA-only
        inspection modes return ``None`` after writing the plot.

    Raises:
        ScientificValidationError: If a reference distribution contains a
            non-finite or out-of-range sample.
    """
    _prepare_trajectories(context, manual_mode)
    _log_scoring_header()
    comparisons = _compare_all_geometries(context, manual_mode)
    result = None
    if not context.scoring.atom_only:
        result = _aggregate_scores(
            context,
            comparisons,
            ignore_dihedrals=ignore_dihedrals,
            record_best_indep_params=record_best_indep_params,
        )
    _render_comparisons(context, comparisons, result, ignore_dihedrals)
    if calc_sasa and not context.scoring.atom_only:
        _compute_optional_sasa(context)
        result = _with_observables(context, result)
    if not manual_mode and not context.scoring.atom_only:
        return result
    return None


def _prepare_trajectories(ns: OptimizationContext, manual_mode: bool) -> None:
    if ns.scoring.atom_only:
        ns.scoring.gyr_aa, ns.scoring.gyr_aa_std = scores.compute_Rg(
            ns.scoring.aa_universe,
            ns.scoring.aa_universe.atoms[: len(ns.scoring.all_atoms)],
            backend=ns.scoring.mda_backend,
        )
        logger.info(
            "Radius of gyration (AA reference, NOT CG-mapped): %s nm",
            ns.scoring.gyr_aa,
        )
        mapped_masses = np.asarray(
            ns.scoring.aa2cg_universe.atoms.masses, dtype=float
        )
        if np.all(np.isfinite(mapped_masses)) and np.all(mapped_masses > 0):
            ns.results.gyr_aa_mapped, ns.results.gyr_aa_mapped_std = (
                scores.compute_Rg(
                    ns.scoring.aa2cg_universe,
                    ns.scoring.aa2cg_universe.atoms[: len(ns.cg_itp.atoms)],
                    backend=ns.scoring.mda_backend,
                    offset=ns.config.reference.aa_rg_offset,
                )
            )
            logger.info(
                "Radius of gyration (AA reference, CG-mapped): %s +/- %s nm",
                ns.results.gyr_aa_mapped,
                ns.results.gyr_aa_mapped_std,
            )
        return

    logger.info("Reading CG trajectory")
    ns.scoring.cg_universe = mda.Universe(
        ns.files.cg_tpr_filename,
        ns.files.cg_traj_filename,
        in_memory=True,
        refresh_offsets=True,
        guess_bonds=False,
    )
    logger.info("  Found %s frames", len(ns.scoring.cg_universe.trajectory))
    if manual_mode or any(atom.mass is None for atom in ns.cg_itp.atoms):
        for bead_id, atom in enumerate(ns.cg_itp.atoms):
            atom.mass = ns.scoring.cg_universe.atoms[bead_id].mass
        masses = np.array([atom.mass for atom in ns.cg_itp.atoms], dtype=float)
        ns.scoring.aa2cg_universe._topology.masses.values = masses
    if ns.cg_itp.virtual_bead_ids:
        fake_bonds = [
            [site.bead_id, defining_bead]
            for site in ns.cg_itp.virtual_sites
            for defining_bead in site.defining_beads
        ]
        ns.scoring.cg_universe.add_bonds(fake_bonds, guessed=False)
    molecule = mda.AtomGroup(
        list(range(len(ns.cg_itp.atoms))), ns.scoring.cg_universe
    )
    for _ in ns.scoring.cg_universe.trajectory:
        mda.lib.mdamath.make_whole(molecule, inplace=True)
    if ns.results.gyr_aa_mapped is None:
        ns.results.gyr_aa_mapped, ns.results.gyr_aa_mapped_std = scores.compute_Rg(
            ns.scoring.aa2cg_universe,
            ns.scoring.aa2cg_universe.atoms[: len(ns.cg_itp.atoms)],
            backend=ns.scoring.mda_backend,
            offset=ns.config.reference.aa_rg_offset,
        )
        logger.info("")
        logger.info(
            "Radius of gyration (AA reference, CG-mapped, no bonds scaling): "
            "%s +/- %s nm",
            ns.results.gyr_aa_mapped,
            ns.results.gyr_aa_mapped_std,
        )
    ns.results.gyr_cg, ns.results.gyr_cg_std = scores.compute_Rg(
        ns.scoring.cg_universe,
        ns.scoring.cg_universe.atoms[: len(ns.cg_itp.atoms)],
        backend=ns.scoring.mda_backend,
    )
    logger.info(
        "Radius of gyration (CG model): %s +/- %s nm",
        ns.results.gyr_cg,
        ns.results.gyr_cg_std,
    )


def _log_scoring_header() -> None:
    logger.info("")
    logger.info(styling.sep_close)
    logger.info(
        "| SCORING AND PLOTTING                                      "
        "                                  |"
    )
    logger.info(styling.sep_close)
    logger.info("")


def _compare_all_geometries(
    ns: OptimizationContext, manual_mode: bool
) -> tuple[GeometryComparisons, ...]:
    return tuple(
        _compare_geometry(ns, kind, manual_mode) for kind in GeometryKind
    )


def _compare_geometry(
    ns: OptimizationContext,
    kind: GeometryKind,
    manual_mode: bool,
) -> GeometryComparisons:
    logger.info("Processing %s ...", kind.plural)
    topology_groups = _topology_groups(ns, kind)
    bins, grid = _histogram_resources(ns, kind)
    comparisons = []
    maximum_range = 0.0
    for index, topology_group in enumerate(topology_groups):
        aa_average, aa_histogram = _reference_distribution(
            ns, kind, index, topology_group, bins, manual_mode
        )
        aa = DistributionStatistics(
            float(aa_average), np.asarray(aa_histogram, dtype=float)
        )
        cg = None
        if not ns.scoring.atom_only:
            cg_average, cg_histogram, cg_values = _cg_distribution(
                ns, kind, index, topology_group, bins
            )
            cg = DistributionStatistics(
                float(cg_average),
                np.asarray(cg_histogram, dtype=float),
                observation=scores.observe_histogram(cg_values, bins),
            )
        _populate_plot_support(aa, cg, bins, periodic=kind == GeometryKind.DIHEDRAL)
        x_range = (float(aa.x[0]), float(aa.x[-1]))
        maximum_range = max(maximum_range, x_range[1] - x_range[0])
        emd_score = (
            _pairwise_score(ns, kind, aa.histogram, cg.histogram, grid)
            if cg is not None
            else aa.average
        )
        comparisons.append(
            GroupComparison(kind, index, aa, cg, emd_score, x_range)
        )
    ordered = list(range(len(comparisons)))
    if ns.scoring.mismatch_order and not ns.scoring.atom_only:
        ordered.sort(
            key=lambda index: comparisons[index].emd_score, reverse=True
        )
    return GeometryComparisons(kind, comparisons, ordered, maximum_range)


def _topology_groups(ns: OptimizationContext, kind: GeometryKind):
    return {
        GeometryKind.CONSTRAINT: ns.cg_itp.constraints,
        GeometryKind.BOND: ns.cg_itp.bonds,
        GeometryKind.ANGLE: ns.cg_itp.angles,
        GeometryKind.DIHEDRAL: ns.cg_itp.dihedrals,
    }[kind]


def _histogram_resources(ns: OptimizationContext, kind: GeometryKind):
    return {
        GeometryKind.CONSTRAINT: (
            ns.scoring.bins_constraints,
            ns.scoring.constraints_grid,
        ),
        GeometryKind.BOND: (ns.scoring.bins_bonds, ns.scoring.bonds_grid),
        GeometryKind.ANGLE: (ns.scoring.bins_angles, ns.scoring.angles_grid),
        GeometryKind.DIHEDRAL: (
            ns.scoring.bins_dihedrals,
            ns.scoring.dihedrals_grid,
        ),
    }[kind]


def _reference_distribution(
    ns: OptimizationContext,
    kind: GeometryKind,
    index: int,
    group,
    bins,
    manual_mode: bool,
) -> tuple[float, np.ndarray]:
    if not manual_mode:
        return group.average, group.histogram
    if kind in (GeometryKind.CONSTRAINT, GeometryKind.BOND):
        average, histogram, _ = scores.get_AA_bonds_distrib(
            ns.scoring.aa2cg_universe,
            beads_ids=group.beads,
            grp_type=f"{kind.plural} group",
            grp_nb=index,
            config=ns.config,
            bins=bins,
            bandwidth=(
                ns.config.optimization.bw_constraints
                if kind == GeometryKind.CONSTRAINT
                else ns.config.optimization.bw_bonds
            ),
        )
        return average, histogram
    getter = (
        scores.get_AA_angles_distrib
        if kind == GeometryKind.ANGLE
        else scores.get_AA_dihedrals_distrib
    )
    average, histogram, _, _ = getter(
        ns.scoring.aa2cg_universe,
        beads_ids=group.beads,
        bins=bins,
        bandwidth=(
            ns.config.optimization.bw_angles
            if kind == GeometryKind.ANGLE
            else ns.config.optimization.bw_dihedrals
        ),
        group_label=f"{kind.value} group {index + 1}",
    )
    return average, histogram


def _cg_distribution(
    ns: OptimizationContext,
    kind: GeometryKind,
    index: int,
    group,
    bins,
) -> tuple[float, np.ndarray, np.ndarray]:
    if kind in (GeometryKind.CONSTRAINT, GeometryKind.BOND):
        return scores.get_CG_bonds_distrib(
            ns.scoring.cg_universe,
            beads_ids=group.beads,
            grp_type=f"{kind.value} group {index + 1}",
            bins=bins,
            bandwidth=(
                ns.config.optimization.bw_constraints
                if kind == GeometryKind.CONSTRAINT
                else ns.config.optimization.bw_bonds
            ),
        )
    getter = (
        scores.get_CG_angles_distrib
        if kind == GeometryKind.ANGLE
        else scores.get_CG_dihedrals_distrib
    )
    average, histogram, values, _ = getter(
        ns.scoring.cg_universe,
        beads_ids=group.beads,
        bins=bins,
        bandwidth=(
            ns.config.optimization.bw_angles
            if kind == GeometryKind.ANGLE
            else ns.config.optimization.bw_dihedrals
        ),
        group_label=f"{kind.value} group {index + 1}",
    )
    return average, histogram, values


def _populate_plot_support(
    aa: DistributionStatistics,
    cg: DistributionStatistics | None,
    edges,
    *,
    periodic: bool = False,
) -> None:
    """Populate inclusive plotting support for typed AA/CG distributions.

    Args:
        aa: Mapped atomistic distribution to populate.
        cg: Optional coarse-grained distribution to populate.
        edges: Histogram edges shared by both distributions.
        periodic: Preserve the full grid when support touches a periodic seam.
    """
    centers = (np.asarray(edges)[:-1] + np.asarray(edges)[1:]) / 2.0
    histograms = [aa.histogram]
    if cg is not None:
        histograms.append(cg.histogram)
    support, selected = scores.support_neighborhood(
        centers, *histograms, periodic=periodic
    )
    aa.x, aa.y = support, selected[0]
    if cg is not None:
        cg.x, cg.y = support, selected[1]


def _pairwise_score(ns, kind, reference, observed, grid) -> float:
    score = scores.earth_movers_distance(reference, observed, grid)
    if kind in (GeometryKind.CONSTRAINT, GeometryKind.BOND):
        score *= ns.config.optimization.bonds2angles_scoring_factor
    return float(score)


def _aggregate_scores(
    ns: OptimizationContext,
    comparisons: tuple[GeometryComparisons, ...],
    *,
    ignore_dihedrals: bool,
    record_best_indep_params: bool,
) -> EvaluationResult:
    pairwise = {
        geometry.kind: tuple(group.emd_score for group in geometry.groups)
        for geometry in comparisons
    }
    if record_best_indep_params:
        for geometry in comparisons:
            if ignore_dihedrals and geometry.kind == GeometryKind.DIHEDRAL:
                continue
            for group in geometry.groups:
                _record_independent_best(ns, group)
    total, constraints_bonds, angles, dihedrals = (
        scores.compose_classwise_l2_score(
            pairwise[GeometryKind.CONSTRAINT],
            pairwise[GeometryKind.BOND],
            pairwise[GeometryKind.ANGLE],
            pairwise[GeometryKind.DIHEDRAL],
        )
    )
    pairwise_text = " ".join(
        str(score)
        for kind in GeometryKind
        for score in pairwise[kind]
    ) + " \n"
    result = EvaluationResult(
        float(total),
        float(constraints_bonds),
        float(angles),
        float(dihedrals),
        pairwise,
        pairwise_text,
    )
    _log_score(ns, result)
    return _with_observables(ns, result)


def _record_independent_best(
    ns: OptimizationContext, comparison: GroupComparison
) -> None:
    plural = comparison.kind.plural
    index = comparison.group_index
    previous = ns.pso.all_best_emd_dist_geoms[plural][index]
    if np.isfinite(previous) and comparison.emd_score >= previous:
        return
    ns.pso.all_best_emd_dist_geoms[plural][index] = comparison.emd_score
    topology_groups = _topology_groups(ns, comparison.kind)
    group = topology_groups[index]
    if comparison.kind == GeometryKind.CONSTRAINT:
        parameters = [ns.out_itp.constraints[index].equilibrium]
    elif comparison.kind == GeometryKind.DIHEDRAL and group.function in (3, 11):
        parameters = list(ns.out_itp.dihedrals[index].gromacs_parameters)
    else:
        output_groups = {
            GeometryKind.BOND: ns.out_itp.bonds,
            GeometryKind.ANGLE: ns.out_itp.angles,
            GeometryKind.DIHEDRAL: ns.out_itp.dihedrals,
        }[comparison.kind]
        parameters = [
            output_groups[index].equilibrium,
            output_groups[index].force_constant,
        ]
    ns.pso.all_best_params_dist_geoms[plural][index]["params"] = parameters


def _log_score(ns: OptimizationContext, result: EvaluationResult) -> None:
    logger.info("")
    logger.info(
        "Using bonds to angles/dihedrals (C) scoring constant: %s",
        ns.config.optimization.bonds2angles_scoring_factor,
    )
    logger.info("")
    logger.info(
        "Global fitness score: %s (lower is better)",
        round(result.total_score, 3),
    )
    logger.info(
        "  Bonds/Constraints constribution to fitness score: %s",
        round(result.constraints_bonds_score, 3),
    )
    logger.info(
        "  Angles constribution to fitness score: %s",
        round(result.angles_score, 3),
    )
    logger.info(
        "  Dihedrals constribution to fitness score: %s",
        round(result.dihedrals_score, 3),
    )


def _render_comparisons(
    ns: OptimizationContext,
    comparisons: tuple[GeometryComparisons, ...],
    result: EvaluationResult | None,
    ignore_dihedrals: bool,
) -> None:
    plt.rcParams["grid.color"] = "k"
    plt.rcParams["grid.linestyle"] = ":"
    plt.rcParams["grid.linewidth"] = 0.5
    populated = [geometry for geometry in comparisons if geometry.groups]
    largest_group = max(len(geometry.groups) for geometry in populated)
    columns = (
        largest_group
        if ns.scoring.ncols_max == 0
        else min(ns.scoring.ncols_max, largest_group)
    )
    _log_plot_order(ns, largest_group, columns)
    figure = plt.figure(figsize=(columns * 3, len(populated) * 3))
    axes = figure.subplots(
        nrows=len(populated), ncols=columns, squeeze=False
    )
    row_limits = []
    for row, geometry in enumerate(populated):
        row_limits.append(
            _render_geometry_row(ns, axes[row], geometry, columns)
        )
    if ns.scoring.row_y_scaling:
        for row, geometry in enumerate(populated):
            lower, upper = row_limits[row]
            for column in range(min(columns, len(geometry.groups))):
                axes[row][column].set_ylim(bottom=lower, top=upper)
    if result is not None:
        plt.tight_layout(rect=[0, 0, 1, 0.9])
        displayed_total = result.total_score
        if ignore_dihedrals and ns.cg_itp.dihedral_count:
            displayed_total -= result.dihedrals_score
        title = (
            f"FITNESS SCORE\nTotal: {round(displayed_total, 3)} -- "
            f"Constraints/Bonds: {round(result.constraints_bonds_score, 3)} -- "
            f"Angles: {round(result.angles_score, 3)} -- "
            f"Dihedrals: {round(result.dihedrals_score, 3)}"
        )
        if ignore_dihedrals and ns.cg_itp.dihedral_count:
            title += " (ignored)"
        plt.suptitle(title)
    else:
        plt.tight_layout()
    plt.savefig(ns.files.plot_filename)
    plt.close(figure)
    logger.info("")
    logger.info(
        "Distributions plot written at location:\n %s", ns.files.plot_filename
    )
    logger.info("")


def _log_plot_order(
    ns: OptimizationContext, largest_group: int, columns: int
) -> None:
    hidden = largest_group - columns
    if hidden > 0 and ns.scoring.atom_only:
        logger.info(
            "Displaying max %s distributions per row using the CG ITP file "
            "ordering of distributions groups (%s more are hidden)",
            columns,
            hidden,
        )
    elif hidden > 0 and not ns.scoring.mismatch_order:
        logger.warning(
            "%sDisplaying max %s distributions groups per row and this can "
            "be MISLEADING because ordering by pairwise AA-mapped vs. CG "
            "distributions mismatch is DISABLED (%s more are hidden)",
            styling.header_warning,
            columns,
            hidden,
        )
    elif hidden > 0:
        logger.info(
            "Displaying max %s distributions groups per row ordered by pairwise "
            "AA-mapped vs. CG distributions difference (%s more are hidden)",
            columns,
            hidden,
        )
    elif not ns.scoring.mismatch_order:
        logger.info("")
        logger.info(
            "Distributions groups will be displayed using the CG ITP file "
            "groups ordering"
        )
    else:
        logger.info("")
        logger.info(
            "Distributions groups will be displayed using ranked mismatch "
            "score between pairwise AA-mapped and CG distributions"
        )


def _render_geometry_row(
    ns: OptimizationContext,
    axes,
    geometry: GeometryComparisons,
    columns: int,
) -> tuple[float, float]:
    logger.info("")
    minimum, maximum = 10.0, 0.0
    for column in range(columns):
        axis = axes[column]
        if column >= len(geometry.groups):
            axis.set_visible(False)
            continue
        comparison = geometry.groups[geometry.ordered_indices[column]]
        _draw_distribution(axis, comparison.aa, "AA-mapped", config.atom_color)
        if np.isfinite(comparison.aa.average):
            axis.plot(
                comparison.aa.average, 0, color=config.atom_color, marker="D"
            )
        if comparison.cg is not None:
            _draw_distribution(axis, comparison.cg, "CG", config.cg_color)
            _annotate_missing_mass(axis, comparison.cg.observation)
            if np.isfinite(comparison.cg.average):
                axis.plot(
                    comparison.cg.average,
                    0,
                    color=config.cg_color,
                    marker="D",
                )
        _title_and_log(axis, comparison)
        axis.grid(zorder=0.5)
        if ns.scoring.row_x_scaling:
            center = np.mean(comparison.x_range)
            radius = geometry.max_x_range / 2 * 1.1
            axis.set_xlim(center - radius, center + radius)
        if column % 2 == 0:
            axis.legend(loc="upper left")
        lower, upper = axis.get_ylim()
        minimum = min(minimum, lower)
        maximum = max(maximum, upper)
    return minimum, maximum


def _draw_distribution(axis, distribution, label: str, color: str) -> None:
    if config.use_hists:
        axis.step(
            distribution.x,
            distribution.y,
            label=label,
            color=color,
            where="mid",
            alpha=config.line_alpha,
        )
        axis.fill_between(
            distribution.x,
            distribution.y,
            color=color,
            step="mid",
            alpha=config.fill_alpha,
        )
    else:
        axis.plot(
            distribution.x,
            distribution.y,
            label=label,
            color=color,
            alpha=config.line_alpha,
        )
        axis.fill_between(
            distribution.x,
            distribution.y,
            color=color,
            alpha=config.fill_alpha,
        )


def _title_and_log(axis, comparison: GroupComparison) -> None:
    label = comparison.kind.value.capitalize()
    number = comparison.group_index + 1
    if comparison.cg is not None:
        axis.set_title(
            f"{label} grp {number} - EMD Δ {round(comparison.emd_score, 3)}"
        )
        if comparison.kind in (GeometryKind.CONSTRAINT, GeometryKind.BOND):
            logger.info(
                "%s %s -- AA Avg: %s nm -- CG Avg: %s%s",
                label,
                number,
                round(comparison.aa.average, 3),
                round(comparison.cg.average, 3),
                " nm" if comparison.kind == GeometryKind.BOND else "",
            )
        elif comparison.kind == GeometryKind.ANGLE:
            logger.info(
                "Angle %s -- AA Avg: %s° -- CG Avg: %s°",
                number,
                round(comparison.aa.average, 1),
                round(comparison.cg.average, 1),
            )
        else:
            logger.info(
                "Dihedral %s -- AA Avg: %s -- CG Avg: %s",
                number,
                _format_circular_mean(comparison.aa.average),
                _format_circular_mean(comparison.cg.average),
            )
        return
    if comparison.kind in (GeometryKind.CONSTRAINT, GeometryKind.BOND):
        average = f"{round(comparison.emd_score, 3)} nm"
    elif comparison.kind == GeometryKind.ANGLE:
        average = f"{round(comparison.emd_score, 1)}°"
    else:
        average = _format_circular_mean(comparison.emd_score)
    axis.set_title(f"{label} grp {number} - Avg {average}")
    logger.info(
        "%s %s -- AA Avg: %s",
        label,
        number,
        (
            _format_circular_mean(comparison.aa.average)
            if comparison.kind == GeometryKind.DIHEDRAL
            else round(
                comparison.aa.average,
                1 if comparison.kind == GeometryKind.ANGLE else 3,
            )
        ),
    )


def _annotate_missing_mass(axis, observation: HistogramObservation) -> None:
    """Annotate a plot when a CG histogram has incomplete sample coverage.

    Args:
        axis: Matplotlib axis receiving the annotation.
        observation: Classified CG histogram masses and missing-data causes.
    """
    missing = max(0.0, 1.0 - observation.coverage)
    if missing > 1e-12:
        axis.text(
            0.98,
            0.94,
            f"CG missing {100.0 * missing:.1f}%\n"
            f"nonfinite={observation.nonfinite_count}, "
            f"below={observation.underflow_count}, "
            f"above={observation.overflow_count}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize="small",
            color=config.cg_color,
        )


def _format_circular_mean(value: float) -> str:
    """Format an optional circular mean for plots and log messages.

    Args:
        value: Circular mean in degrees, potentially ``NaN``.

    Returns:
        One-decimal degree text or ``"unavailable"``.
    """
    return f"{float(value):.1f}°" if np.isfinite(value) else "unavailable"


def _compute_optional_sasa(ns: OptimizationContext) -> None:
    try:
        if ns.results.sasa_aa_mapped is None:
            ns.results.sasa_aa_mapped, ns.results.sasa_aa_mapped_std = (
                scores.compute_SASA(ns, traj_type="AA_mapped")
            )
        ns.results.sasa_cg, ns.results.sasa_cg_std = scores.compute_SASA(
            ns, traj_type="CG"
        )
        logger.info(
            "SASA (AA reference, CG-mapped): %s +/- %s nm2",
            ns.results.sasa_aa_mapped,
            ns.results.sasa_aa_mapped_std,
        )
        logger.info(
            "SASA (CG model): %s +/- %s nm2",
            ns.results.sasa_cg,
            ns.results.sasa_cg_std,
        )
    except Exception as exc:
        ns.results.sasa_cg = ns.results.sasa_cg_std = None
        logger.warning(
            "Optional SASA diagnostic failed and will not affect fitness: %s",
            exc,
        )
    logger.info("")


def _with_observables(
    ns: OptimizationContext, result: EvaluationResult
) -> EvaluationResult:
    return EvaluationResult(
        total_score=result.total_score,
        constraints_bonds_score=result.constraints_bonds_score,
        angles_score=result.angles_score,
        dihedrals_score=result.dihedrals_score,
        pairwise_scores=result.pairwise_scores,
        pairwise_text=result.pairwise_text,
        rg_aa_mapped=ObservableStatistics(
            ns.results.gyr_aa_mapped, ns.results.gyr_aa_mapped_std
        ),
        rg_cg=ObservableStatistics(ns.results.gyr_cg, ns.results.gyr_cg_std),
        sasa_aa_mapped=ObservableStatistics(
            ns.results.sasa_aa_mapped, ns.results.sasa_aa_mapped_std
        ),
        sasa_cg=ObservableStatistics(
            ns.results.sasa_cg, ns.results.sasa_cg_std
        ),
    )
