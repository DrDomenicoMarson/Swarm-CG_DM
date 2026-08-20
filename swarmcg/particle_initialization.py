"""Particle initialization for typed optimization parameter layouts."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from swarmcg.config_types import SwarmConfig
from swarmcg.optimization_types import ParameterVectorLayout, _ParameterSlot
from swarmcg.shared.math_utils import draw_float
from swarmcg.topology import CGTopology, GeometryKind


def initialize_particles(
    particle_count: int,
    layout: ParameterVectorLayout,
    baseline: CGTopology,
    staged: CGTopology,
    best_scores: Mapping[str, Mapping[int, float]],
    best_parameters: Mapping[str, Mapping[int, Mapping[str, Sequence[float]]]],
    config: SwarmConfig,
    *,
    use_input_seed: bool = False,
    equilibrium_guess_factor: float = 1.0,
    force_guess_factor: float = 0.5,
) -> list[list[float]]:
    """Build deterministic and exploratory PSO particle positions.

    Args:
        particle_count: Number of particles to initialize.
        layout: Authoritative parameter ordering and bounds.
        baseline: Parsed baseline topology used for function metadata.
        staged: Working topology containing the current cycle parameters.
        best_scores: Best known per-group EMD scores.
        best_parameters: Parameters associated with the independent scores.
        config: Validated application configuration.
        use_input_seed: Include original user parameters in the first cycle.
        equilibrium_guess_factor: Equilibrium exploration multiplier.
        force_guess_factor: Force-parameter exploration multiplier.

    Returns:
        One flat parameter vector per particle.

    Raises:
        ValueError: If ``particle_count`` is not positive.
    """
    if particle_count <= 0:
        raise ValueError("particle_count must be positive")
    particles: list[list[float]] = []
    _append_unique(particles, layout.encode(staged))

    if layout.cycle.number > 1:
        _append_unique(
            particles,
            layout.encode_independent_best(staged, best_scores, best_parameters),
        )
    elif use_input_seed:
        _append_unique(particles, layout.encode(staged, input_parameters=True))

    while len(particles) < particle_count:
        particles.append(
            _draw_particle(
                layout,
                baseline,
                staged,
                best_scores,
                config,
                equilibrium_guess_factor,
                force_guess_factor,
            )
        )
    return particles[:particle_count]


def _append_unique(particles: list[list[float]], candidate: Sequence[float]) -> None:
    candidate_array = np.asarray(candidate, dtype=float)
    if not any(
        candidate_array.shape == np.asarray(existing).shape
        and np.allclose(candidate_array, existing, rtol=0.0, atol=1e-12)
        for existing in particles
    ):
        particles.append(list(candidate))


def _finite_history_score(
    best_scores: Mapping[str, Mapping[int, float]], slot: _ParameterSlot
) -> float | None:
    try:
        score = float(best_scores[slot.kind.plural][slot.group_index])
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return score if np.isfinite(score) else None


def _history_factor(
    best_scores: Mapping[str, Mapping[int, float]],
    slot: _ParameterSlot,
    divisor: float,
) -> float:
    score = _finite_history_score(best_scores, slot)
    return 1.0 if score is None else max(1.0, score / divisor)


def _draw_particle(
    layout: ParameterVectorLayout,
    baseline: CGTopology,
    staged: CGTopology,
    best_scores: Mapping[str, Mapping[int, float]],
    config: SwarmConfig,
    equilibrium_factor: float,
    force_factor: float,
) -> list[float]:
    particle = []
    for slot in layout.slots:
        if slot.component == "equilibrium":
            particle.append(
                _draw_equilibrium(
                    layout,
                    staged,
                    best_scores,
                    slot,
                    config,
                    equilibrium_factor,
                )
            )
        elif slot.component == "coefficient":
            particle.append(
                _draw_polynomial_coefficient(
                    layout,
                    staged,
                    best_scores,
                    slot,
                    config,
                    force_factor,
                )
            )
        else:
            particle.append(
                _draw_force_constant(
                    layout,
                    baseline,
                    staged,
                    best_scores,
                    slot,
                    config,
                    force_factor,
                )
            )
    return particle


def _draw_equilibrium(
    layout: ParameterVectorLayout,
    staged: CGTopology,
    best_scores: Mapping[str, Mapping[int, float]],
    slot: _ParameterSlot,
    config: SwarmConfig,
    guess_factor: float,
) -> float:
    value = layout._slot_value(staged, slot, input_parameters=False)
    divisor = 5.0 if slot.kind == GeometryKind.DIHEDRAL else 2.0
    history_factor = _history_factor(best_scores, slot, divisor)
    variation = {
        GeometryKind.CONSTRAINT: config.optimization.bond_dist_guess_variation,
        GeometryKind.BOND: config.optimization.bond_dist_guess_variation,
        GeometryKind.ANGLE: config.optimization.angle_value_guess_variation,
        GeometryKind.DIHEDRAL: config.optimization.dihedral_value_guess_variation,
    }[slot.kind]
    radius = variation * guess_factor * history_factor
    lower = max(value - radius, slot.bounds[0])
    upper = min(value + radius, slot.bounds[1])
    return draw_float(lower, upper, 3)


def _draw_force_constant(
    layout: ParameterVectorLayout,
    baseline: CGTopology,
    staged: CGTopology,
    best_scores: Mapping[str, Mapping[int, float]],
    slot: _ParameterSlot,
    config: SwarmConfig,
    guess_factor: float,
) -> float:
    value = layout._slot_value(staged, slot, input_parameters=False)
    divisor = 5.0 if slot.kind == GeometryKind.DIHEDRAL else 2.0
    history_factor = _history_factor(best_scores, slot, divisor)
    if slot.kind == GeometryKind.BOND:
        flat_radius = config.optimization.fct_guess_min_flat_diff_bonds
    elif slot.kind == GeometryKind.ANGLE:
        flat_radius = config.optimization.fct_guess_min_flat_diff_angles
    else:
        function = baseline.dihedrals[slot.group_index].function
        flat_radius = (
            config.optimization.fct_guess_min_flat_diff_dihedrals_without_mult
            if function == 2
            else config.optimization.fct_guess_min_flat_diff_dihedrals_with_mult
        )
    relative_radius = guess_factor * history_factor
    if value > 0:
        relative_lower = value * (1 - relative_radius)
        relative_upper = value * (1 + relative_radius)
    else:
        relative_lower = value * (1 + relative_radius)
        relative_upper = value * (1 - relative_radius)
    lower = max(min(relative_lower, value - flat_radius), slot.bounds[0])
    upper = min(max(relative_upper, value + flat_radius), slot.bounds[1])
    return draw_float(lower, upper, 3)


def _draw_polynomial_coefficient(
    layout: ParameterVectorLayout,
    staged: CGTopology,
    best_scores: Mapping[str, Mapping[int, float]],
    slot: _ParameterSlot,
    config: SwarmConfig,
    guess_factor: float,
) -> float:
    if _finite_history_score(best_scores, slot) is None:
        return draw_float(slot.bounds[0], slot.bounds[1], 3)
    value = layout._slot_value(staged, slot, input_parameters=False)
    history_factor = _history_factor(best_scores, slot, 5.0)
    radius = guess_factor * history_factor
    if value > 0:
        relative_lower = value * (1 - radius)
        relative_upper = value * (1 + radius)
    else:
        relative_lower = value * (1 + radius)
        relative_upper = value * (1 - radius)
    flat_radius = (
        config.optimization.fct_guess_min_flat_diff_dihedrals_without_mult
    )
    lower = max(min(relative_lower, value - flat_radius), slot.bounds[0])
    upper = min(max(relative_upper, value + flat_radius), slot.bounds[1])
    return draw_float(lower, upper, 3)
