"""Typed internal models for optimization cycles and parameter vectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from swarmcg.config_types import SwarmConfig
from swarmcg.shared.periodic import (
    PeriodicDihedralParameters,
    normalize_periodic_degrees,
    unwrap_degrees_around,
)
from swarmcg.simulations.polynomial import CBTParameters, RBParameters
from swarmcg.topology import CGTopology, GeometryKind, HarmonicParameters


@dataclass(frozen=True)
class GeometryCounts:
    """Numbers of active groups for each bonded geometry kind.

    Args:
        constraints: Number of active constraint groups.
        bonds: Number of active bond groups.
        angles: Number of active angle groups.
        dihedrals: Number of active dihedral groups.
    """

    constraints: int = 0
    bonds: int = 0
    angles: int = 0
    dihedrals: int = 0

    def for_kind(self, kind: GeometryKind) -> int:
        """Return the number of active groups for one geometry kind.

        Args:
            kind: Geometry kind to query.

        Returns:
            Number of active groups of ``kind``.
        """
        return {
            GeometryKind.CONSTRAINT: self.constraints,
            GeometryKind.BOND: self.bonds,
            GeometryKind.ANGLE: self.angles,
            GeometryKind.DIHEDRAL: self.dihedrals,
        }[kind]

    @property
    def total(self) -> int:
        """Return the total number of active geometry groups."""
        return self.constraints + self.bonds + self.angles + self.dihedrals


@dataclass(frozen=True)
class OptimizationCycle:
    """One staged optimization cycle.

    Args:
        number: One-based cycle number.
        geometries: Geometry kinds optimized during the cycle.
        counts: Number of active groups for each geometry kind.
    """

    number: int
    geometries: tuple[GeometryKind, ...]
    counts: GeometryCounts

    @classmethod
    def from_topology(
        cls,
        number: int,
        geometries: Sequence[str | GeometryKind],
        topology: CGTopology,
    ) -> "OptimizationCycle":
        """Build a cycle and derive its active counts from a topology.

        Args:
            number: One-based cycle number.
            geometries: Geometry names active during this cycle.
            topology: Topology whose groups will be optimized.

        Returns:
            Typed optimization-cycle description.
        """
        kinds = tuple(GeometryKind(geometry) for geometry in geometries)
        active = set(kinds)
        return cls(
            number=number,
            geometries=kinds,
            counts=GeometryCounts(
                constraints=(
                    topology.constraint_count
                    if GeometryKind.CONSTRAINT in active
                    else 0
                ),
                bonds=topology.bond_count if GeometryKind.BOND in active else 0,
                angles=topology.angle_count if GeometryKind.ANGLE in active else 0,
                dihedrals=(
                    topology.dihedral_count
                    if GeometryKind.DIHEDRAL in active
                    else 0
                ),
            ),
        )

    def includes(self, kind: str | GeometryKind) -> bool:
        """Return whether a geometry kind is active in this cycle.

        Args:
            kind: Geometry kind or its singular string value.

        Returns:
            ``True`` when ``kind`` is active.
        """
        return GeometryKind(kind) in self.geometries


@dataclass(frozen=True)
class SimulationSetup:
    """Simulation and swarm settings for one optimization cycle.

    Args:
        duration_ns: Production simulation duration in nanoseconds.
        frame_count: Number of production trajectory frames.
        max_swarm_iterations: Maximum number of swarm iterations, or ``None``
            when derived from the parameter-vector dimension.
        max_iterations_without_improvement: Early-stop iteration count.
        equilibrium_guess_factor: Equilibrium-value exploration multiplier.
        force_guess_factor: Force-parameter exploration multiplier.
    """

    duration_ns: float
    frame_count: int
    max_swarm_iterations: int | None
    max_iterations_without_improvement: int
    equilibrium_guess_factor: float
    force_guess_factor: float

    @classmethod
    def from_mapping(cls, values: Mapping[str, float | int | None]) -> "SimulationSetup":
        """Create typed settings from the strategy's internal mapping.

        Args:
            values: Strategy values using the historical setting names.

        Returns:
            Validated immutable simulation setup.
        """
        max_iterations = values["max_swarm_iter"]
        return cls(
            duration_ns=float(values["sim_duration"]),
            frame_count=int(values["prod_nb_frames"]),
            max_swarm_iterations=(
                None if max_iterations is None else int(max_iterations)
            ),
            max_iterations_without_improvement=int(
                values["max_swarm_iter_without_new_global_best"]
            ),
            equilibrium_guess_factor=float(values["val_guess_fact"]),
            force_guess_factor=float(values["fct_guess_fact"]),
        )


@dataclass(frozen=True)
class EvaluationResult:
    """Typed result of comparing one coarse-grained model to its reference.

    Args:
        total_score: Aggregate mismatch across all geometry classes.
        constraints_bonds_score: Combined constraint and bond mismatch.
        angles_score: Aggregate angle mismatch.
        dihedrals_score: Aggregate dihedral mismatch.
        pairwise_scores: Per-group mismatch scores by geometry kind.
        pairwise_text: Legacy whitespace serialization retained until the
            versioned history cutover.
    """

    total_score: float
    constraints_bonds_score: float
    angles_score: float
    dihedrals_score: float
    pairwise_scores: Mapping[GeometryKind, tuple[float, ...]]
    pairwise_text: str = ""


@dataclass(frozen=True)
class _ParameterSlot:
    kind: GeometryKind
    group_index: int
    component: str
    bounds: tuple[float, float]
    coefficient_index: int | None = None


@dataclass(frozen=True)
class ParameterVectorLayout:
    """Authoritative ordering and bounds of a cycle's free PSO parameters.

    Args:
        cycle: Active optimization cycle.
        execution_mode: ``1`` to optimize equilibria and force terms, or ``2``
            to optimize force terms only.
        slots: Ordered internal parameter descriptors.
    """

    cycle: OptimizationCycle
    execution_mode: int
    slots: tuple[_ParameterSlot, ...]

    @classmethod
    def build(
        cls,
        topology: CGTopology,
        cycle: OptimizationCycle,
        domains: Mapping[str, Sequence[Sequence[float] | None]],
        execution_mode: int,
        config: SwarmConfig,
    ) -> "ParameterVectorLayout":
        """Build the free-parameter layout for one topology and cycle.

        Args:
            topology: Baseline coarse-grained topology.
            cycle: Active optimization cycle.
            domains: Reference-derived equilibrium search domains.
            execution_mode: Optimization execution mode, either ``1`` or ``2``.
            config: Validated application configuration.

        Returns:
            Parameter-vector layout in canonical PSO order.

        Raises:
            ValueError: If the execution mode, topology function, or required
                domain/coefficient bound is invalid.
        """
        if execution_mode not in (1, 2):
            raise ValueError("execution_mode must be 1 or 2")
        slots: list[_ParameterSlot] = []
        opt = config.optimization

        if execution_mode == 1:
            slots.extend(
                cls._equilibrium_slots(
                    GeometryKind.CONSTRAINT,
                    cycle.counts.constraints,
                    domains,
                )
            )
            slots.extend(
                cls._equilibrium_slots(
                    GeometryKind.BOND, cycle.counts.bonds, domains
                )
            )
        slots.extend(
            _ParameterSlot(
                GeometryKind.BOND,
                index,
                "force_constant",
                (0.0, float(opt.default_max_fct_bonds_opti)),
            )
            for index in range(cycle.counts.bonds)
        )

        if execution_mode == 1:
            slots.extend(
                cls._equilibrium_slots(
                    GeometryKind.ANGLE, cycle.counts.angles, domains
                )
            )
        angle_maxima = {
            1: opt.default_max_fct_angles_opti_f1,
            2: opt.default_max_fct_angles_opti_f2,
            10: opt.default_max_fct_angles_opti_f10,
        }
        for index in range(cycle.counts.angles):
            function = topology.angles[index].function
            if function not in angle_maxima:
                raise ValueError(f"unsupported angle function {function}")
            slots.append(
                _ParameterSlot(
                    GeometryKind.ANGLE,
                    index,
                    "force_constant",
                    (0.0, float(angle_maxima[function])),
                )
            )

        for index in range(cycle.counts.dihedrals):
            group = topology.dihedrals[index]
            if group.function in (3, 11):
                if group.coefficient_bound is None or group.coefficient_bound <= 0:
                    raise ValueError(
                        f"dihedral group {index + 1} has no positive coefficient bound"
                    )
                slots.extend(
                    _ParameterSlot(
                        GeometryKind.DIHEDRAL,
                        index,
                        "coefficient",
                        (-float(group.coefficient_bound), float(group.coefficient_bound)),
                        coefficient_index=coefficient_index,
                    )
                    for coefficient_index in range(5)
                )
                continue
            if execution_mode == 1:
                slots.extend(
                    cls._equilibrium_slots(
                        GeometryKind.DIHEDRAL, 1, domains, start=index
                    )
                )
            maximum = (
                opt.default_abs_range_fct_dihedrals_opti_func_without_mult
                if group.function == 2
                else opt.default_abs_range_fct_dihedrals_opti_func_with_mult
            )
            lower = 0.0 if group.function in (1, 4) else -float(maximum)
            slots.append(
                _ParameterSlot(
                    GeometryKind.DIHEDRAL,
                    index,
                    "force_constant",
                    (lower, float(maximum)),
                )
            )
        return cls(cycle=cycle, execution_mode=execution_mode, slots=tuple(slots))

    @staticmethod
    def _equilibrium_slots(
        kind: GeometryKind,
        count: int,
        domains: Mapping[str, Sequence[Sequence[float] | None]],
        start: int = 0,
    ) -> list[_ParameterSlot]:
        slots = []
        for index in range(start, start + count):
            domain = domains[kind.value][index]
            if domain is None or len(domain) != 2:
                raise ValueError(
                    f"{kind.value} group {index + 1} has no finite search domain"
                )
            lower, upper = map(float, domain)
            if not np.isfinite((lower, upper)).all() or lower > upper:
                raise ValueError(
                    f"{kind.value} group {index + 1} has an invalid search domain"
                )
            slots.append(
                _ParameterSlot(kind, index, "equilibrium", (lower, upper))
            )
        return slots

    @property
    def dimension(self) -> int:
        """Return the number of free parameters in the vector."""
        return len(self.slots)

    @property
    def bounds(self) -> list[list[float]]:
        """Return mutable lower/upper pairs accepted by FST-PSO."""
        return [[slot.bounds[0], slot.bounds[1]] for slot in self.slots]

    def encode(
        self, topology: CGTopology, *, input_parameters: bool = False
    ) -> list[float]:
        """Encode active or original topology parameters in PSO order.

        Args:
            topology: Topology providing parameter values.
            input_parameters: Encode original input values instead of active
                staged values.

        Returns:
            Clipped flat vector in this layout's canonical order.
        """
        values = [
            self._slot_value(topology, slot, input_parameters=input_parameters)
            for slot in self.slots
        ]
        return self.clamp(values)

    def encode_independent_best(
        self,
        topology: CGTopology,
        best_scores: Mapping[str, Mapping[int, float]],
        best_parameters: Mapping[str, Mapping[int, Mapping[str, Sequence[float]]]],
    ) -> list[float]:
        """Encode finite per-group best parameters with staged fallbacks.

        Args:
            topology: Staged topology used when no finite group best exists.
            best_scores: Per-geometry independent best EMD values.
            best_parameters: Parameter sequences associated with those scores.

        Returns:
            Clipped flat vector in canonical layout order.
        """
        staged = self.encode(topology)
        values = []
        for position, slot in enumerate(self.slots):
            plural = slot.kind.plural
            try:
                score = float(best_scores[plural][slot.group_index])
                parameters = best_parameters[plural][slot.group_index]["params"]
            except (KeyError, IndexError, TypeError, ValueError):
                values.append(staged[position])
                continue
            if not np.isfinite(score):
                values.append(staged[position])
                continue
            try:
                values.append(self._history_value(topology, slot, parameters))
            except (IndexError, TypeError, ValueError):
                values.append(staged[position])
        return self.clamp(values)

    def clamp(self, values: Sequence[float]) -> list[float]:
        """Validate and clip a vector to this layout's bounds.

        Args:
            values: Candidate values in canonical parameter order.

        Returns:
            Finite values clipped to their corresponding bounds.

        Raises:
            ValueError: If the vector has the wrong dimension or contains a
                non-finite value.
        """
        vector = self._validate_vector(values)
        return [
            float(min(max(value, slot.bounds[0]), slot.bounds[1]))
            for value, slot in zip(vector, self.slots)
        ]

    def apply(self, topology: CGTopology, values: Sequence[float]) -> None:
        """Apply one PSO vector to a topology in place.

        Args:
            topology: Topology whose active parameters will be replaced.
            values: Candidate vector in canonical order.

        Raises:
            ValueError: If the vector has the wrong dimension or contains a
                non-finite value.
        """
        vector = self._validate_vector(values)
        polynomial: dict[int, list[float]] = {}
        for value, slot in zip(vector, self.slots):
            if slot.kind == GeometryKind.CONSTRAINT:
                topology.constraints[slot.group_index].equilibrium = round(value, 3)
            elif slot.kind == GeometryKind.BOND:
                group = topology.bonds[slot.group_index]
                if slot.component == "equilibrium":
                    group.equilibrium = round(value, 3)
                else:
                    group.force_constant = round(value, 3)
            elif slot.kind == GeometryKind.ANGLE:
                group = topology.angles[slot.group_index]
                if slot.component == "equilibrium":
                    group.equilibrium = round(value, 2)
                else:
                    group.force_constant = round(value, 2)
            elif slot.component == "coefficient":
                polynomial.setdefault(slot.group_index, [0.0] * 5)[
                    slot.coefficient_index
                ] = round(value, 8)
            else:
                self._apply_dihedral_value(topology, slot, value)
        for index, coefficients in polynomial.items():
            function = topology.dihedrals[index].function
            topology.dihedrals[index].parameters = (
                RBParameters(tuple(coefficients))
                if function == 3
                else CBTParameters(tuple(coefficients))
            )

    def _validate_vector(self, values: Sequence[float]) -> np.ndarray:
        if len(values) != self.dimension:
            raise ValueError(
                f"parameter vector has dimension {len(values)}, expected {self.dimension}"
            )
        vector = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(vector)):
            raise ValueError("parameter vector must contain only finite values")
        return vector

    def _slot_value(
        self,
        topology: CGTopology,
        slot: _ParameterSlot,
        *,
        input_parameters: bool,
    ) -> float:
        groups = {
            GeometryKind.CONSTRAINT: topology.constraints,
            GeometryKind.BOND: topology.bonds,
            GeometryKind.ANGLE: topology.angles,
            GeometryKind.DIHEDRAL: topology.dihedrals,
        }
        group = groups[slot.kind][slot.group_index]
        prefix = "input_" if input_parameters else ""
        if slot.component == "coefficient":
            parameters = (
                group.input_parameters if input_parameters else group.parameters
            )
            coefficients = (
                parameters.coefficients
                if isinstance(parameters, RBParameters)
                else parameters.effective_coefficients
            )
            return float(coefficients[slot.coefficient_index])
        value = getattr(group, f"{prefix}{slot.component}")
        if slot.kind == GeometryKind.DIHEDRAL and slot.component == "equilibrium":
            center = (slot.bounds[0] + slot.bounds[1]) / 2.0
            value = unwrap_degrees_around(np.array([value]), center)[0]
        return float(value)

    def _history_value(
        self,
        topology: CGTopology,
        slot: _ParameterSlot,
        parameters: Sequence[float],
    ) -> float:
        if slot.component == "coefficient":
            group = topology.dihedrals[slot.group_index]
            typed = (
                RBParameters.from_gromacs(parameters)
                if group.function == 3
                else CBTParameters.from_gromacs(parameters)
            )
            coefficients = (
                typed.coefficients
                if isinstance(typed, RBParameters)
                else typed.effective_coefficients
            )
            return coefficients[slot.coefficient_index]
        parameter_index = 0 if slot.component == "equilibrium" else 1
        value = float(parameters[parameter_index])
        if slot.kind == GeometryKind.DIHEDRAL and slot.component == "equilibrium":
            center = (slot.bounds[0] + slot.bounds[1]) / 2.0
            value = unwrap_degrees_around(np.array([value]), center)[0]
        return float(value)

    @staticmethod
    def _apply_dihedral_value(
        topology: CGTopology, slot: _ParameterSlot, value: float
    ) -> None:
        group = topology.dihedrals[slot.group_index]
        if slot.component == "equilibrium":
            group.equilibrium = round(normalize_periodic_degrees(value), 2)
            return
        force_constant = round(value, 2)
        equilibrium = group.equilibrium
        if group.function in (1, 4):
            group.parameters = PeriodicDihedralParameters.from_gromacs(
                equilibrium, force_constant, group.multiplicity
            )
        else:
            group.parameters = HarmonicParameters(equilibrium, force_constant)
