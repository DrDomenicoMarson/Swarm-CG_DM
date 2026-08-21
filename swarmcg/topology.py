"""Internal typed representation of coarse-grained GROMACS topologies."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, TypeAlias

from swarmcg.shared import exceptions
from swarmcg.shared.periodic import PeriodicDihedralParameters
from swarmcg.simulations.polynomial import CBTParameters, RBParameters


class GeometryKind(StrEnum):
    """Kinds of bonded geometry represented by an optimization topology."""

    CONSTRAINT = "constraint"
    BOND = "bond"
    ANGLE = "angle"
    DIHEDRAL = "dihedral"

    @property
    def plural(self) -> str:
        """Return the plural spelling used in reports and serialized history."""
        return f"{self.value}s"


@dataclass(frozen=True)
class ConstraintParameters:
    """Immutable parameters for a distance constraint.

    Args:
        length: Constrained distance in nanometers.
    """

    length: float

    def __post_init__(self) -> None:
        """Validate and normalize the constraint length."""
        if not math.isfinite(self.length) or self.length < 0:
            raise ValueError("Constraint length must be finite and nonnegative.")
        object.__setattr__(self, "length", float(self.length))


@dataclass(frozen=True)
class HarmonicParameters:
    """Immutable equilibrium and force parameters for a bonded potential.

    Args:
        equilibrium: Equilibrium distance or angle.
        force_constant: GROMACS force constant in the section's native units.
    """

    equilibrium: float
    force_constant: float

    def __post_init__(self) -> None:
        """Validate and normalize both harmonic parameters."""
        if not math.isfinite(self.equilibrium) or not math.isfinite(
            self.force_constant
        ):
            raise ValueError("Harmonic parameters must be finite.")
        object.__setattr__(self, "equilibrium", float(self.equilibrium))
        object.__setattr__(self, "force_constant", float(self.force_constant))


DihedralParameters: TypeAlias = (
    HarmonicParameters | PeriodicDihedralParameters | RBParameters | CBTParameters
)


@dataclass
class MoleculeType:
    """Metadata from the GROMACS ``moleculetype`` section.

    Args:
        name: Molecule type name.
        exclusion_depth: Number of bonded neighbors excluded from nonbonded terms.
    """

    name: str = ""
    exclusion_depth: int = 0


@dataclass
class Atom:
    """One coarse-grained atom or virtual-site bead.

    Args:
        bead_id: Zero-based internal bead identifier.
        bead_type: GROMACS atom type.
        residue_number: Residue number.
        residue_name: Residue name.
        atom_name: Atom name.
        charge_group: Charge-group number.
        charge: Partial charge.
        mass: Optional explicit mass.
        virtual_site_kind: ``2``, ``3``, ``4``, or ``"n"`` when defined as a
            virtual site.
    """

    bead_id: int
    bead_type: str
    residue_number: int
    residue_name: str
    atom_name: str
    charge_group: int
    charge: float
    mass: float | None = None
    virtual_site_kind: int | str | None = None

    @property
    def is_virtual(self) -> bool:
        """Return whether this atom is defined in a virtual-site section."""
        return self.virtual_site_kind is not None


@dataclass
class ConstraintGroup:
    """A group of constraints that share one parameter set.

    Args:
        geometry_type: Stable group label used in plots and written comments.
        beads: Zero-based bead pairs in topology order.
        function: GROMACS constraint function identifier.
        parameters: Active constraint parameters.
        input_parameters: Original parameters read from the input topology.
    """

    geometry_type: str
    beads: list[tuple[int, int]]
    function: int
    parameters: ConstraintParameters
    input_parameters: ConstraintParameters
    average: float | None = field(default=None, compare=False)
    histogram: Any = field(default=None, repr=False, compare=False)

    @property
    def equilibrium(self) -> float:
        """Return the active constrained distance."""
        return self.parameters.length

    @equilibrium.setter
    def equilibrium(self, value: float) -> None:
        """Replace the active constrained distance."""
        self.parameters = ConstraintParameters(value)

    @property
    def input_equilibrium(self) -> float:
        """Return the constrained distance supplied by the user."""
        return self.input_parameters.length


@dataclass
class BondGroup:
    """A group of bonds that share one parameter set.

    Args:
        geometry_type: Stable group label used in plots and written comments.
        beads: Zero-based bead pairs in topology order.
        function: GROMACS bond function identifier.
        parameters: Active bond parameters.
        input_parameters: Original parameters read from the input topology.
    """

    geometry_type: str
    beads: list[tuple[int, int]]
    function: int
    parameters: HarmonicParameters
    input_parameters: HarmonicParameters
    average: float | None = field(default=None, compare=False)
    histogram: Any = field(default=None, repr=False, compare=False)

    @property
    def equilibrium(self) -> float:
        """Return the active equilibrium distance."""
        return self.parameters.equilibrium

    @equilibrium.setter
    def equilibrium(self, value: float) -> None:
        """Replace the active equilibrium distance."""
        self.parameters = HarmonicParameters(value, self.parameters.force_constant)

    @property
    def force_constant(self) -> float:
        """Return the active force constant."""
        return self.parameters.force_constant

    @force_constant.setter
    def force_constant(self, value: float) -> None:
        """Replace the active force constant."""
        self.parameters = HarmonicParameters(self.parameters.equilibrium, value)

    @property
    def input_equilibrium(self) -> float:
        """Return the user-supplied equilibrium distance."""
        return self.input_parameters.equilibrium

    @property
    def input_force_constant(self) -> float:
        """Return the user-supplied force constant."""
        return self.input_parameters.force_constant


@dataclass
class AngleGroup:
    """A group of angles that share one parameter set.

    Args:
        geometry_type: Stable group label used in plots and written comments.
        beads: Zero-based bead triples in topology order.
        function: GROMACS angle function identifier.
        parameters: Active angle parameters.
        input_parameters: Original parameters read from the input topology.
    """

    geometry_type: str
    beads: list[tuple[int, int, int]]
    function: int
    parameters: HarmonicParameters
    input_parameters: HarmonicParameters
    average: float | None = field(default=None, compare=False)
    histogram: Any = field(default=None, repr=False, compare=False)

    @property
    def equilibrium(self) -> float:
        """Return the active equilibrium angle."""
        return self.parameters.equilibrium

    @equilibrium.setter
    def equilibrium(self, value: float) -> None:
        """Replace the active equilibrium angle."""
        self.parameters = HarmonicParameters(value, self.parameters.force_constant)

    @property
    def force_constant(self) -> float:
        """Return the active force constant."""
        return self.parameters.force_constant

    @force_constant.setter
    def force_constant(self, value: float) -> None:
        """Replace the active force constant."""
        self.parameters = HarmonicParameters(self.parameters.equilibrium, value)

    @property
    def input_equilibrium(self) -> float:
        """Return the user-supplied equilibrium angle."""
        return self.input_parameters.equilibrium

    @property
    def input_force_constant(self) -> float:
        """Return the user-supplied force constant."""
        return self.input_parameters.force_constant


@dataclass
class DihedralGroup:
    """A group of dihedrals that share one typed parameter set.

    Args:
        geometry_type: Stable group label used in plots and written comments.
        beads: Zero-based bead quadruples in topology order.
        function: GROMACS dihedral function identifier.
        parameters: Active typed dihedral parameters.
        input_parameters: Original typed parameters read from the input topology.
    """

    geometry_type: str
    beads: list[tuple[int, int, int, int]]
    function: int
    parameters: DihedralParameters
    input_parameters: DihedralParameters
    average: float | None = field(default=None, compare=False)
    histogram: Any = field(default=None, repr=False, compare=False)
    phase_moment_resultant: float | None = field(default=None, compare=False)
    polynomial_symmetry_tv: float | None = field(default=None, compare=False)
    coefficient_bound: float | None = field(default=None, compare=False)

    @property
    def equilibrium(self) -> float | None:
        """Return the active phase/equilibrium angle, if the form has one."""
        if isinstance(self.parameters, PeriodicDihedralParameters):
            return self.parameters.phase_degrees
        if isinstance(self.parameters, HarmonicParameters):
            return self.parameters.equilibrium
        return None

    @equilibrium.setter
    def equilibrium(self, value: float) -> None:
        """Replace the active phase/equilibrium angle."""
        if isinstance(self.parameters, PeriodicDihedralParameters):
            self.parameters = PeriodicDihedralParameters(
                value,
                self.parameters.force_constant,
                self.parameters.multiplicity,
            )
        elif isinstance(self.parameters, HarmonicParameters):
            self.parameters = HarmonicParameters(value, self.parameters.force_constant)
        else:
            raise TypeError("Polynomial dihedrals do not have an equilibrium angle.")

    @property
    def force_constant(self) -> float | None:
        """Return the active force constant, if the form has one."""
        if isinstance(
            self.parameters, (PeriodicDihedralParameters, HarmonicParameters)
        ):
            return self.parameters.force_constant
        return None

    @force_constant.setter
    def force_constant(self, value: float) -> None:
        """Replace the active force constant."""
        if isinstance(self.parameters, PeriodicDihedralParameters):
            self.parameters = PeriodicDihedralParameters(
                self.parameters.phase_degrees,
                value,
                self.parameters.multiplicity,
            )
        elif isinstance(self.parameters, HarmonicParameters):
            self.parameters = HarmonicParameters(self.parameters.equilibrium, value)
        else:
            raise TypeError("Polynomial dihedrals do not have a force constant.")

    @property
    def multiplicity(self) -> int | None:
        """Return the active periodic multiplicity, if applicable."""
        return (
            self.parameters.multiplicity
            if isinstance(self.parameters, PeriodicDihedralParameters)
            else None
        )

    @property
    def input_equilibrium(self) -> float | None:
        """Return the user-supplied phase/equilibrium angle, if applicable."""
        if isinstance(self.input_parameters, PeriodicDihedralParameters):
            return self.input_parameters.phase_degrees
        if isinstance(self.input_parameters, HarmonicParameters):
            return self.input_parameters.equilibrium
        return None

    @property
    def input_force_constant(self) -> float | None:
        """Return the user-supplied force constant, if applicable."""
        if isinstance(
            self.input_parameters,
            (PeriodicDihedralParameters, HarmonicParameters),
        ):
            return self.input_parameters.force_constant
        return None

    @property
    def gromacs_parameters(self) -> tuple[float, ...]:
        """Return the active canonical GROMACS parameter sequence."""
        if isinstance(self.parameters, PeriodicDihedralParameters):
            return (
                self.parameters.phase_degrees,
                self.parameters.force_constant,
            )
        if isinstance(self.parameters, HarmonicParameters):
            return (
                self.parameters.equilibrium,
                self.parameters.force_constant,
            )
        return self.parameters.to_gromacs()

    @property
    def input_gromacs_parameters(self) -> tuple[float, ...]:
        """Return the user-supplied canonical GROMACS parameter sequence."""
        if isinstance(self.input_parameters, PeriodicDihedralParameters):
            return (
                self.input_parameters.phase_degrees,
                self.input_parameters.force_constant,
            )
        if isinstance(self.input_parameters, HarmonicParameters):
            return (
                self.input_parameters.equilibrium,
                self.input_parameters.force_constant,
            )
        return self.input_parameters.to_gromacs()


@dataclass
class VirtualSite:
    """Definition of one GROMACS virtual site.

    Args:
        bead_id: Zero-based identifier of the virtual bead.
        kind: Section kind, one of ``2``, ``3``, ``4``, or ``"n"``.
        function: GROMACS virtual-site function identifier.
        defining_beads: Zero-based identifiers used to construct the site.
        parameters: Function parameters, or ``None`` for parameter-free forms.
    """

    bead_id: int
    kind: int | str
    function: int
    defining_beads: tuple[int, ...]
    parameters: tuple[float, ...] | None = None


@dataclass
class CGTopology:
    """Mutable typed coarse-grained topology used by Swarm-CG internals.

    Args:
        molecule: Molecule type metadata.
        atoms: Atoms in ITP order.
        constraints: Constraint groups in topology order.
        bonds: Bond groups in topology order.
        angles: Angle groups in topology order.
        dihedrals: Dihedral groups in topology order.
        virtual_sites: Virtual-site definitions.
        exclusions: Zero-based exclusion tuples.
    """

    molecule: MoleculeType = field(default_factory=MoleculeType)
    atoms: list[Atom] = field(default_factory=list)
    constraints: list[ConstraintGroup] = field(default_factory=list)
    bonds: list[BondGroup] = field(default_factory=list)
    angles: list[AngleGroup] = field(default_factory=list)
    dihedrals: list[DihedralGroup] = field(default_factory=list)
    virtual_sites: list[VirtualSite] = field(default_factory=list)
    exclusions: list[tuple[int, ...]] = field(default_factory=list)

    @property
    def real_bead_ids(self) -> tuple[int, ...]:
        """Return real bead identifiers in topology order."""
        return tuple(atom.bead_id for atom in self.atoms if not atom.is_virtual)

    @property
    def virtual_bead_ids(self) -> tuple[int, ...]:
        """Return virtual bead identifiers in topology order."""
        return tuple(atom.bead_id for atom in self.atoms if atom.is_virtual)

    @property
    def constraint_count(self) -> int:
        """Return the number of constraint groups."""
        return len(self.constraints)

    @property
    def bond_count(self) -> int:
        """Return the number of bond groups."""
        return len(self.bonds)

    @property
    def angle_count(self) -> int:
        """Return the number of angle groups."""
        return len(self.angles)

    @property
    def dihedral_count(self) -> int:
        """Return the number of dihedral groups."""
        return len(self.dihedrals)

    def groups(self, kind: GeometryKind):
        """Return geometry groups for a typed geometry kind.

        Args:
            kind: Geometry section to retrieve.

        Returns:
            The mutable group list for ``kind``.
        """
        return getattr(self, kind.plural)

    def virtual_sites_of_kind(self, kind: int | str) -> tuple[VirtualSite, ...]:
        """Return virtual sites belonging to one GROMACS section kind.

        Args:
            kind: Section kind, one of ``2``, ``3``, ``4``, or ``"n"``.

        Returns:
            Matching sites in topology order.
        """
        return tuple(site for site in self.virtual_sites if site.kind == kind)

    def validate(self) -> None:
        """Validate bead references, parameter types, and finite numeric state.

        Raises:
            MissformattedFile: If the topology is internally inconsistent.
        """
        if not self.molecule.name or self.molecule.exclusion_depth < 0:
            raise exceptions.MissformattedFile(
                "CG topology requires a molecule name and nonnegative nrexcl."
            )
        expected_ids = list(range(len(self.atoms)))
        if [atom.bead_id for atom in self.atoms] != expected_ids:
            raise exceptions.MissformattedFile(
                "CG topology atom identifiers must be consecutive and start at zero."
            )
        for atom in self.atoms:
            if not math.isfinite(atom.charge) or (
                atom.mass is not None and not math.isfinite(atom.mass)
            ):
                raise exceptions.MissformattedFile(
                    f"CG topology atom {atom.bead_id + 1} has non-finite numeric data."
                )

        atom_count = len(self.atoms)
        specs = (
            (GeometryKind.CONSTRAINT, self.constraints, 2, ConstraintParameters),
            (GeometryKind.BOND, self.bonds, 2, HarmonicParameters),
            (GeometryKind.ANGLE, self.angles, 3, HarmonicParameters),
        )
        for kind, groups, arity, parameter_type in specs:
            for index, group in enumerate(groups, 1):
                self._validate_bead_tuples(kind, index, group.beads, arity, atom_count)
                if not isinstance(group.parameters, parameter_type) or not isinstance(
                    group.input_parameters, parameter_type
                ):
                    raise exceptions.MissformattedFile(
                        f"{kind.value.title()} group {index} has invalid parameter types."
                    )

        expected_dihedral_types = {
            1: PeriodicDihedralParameters,
            2: HarmonicParameters,
            3: RBParameters,
            4: PeriodicDihedralParameters,
            11: CBTParameters,
        }
        for index, group in enumerate(self.dihedrals, 1):
            self._validate_bead_tuples(
                GeometryKind.DIHEDRAL, index, group.beads, 4, atom_count
            )
            expected = expected_dihedral_types.get(group.function)
            if expected is None or not isinstance(group.parameters, expected) or not isinstance(
                group.input_parameters, expected
            ):
                raise exceptions.MissformattedFile(
                    f"Dihedral group {index} has parameters inconsistent with function {group.function}."
                )

        virtual_ids = set(self.virtual_bead_ids)
        defined_virtual_ids: set[int] = set()
        for site in self.virtual_sites:
            if site.kind not in (2, 3, 4, "n") or site.bead_id not in virtual_ids:
                raise exceptions.MissformattedFile(
                    f"Invalid virtual-site definition for bead {site.bead_id + 1}."
                )
            if site.bead_id in defined_virtual_ids:
                raise exceptions.MissformattedFile(
                    f"Virtual bead {site.bead_id + 1} is defined more than once."
                )
            defined_virtual_ids.add(site.bead_id)
            self._validate_indices(site.defining_beads, atom_count, "virtual site")
            if site.parameters is not None and not all(
                math.isfinite(value) for value in site.parameters
            ):
                raise exceptions.MissformattedFile(
                    f"Virtual site {site.bead_id + 1} has non-finite parameters."
                )

        for exclusion in self.exclusions:
            self._validate_indices(exclusion, atom_count, "exclusion")

    @staticmethod
    def _validate_indices(indices, atom_count: int, label: str) -> None:
        """Validate a sequence of zero-based bead identifiers."""
        if not indices or any(index < 0 or index >= atom_count for index in indices):
            raise exceptions.MissformattedFile(f"Invalid bead identifiers in {label}.")

    @classmethod
    def _validate_bead_tuples(
        cls, kind, group_index: int, bead_tuples, arity: int, atom_count: int
    ) -> None:
        """Validate arity and identifiers for one geometry group."""
        if not bead_tuples:
            raise exceptions.MissformattedFile(
                f"{kind.value.title()} group {group_index} contains no interactions."
            )
        for beads in bead_tuples:
            if len(beads) != arity:
                raise exceptions.MissformattedFile(
                    f"Invalid bead arity in {kind.value} group {group_index}."
                )
            cls._validate_indices(beads, atom_count, f"{kind.value} group {group_index}")
