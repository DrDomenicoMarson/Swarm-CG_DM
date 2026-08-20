"""Section-oriented parser and writer for the internal typed topology model."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Sequence

from swarmcg import config
from swarmcg.shared import exceptions
from swarmcg.shared.periodic import PeriodicDihedralParameters, normalize_periodic_degrees
from swarmcg.simulations.polynomial import CBTParameters, RBParameters
from swarmcg.topology import (
    AngleGroup,
    Atom,
    BondGroup,
    CGTopology,
    ConstraintGroup,
    ConstraintParameters,
    DihedralGroup,
    GeometryKind,
    HarmonicParameters,
    MoleculeType,
    VirtualSite,
)


_SECTION_ALIASES = {
    "moleculetype": "moleculetype",
    "atoms": "atoms",
    "constraints": "constraints",
    "bonds": "bonds",
    "angles": "angles",
    "dihedrals": "dihedrals",
    "virtual_sites2": "virtual_sites2",
    "virtual_sites3": "virtual_sites3",
    "virtual_sites4": "virtual_sites4",
    "virtual_sitesn": "virtual_sitesn",
    "exclusions": "exclusions",
}


@dataclass(frozen=True)
class _SectionRecord:
    """One tokenized ITP data line with grouping metadata."""

    line_number: int
    tokens: tuple[str, ...]
    starts_group: bool
    group_label: str | None


def _finite_float(token: str, section: str, line_number: int, field: str) -> float:
    """Parse a finite float and attach ITP source diagnostics to failures."""
    try:
        value = float(token)
    except (TypeError, ValueError) as exc:
        raise exceptions.MissformattedFile(
            f"Invalid {field} in [{section}] at ITP line {line_number}: {token!r}."
        ) from exc
    if not math.isfinite(value):
        raise exceptions.MissformattedFile(
            f"Non-finite {field} in [{section}] at ITP line {line_number}: {token!r}."
        )
    return value


def _integer(token: str, section: str, line_number: int, field: str) -> int:
    """Parse an integer and attach ITP source diagnostics to failures."""
    try:
        return int(token)
    except (TypeError, ValueError) as exc:
        raise exceptions.MissformattedFile(
            f"Invalid {field} in [{section}] at ITP line {line_number}: {token!r}."
        ) from exc


def _collect_sections(path: Path) -> dict[str, list[_SectionRecord]]:
    """Tokenize supported ITP sections while retaining explicit group boundaries."""
    try:
        lines = path.read_text().splitlines()
    except UnicodeDecodeError as exc:
        raise exceptions.MissformattedFile(
            "Cannot read CG ITP; the supplied file appears to be binary."
        ) from exc

    sections = {name: [] for name in _SECTION_ALIASES.values()}
    current: str | None = None
    boundary = True
    pending_label: str | None = None
    for line_number, raw_line in enumerate(lines, 1):
        stripped = raw_line.strip()
        header = re.fullmatch(r"\[\s*([^]]+?)\s*\]", stripped)
        if header:
            current = _SECTION_ALIASES.get(header.group(1).strip().lower())
            boundary = True
            pending_label = None
            continue
        if not stripped:
            boundary = True
            pending_label = None
            continue
        if stripped.startswith(";"):
            boundary = True
            comment = stripped[1:].strip()
            group_match = re.match(
                r"(?:constraint|bond|angle|dihedral)\s+(?:type|group)\s+(.+)$",
                comment,
                flags=re.IGNORECASE,
            )
            pending_label = group_match.group(1).strip() if group_match else None
            continue
        if current is None:
            continue
        content = raw_line.split(";", 1)[0].strip()
        if not content:
            continue
        sections[current].append(
            _SectionRecord(
                line_number=line_number,
                tokens=tuple(content.split()),
                starts_group=boundary,
                group_label=pending_label,
            )
        )
        boundary = False
        pending_label = None
    return sections


def _require_fields(record: _SectionRecord, section: str, count: int) -> None:
    """Require an exact token count for one section record."""
    if len(record.tokens) != count:
        raise exceptions.MissformattedFile(
            f"Expected {count} fields in [{section}] at ITP line "
            f"{record.line_number}, found {len(record.tokens)}."
        )


def _function(record: _SectionRecord, section: str, token_index: int) -> int:
    """Parse and validate a supported GROMACS function identifier."""
    function = _integer(
        record.tokens[token_index], section, record.line_number, "function"
    )
    lookup = section[:-1] if section in {"constraints", "bonds", "angles", "dihedrals"} else section
    if function not in config.handled_functions[lookup]:
        supported = ", ".join(str(value) for value in config.handled_functions[lookup])
        raise exceptions.MissformattedFile(
            f"Unsupported function {function} in [{section}] at ITP line "
            f"{record.line_number}; supported functions are {supported}."
        )
    return function


def _beads(
    record: _SectionRecord, section: str, count: int, *, start: int = 0
) -> tuple[int, ...]:
    """Parse a fixed number of one-based ITP bead identifiers as zero-based IDs."""
    return tuple(
        _integer(token, section, record.line_number, "bead identifier") - 1
        for token in record.tokens[start : start + count]
    )


def _group_records(
    records: Sequence[_SectionRecord], section: str
) -> list[tuple[str, list[_SectionRecord]]]:
    """Split a bonded section into its explicitly delimited geometry groups."""
    groups: list[tuple[str, list[_SectionRecord]]] = []
    for record in records:
        if record.starts_group or not groups:
            label = record.group_label or str(len(groups) + 1)
            groups.append((label, [record]))
        else:
            groups[-1][1].append(record)
    return groups


def _require_shared_parameters(
    section: str,
    group_index: int,
    values: Sequence[object],
) -> object:
    """Return a group's common parameter value or reject heterogeneous groups."""
    first = values[0]
    if any(value != first for value in values[1:]):
        raise exceptions.MissformattedFile(
            f"Grouped {section} group {group_index} contains different parameters; "
            "separate groups with a blank or comment line."
        )
    return first


def _parse_molecule(records: Sequence[_SectionRecord]) -> MoleculeType:
    """Parse the required ``moleculetype`` section."""
    if len(records) != 1:
        raise exceptions.MissformattedFile(
            "The CG ITP must contain exactly one moleculetype record."
        )
    record = records[0]
    _require_fields(record, "moleculetype", 2)
    return MoleculeType(
        name=record.tokens[0],
        exclusion_depth=_integer(
            record.tokens[1], "moleculetype", record.line_number, "nrexcl"
        ),
    )


def _parse_atoms(records: Sequence[_SectionRecord]) -> list[Atom]:
    """Parse atoms and optional explicit masses in topology order."""
    atoms: list[Atom] = []
    for record in records:
        if len(record.tokens) not in (7, 8):
            raise exceptions.MissformattedFile(
                f"Expected 7 or 8 fields in [atoms] at ITP line {record.line_number}."
            )
        bead_id = _integer(record.tokens[0], "atoms", record.line_number, "atom id") - 1
        if bead_id != len(atoms):
            raise exceptions.MissformattedFile(
                "CG ITP atom identifiers must be consecutive and start at one "
                f"(ITP line {record.line_number})."
            )
        mass = (
            _finite_float(record.tokens[7], "atoms", record.line_number, "mass")
            if len(record.tokens) == 8
            else None
        )
        atoms.append(
            Atom(
                bead_id=bead_id,
                bead_type=record.tokens[1],
                residue_number=_integer(
                    record.tokens[2], "atoms", record.line_number, "residue number"
                ),
                residue_name=record.tokens[3],
                atom_name=record.tokens[4],
                charge_group=_integer(
                    record.tokens[5], "atoms", record.line_number, "charge group"
                ),
                charge=_finite_float(
                    record.tokens[6], "atoms", record.line_number, "charge"
                ),
                mass=mass,
            )
        )
    return atoms


def _parse_constraints(records: Sequence[_SectionRecord]) -> list[ConstraintGroup]:
    """Parse grouped constraint records."""
    result: list[ConstraintGroup] = []
    for group_index, (label, grouped) in enumerate(
        _group_records(records, "constraints"), 1
    ):
        parsed = []
        for record in grouped:
            _require_fields(record, "constraints", 4)
            parsed.append(
                (
                    _beads(record, "constraints", 2),
                    _function(record, "constraints", 2),
                    ConstraintParameters(
                        _finite_float(
                            record.tokens[3],
                            "constraints",
                            record.line_number,
                            "length",
                        )
                    ),
                )
            )
        function = _require_shared_parameters(
            "constraints", group_index, [item[1] for item in parsed]
        )
        parameters = _require_shared_parameters(
            "constraints", group_index, [item[2] for item in parsed]
        )
        result.append(
            ConstraintGroup(
                geometry_type=label,
                beads=[item[0] for item in parsed],
                function=function,
                parameters=parameters,
                input_parameters=replace(parameters),
            )
        )
    return result


def _parse_harmonic_groups(
    records: Sequence[_SectionRecord],
    section: str,
    arity: int,
    group_type: Callable,
) -> list[BondGroup] | list[AngleGroup]:
    """Parse grouped bond or angle records sharing harmonic parameters."""
    result = []
    for group_index, (label, grouped) in enumerate(
        _group_records(records, section), 1
    ):
        parsed = []
        for record in grouped:
            _require_fields(record, section, arity + 3)
            function = _function(record, section, arity)
            equilibrium = _finite_float(
                record.tokens[arity + 1], section, record.line_number, "equilibrium value"
            )
            force_constant = _finite_float(
                record.tokens[arity + 2], section, record.line_number, "force constant"
            )
            if section == "angles" and function == 10 and not 10.0 <= equilibrium <= 170.0:
                raise exceptions.MissformattedFile(
                    "Restricted-bending angle function 10 requires an equilibrium "
                    f"angle in [10, 170] degrees (ITP line {record.line_number})."
                )
            parsed.append(
                (
                    _beads(record, section, arity),
                    function,
                    HarmonicParameters(equilibrium, force_constant),
                )
            )
        function = _require_shared_parameters(
            section, group_index, [item[1] for item in parsed]
        )
        parameters = _require_shared_parameters(
            section, group_index, [item[2] for item in parsed]
        )
        result.append(
            group_type(
                geometry_type=label,
                beads=[item[0] for item in parsed],
                function=function,
                parameters=parameters,
                input_parameters=replace(parameters),
            )
        )
    return result


def _parse_dihedral_parameters(record: _SectionRecord, function: int):
    """Parse one typed dihedral parameter record."""
    if function in (3, 11):
        _require_fields(record, "dihedrals", 11)
        raw = tuple(
            _finite_float(token, "dihedrals", record.line_number, "polynomial parameter")
            for token in record.tokens[5:11]
        )
        return RBParameters.from_gromacs(raw) if function == 3 else CBTParameters.from_gromacs(raw)
    expected = 8 if function in config.dihedral_func_with_mult else 7
    _require_fields(record, "dihedrals", expected)
    equilibrium = _finite_float(
        record.tokens[5], "dihedrals", record.line_number, "phase/equilibrium angle"
    )
    force_constant = _finite_float(
        record.tokens[6], "dihedrals", record.line_number, "force constant"
    )
    if function in config.dihedral_func_with_mult:
        multiplicity = _integer(
            record.tokens[7], "dihedrals", record.line_number, "multiplicity"
        )
        if multiplicity <= 0:
            raise exceptions.MissformattedFile(
                f"Periodic dihedral function {function} at ITP line "
                f"{record.line_number} requires a positive integer multiplicity."
            )
        try:
            return PeriodicDihedralParameters.from_gromacs(
                equilibrium, force_constant, multiplicity
            )
        except ValueError as exc:
            raise exceptions.MissformattedFile(str(exc)) from exc
    return HarmonicParameters(normalize_periodic_degrees(equilibrium), force_constant)


def _parse_dihedrals(records: Sequence[_SectionRecord]) -> list[DihedralGroup]:
    """Parse grouped dihedrals into function-specific parameter objects."""
    result: list[DihedralGroup] = []
    for group_index, (label, grouped) in enumerate(
        _group_records(records, "dihedrals"), 1
    ):
        parsed = []
        for record in grouped:
            if len(record.tokens) < 5:
                raise exceptions.MissformattedFile(
                    f"Incomplete [dihedrals] record at ITP line {record.line_number}."
                )
            function = _function(record, "dihedrals", 4)
            parsed.append(
                (
                    _beads(record, "dihedrals", 4),
                    function,
                    _parse_dihedral_parameters(record, function),
                )
            )
        function = _require_shared_parameters(
            "dihedrals", group_index, [item[1] for item in parsed]
        )
        parameters = _require_shared_parameters(
            "dihedrals", group_index, [item[2] for item in parsed]
        )
        result.append(
            DihedralGroup(
                geometry_type=label,
                beads=[item[0] for item in parsed],
                function=function,
                parameters=parameters,
                input_parameters=replace(parameters),
            )
        )
    return result


def _parse_virtual_sites(
    sections: dict[str, list[_SectionRecord]], atoms: list[Atom]
) -> list[VirtualSite]:
    """Parse all four supported virtual-site sections."""
    sites: list[VirtualSite] = []
    fixed_specs = {
        "virtual_sites2": (2, 2, 3),
        "virtual_sites3": (3, 3, 4),
        "virtual_sites4": (4, 4, 5),
    }
    for section, (kind, arity, function_index) in fixed_specs.items():
        for record in sections[section]:
            minimum = arity + 2
            if len(record.tokens) < minimum:
                raise exceptions.MissformattedFile(
                    f"Incomplete [{section}] record at ITP line {record.line_number}."
                )
            bead_id = _integer(
                record.tokens[0], section, record.line_number, "virtual bead id"
            ) - 1
            function = _function(record, section, function_index)
            expected_params = 1 if kind == 2 else (3 if function == 4 or kind == 4 else 2)
            _require_fields(record, section, arity + 2 + expected_params)
            parameters = tuple(
                _finite_float(token, section, record.line_number, "virtual-site parameter")
                for token in record.tokens[function_index + 1 :]
            )
            sites.append(
                VirtualSite(
                    bead_id=bead_id,
                    kind=kind,
                    function=function,
                    defining_beads=_beads(record, section, arity, start=1),
                    parameters=parameters,
                )
            )

    for record in sections["virtual_sitesn"]:
        if len(record.tokens) < 3:
            raise exceptions.MissformattedFile(
                f"Incomplete [virtual_sitesn] record at ITP line {record.line_number}."
            )
        bead_id = _integer(
            record.tokens[0], "virtual_sitesn", record.line_number, "virtual bead id"
        ) - 1
        function = _function(record, "virtual_sitesn", 1)
        if function == 3:
            definition = record.tokens[2:]
            if len(definition) % 2 != 0:
                raise exceptions.MissformattedFile(
                    f"Virtual-sitesn function 3 requires bead/weight pairs at ITP line {record.line_number}."
                )
            defining_beads = tuple(
                _integer(definition[index], "virtual_sitesn", record.line_number, "bead identifier") - 1
                for index in range(0, len(definition), 2)
            )
            parameters = tuple(
                _finite_float(
                    definition[index],
                    "virtual_sitesn",
                    record.line_number,
                    "virtual-site weight",
                )
                for index in range(1, len(definition), 2)
            )
        else:
            defining_beads = tuple(
                _integer(token, "virtual_sitesn", record.line_number, "bead identifier") - 1
                for token in record.tokens[2:]
            )
            parameters = None
        sites.append(
            VirtualSite(
                bead_id=bead_id,
                kind="n",
                function=function,
                defining_beads=defining_beads,
                parameters=parameters,
            )
        )

    for site in sites:
        if site.bead_id < 0 or site.bead_id >= len(atoms):
            raise exceptions.MissformattedFile(
                f"Virtual-site bead {site.bead_id + 1} is outside the atom table."
            )
        if not atoms[site.bead_id].is_virtual:
            raise exceptions.MissformattedFile(
                f"Virtual-site bead {site.bead_id + 1} does not use a type starting with 'v'."
            )
        atoms[site.bead_id].virtual_site_kind = site.kind
    return sites


def _parse_exclusions(records: Sequence[_SectionRecord]) -> list[tuple[int, ...]]:
    """Parse zero-based exclusion tuples."""
    return [
        tuple(
            _integer(token, "exclusions", record.line_number, "bead identifier") - 1
            for token in record.tokens
        )
        for record in records
    ]


def read_cg_topology(path: str | Path) -> CGTopology:
    """Read a coarse-grained GROMACS ITP into the internal typed model.

    Args:
        path: Input ITP path.

    Returns:
        Parsed and centrally validated topology.

    Raises:
        MissformattedFile: If syntax, supported functions, grouping, numeric
            values, or bead references are invalid.
    """
    sections = _collect_sections(Path(path))
    topology = CGTopology(
        molecule=_parse_molecule(sections["moleculetype"]),
        atoms=_parse_atoms(sections["atoms"]),
        constraints=_parse_constraints(sections["constraints"]),
        bonds=_parse_harmonic_groups(
            sections["bonds"], "bonds", 2, BondGroup
        ),
        angles=_parse_harmonic_groups(
            sections["angles"], "angles", 3, AngleGroup
        ),
        dihedrals=_parse_dihedrals(sections["dihedrals"]),
        exclusions=_parse_exclusions(sections["exclusions"]),
    )
    topology.virtual_sites = _parse_virtual_sites(sections, topology.atoms)
    topology.validate()
    return topology


def _write_interactions(
    lines: list[str],
    heading: str,
    label: str,
    groups: Iterable,
    render: Callable,
) -> None:
    """Append one grouped bonded section to an output line buffer."""
    groups = list(groups)
    if not groups:
        return
    lines.extend(("", "", f"[ {heading} ]"))
    for group in groups:
        lines.extend(("", f"; {label} type {group.geometry_type}"))
        lines.extend(render(group, beads) for beads in group.beads)


def _format_dihedral(group: DihedralGroup, beads: tuple[int, ...]) -> str:
    """Serialize one typed dihedral interaction."""
    prefix = " ".join(f"{bead + 1:>5}" for bead in beads)
    parameters = group.parameters
    if isinstance(parameters, RBParameters):
        values = parameters.to_gromacs()
    elif isinstance(parameters, CBTParameters):
        values = parameters.to_gromacs()
    elif isinstance(parameters, PeriodicDihedralParameters):
        values = parameters.to_gromacs()
    else:
        values = (normalize_periodic_degrees(parameters.equilibrium), parameters.force_constant)
    serialized = " ".join(f"{value:.17g}" for value in values)
    return f"{prefix} {group.function:>7} {serialized}     ; {group.geometry_type}"


def _write_virtual_sites(lines: list[str], topology: CGTopology) -> None:
    """Append every populated virtual-site section."""
    for kind in (2, 3, 4, "n"):
        sites = topology.virtual_sites_of_kind(kind)
        if not sites:
            continue
        lines.extend(("", "", f"[ virtual_sites{kind} ]"))
        for site in sites:
            if kind == "n":
                if site.function == 3:
                    definition = " ".join(
                        f"{bead + 1} {weight:.17g}"
                        for bead, weight in zip(
                            site.defining_beads, site.parameters or ()
                        )
                    )
                else:
                    definition = " ".join(
                        str(bead + 1) for bead in site.defining_beads
                    )
                lines.append(
                    f"{site.bead_id + 1:>5} {site.function:>5} {definition}"
                )
            else:
                bead_text = " ".join(f"{bead + 1:>5}" for bead in site.defining_beads)
                parameter_text = " ".join(
                    f"{value:.17g}" for value in site.parameters or ()
                )
                lines.append(
                    f"{site.bead_id + 1:>5} {bead_text} {site.function:>5} {parameter_text}"
                    .rstrip()
                )


def write_cg_topology(
    topology: CGTopology,
    path: str | Path,
    sections: Iterable[GeometryKind | str] = (
        "constraint",
        "bond",
        "angle",
        "dihedral",
        "exclusion",
    ),
) -> None:
    """Write a canonical ITP from the internal typed topology model.

    Args:
        topology: Topology to validate and serialize.
        path: Destination ITP path.
        sections: Bonded sections to include. Virtual-site definitions are
            always retained because atoms may depend on them.

    Returns:
        ``None``.
    """
    topology.validate()
    selected = {
        item.value if hasattr(item, "value") else str(item) for item in sections
    }
    lines = [
        "[ moleculetype ]",
        "; molname        nrexcl",
        f"{topology.molecule.name:<4} {topology.molecule.exclusion_depth:>13}",
        "",
        "",
        "[ atoms ]",
        "; id type resnr residue atom cgnr charge mass",
        "",
    ]
    for atom in topology.atoms:
        record = (
            f"{atom.bead_id + 1:<4} {atom.bead_type:>4} {atom.residue_number:>5} "
            f"{atom.residue_name:>6} {atom.atom_name:>6} {atom.charge_group:>5} "
            f"{atom.charge:.17g}"
        )
        if atom.mass is not None:
            record += f" {atom.mass:.17g}"
        lines.append(record)

    if "constraint" in selected:
        _write_interactions(
            lines,
            "constraints",
            "constraint",
            topology.constraints,
            lambda group, beads: (
                f"{beads[0] + 1:>5} {beads[1] + 1:>5} {group.function:>7} "
                f"{group.parameters.length:.17g}      ; {group.geometry_type}"
            ),
        )
    if "bond" in selected:
        _write_interactions(
            lines,
            "bonds",
            "bond",
            topology.bonds,
            lambda group, beads: (
                f"{beads[0] + 1:>5} {beads[1] + 1:>5} {group.function:>7} "
                f"{group.parameters.equilibrium:.17g} "
                f"{group.parameters.force_constant:.17g}      ; {group.geometry_type}"
            ),
        )
    if "angle" in selected:
        _write_interactions(
            lines,
            "angles",
            "angle",
            topology.angles,
            lambda group, beads: (
                f"{beads[0] + 1:>5} {beads[1] + 1:>5} {beads[2] + 1:>5} "
                f"{group.function:>7} {group.parameters.equilibrium:.17g} "
                f"{group.parameters.force_constant:.17g}      ; {group.geometry_type}"
            ),
        )
    if "dihedral" in selected:
        _write_interactions(
            lines,
            "dihedrals",
            "dihedral",
            topology.dihedrals,
            _format_dihedral,
        )

    _write_virtual_sites(lines, topology)
    if "exclusion" in selected and topology.exclusions:
        lines.extend(("", "", "[ exclusions ]", "; bead identifiers", ""))
        lines.extend(
            " ".join(f"{bead + 1:>4}" for bead in exclusion)
            for exclusion in topology.exclusions
        )
    lines.extend(("", ""))
    Path(path).write_text("\n".join(lines))
