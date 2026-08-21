"""Reproducible atomistic and Martini 3 SASA validation."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path
from typing import Mapping, Sequence

import MDAnalysis as mda
import numpy as np
from MDAnalysis.exceptions import NoDataError

from swarmcg.context import OptimizationContext
from swarmcg.io import read_xvg_col
from swarmcg.sasa_types import SasaMeasurement, SasaProtocol, SasaRepresentation
from swarmcg.shared import exceptions
from swarmcg.simulations.runner import GromacsCommandResult, exec_gmx
from swarmcg.topology import CGTopology


_ROWLAND_TAYLOR_RESOURCE = "rowland_taylor_vdw_radii.dat"
_ATOMIC_MASSES = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "S": 32.06,
    "Cl": 35.45,
    "Br": 79.904,
    "I": 126.904,
}
_MASS_TOLERANCE_DA = 0.75
_NO_RADIUS_MARTINI_TYPES = frozenset({"DUM", "V"})
_MISSING_RADIUS_PATTERNS = (
    "could not find a van der waals radius",
    "using default radius 0.14",
    "using default van der waals radius",
)


@dataclass(frozen=True)
class ResolvedRadii:
    """Resolved radius for every staged atom.

    Args:
        values_nm: Radius in nanometres for each atom in staging order. Zero
            explicitly marks a non-surface virtual site.
        source: Human-readable origin of the resolved radii.
    """

    values_nm: tuple[float, ...]
    source: str


@dataclass(frozen=True)
class _OverrideEntry:
    residue_pattern: str
    atom_pattern: str
    radius_nm: float
    line_number: int


@dataclass(frozen=True)
class _SasaSelection:
    universe: mda.Universe
    atom_ids: tuple[int, ...]
    surface_local_ids: tuple[int, ...]
    radii: ResolvedRadii
    dimensions_source: mda.Universe | None = None


def resolve_aa_radii(
    universe: mda.Universe,
    atom_ids: Sequence[int],
    override_path: str | Path | None = None,
) -> ResolvedRadii:
    """Resolve strict atomistic radii for a target molecule.

    The built-in profile infers each element independently from atom name and
    type, requires the two labels to agree when both are informative, and
    cross-checks the result against atom mass. An override is parsed using
    GROMACS ``vdwradii.dat`` residue and atom-prefix matching, then reduced to
    exact entries for the selected atoms.

    Args:
        universe: Atomistic MDAnalysis universe containing names, types,
            residue names, and masses.
        atom_ids: Zero-based target atom identifiers in output order.
        override_path: Optional user ``vdwradii.dat`` file.

    Returns:
        One positive radius per selected atom and its source description.

    Raises:
        ComputationError: If an atom is ambiguous, conflicting, unsupported,
            unresolved, duplicated with a conflicting radius, or inconsistent
            with its mass.
        FileNotFoundError: If an override path does not exist.
    """
    atoms = universe.atoms[list(atom_ids)]
    _validate_gromacs_labels(atoms)
    if override_path is not None:
        source_path = Path(override_path).expanduser().resolve()
        entries = _parse_override(source_path)
        values = tuple(_resolve_override_atom(atom, entries) for atom in atoms)
        _validate_exact_radius_keys(atoms, values)
        return ResolvedRadii(values, f"override:{source_path}")

    element_radii = _load_rowland_taylor_radii()
    values = []
    for atom in atoms:
        name_symbol = _element_from_label(atom.name)
        try:
            atom_type = str(atom.type)
        except NoDataError:
            atom_type = ""
        type_symbol = _element_from_label(atom_type)
        informative = {symbol for symbol in (name_symbol, type_symbol) if symbol}
        if len(informative) > 1:
            raise exceptions.ComputationError(
                f"Conflicting AA element labels for {atom.resname}:{atom.name}: "
                f"name implies {name_symbol}, type {atom_type!r} implies {type_symbol}."
            )
        if not informative:
            raise exceptions.ComputationError(
                f"Cannot resolve an element for AA atom {atom.resname}:{atom.name} "
                f"with type {atom_type!r}; provide --sasa-aa-radii."
            )
        symbol = informative.pop()
        if symbol not in element_radii:
            raise exceptions.ComputationError(
                f"AA element {symbol} for {atom.resname}:{atom.name} is not in "
                "the built-in Rowland--Taylor profile; provide --sasa-aa-radii."
            )
        try:
            mass = float(atom.mass)
        except NoDataError as exc:
            raise exceptions.ComputationError(
                f"AA atom {atom.resname}:{atom.name} has no mass for the "
                "required element cross-check."
            ) from exc
        expected_mass = _ATOMIC_MASSES[symbol]
        if not math.isfinite(mass) or abs(mass - expected_mass) > _MASS_TOLERANCE_DA:
            raise exceptions.ComputationError(
                f"AA atom {atom.resname}:{atom.name} resolves to {symbol}, but "
                f"its mass {mass!r} Da conflicts with the expected elemental mass; "
                "provide an explicit --sasa-aa-radii file if this is intentional."
            )
        values.append(element_radii[symbol])
    resolved = tuple(values)
    _validate_exact_radius_keys(atoms, resolved)
    return ResolvedRadii(resolved, "Rowland--Taylor 1996 (packaged)")


def resolve_martini3_radii(
    topology: CGTopology,
    explicit_radii: Mapping[str, float] | None = None,
) -> ResolvedRadii:
    """Resolve Martini 3 regular, small, and tiny bead radii.

    Args:
        topology: Typed CG topology in atom order.
        explicit_radii: Optional exact radii by custom atom type. This internal
            seam permits explicitly supported extensions without silently
            classifying a custom type as Martini 3.

    Returns:
        Radii in topology order. Explicit dummy virtual-site types receive
        zero and are excluded from the surface group.

    Raises:
        ComputationError: If a bead type is empty, custom, or has an invalid
            explicit radius.
    """
    explicit = explicit_radii or {}
    values = []
    for atom in topology.atoms:
        bead_type = atom.bead_type.strip()
        if bead_type in explicit:
            radius = float(explicit[bead_type])
            if not math.isfinite(radius) or radius < 0:
                raise exceptions.ComputationError(
                    f"Explicit radius for CG type {bead_type!r} must be finite and nonnegative."
                )
            values.append(radius)
            continue
        normalized = bead_type.upper()
        if normalized in _NO_RADIUS_MARTINI_TYPES and atom.is_virtual:
            values.append(0.0)
            continue
        size_prefix = normalized[:1]
        base = normalized[1:] if size_prefix in {"S", "T"} else normalized
        if not re.fullmatch(
            r"(?:[PN][1-6](?:A|D|R|E)?|C[1-6](?:R|E)?|"
            r"Q[1-5](?:P|N|D)?|X[1-4]E?|W|D)",
            base,
        ):
            raise exceptions.ComputationError(
                f"CG atom {atom.atom_name!r} uses unsupported non-Martini-3 "
                f"type {bead_type!r}; an explicit supported radius is required."
            )
        values.append(
            0.191 if size_prefix == "T" else 0.230 if size_prefix == "S" else 0.264
        )
    return ResolvedRadii(tuple(values), "Martini 3 bead-size classes")


def validate_sasa_inputs(context: OptimizationContext) -> None:
    """Validate AA and Martini 3 radii before a requested SASA workflow.

    Args:
        context: Initialized context with mapped AA data and a typed topology.

    Returns:
        ``None`` after all atom labels and radii have resolved.

    Raises:
        ComputationError: If target atoms, radii, or surface selections are
            invalid.
    """
    _selection_for(context, SasaRepresentation.AA)
    _selection_for(context, SasaRepresentation.AA_MAPPED)


def compute_sasa(
    context: OptimizationContext,
    representation: SasaRepresentation | str,
    output_dir: str | Path,
) -> SasaMeasurement:
    """Run one isolated GROMACS SASA calculation with strict validation.

    Args:
        context: Initialized optimization or standalone-evaluation context.
        representation: ``aa``, ``aa_mapped``, or ``cg``.
        output_dir: Dedicated directory for generated structure, trajectory,
            index, radii, raw XVG, command logs, and protocol metadata.

    Returns:
        Full-precision SASA measurement and reproduction metadata.

    Raises:
        InvalidArgument: If *representation* is unknown.
        ComputationError: If radii are unresolved, GROMACS warns about a
            missing radius, output is incomplete, a frame is non-finite, or
            the frame count differs from the input trajectory.
    """
    try:
        selected_representation = SasaRepresentation(representation)
    except ValueError as exc:
        raise exceptions.InvalidArgument(
            f"Unsupported SASA representation {representation!r}."
        ) from exc
    selection = _selection_for(context, selected_representation)
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    structure_path = destination / "structure.gro"
    trajectory_path = destination / "trajectory.xtc"
    index_path = destination / "surface.ndx"
    radii_path = destination / "vdwradii.dat"
    output_path = destination / "sasa.xvg"
    metadata_path = destination / "protocol.json"
    output_path.unlink(missing_ok=True)
    metadata_path.unlink(missing_ok=True)

    frame_count = _write_analysis_coordinates(
        selection, structure_path, trajectory_path
    )
    _write_surface_index(index_path, selection.surface_local_ids)
    staged_radii = _write_exact_vdwradii(
        radii_path,
        selection.universe.atoms[list(selection.atom_ids)],
        selection.radii.values_nm,
    )
    radii_hash = hashlib.sha256(staged_radii).hexdigest()
    command = [
        context.config.gromacs.gmx_path,
        "sasa",
        "-s",
        structure_path.name,
        "-f",
        trajectory_path.name,
        "-n",
        index_path.name,
        "-surface",
        "sasa_surface",
        "-output",
        "sasa_surface",
        "-o",
        output_path.name,
        "-probe",
        str(context.config.sasa.probe_radius_nm),
        "-ndots",
        str(context.config.sasa.sphere_points),
    ]
    result = exec_gmx(command, cwd=str(destination))
    _persist_command_output(destination, result)
    _validate_gromacs_result(result, selected_representation)
    if not output_path.is_file():
        raise exceptions.ComputationError(
            f"GROMACS did not create the expected SASA output: {output_path}"
        )
    per_frame = np.asarray(read_xvg_col(str(output_path), 1), dtype=float)
    if per_frame.size != frame_count:
        raise exceptions.ComputationError(
            f"{selected_representation.value} SASA produced {per_frame.size} frames; "
            f"expected exactly {frame_count}."
        )
    if per_frame.size == 0 or not np.all(np.isfinite(per_frame)):
        raise exceptions.ComputationError(
            f"{selected_representation.value} SASA contains missing or non-finite frame values."
        )
    protocol = SasaProtocol(
        probe_radius_nm=context.config.sasa.probe_radius_nm,
        sphere_points=context.config.sasa.sphere_points,
        radii_source=selection.radii.source,
        radii_sha256=radii_hash,
    )
    measurement = SasaMeasurement(
        representation=selected_representation,
        mean=float(np.mean(per_frame)),
        standard_deviation=float(np.std(per_frame)),
        frame_count=frame_count,
        protocol=protocol,
    )
    metadata = asdict(measurement)
    metadata["representation"] = selected_representation.value
    metadata_path.write_text(
        json.dumps(metadata, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return measurement


def _selection_for(
    context: OptimizationContext, representation: SasaRepresentation
) -> _SasaSelection:
    if context.cg_itp is None:
        raise exceptions.ComputationError("SASA requires an initialized CG topology.")
    if representation is SasaRepresentation.AA:
        if not context.scoring.all_aa_mols:
            raise exceptions.ComputationError(
                "SASA requires a selected atomistic target molecule."
            )
        universe = context.scoring.aa_universe
        atom_ids = tuple(
            int(value) for value in context.scoring.all_aa_mols[0].indices
        )
        radii = resolve_aa_radii(
            universe, atom_ids, context.config.sasa.aa_radii_filename
        )
        surface = tuple(range(len(atom_ids)))
        return _SasaSelection(universe, atom_ids, surface, radii)
    if representation is SasaRepresentation.AA_MAPPED:
        universe = context.scoring.aa2cg_universe
        dimensions_source = context.scoring.aa_universe
    else:
        universe = context.scoring.cg_universe
        dimensions_source = None
        if universe is None:
            raise exceptions.ComputationError(
                "CG SASA requires a successfully loaded CG trajectory."
            )
    atom_ids = tuple(range(len(context.cg_itp.atoms)))
    _validate_gromacs_labels(universe.atoms[list(atom_ids)])
    radii = resolve_martini3_radii(context.cg_itp)
    surface = tuple(
        index for index, radius in enumerate(radii.values_nm) if radius > 0
    )
    if not surface:
        raise exceptions.ComputationError(
            f"{representation.value} SASA has no atoms with a positive resolved radius."
        )
    return _SasaSelection(universe, atom_ids, surface, radii, dimensions_source)


def _load_rowland_taylor_radii() -> dict[str, float]:
    resource = resources.files("swarmcg").joinpath(
        "data", _ROWLAND_TAYLOR_RESOURCE
    )
    result = {}
    for line_number, raw_line in enumerate(
        resource.read_text(encoding="utf-8").splitlines(), 1
    ):
        payload = raw_line.split(";", 1)[0].strip()
        if not payload:
            continue
        tokens = payload.split()
        if len(tokens) != 2:
            raise exceptions.ComputationError(
                f"Invalid packaged Rowland--Taylor entry on line {line_number}."
            )
        symbol, raw_radius = tokens
        radius = float(raw_radius)
        if symbol in result or not math.isfinite(radius) or radius <= 0:
            raise exceptions.ComputationError(
                f"Invalid packaged Rowland--Taylor radius for {symbol!r}."
            )
        result[symbol] = radius
    return result


def _element_from_label(label: str) -> str | None:
    letters = re.sub(r"^[0-9]+", "", str(label).strip())
    match = re.match(r"[A-Za-z]+", letters)
    if match is None:
        return None
    token = match.group(0)
    if len(token) > 2:
        return None
    if len(token) >= 2:
        two_letter = token[0].upper() + token[1].lower()
        if two_letter in _ATOMIC_MASSES:
            return two_letter
    one_letter = token[0].upper()
    return one_letter if one_letter in _ATOMIC_MASSES else None


def _parse_override(path: Path) -> tuple[_OverrideEntry, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find SASA AA radii override: {path}")
    entries = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        payload = raw_line.split(";", 1)[0].strip()
        if not payload or payload.startswith("#"):
            continue
        tokens = payload.split()
        if len(tokens) != 3:
            raise exceptions.ComputationError(
                f"Invalid vdwradii.dat entry on line {line_number}: expected "
                "residue, atom prefix, and radius."
            )
        try:
            radius = float(tokens[2])
        except ValueError as exc:
            raise exceptions.ComputationError(
                f"Invalid vdwradii.dat radius on line {line_number}."
            ) from exc
        if not math.isfinite(radius) or radius <= 0:
            raise exceptions.ComputationError(
                f"vdwradii.dat radius on line {line_number} must be finite and positive."
            )
        entries.append(
            _OverrideEntry(tokens[0], tokens[1], radius, line_number)
        )
    if not entries:
        raise exceptions.ComputationError(
            "The SASA AA radii override contains no entries."
        )
    return tuple(entries)


def _resolve_override_atom(atom, entries: Sequence[_OverrideEntry]) -> float:
    matches = []
    for entry in entries:
        residue_matches = entry.residue_pattern in {"???", "*"} or fnmatch.fnmatchcase(
            atom.resname, entry.residue_pattern
        )
        atom_matches = (
            fnmatch.fnmatchcase(atom.name, entry.atom_pattern)
            if any(character in entry.atom_pattern for character in "*?[]")
            else atom.name.startswith(entry.atom_pattern)
        )
        if residue_matches and atom_matches:
            residue_exact = int(entry.residue_pattern == atom.resname)
            atom_specificity = len(entry.atom_pattern.rstrip("*?"))
            matches.append((residue_exact, atom_specificity, entry))
    if not matches:
        raise exceptions.ComputationError(
            f"No AA radius override matches {atom.resname}:{atom.name}."
        )
    top_key = max((match[0], match[1]) for match in matches)
    best = [
        entry
        for exact, specificity, entry in matches
        if (exact, specificity) == top_key
    ]
    radii = {entry.radius_nm for entry in best}
    if len(radii) != 1:
        lines = ", ".join(str(entry.line_number) for entry in best)
        raise exceptions.ComputationError(
            f"Conflicting AA radii override entries for {atom.resname}:{atom.name} "
            f"on lines {lines}."
        )
    return radii.pop()


def _validate_gromacs_labels(atoms) -> None:
    for atom in atoms:
        if not atom.name or not atom.resname:
            raise exceptions.ComputationError(
                "SASA atoms require non-empty atom and residue names."
            )
        if len(atom.name) > 5 or len(atom.resname) > 5:
            raise exceptions.ComputationError(
                f"GROMACS GRO SASA staging cannot preserve label "
                f"{atom.resname}:{atom.name}; atom and residue names must "
                "contain at most five characters."
            )


def _validate_exact_radius_keys(atoms, radii: Sequence[float]) -> None:
    seen = {}
    for atom, radius in zip(atoms, radii, strict=True):
        key = (atom.resname, atom.name)
        previous = seen.setdefault(key, radius)
        if previous != radius:
            raise exceptions.ComputationError(
                f"Duplicate AA name {atom.resname}:{atom.name} resolves to conflicting radii."
            )


def _write_analysis_coordinates(
    selection: _SasaSelection, structure_path: Path, trajectory_path: Path
) -> int:
    atom_group = selection.universe.atoms[list(selection.atom_ids)]
    frame_count = len(selection.universe.trajectory)
    if frame_count <= 0:
        raise exceptions.ComputationError("SASA source trajectory contains no frames.")
    if (
        selection.dimensions_source is not None
        and len(selection.dimensions_source.trajectory) != frame_count
    ):
        raise exceptions.ComputationError(
            "SASA coordinate and periodic-box trajectories have different frame counts."
        )
    selection.universe.trajectory[0]
    if selection.dimensions_source is not None:
        selection.dimensions_source.trajectory[0]
        selection.universe.trajectory.ts.dimensions = (
            selection.dimensions_source.trajectory.ts.dimensions
        )
    with mda.Writer(str(structure_path), n_atoms=len(atom_group)) as writer:
        writer.write(atom_group)
    with mda.Writer(str(trajectory_path), n_atoms=len(atom_group)) as writer:
        for frame_index, _ in enumerate(selection.universe.trajectory):
            if selection.dimensions_source is not None:
                selection.dimensions_source.trajectory[frame_index]
                selection.universe.trajectory.ts.dimensions = (
                    selection.dimensions_source.trajectory.ts.dimensions
                )
            writer.write(atom_group)
    return frame_count


def _write_surface_index(path: Path, local_ids: Sequence[int]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("[ sasa_surface ]\n")
        for start in range(0, len(local_ids), 15):
            handle.write(
                " ".join(
                    str(index + 1) for index in local_ids[start : start + 15]
                )
            )
            handle.write("\n")


def _write_exact_vdwradii(
    path: Path, atoms, radii: Sequence[float]
) -> bytes:
    entries = {}
    for atom, radius in zip(atoms, radii, strict=True):
        key = (atom.resname, atom.name)
        previous = entries.setdefault(key, radius)
        if previous != radius:
            raise exceptions.ComputationError(
                f"Cannot stage conflicting radii for duplicate name "
                f"{atom.resname}:{atom.name}."
            )
    ordered = sorted(
        entries.items(), key=lambda item: (-len(item[0][1]), item[0])
    )
    text = "; Exact SASA radii generated by Swarm-CG (nm)\n" + "".join(
        f"{resname:<5s} {atom_name:<5s} {radius:.9f}\n"
        for (resname, atom_name), radius in ordered
    )
    data = text.encode("utf-8")
    path.write_bytes(data)
    return data


def _persist_command_output(
    destination: Path, result: GromacsCommandResult
) -> None:
    (destination / "gromacs_stdout.txt").write_text(
        result.stdout, encoding="utf-8"
    )
    (destination / "gromacs_stderr.txt").write_text(
        result.stderr, encoding="utf-8"
    )


def _validate_gromacs_result(
    result: GromacsCommandResult, representation: SasaRepresentation
) -> None:
    if result.returncode != 0:
        raise exceptions.ComputationError(
            f"GROMACS failed while calculating {representation.value} SASA "
            f"(exit code {result.returncode})."
        )
    output = result.output.lower()
    warning = next(
        (pattern for pattern in _MISSING_RADIUS_PATTERNS if pattern in output),
        None,
    )
    if warning is not None:
        raise exceptions.ComputationError(
            f"GROMACS reported an unresolved radius while calculating "
            f"{representation.value} SASA: {warning}."
        )
