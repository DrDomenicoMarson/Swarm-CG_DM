"""Temporary conversions between the typed topology and legacy mapping model."""

from __future__ import annotations

from swarmcg.io.itp import CGITP
from swarmcg.shared.periodic import PeriodicDihedralParameters
from swarmcg.simulations.polynomial import CBTParameters, RBParameters
from swarmcg.topology import (
    AngleGroup,
    Atom,
    BondGroup,
    CGTopology,
    ConstraintGroup,
    ConstraintParameters,
    DihedralGroup,
    HarmonicParameters,
    MoleculeType,
    VirtualSite,
)


def _typed_dihedral(function: int, group: dict, source: str):
    """Build one typed dihedral parameter object from a legacy group."""
    if function in (1, 4):
        return PeriodicDihedralParameters.from_gromacs(
            group[f"value{source}"], group[f"fct{source}"], group["mult"]
        )
    if function == 2:
        return HarmonicParameters(group[f"value{source}"], group[f"fct{source}"])
    values = group[f"params{source}"]
    return (
        RBParameters.from_gromacs(values)
        if function == 3
        else CBTParameters.from_gromacs(values)
    )


def legacy_to_topology(legacy: CGITP) -> CGTopology:
    """Convert the temporary legacy mapping representation to typed topology.

    Args:
        legacy: Parsed legacy topology mapping.

    Returns:
        Equivalent validated typed topology.
    """
    topology = CGTopology(
        molecule=MoleculeType(
            legacy["moleculetype"]["molname"],
            legacy["moleculetype"]["nrexcl"],
        ),
        atoms=[
            Atom(
                bead_id=atom["bead_id"],
                bead_type=atom["bead_type"],
                residue_number=atom["resnr"],
                residue_name=atom["residue"],
                atom_name=atom["atom"],
                charge_group=atom["cgnr"],
                charge=atom["charge"],
                mass=atom["mass"],
                virtual_site_kind=atom.get("vs_type"),
            )
            for atom in legacy["atoms"]
        ],
        constraints=[
            ConstraintGroup(
                geometry_type=group["geom_type"],
                beads=[tuple(beads) for beads in group["beads"]],
                function=group["func"],
                parameters=ConstraintParameters(group["value"]),
                input_parameters=ConstraintParameters(group["value_user"]),
                average=group.get("avg"),
                histogram=group.get("hist"),
            )
            for group in legacy["constraint"]
        ],
        bonds=[
            BondGroup(
                geometry_type=group["geom_type"],
                beads=[tuple(beads) for beads in group["beads"]],
                function=group["func"],
                parameters=HarmonicParameters(group["value"], group["fct"]),
                input_parameters=HarmonicParameters(
                    group["value_user"], group["fct_user"]
                ),
                average=group.get("avg"),
                histogram=group.get("hist"),
            )
            for group in legacy["bond"]
        ],
        angles=[
            AngleGroup(
                geometry_type=group["geom_type"],
                beads=[tuple(beads) for beads in group["beads"]],
                function=group["func"],
                parameters=HarmonicParameters(group["value"], group["fct"]),
                input_parameters=HarmonicParameters(
                    group["value_user"], group["fct_user"]
                ),
                average=group.get("avg"),
                histogram=group.get("hist"),
            )
            for group in legacy["angle"]
        ],
        dihedrals=[
            DihedralGroup(
                geometry_type=group["geom_type"],
                beads=[tuple(beads) for beads in group["beads"]],
                function=group["func"],
                parameters=_typed_dihedral(group["func"], group, ""),
                input_parameters=_typed_dihedral(group["func"], group, "_user"),
                average=group.get("avg"),
                histogram=group.get("hist"),
                phase_moment_resultant=group.get("phase_moment_resultant"),
                polynomial_symmetry_tv=group.get("polynomial_symmetry_tv"),
                coefficient_bound=group.get("coefficient_bound"),
            )
            for group in legacy["dihedral"]
        ],
        virtual_sites=[
            VirtualSite(
                bead_id=site["bead_id"],
                kind=kind,
                function=site["func"],
                defining_beads=tuple(site["vs_def_beads_ids"]),
                parameters=(
                    None
                    if site["vs_params"] is None
                    else (
                        tuple(site["vs_params"])
                        if isinstance(site["vs_params"], (list, tuple))
                        else (site["vs_params"],)
                    )
                ),
            )
            for section, kind in (
                ("virtual_sites2", 2),
                ("virtual_sites3", 3),
                ("virtual_sites4", 4),
                ("virtual_sitesn", "n"),
            )
            for site in legacy[section].values()
        ],
        exclusions=[tuple(exclusion) for exclusion in legacy["exclusion"]],
    )
    topology.validate()
    return topology


def _legacy_dihedral_parameters(parameters) -> tuple[list[float], float | None, float | None, int | None]:
    """Return legacy parameter, value, force, and multiplicity fields."""
    if isinstance(parameters, PeriodicDihedralParameters):
        return (
            [parameters.phase_degrees, parameters.force_constant],
            parameters.phase_degrees,
            parameters.force_constant,
            parameters.multiplicity,
        )
    if isinstance(parameters, HarmonicParameters):
        return (
            [parameters.equilibrium, parameters.force_constant],
            parameters.equilibrium,
            parameters.force_constant,
            None,
        )
    if isinstance(parameters, (RBParameters, CBTParameters)):
        return list(parameters.to_gromacs()), None, None, None
    raise TypeError(f"Unsupported dihedral parameter type: {type(parameters)!r}")


def topology_to_legacy(topology: CGTopology) -> CGITP:
    """Convert typed topology to the temporary legacy mapping representation.

    Args:
        topology: Validated typed topology.

    Returns:
        Equivalent legacy mapping used only during the package migration.
    """
    topology.validate()
    legacy = CGITP()
    legacy["moleculetype"] = {
        "molname": topology.molecule.name,
        "nrexcl": topology.molecule.exclusion_depth,
    }
    legacy["atoms"] = [
        {
            "bead_id": atom.bead_id,
            "bead_type": atom.bead_type,
            "resnr": atom.residue_number,
            "residue": atom.residue_name,
            "atom": atom.atom_name,
            "cgnr": atom.charge_group,
            "charge": atom.charge,
            "mass": atom.mass,
            "vs_type": atom.virtual_site_kind,
        }
        for atom in topology.atoms
    ]
    legacy["constraint"] = [
        {
            "geom_type": group.geometry_type,
            "beads": [list(beads) for beads in group.beads],
            "func": group.function,
            "value": group.parameters.length,
            "value_user": group.input_parameters.length,
            "avg": group.average,
            "hist": group.histogram,
        }
        for group in topology.constraints
    ]
    for name, groups in (("bond", topology.bonds), ("angle", topology.angles)):
        legacy[name] = [
            {
                "geom_type": group.geometry_type,
                "beads": [list(beads) for beads in group.beads],
                "func": group.function,
                "value": group.parameters.equilibrium,
                "value_user": group.input_parameters.equilibrium,
                "fct": group.parameters.force_constant,
                "fct_user": group.input_parameters.force_constant,
                "avg": group.average,
                "hist": group.histogram,
            }
            for group in groups
        ]
    legacy["dihedral"] = []
    for group in topology.dihedrals:
        params, value, force, multiplicity = _legacy_dihedral_parameters(
            group.parameters
        )
        input_params, input_value, input_force, input_multiplicity = (
            _legacy_dihedral_parameters(group.input_parameters)
        )
        if multiplicity != input_multiplicity:
            raise ValueError("Active and input periodic multiplicities must match.")
        legacy["dihedral"].append(
            {
                "geom_type": group.geometry_type,
                "beads": [list(beads) for beads in group.beads],
                "func": group.function,
                "value": value,
                "value_user": input_value,
                "fct": force,
                "fct_user": input_force,
                "params": params,
                "params_user": input_params,
                "mult": multiplicity,
                "avg": group.average,
                "hist": group.histogram,
                "phase_moment_resultant": group.phase_moment_resultant,
                "polynomial_symmetry_tv": group.polynomial_symmetry_tv,
                "coefficient_bound": group.coefficient_bound,
            }
        )
    for section, kind in (
        ("virtual_sites2", 2),
        ("virtual_sites3", 3),
        ("virtual_sites4", 4),
        ("virtual_sitesn", "n"),
    ):
        legacy[section] = {
            site.bead_id: {
                "bead_id": site.bead_id,
                "func": site.function,
                "vs_def_beads_ids": list(site.defining_beads),
                "vs_params": (
                    None
                    if site.parameters is None
                    else (
                        site.parameters[0]
                        if kind == 2
                        else list(site.parameters)
                    )
                ),
            }
            for site in topology.virtual_sites_of_kind(kind)
        }
    legacy["exclusion"] = [list(exclusion) for exclusion in topology.exclusions]
    legacy["real_beads_ids"] = list(topology.real_bead_ids)
    legacy["vs_beads_ids"] = list(topology.virtual_bead_ids)
    legacy["nb_constraints"] = topology.constraint_count
    legacy["nb_bonds"] = topology.bond_count
    legacy["nb_angles"] = topology.angle_count
    legacy["nb_dihedrals"] = topology.dihedral_count
    legacy.validate()
    return legacy
