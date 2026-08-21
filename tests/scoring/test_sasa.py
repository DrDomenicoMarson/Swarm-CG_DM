"""Tests for strict AA/Martini 3 SASA radii and execution."""

from pathlib import Path
from unittest.mock import patch

import MDAnalysis as mda
import numpy as np
import pytest

from swarmcg.config_types import SasaConfig, SwarmConfig
from swarmcg.context import OptimizationContext
from swarmcg.io import read_cg_topology
from swarmcg.sasa_types import SasaRepresentation
from swarmcg.scoring.sasa import (
    compute_sasa,
    resolve_aa_radii,
    resolve_martini3_radii,
)
from swarmcg.shared.exceptions import ComputationError
from swarmcg.simulations.runner import GromacsCommandResult
from swarmcg.topology import Atom, CGTopology, MoleculeType


def _universe(names, types, masses, *, resname="MOL", frames=1):
    universe = mda.Universe.empty(
        len(names),
        n_residues=1,
        atom_resindex=np.zeros(len(names), dtype=int),
        trajectory=True,
    )
    universe.add_TopologyAttr("names", names)
    universe.add_TopologyAttr("types", types)
    universe.add_TopologyAttr("resnames", [resname])
    universe.add_TopologyAttr("resids", [1])
    universe.add_TopologyAttr("segids", ["SYSTEM"])
    universe.add_TopologyAttr("masses", masses)
    coordinates = np.zeros((frames, len(names), 3), dtype=float)
    coordinates[:, :, 0] = np.arange(len(names), dtype=float)
    universe.load_new(coordinates, format=mda.coordinates.memory.MemoryReader)
    for frame in universe.trajectory:
        frame.dimensions = [50.0, 50.0, 50.0, 90.0, 90.0, 90.0]
    return universe


def test_sasa_config_validates_protocol_values():
    assert SasaConfig().probe_radius_nm == 0.191
    assert SasaConfig().sphere_points == 4800
    with pytest.raises(ValueError, match="probe_radius_nm"):
        SasaConfig(probe_radius_nm=float("nan"))
    with pytest.raises(ValueError, match="sphere_points"):
        SasaConfig(sphere_points=0)


def test_rowland_taylor_resolution_uses_labels_and_mass():
    universe = _universe(
        ["C1", "CL1", "H1"],
        ["c3", "cl", "h1"],
        [12.011, 35.45, 1.008],
    )

    radii = resolve_aa_radii(universe, [0, 1, 2])

    assert radii.values_nm == pytest.approx((0.177, 0.176, 0.110))
    assert "Rowland" in radii.source


def test_non_elemental_force_field_type_defers_to_name_and_mass():
    universe = _universe(["C1"], ["opls_135"], [12.011])

    assert resolve_aa_radii(universe, [0]).values_nm == pytest.approx((0.177,))


def test_missing_atom_types_can_resolve_from_name_and_mass():
    universe = _universe(["O1"], ["o"], [15.999])
    del universe._topology.types

    assert resolve_aa_radii(universe, [0]).values_nm == pytest.approx((0.158,))


def test_duplicate_atom_names_are_allowed_only_when_radius_is_identical():
    universe = _universe(["C1", "C1"], ["c3", "c3"], [12.011, 12.011])

    assert resolve_aa_radii(universe, [0, 1]).values_nm == pytest.approx(
        (0.177, 0.177)
    )


@pytest.mark.parametrize(
    ("names", "types", "masses", "message"),
    [
        (["C1"], ["h1"], [12.011], "Conflicting"),
        (["X1"], ["xx"], [12.011], "Cannot resolve"),
        (["C1"], ["c3"], [1.008], "conflicts with"),
    ],
)
def test_rowland_taylor_resolution_rejects_ambiguous_atoms(
    names, types, masses, message
):
    universe = _universe(names, types, masses)

    with pytest.raises(ComputationError, match=message):
        resolve_aa_radii(universe, [0])


def test_gromacs_override_is_resolved_to_every_atom(tmp_path):
    universe = _universe(
        ["C1", "H1"], ["custom", "custom"], [99.0, 99.0], resname="LIG1"
    )
    override = tmp_path / "vdwradii.dat"
    override.write_text("??? C 0.201\nLIG1 H1 0.123\n", encoding="utf-8")

    radii = resolve_aa_radii(universe, [0, 1], override)

    assert radii.values_nm == pytest.approx((0.201, 0.123))


def test_gromacs_override_rejects_unresolved_and_conflicting_names(tmp_path):
    universe = _universe(["C1"], ["custom"], [99.0])
    unresolved = tmp_path / "unresolved.dat"
    unresolved.write_text("MOL H 0.1\n", encoding="utf-8")
    with pytest.raises(ComputationError, match="No AA radius override"):
        resolve_aa_radii(universe, [0], unresolved)

    conflicting = tmp_path / "conflicting.dat"
    conflicting.write_text("MOL C 0.1\nMOL C 0.2\n", encoding="utf-8")
    with pytest.raises(ComputationError, match="Conflicting AA radii"):
        resolve_aa_radii(universe, [0], conflicting)


def test_martini3_radii_cover_regular_small_tiny_and_virtual_sites():
    topology = CGTopology(
        molecule=MoleculeType("MOL", 1),
        atoms=[
            Atom(0, "C1", 1, "MOL", "R1", 1, 0.0),
            Atom(1, "SC1", 1, "MOL", "S1", 2, 0.0),
            Atom(2, "TC5", 1, "MOL", "V1", 3, 0.0, virtual_site_kind=2),
            Atom(3, "DUM", 1, "MOL", "D1", 4, 0.0, virtual_site_kind=2),
            Atom(4, "TN6d", 1, "MOL", "T1", 5, 0.0),
            Atom(5, "SQ4p", 1, "MOL", "Q1", 6, 1.0),
            Atom(6, "D", 1, "MOL", "I1", 7, 2.0),
        ],
    )

    radii = resolve_martini3_radii(topology)

    assert radii.values_nm == pytest.approx(
        (0.264, 0.230, 0.191, 0.0, 0.191, 0.230, 0.264)
    )
    with pytest.raises(ComputationError, match="unsupported non-Martini-3"):
        resolve_martini3_radii(
            CGTopology(
                atoms=[Atom(0, "custom", 1, "MOL", "X1", 1, 0.0)]
            )
        )


def test_topology_defines_virtual_sites_independently_of_type_prefix(tmp_path):
    topology_file = tmp_path / "virtual.itp"
    topology_file.write_text(
        """[ moleculetype ]
MOL 1
[ atoms ]
1 C1  1 MOL R1 1 0
2 C1  1 MOL R2 2 0
3 TC5 1 MOL V1 3 0
[ virtual_sites2 ]
3 1 2 1 0.5
""",
        encoding="utf-8",
    )

    topology = read_cg_topology(topology_file)

    assert topology.virtual_bead_ids == (2,)
    assert topology.real_bead_ids == (0, 1)
    assert resolve_martini3_radii(topology).values_nm[2] == 0.191


def test_compute_sasa_uses_named_full_surface_and_strict_protocol(tmp_path):
    topology = CGTopology(
        molecule=MoleculeType("MOL", 1),
        atoms=[
            Atom(0, "C1", 1, "MOL", "R1", 1, 0.0),
            Atom(1, "SC1", 1, "MOL", "S1", 2, 0.0),
            Atom(2, "TC5", 1, "MOL", "V1", 3, 0.0, virtual_site_kind=2),
            Atom(3, "DUM", 1, "MOL", "D1", 4, 0.0, virtual_site_kind=2),
        ],
    )
    context = OptimizationContext(config=SwarmConfig())
    context.cg_itp = topology
    context.scoring.cg_universe = _universe(
        ["R1", "S1", "V1", "D1"],
        ["C1", "SC1", "TC5", "DUM"],
        [72.0, 54.0, 36.0, 0.0],
        frames=2,
    )
    observed = {}

    def fake_exec(command, *, stdin_text=None, cwd=None):
        observed["command"] = command
        Path(cwd, "sasa.xvg").write_text(
            "@ title synthetic\n0 1.25\n1 1.75\n", encoding="utf-8"
        )
        return GromacsCommandResult(tuple(command), 0, "complete", "")

    with patch("swarmcg.scoring.sasa.exec_gmx", side_effect=fake_exec):
        measurement = compute_sasa(
            context, SasaRepresentation.CG, tmp_path / "cg"
        )

    assert measurement.mean == 1.5
    assert measurement.standard_deviation == 0.25
    assert measurement.frame_count == 2
    assert measurement.protocol.probe_radius_nm == 0.191
    assert measurement.protocol.sphere_points == 4800
    assert "-ndots" in observed["command"]
    assert "sasa_surface" in observed["command"]
    assert (tmp_path / "cg" / "surface.ndx").read_text().split()[-3:] == [
        "1",
        "2",
        "3",
    ]
    assert (tmp_path / "cg" / "structure.gro").read_text().splitlines()[1].strip() == "4"
    assert (tmp_path / "cg" / "protocol.json").is_file()
    assert (tmp_path / "cg" / "gromacs_stdout.txt").read_text() == "complete"


def test_compute_sasa_rejects_radius_warning(tmp_path):
    context = OptimizationContext(config=SwarmConfig())
    context.cg_itp = CGTopology(
        atoms=[Atom(0, "C1", 1, "MOL", "R1", 1, 0.0)]
    )
    context.scoring.cg_universe = _universe(["R1"], ["C1"], [72.0])
    warning = GromacsCommandResult(
        ("gmx", "sasa"),
        0,
        "",
        "Could not find a Van der Waals radius for atom R1",
    )

    with (
        patch("swarmcg.scoring.sasa.exec_gmx", return_value=warning),
        pytest.raises(ComputationError, match="unresolved radius"),
    ):
        compute_sasa(context, SasaRepresentation.CG, tmp_path / "cg")


def test_compute_sasa_never_accepts_stale_output(tmp_path):
    context = OptimizationContext(config=SwarmConfig())
    context.cg_itp = CGTopology(
        atoms=[Atom(0, "C1", 1, "MOL", "R1", 1, 0.0)]
    )
    context.scoring.cg_universe = _universe(["R1"], ["C1"], [72.0])
    destination = tmp_path / "cg"
    destination.mkdir()
    (destination / "sasa.xvg").write_text("0 999\n", encoding="utf-8")
    success_without_output = GromacsCommandResult(
        ("gmx", "sasa"), 0, "complete", ""
    )

    with (
        patch(
            "swarmcg.scoring.sasa.exec_gmx",
            return_value=success_without_output,
        ),
        pytest.raises(ComputationError, match="did not create"),
    ):
        compute_sasa(context, SasaRepresentation.CG, destination)
