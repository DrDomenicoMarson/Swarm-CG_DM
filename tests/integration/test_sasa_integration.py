"""Real-GROMACS validation of the Martini 3 SASA protocol."""

import math
import shutil

import MDAnalysis as mda
import numpy as np
import pytest

from swarmcg.config_types import SwarmConfig
from swarmcg.context import OptimizationContext
from swarmcg.sasa_types import SasaRepresentation
from swarmcg.scoring.sasa import compute_sasa
from swarmcg.topology import Atom, CGTopology


gmx_path = shutil.which("gmx")
pytestmark = pytest.mark.skipif(gmx_path is None, reason="GROMACS is not on PATH")


def _context(types, positions_angstrom, *, virtual_ids=()):
    atom_count = len(types)
    universe = mda.Universe.empty(
        atom_count,
        n_residues=1,
        atom_resindex=np.zeros(atom_count, dtype=int),
        trajectory=True,
    )
    names = [f"B{index + 1}" for index in range(atom_count)]
    universe.add_TopologyAttr("names", names)
    universe.add_TopologyAttr("types", types)
    universe.add_TopologyAttr("resnames", ["MOL"])
    universe.add_TopologyAttr("resids", [1])
    universe.add_TopologyAttr("segids", ["SYSTEM"])
    universe.add_TopologyAttr("masses", np.full(atom_count, 72.0))
    universe.atoms.positions = np.asarray(positions_angstrom, dtype=float)
    universe.dimensions = [100.0, 100.0, 100.0, 90.0, 90.0, 90.0]
    topology = CGTopology(
        atoms=[
            Atom(
                index,
                bead_type,
                1,
                "MOL",
                names[index],
                index + 1,
                0.0,
                virtual_site_kind=2 if index in virtual_ids else None,
            )
            for index, bead_type in enumerate(types)
        ]
    )
    config = SwarmConfig()
    config.gromacs.gmx_path = gmx_path
    context = OptimizationContext(config=config)
    context.cg_itp = topology
    context.scoring.cg_universe = universe
    return context


def test_single_and_two_sphere_areas_match_analytic_surfaces(tmp_path):
    radius = 0.264 + 0.191
    single = compute_sasa(
        _context(["C1"], [[50.0, 50.0, 50.0]]),
        SasaRepresentation.CG,
        tmp_path / "single",
    )
    expected_single = 4 * math.pi * radius**2
    assert single.mean == pytest.approx(expected_single, rel=2e-3)

    separated = compute_sasa(
        _context(["C1", "C1"], [[30.0, 50.0, 50.0], [70.0, 50.0, 50.0]]),
        SasaRepresentation.CG,
        tmp_path / "separated",
    )
    assert separated.mean == pytest.approx(2 * expected_single, rel=2e-3)

    distance_nm = 0.4
    overlapping = compute_sasa(
        _context(["C1", "C1"], [[48.0, 50.0, 50.0], [52.0, 50.0, 50.0]]),
        SasaRepresentation.CG,
        tmp_path / "overlapping",
    )
    expected_union = 4 * math.pi * radius**2 + 2 * math.pi * radius * distance_nm
    assert overlapping.mean == pytest.approx(expected_union, rel=5e-3)


def test_regular_small_tiny_and_surface_virtual_site_are_included(tmp_path):
    context = _context(
        ["C1", "SC1", "TC5", "DUM"],
        [
            [20.0, 50.0, 50.0],
            [50.0, 50.0, 50.0],
            [80.0, 50.0, 50.0],
            [90.0, 50.0, 50.0],
        ],
        virtual_ids={2, 3},
    )

    measurement = compute_sasa(
        context, SasaRepresentation.CG, tmp_path / "rst-virtual"
    )

    expected = 4 * math.pi * sum(
        (radius + 0.191) ** 2 for radius in (0.264, 0.230, 0.191)
    )
    assert measurement.mean == pytest.approx(expected, rel=2e-3)
    assert (tmp_path / "rst-virtual" / "surface.ndx").read_text().split()[-3:] == [
        "1",
        "2",
        "3",
    ]
    assert (
        tmp_path / "rst-virtual" / "structure.gro"
    ).read_text().splitlines()[1].strip() == "4"


def test_sasa_is_invariant_after_cross_boundary_molecule_is_made_whole(tmp_path):
    centered = _context(
        ["C1", "C1"], [[49.0, 50.0, 50.0], [51.0, 50.0, 50.0]]
    )
    crossed = _context(
        ["C1", "C1"], [[99.0, 50.0, 50.0], [1.0, 50.0, 50.0]]
    )
    crossed.scoring.cg_universe.add_bonds([(0, 1)], guessed=False)
    for _ in crossed.scoring.cg_universe.trajectory:
        mda.lib.mdamath.make_whole(
            crossed.scoring.cg_universe.atoms, inplace=True
        )

    centered_measurement = compute_sasa(
        centered, SasaRepresentation.CG, tmp_path / "centered"
    )
    crossed_measurement = compute_sasa(
        crossed, SasaRepresentation.CG, tmp_path / "crossed"
    )

    assert crossed_measurement.mean == pytest.approx(
        centered_measurement.mean, abs=1e-12
    )


def test_full_aa_and_mapped_selections_include_hydrogen_and_virtual_site(tmp_path):
    context = _context(
        ["C1", "TC5"],
        [[45.0, 50.0, 50.0], [55.0, 50.0, 50.0]],
        virtual_ids={1},
    )
    mapped = context.scoring.cg_universe
    aa = mda.Universe.empty(
        3,
        n_residues=1,
        atom_resindex=[0, 0, 0],
        trajectory=True,
    )
    aa.add_TopologyAttr("names", ["C1", "H1", "O1"])
    aa.add_TopologyAttr("types", ["c3", "h1", "o"])
    aa.add_TopologyAttr("resnames", ["MOL"])
    aa.add_TopologyAttr("resids", [1])
    aa.add_TopologyAttr("segids", ["SYSTEM"])
    aa.add_TopologyAttr("masses", [12.011, 1.008, 15.999])
    aa.atoms.positions = [[48.0, 50.0, 50.0], [50.0, 50.0, 50.0], [52.0, 50.0, 50.0]]
    aa.dimensions = mapped.dimensions
    context.scoring.aa_universe = aa
    context.scoring.aa2cg_universe = mapped
    context.scoring.all_aa_mols = [aa.atoms]

    aa_measurement = compute_sasa(
        context, SasaRepresentation.AA, tmp_path / "aa"
    )
    mapped_measurement = compute_sasa(
        context, SasaRepresentation.AA_MAPPED, tmp_path / "aa_mapped"
    )

    assert aa_measurement.frame_count == mapped_measurement.frame_count == 1
    assert aa_measurement.mean > 0
    assert mapped_measurement.mean > 0
    assert (tmp_path / "aa" / "surface.ndx").read_text().split()[-3:] == [
        "1",
        "2",
        "3",
    ]
    assert (tmp_path / "aa_mapped" / "surface.ndx").read_text().split()[-2:] == [
        "1",
        "2",
    ]
