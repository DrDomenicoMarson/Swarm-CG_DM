"""Validation tests for the typed coarse-grained topology reader."""

import pytest

from swarmcg.io.topology import read_cg_topology, write_cg_topology
from swarmcg.io.validation import validate_mapping_bead_count
from swarmcg.shared import exceptions
from swarmcg.simulations.polynomial import CBTParameters, RBParameters
from swarmcg.topology import CGTopology


def _minimal_itp(section: str) -> str:
    """Return a four-atom ITP with one caller-supplied section."""
    return f"""[ moleculetype ]
MOL 1

[ atoms ]
1 P1 1 MOL B1 1 0 72
2 P1 1 MOL B2 2 0 72
3 P1 1 MOL B3 3 0 72
4 P1 1 MOL B4 4 0 72

{section}
"""


def test_read_cg_topology_and_validate_mapping_count():
    """The reader returns typed counts and mapping validation uses real beads."""
    topology = read_cg_topology("tests/data/test.itp")

    assert isinstance(topology, CGTopology)
    assert topology.bond_count == 23
    assert topology.angle_count == 6
    assert topology.dihedral_count == 1
    assert topology.constraint_count == 0
    assert len(topology.virtual_sites_of_kind(2)) == 1
    with pytest.raises(exceptions.MissformattedFile):
        validate_mapping_bead_count(topology, {})
    validate_mapping_bead_count(topology, dict.fromkeys(range(26)))


def test_read_cg_topology_basic_counts():
    """The basic bundled topology derives its expected group counts."""
    topology = read_cg_topology("tests/data/cg_model.itp")

    assert topology.bond_count == 4
    assert topology.angle_count == 5
    assert topology.dihedral_count == 0
    assert topology.constraint_count == 0


def test_restricted_bending_rejects_unsafe_input_equilibrium(tmp_path):
    """Function 10 rejects unsafe equilibrium angles at parse time."""
    source = open("tests/data/restricted_bending_safe.itp").read()
    unsafe = tmp_path / "unsafe_reb.itp"
    unsafe.write_text(source.replace("120.0   25.0", "180.0   25.0"))

    with pytest.raises(exceptions.MissformattedFile, match="10, 170"):
        read_cg_topology(unsafe)


def test_standard_rb_and_cbt_records_are_canonicalized(tmp_path):
    """Polynomial records use force-relevant typed canonical parameters."""
    path = tmp_path / "polynomial.itp"
    path.write_text(
        _minimal_itp(
            """[ dihedrals ]
; dihedral type RB
1 2 3 4 3  9 1 2 3 4 5
; dihedral type CBT
1 2 3 4 11  2 1 -2 3 -4 5"""
        )
    )

    topology = read_cg_topology(path)

    assert topology.dihedrals[0].parameters == RBParameters((1, 2, 3, 4, 5))
    assert topology.dihedrals[1].parameters == CBTParameters((2, -4, 6, -8, 10))
    assert topology.dihedrals[0].gromacs_parameters == (-15, 1, 2, 3, 4, 5)
    assert topology.dihedrals[1].gromacs_parameters == (
        10,
        0.2,
        -0.4,
        0.6,
        -0.8,
        1,
    )


@pytest.mark.parametrize(
    "section,match",
    [
        ("[ constraints ]\n1 2 1 nan", "Non-finite length"),
        ("[ bonds ]\n1 2 1 nan 100", "Non-finite equilibrium"),
        ("[ bonds ]\n1 2 1 0.3 inf", "Non-finite force constant"),
        ("[ angles ]\n1 2 3 1 nan 100", "Non-finite equilibrium"),
        ("[ angles ]\n1 2 3 1 120 -inf", "Non-finite force constant"),
        ("[ dihedrals ]\n1 2 3 4 1 nan 2 1", "Non-finite phase"),
        ("[ dihedrals ]\n1 2 3 4 1 30 inf 1", "Non-finite force constant"),
        ("[ dihedrals ]\n1 2 3 4 3 0 1 2 nan 4 5", "Non-finite polynomial"),
        ("[ virtual_sites2 ]\n1 2 3 1 inf", "Non-finite virtual-site"),
        ("[ virtual_sites3 ]\n1 2 3 4 1 nan 0", "Non-finite virtual-site"),
        ("[ virtual_sites4 ]\n1 2 3 4 5 2 nan 0 0", "Non-finite virtual-site"),
        ("[ virtual_sitesn ]\n1 3 2 nan 3 0.5", "Non-finite virtual-site"),
    ],
)
def test_itp_reader_rejects_nonfinite_numeric_fields(tmp_path, section, match):
    """Every supported section rejects non-finite serialized values."""
    path = tmp_path / "nonfinite.itp"
    source = _minimal_itp(section)
    if section.startswith("[ virtual_sites"):
        source = source.replace("1 P1 1 MOL B1", "1 vP 1 MOL B1", 1)
    if section.startswith("[ virtual_sites4"):
        source = source.replace(
            "4 P1 1 MOL B4 4 0 72",
            "4 P1 1 MOL B4 4 0 72\n5 P1 1 MOL B5 5 0 72",
        )
    path.write_text(source)

    with pytest.raises(exceptions.MissformattedFile, match=match):
        read_cg_topology(path)


@pytest.mark.parametrize(
    "atom_record,match",
    [
        ("1 P1 1 MOL B1 1 nan 72", "Non-finite charge"),
        ("1 P1 1 MOL B1 1 0 inf", "Non-finite mass"),
    ],
)
def test_itp_reader_rejects_nonfinite_atom_fields(tmp_path, atom_record, match):
    """Atom charges and explicit masses must be finite."""
    path = tmp_path / "nonfinite_atom.itp"
    path.write_text(
        _minimal_itp("").replace("1 P1 1 MOL B1 1 0 72", atom_record)
    )

    with pytest.raises(exceptions.MissformattedFile, match=match):
        read_cg_topology(path)


@pytest.mark.parametrize("multiplicity", [0, -1])
def test_periodic_dihedral_requires_positive_multiplicity(tmp_path, multiplicity):
    """Periodic functions reject non-positive multiplicities."""
    path = tmp_path / "bad_mult.itp"
    path.write_text(
        _minimal_itp(f"[ dihedrals ]\n1 2 3 4 1 30 2 {multiplicity}")
    )

    with pytest.raises(exceptions.MissformattedFile, match="positive integer"):
        read_cg_topology(path)


def test_periodic_negative_force_is_read_and_written_canonically(tmp_path):
    """Negative periodic forces canonicalize without changing semantics."""
    path = tmp_path / "periodic.itp"
    path.write_text(_minimal_itp("[ dihedrals ]\n1 2 3 4 1 35 -3 2"))

    topology = read_cg_topology(path)
    group = topology.dihedrals[0]
    assert group.equilibrium == -145.0
    assert group.force_constant == 3.0

    output = tmp_path / "canonical.itp"
    write_cg_topology(topology, output)
    reparsed = read_cg_topology(output)
    assert reparsed.dihedrals[0].equilibrium == -145.0
    assert reparsed.dihedrals[0].force_constant == 3.0
