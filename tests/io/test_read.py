import os

import pytest

from swarmcg.shared import exceptions
from swarmcg.io.read import read_itp, read_cg_itp_file, validate_cg_itp
from swarmcg.io.write import write_cg_itp_file
from swarmcg.config_types import SwarmConfig
from swarmcg.io.itp import CGITP

required_itp_fields = ["real_beads_ids", "vs_beads_ids", "nb_bonds", "nb_angles",
                       "nb_dihedrals", "nb_constraints",
                       "moleculetype", "atoms", "constraint", "bond", "angle", "dihedral",
                       "virtual_sites2", "virtual_sites3", "virtual_sites4", "virtual_sitesn",
                       "exclusion"]


def check_ipt_dict(cg_itp):
    assert len(cg_itp["bond"]) == 23
    assert len(cg_itp["bond"]) == cg_itp["nb_bonds"]
    assert len(cg_itp["angle"]) == 6
    assert len(cg_itp["angle"]) == cg_itp["nb_angles"]
    assert len(cg_itp["dihedral"]) == 1
    assert len(cg_itp["dihedral"]) == cg_itp["nb_dihedrals"]
    assert len(cg_itp["constraint"]) == 0
    assert len(cg_itp["virtual_sites2"]) == 1


# ... (lines 9-44 skipped in view but consistent)

def test_read_cg_itp_file(ns_opt):
    # when:
    filename = f"tests/data/test.itp"
    ns = ns_opt(cg_itp_filename=filename)
    config = SwarmConfig.from_namespace(ns)

    from swarmcg.io.itp import CGITP
    result = read_cg_itp_file(config)
    assert isinstance(result, CGITP)
    assert all([field in result for field in required_itp_fields])
    check_ipt_dict(result)

    # when:
    all_beads = []

    # then:
    with pytest.raises(exceptions.MissformattedFile):
        _ = validate_cg_itp(result, all_beads=all_beads)

    # when:
    all_beads = list(range(26))

    # then:
    _ = validate_cg_itp(result, all_beads=all_beads)


def test_read_cg_itp_file_basic(ns_opt):
    # when:
    filename = f"tests/data/cg_model.itp"
    ns = ns_opt(cg_itp_filename=filename)
    config = SwarmConfig.from_namespace(ns)

    # then:
    result = read_cg_itp_file(config)
    assert len(result["bond"]) == 4
    assert len(result["bond"]) == result["nb_bonds"]
    assert len(result["angle"]) == 5
    assert len(result["angle"]) == result["nb_angles"]
    assert result["nb_dihedrals"] == 0
    assert result["nb_constraints"] == 0
    assert all([field in result for field in required_itp_fields])


def test_restricted_bending_rejects_unsafe_input_equilibrium(tmp_path, ns_opt):
    source = open("tests/data/restricted_bending_safe.itp").read()
    unsafe = tmp_path / "unsafe_reb.itp"
    unsafe.write_text(source.replace("120.0   25.0", "180.0   25.0"))
    config = SwarmConfig.from_namespace(ns_opt(cg_itp_filename=str(unsafe)))

    with pytest.raises(exceptions.MissformattedFile, match="10, 170"):
        read_cg_itp_file(config)


def test_standard_rb_and_cbt_records_are_read_and_canonicalized(tmp_path, ns_opt):
    topology = tmp_path / "polynomial.itp"
    topology.write_text(
        """[ moleculetype ]
MOL 1

[ atoms ]
1 P1 1 MOL B1 1 0 72
2 P1 1 MOL B2 2 0 72
3 P1 1 MOL B3 3 0 72
4 P1 1 MOL B4 4 0 72

[ dihedrals ]
; dihedral type RB
1 2 3 4 3  9 1 2 3 4 5

; dihedral type CBT
1 2 3 4 11  2 1 -2 3 -4 5
"""
    )
    cfg = SwarmConfig.from_namespace(ns_opt(cg_itp_filename=str(topology)))

    parsed = read_cg_itp_file(cfg)

    assert parsed["dihedral"][0]["params"] == [-15.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert parsed["dihedral"][1]["params"] == [10.0, 0.2, -0.4, 0.6, -0.8, 1.0]


def _minimal_itp(section):
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


@pytest.mark.parametrize(
    "section,match",
    [
        ("[ constraints ]\n1 2 1 nan", "Non-finite length"),
        ("[ bonds ]\n1 2 1 nan 100", "Non-finite length"),
        ("[ bonds ]\n1 2 1 0.3 inf", "Non-finite force constant"),
        ("[ angles ]\n1 2 3 1 nan 100", "Non-finite equilibrium angle"),
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
def test_itp_reader_rejects_nonfinite_numeric_fields(
    tmp_path, ns_opt, section, match
):
    topology = tmp_path / "nonfinite.itp"
    source = _minimal_itp(section)
    if section.startswith("[ virtual_sites"):
        source = source.replace("1 P1 1 MOL B1", "1 vP 1 MOL B1", 1)
    if section.startswith("[ virtual_sites4"):
        source = source.replace(
            "4 P1 1 MOL B4 4 0 72",
            "4 P1 1 MOL B4 4 0 72\n5 P1 1 MOL B5 5 0 72",
        )
    topology.write_text(source)
    cfg = SwarmConfig.from_namespace(ns_opt(cg_itp_filename=str(topology)))

    with pytest.raises(exceptions.MissformattedFile, match=match):
        read_cg_itp_file(cfg)


@pytest.mark.parametrize(
    "atom_record,match",
    [
        ("1 P1 1 MOL B1 1 nan 72", "Non-finite charge"),
        ("1 P1 1 MOL B1 1 0 inf", "Non-finite mass"),
    ],
)
def test_itp_reader_rejects_nonfinite_atom_fields(
    tmp_path, ns_opt, atom_record, match
):
    topology = tmp_path / "nonfinite_atom.itp"
    topology.write_text(_minimal_itp("").replace(
        "1 P1 1 MOL B1 1 0 72", atom_record
    ))
    cfg = SwarmConfig.from_namespace(ns_opt(cg_itp_filename=str(topology)))

    with pytest.raises(exceptions.MissformattedFile, match=match):
        read_cg_itp_file(cfg)


def test_cgitp_validate_defensively_rejects_nonfinite_state(ns_opt):
    cfg = SwarmConfig.from_namespace(
        ns_opt(cg_itp_filename="tests/data/cg_model.itp")
    )
    topology = read_cg_itp_file(cfg)
    assert isinstance(topology, CGITP)
    topology["angle"][0]["fct"] = float("nan")

    with pytest.raises(exceptions.MissformattedFile, match="finite numeric"):
        topology.validate()


@pytest.mark.parametrize("multiplicity", [0, -1])
def test_periodic_dihedral_requires_positive_multiplicity(
    tmp_path, ns_opt, multiplicity
):
    topology = tmp_path / "bad_mult.itp"
    topology.write_text(
        _minimal_itp(f"[ dihedrals ]\n1 2 3 4 1 30 2 {multiplicity}")
    )
    cfg = SwarmConfig.from_namespace(ns_opt(cg_itp_filename=str(topology)))

    with pytest.raises(exceptions.MissformattedFile, match="positive integer"):
        read_cg_itp_file(cfg)


def test_periodic_negative_force_is_read_and_written_canonically(
    tmp_path, ns_opt
):
    topology = tmp_path / "periodic.itp"
    topology.write_text(
        _minimal_itp("[ dihedrals ]\n1 2 3 4 1 35 -3 2")
    )
    cfg = SwarmConfig.from_namespace(ns_opt(cg_itp_filename=str(topology)))

    parsed = read_cg_itp_file(cfg)

    assert parsed["dihedral"][0]["value"] == -145.0
    assert parsed["dihedral"][0]["fct"] == 3.0
    output = tmp_path / "canonical.itp"
    write_cg_itp_file(parsed, output)
    reparsed_cfg = SwarmConfig.from_namespace(
        ns_opt(cg_itp_filename=str(output))
    )
    reparsed = read_cg_itp_file(reparsed_cfg)
    assert reparsed["dihedral"][0]["value"] == -145.0
    assert reparsed["dihedral"][0]["fct"] == 3.0
