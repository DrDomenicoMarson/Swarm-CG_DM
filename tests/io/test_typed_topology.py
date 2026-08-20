"""Regression tests for the internal typed topology reader and writer."""

import pytest

from swarmcg.io.topology import read_cg_topology, write_cg_topology
from swarmcg.shared import exceptions
from swarmcg.shared.periodic import PeriodicDihedralParameters
from swarmcg.simulations.polynomial import CBTParameters, RBParameters
from swarmcg.topology import (
    CGTopology,
    ConstraintParameters,
    HarmonicParameters,
)


def _parameter_values(parameters):
    """Return a uniform numeric signature for a typed parameter object."""
    if isinstance(parameters, ConstraintParameters):
        return (parameters.length,)
    if isinstance(parameters, HarmonicParameters):
        return (parameters.equilibrium, parameters.force_constant)
    if isinstance(parameters, PeriodicDihedralParameters):
        return parameters.to_gromacs()
    return parameters.to_gromacs()


def _assert_semantically_equal(left: CGTopology, right: CGTopology) -> None:
    """Assert topology semantics while tolerating equivalent float serialization."""
    assert left.molecule == right.molecule
    assert left.atoms == right.atoms
    assert left.real_bead_ids == right.real_bead_ids
    assert left.virtual_bead_ids == right.virtual_bead_ids
    assert left.exclusions == right.exclusions
    assert left.virtual_sites == right.virtual_sites
    for left_groups, right_groups in (
        (left.constraints, right.constraints),
        (left.bonds, right.bonds),
        (left.angles, right.angles),
        (left.dihedrals, right.dihedrals),
    ):
        assert len(left_groups) == len(right_groups)
        for left_group, right_group in zip(left_groups, right_groups):
            assert left_group.geometry_type == right_group.geometry_type
            assert left_group.beads == right_group.beads
            assert left_group.function == right_group.function
            assert _parameter_values(left_group.parameters) == pytest.approx(
                _parameter_values(right_group.parameters), abs=1e-12
            )
            assert _parameter_values(left_group.input_parameters) == pytest.approx(
                _parameter_values(right_group.input_parameters), abs=1e-12
            )


def test_typed_reader_exposes_computed_counts_and_typed_tuples():
    """The typed model derives metadata instead of storing duplicate counters."""
    topology = read_cg_topology("tests/data/test.itp")

    assert isinstance(topology, CGTopology)
    assert topology.bond_count == 23
    assert topology.angle_count == 6
    assert topology.dihedral_count == 1
    assert topology.constraint_count == 0
    assert len(topology.real_bead_ids) == 26
    assert len(topology.virtual_bead_ids) == 10
    assert all(isinstance(beads, tuple) for group in topology.bonds for beads in group.beads)
    assert all(isinstance(exclusion, tuple) for exclusion in topology.exclusions)
    assert topology.bonds[0].parameters is not topology.bonds[0].input_parameters


def test_typed_reader_supports_every_virtual_site_function():
    """The bundled fixture exercises all supported virtual-site function IDs."""
    topology = read_cg_topology("tests/data/test.itp")

    observed = {(site.kind, site.function) for site in topology.virtual_sites}
    assert observed == {
        (2, 1),
        (3, 1),
        (3, 2),
        (3, 3),
        (3, 4),
        (4, 2),
        ("n", 1),
        ("n", 2),
        ("n", 3),
    }


def test_typed_reader_supports_every_bonded_function(tmp_path):
    """All supported constraint, bond, angle, and dihedral forms are typed."""
    path = tmp_path / "all_functions.itp"
    path.write_text(
        """[ moleculetype ]
MOL 1

[ atoms ]
1 P1 1 MOL B1 1 0 72
2 P1 1 MOL B2 2 0 72
3 P1 1 MOL B3 3 0 72
4 P1 1 MOL B4 4 0 72

[ constraints ]
1 2 1 0.2

[ bonds ]
1 2 1 0.2 1000

[ angles ]
; angle type harmonic
1 2 3 1 120 25
; angle type cosine
1 2 3 2 110 30
; angle type restricted
1 2 3 10 100 35

[ dihedrals ]
; dihedral type periodic1
1 2 3 4 1 30 -2 1
; dihedral type harmonic
1 2 3 4 2 40 20
; dihedral type rb
1 2 3 4 3 0 1 2 3 4 5
; dihedral type periodic4
1 2 3 4 4 -70 3 2
; dihedral type cbt
1 2 3 4 11 2 1 -2 3 -4 5
"""
    )

    topology = read_cg_topology(path)

    assert [group.function for group in topology.angles] == [1, 2, 10]
    assert [group.function for group in topology.dihedrals] == [1, 2, 3, 4, 11]
    assert isinstance(topology.dihedrals[0].parameters, PeriodicDihedralParameters)
    assert isinstance(topology.dihedrals[1].parameters, HarmonicParameters)
    assert isinstance(topology.dihedrals[2].parameters, RBParameters)
    assert isinstance(topology.dihedrals[3].parameters, PeriodicDihedralParameters)
    assert isinstance(topology.dihedrals[4].parameters, CBTParameters)
    assert topology.dihedrals[0].parameters.force_constant == 2
    assert topology.dihedrals[0].parameters.phase_degrees == -150
    output = tmp_path / "all_functions_roundtrip.itp"
    write_cg_topology(topology, output)
    _assert_semantically_equal(topology, read_cg_topology(output))


@pytest.mark.parametrize(
    "fixture",
    [
        "tests/data/test.itp",
        "tests/data/cg_model.itp",
        "tests/data/restricted_bending_safe.itp",
    ],
)
def test_typed_parse_write_parse_is_semantically_equal(tmp_path, fixture):
    """Canonical writing preserves the typed topology semantics."""
    topology = read_cg_topology(fixture)
    output = tmp_path / "roundtrip.itp"

    write_cg_topology(topology, output)
    reparsed = read_cg_topology(output)

    _assert_semantically_equal(topology, reparsed)


def test_central_validation_rejects_nonfinite_mutated_state():
    """CGTopology.validate catches invalid state introduced after parsing."""
    topology = read_cg_topology("tests/data/cg_model.itp")
    topology.atoms[0].charge = float("nan")

    with pytest.raises(exceptions.MissformattedFile, match="non-finite"):
        topology.validate()
