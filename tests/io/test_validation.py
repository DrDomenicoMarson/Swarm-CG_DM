"""Scientific validation tests for restricted-bending start coordinates."""

import math

import pytest

from swarmcg.io.validation import validate_restricted_bending_start
from swarmcg.shared import exceptions


def _reb_topology():
    """Return a minimal three-bead restricted-bending topology."""
    return {
        "atoms": [{"bead_id": index} for index in range(3)],
        "angle": [{"func": 10, "beads": [[0, 1, 2]]}],
    }


def _write_gro(path, angle_degrees, *, atom_count=3, nonfinite=False, pbc=False):
    """Write a minimal GRO with a requested first-three-atom angle."""
    theta = math.radians(angle_degrees)
    center = (0.95, 0.5, 0.5) if pbc else (0.5, 0.5, 0.5)
    third = (0.05, 0.5, 0.5) if pbc else (0.7, 0.5, 0.5)
    bond = 0.1 if pbc else 0.2
    first = (
        center[0] + bond * math.cos(theta),
        center[1] + bond * math.sin(theta),
        center[2],
    )
    coordinates = [first, center, third][:atom_count]
    if nonfinite:
        coordinates[0] = (float("nan"), coordinates[0][1], coordinates[0][2])
    lines = ["ReB validation", f"{atom_count:5d}"]
    for index, (x, y, z) in enumerate(coordinates, start=1):
        lines.append(
            f"{1:5d}{'MOL':<5}{f'B{index}':>5}{index:5d}"
            f"{x:8.3f}{y:8.3f}{z:8.3f}"
        )
    lines.append("   1.00000   1.00000   1.00000")
    path.write_text("\n".join(lines) + "\n")


def test_safe_restricted_bending_start_is_accepted(tmp_path):
    gro = tmp_path / "safe.gro"
    _write_gro(gro, 100.0)

    validate_restricted_bending_start(str(gro), _reb_topology())


def test_restricted_bending_start_uses_periodic_minimum_image(tmp_path):
    gro = tmp_path / "periodic.gro"
    _write_gro(gro, 100.0, pbc=True)

    validate_restricted_bending_start(str(gro), _reb_topology())


@pytest.mark.parametrize("angle", [5.0, 175.0])
def test_unsafe_restricted_bending_start_is_rejected(tmp_path, angle):
    gro = tmp_path / "unsafe.gro"
    _write_gro(gro, angle)

    with pytest.raises(exceptions.ScientificValidationError, match="unsafe angles"):
        validate_restricted_bending_start(str(gro), _reb_topology())


def test_restricted_bending_start_rejects_too_few_atoms(tmp_path):
    gro = tmp_path / "short.gro"
    _write_gro(gro, 100.0, atom_count=2)

    with pytest.raises(exceptions.ScientificValidationError, match="requires its first 3"):
        validate_restricted_bending_start(str(gro), _reb_topology())


def test_restricted_bending_start_rejects_nonfinite_coordinates(tmp_path):
    gro = tmp_path / "nonfinite.gro"
    _write_gro(gro, 100.0, nonfinite=True)

    with pytest.raises(exceptions.ScientificValidationError, match="must be finite"):
        validate_restricted_bending_start(str(gro), _reb_topology())
