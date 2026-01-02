
import pytest
import numpy as np
from unittest.mock import MagicMock
import MDAnalysis as mda
from swarmcg.scoring import bonds, angles, dihedrals
from swarmcg.config_types import SwarmConfig

@pytest.fixture
def simple_universe():
    # create a simple universe with 4 atoms in a line at 0, 1, 2, 3 on x axis
    u = mda.Universe.empty(4, trajectory=True)
    u.add_TopologyAttr("masses", [1, 1, 1, 1])
    
    coords = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0], # dist 0.1 nm (1.0 A)
        [0.3, 0.0, 0.0], # dist 0.2 nm (2.0 A) from prev
        [0.6, 0.0, 0.0]  # dist 0.3 nm (3.0 A) from prev
    ])
    # MDAnalysis uses Angstroms, SwarmCG uses nm for bonds?
    # Actually SwarmCG converts MDA (Angstrom) to nm by dividing by 10.
    # So if we put coords in Angstroms:
    
    coords_angstrom = coords * 10
    u.load_new(coords_angstrom[None, :, :], format=mda.coordinates.memory.MemoryReader)
    return u

def test_get_AA_bonds_distrib(simple_universe):
    # Bead 0-1: 0.1 nm
    # Bead 1-2: 0.2 nm
    beads_ids = [[0, 1], [1, 2]]
    
    config = MagicMock(spec=SwarmConfig)
    # MagicMock doesn't automatically create sub-mocks for non-existent attributes if spec is strict/pydantic maybe?
    # Actually SwarmConfig has 'optimization' field.
    # But let's be explicit.
    config.optimization = MagicMock()
    config.optimization.bonds_scaling = 1.0
    config.optimization.min_bonds_length = 0.0
    config.optimization.bonds_scaling_specific = None
    
    avg, hist, values = bonds.get_AA_bonds_distrib(
        simple_universe, beads_ids, "bond", 0, config
    )
    
    expected_values = np.array([0.1, 0.2])
    np.testing.assert_allclose(values, expected_values, rtol=1e-5)
    assert avg == 0.15

def test_get_AA_angles_distrib(simple_universe):
    # Create simplified coords for 90 degree angle
    u = mda.Universe.empty(3, trajectory=True)
    coords = np.array([
        [10.0, 0.0, 0.0], # 1
        [0.0, 0.0, 0.0],  # 0 (center)
        [0.0, 10.0, 0.0]  # 2
    ])
    u.load_new(coords[None, :, :], format=mda.coordinates.memory.MemoryReader)
    
    beads_ids = [[0, 1, 2]] # 1-0-2 angle should be 90 degrees
    
    avg, hist, values, values_rad = angles.get_AA_angles_distrib(u, beads_ids)
    
    assert round(avg, 1) == 90.0

def test_get_AA_dihedrals_distrib():
    # Torsion check
    u = mda.Universe.empty(4, trajectory=True)
    coords = np.array([
        [10.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 10.0, 10.0]
    ])
    # 1-0-2-3 dihedral should be -90 or 90 depending on definition?
    # Flat 0, 90 deg twist.
    u.load_new(coords[None, :, :], format=mda.coordinates.memory.MemoryReader)
    
    beads_ids = [[0, 1, 2, 3]]
    avg, hist, values, values_rad = dihedrals.get_AA_dihedrals_distrib(u, beads_ids)
    
    # Check value is close to 90 or -90
    assert abs(round(avg, 1)) == 90.0
