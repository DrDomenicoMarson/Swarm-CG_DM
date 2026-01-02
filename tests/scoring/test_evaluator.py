
import pytest
from unittest.mock import MagicMock, patch
from swarmcg.scoring.evaluator import SwarmEvaluator
from swarmcg.config_types import SwarmConfig
import numpy as np

@pytest.fixture
def mock_config():
    config = MagicMock(spec=SwarmConfig)
    return config

@pytest.fixture
def evaluator(mock_config):
    return SwarmEvaluator(mock_config)

def test_init(evaluator, mock_config):
    assert evaluator.config == mock_config
    assert evaluator.ns is None

@patch("swarmcg.scoring.evaluator.Mapping")
@patch("swarmcg.scoring.evaluator.io")
@patch("swarmcg.scoring.evaluator.scores")
def test_initialize(mock_scores, mock_io, mock_mapping_cls, evaluator):
    context = MagicMock()
    evaluator.config.reference = "ref.tpr"
    
    mock_mapping_instance = mock_mapping_cls.return_value
    mock_mapping_instance.all_beads = "beads"
    mock_mapping_instance.atom_w = "weights"
    
    mock_io.read_cg_itp_file.return_value = {
        "nb_constraints": 0,
        "atoms": [{"mass": 10.0, "atom": "A1", "residue": "R1", "resnr": 1, "bead_type": "B1"}]
    }
    
    evaluator.initialize(context)
    
    assert evaluator.ns == context
    mock_scores.create_bins_and_dist_matrices.assert_called_with(context)
    mock_mapping_instance.read_ndx_atoms2beads.assert_called()
    mock_io.read_aa_traj.assert_called_with("ref.tpr")
    mock_mapping_instance.map_aa2cg_traj.assert_called()
