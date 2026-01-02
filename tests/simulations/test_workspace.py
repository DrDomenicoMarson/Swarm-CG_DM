
import os
import shutil
import pytest
from unittest.mock import MagicMock
from swarmcg.simulations.workspace import WorkspaceManager
from swarmcg.config_types import SwarmConfig
from swarmcg.shared import exceptions

@pytest.fixture
def mock_config():
    config = MagicMock(spec=SwarmConfig)
    config.optimization = MagicMock()
    config.cg_model = MagicMock()
    config.simulation = MagicMock()
    
    # Defaults
    config.optimization.keep_all_sims = False
    
    # File paths
    config.cg_model.cg_itp_filename = "test.itp"
    config.cg_model.gro_input_filename = "test.gro"
    config.cg_model.top_input_filename = "test.top"
    config.simulation.mdp_minimization_filename = "mini.mdp"
    config.simulation.mdp_equi_filename = "equi.mdp"
    config.simulation.mdp_md_filename = "md.mdp"
    
    return config

@pytest.fixture
def workspace_mgr(mock_config):
    return WorkspaceManager(mock_config)

def test_setup_execution_folder_default(workspace_mgr, tmp_path):
    # Test default folder creation
    os.chdir(tmp_path)
    exec_folder = workspace_mgr.setup_execution_folder()
    
    assert os.path.isdir(exec_folder)
    assert os.path.isdir(os.path.join(exec_folder, ".internal"))
    assert "MODEL_OPTI__STARTED" in exec_folder

def test_setup_execution_folder_custom(workspace_mgr, tmp_path):
    # Test named folder creation
    os.chdir(tmp_path)
    folder_name = "MY_MODEL"
    exec_folder = workspace_mgr.setup_execution_folder(folder_name)
    
    assert os.path.abspath(folder_name) == os.path.abspath(exec_folder)
    assert os.path.isdir(folder_name)

def test_setup_execution_folder_exists_error(workspace_mgr, tmp_path):
    # Test overwrite protection
    os.chdir(tmp_path)
    os.mkdir("EXISTING_FOLDER")
    
    with pytest.raises(exceptions.AvoidOverwritingFolder):
        workspace_mgr.setup_execution_folder("EXISTING_FOLDER")

def test_prepare_simulation_input(workspace_mgr, tmp_path):
    os.chdir(tmp_path)
    workspace_mgr.setup_execution_folder("EXEC_DIR")
    
    # Create dummy source files
    for f in ["test.itp", "test.gro", "test.top", "mini.mdp", "equi.mdp", "md.mdp"]:
        with open(f, "w") as fp:
            fp.write(f"content of {f}")
            
    # Add dummy TOP content with includes
    with open("test.top", "w") as fp:
        fp.write("#include \"something.itp\"\n[ molecules ]\n")
            
    workspace_mgr.prepare_simulation_input(["test.itp"])
    
    input_dir = os.path.join(workspace_mgr.exec_folder, "input_sim_files")
    assert os.path.isdir(input_dir)
    assert os.path.isfile(os.path.join(input_dir, "test.itp"))
    assert os.path.isfile(os.path.join(input_dir, "test.gro"))
