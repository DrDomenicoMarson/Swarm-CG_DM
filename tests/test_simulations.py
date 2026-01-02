import os
import pytest
from unittest.mock import MagicMock, patch, mock_open
from swarmcg.simulations.runner import SimulationStep, SimulationManager, config_to_runner
from swarmcg.config_types import SwarmConfig

@pytest.fixture
def mock_sim_setup():
    return {
        "exec": "gmx",
        "gro": "input.gro",
        "mdp": "minim.mdp",
        "top": "topol.top",
        "md_output": "output",
        "maxwarn": 1,
        "swarmcg_flag": "flag",
        "step_name": "minimization",
        "nb_threads": 1,
        "gpu_id": "",
        "mpi_tasks": 0,
        "monitor_file": "output.log",
        "keep_alive_n_cycles": 10,
        "seconds_between_checks": 1,
        "simulation_config": MagicMock()
    }

def test_simulation_step_init(mock_sim_setup):
    step = SimulationStep(mock_sim_setup)
    assert step.step_name == "minimization"
    assert step.swarmcg_flag == "flag"
    assert step.output_gro == "output.gro"

def test_simulation_step_validation(mock_sim_setup):
    del mock_sim_setup["exec"]
    with pytest.raises(Exception): # expecting InputArgumentError which inherits from Exception
        SimulationStep(mock_sim_setup)

@patch("subprocess.Popen")
@patch("os.path.isfile", return_value=True)
@patch("os.path.getsize", return_value=100)
def test_simulation_step_run_md(mock_getsize, mock_isfile, mock_popen, mock_sim_setup):
    # Mocking process behavior
    process_mock = MagicMock()
    process_mock.poll.side_effect = [None, 0] # Alive once, then finished
    process_mock.communicate.return_value = (b"", b"")
    process_mock.returncode = 0
    mock_popen.return_value.__enter__.return_value = process_mock
    
    step = SimulationStep(mock_sim_setup)
    # Testing internal _run_md directly or via run?
    # run requires chain which is harder to mock without full integration
    # Let's test _run_md
    
    cmd = "gmx mdrun -deffnm output"
    ret_code = step._run_md(cmd)
    
    assert ret_code == 0
    mock_popen.assert_called()

@patch("os.chdir")
@patch("swarmcg.simulations.runner.SimulationStep")
@patch("os.path.isfile", return_value=True)
@patch("builtins.open", new_callable=mock_open, read_data="nsteps=1000\nnstlog=100\ndt=0.002")
def test_simulation_manager_run(mock_open, mock_isfile, mock_sim_step_cls, mock_chdir):
    config = SwarmConfig()
    manager = SimulationManager(config)
    
    # Mock step instance
    mock_step_instance = MagicMock()
    mock_step_instance.output_gro = "next_step.gro"
    mock_sim_step_cls.return_value = mock_step_instance
    
    # Run
    manager.run_simulation("/tmp/test_dir")
    
    # Verify directory change
    mock_chdir.assert_any_call("/tmp/test_dir")
    
    # Verify SimulationStep creation (3 times: mini, equi, prod)
    assert mock_sim_step_cls.call_count == 3
