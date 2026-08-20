import pytest
from unittest.mock import MagicMock, patch, mock_open
from swarmcg.simulations.runner import SimulationStep, SimulationManager
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
        "ntomp": 2,
        "gpu_id": "",
        "gmx_args": (),
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


def test_custom_gromacs_arguments_replace_thread_and_gpu_flags(mock_sim_setup):
    mock_sim_setup.update(
        {
            "nb_threads": 8,
            "ntomp": 4,
            "gpu_id": "0",
            "gmx_args": ("-pin", "on", "-nt", "3"),
            "mpi_tasks": 2,
        }
    )
    command = SimulationStep(mock_sim_setup)._run_cmd()

    assert command == [
        "mpirun", "-np", "2", "gmx", "mdrun", "-deffnm", "output",
        "-pin", "on", "-nt", "3",
    ]
    assert "-ntomp" not in command
    assert "-gpu_id" not in command


def test_gromacs_command_preserves_paths_with_spaces(mock_sim_setup):
    mock_sim_setup["gro"] = "/tmp/a path/input.gro"
    mock_sim_setup["mdp"] = "/tmp/a path/run.mdp"
    command = SimulationStep(mock_sim_setup)._prepare_cmd()

    assert "/tmp/a path/input.gro" in command
    assert "/tmp/a path/run.mdp" in command

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

@patch("swarmcg.simulations.runner.SimulationStep")
@patch("os.path.isfile", return_value=True)
@patch("builtins.open", new_callable=mock_open, read_data="nsteps=1000\nnstlog=100\ndt=0.002")
def test_simulation_manager_run(mock_open, mock_isfile, mock_sim_step_cls):
    config = SwarmConfig()
    manager = SimulationManager(config)
    
    # Mock step instance
    mock_step_instance = MagicMock()
    mock_step_instance.output_gro = "next_step.gro"
    mock_sim_step_cls.return_value = mock_step_instance
    
    # Run
    manager.run_simulation("/tmp/test_dir")
    
    # Verify SimulationStep creation (3 times: mini, equi, prod)
    assert mock_sim_step_cls.call_count == 3
    
    # Verify we ran the step in the directory
    mock_step_instance.run.assert_called_with("/tmp/test_dir")
