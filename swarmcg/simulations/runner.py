import os
import shlex
import signal
import subprocess
import tempfile
import time
from collections.abc import Sequence
import swarmcg.shared.exceptions as exceptions
import swarmcg.config as config
from swarmcg.utils import print_stdout_forced
from swarmcg.simulations.simulation_steps import select_class
from swarmcg.config_types import SwarmConfig


def exec_gmx(gmx_cmd, *, stdin_text=None, cwd=None):
    """Execute a GROMACS command without a shell and return its exit code.

    Args:
        gmx_cmd: Argument sequence. A legacy string is accepted and parsed with
            :func:`shlex.split`, but shell operators are never interpreted.
        stdin_text: Optional text supplied to the command's standard input.
        cwd: Optional working directory.

    Returns:
        Process exit code.
    """
    cmd = shlex.split(gmx_cmd) if isinstance(gmx_cmd, str) else list(gmx_cmd)
    completed = subprocess.run(
        cmd,
        input=stdin_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        check=False,
    )
    if completed.returncode != 0:
        print_stdout_forced(
            "NON-ZERO EXIT CODE FOR COMMAND:",
            shlex.join(cmd),
            "\n\nCOMMAND OUTPUT:\n\n",
            completed.stdout + completed.stderr,
            "\n\n",
        )
    return completed.returncode


class SimulationStep:
    """Prepare and run one GROMACS simulation stage.

    Args:
        sim_setup: Runner settings containing executable, input basenames,
            resource options, monitoring interval, and stage configuration.
    """
    REQUIRED_FIELDS = ["exec", "gro", "mdp", "top", "md_output"]

    def __init__(self, sim_setup):
        self.sim_setup = sim_setup
        self.step_name = sim_setup.get("step_name")
        self._validate_init()

    def _validate_init(self):
        missing_args = ", ".join([i for i in SimulationStep.REQUIRED_FIELDS if i not in self.sim_setup.keys()])
        if missing_args:
            msg = (
                f"The following arguments are missing: {missing_args}. Please check your input."
            )
            raise exceptions.InputArgumentError(msg)

    @staticmethod
    def _validate_exec(exec):
        """Validate that a GROMACS executable can be launched.

        Args:
            exec: Executable name or path.

        Raises:
            ExecutableNotFound: If the executable cannot be launched.
        """
        with open(os.devnull, 'w') as devnull:
            try:
                subprocess.call([exec, "--version"], stdout=devnull, stderr=devnull)
            except OSError:
                msg = (
                    f"Cannot find GROMACS using alias {exec}, please provide "
                    f"the right GROMACS alias or path"
                )
                raise exceptions.ExecutableNotFound(msg)

    @property
    def swarmcg_flag(self):
        return self.sim_setup.get("swarmcg_flag")

    @property
    def output_gro(self):
        return f"{self.sim_setup.get('md_output')}.gro"

    def _prepare_cmd(self, **kwargs):
        setup = {**self.sim_setup, **kwargs}
        return [
            setup["exec"],
            "grompp",
            "-c",
            setup["gro"],
            "-f",
            setup["mdp"],
            "-p",
            setup["top"],
            "-o",
            setup["md_output"],
            "-maxwarn",
            str(setup["maxwarn"]),
        ]

    def _run_cmd(self, aux_command="", mpi=True):
        cmd = [self.sim_setup["exec"], "mdrun", "-deffnm", self.sim_setup["md_output"]]
        if aux_command:
            cmd.extend(shlex.split(aux_command) if isinstance(aux_command, str) else aux_command)

        custom_args = self.sim_setup.get("gmx_args", ())
        if custom_args:
            cmd.extend(custom_args)
        else:
            threads = int(self.sim_setup.get("nb_threads"))
            if threads > 0:
                cmd.extend(["-nt", str(threads)])

            omp_threads = int(self.sim_setup.get("ntomp", 0))
            if omp_threads > 0:
                cmd.extend(["-ntomp", str(omp_threads)])

            gpu = self.sim_setup.get("gpu_id")
            if gpu:
                cmd.extend(["-gpu_id", str(gpu)])

        mpi_tasks = int(self.sim_setup.get("mpi_tasks"))
        if mpi and mpi_tasks > 0:
            cmd = ["mpirun", "-np", str(mpi_tasks), *cmd]
        return cmd

    def _run_setup(self, exec_path):
        sim_time = self.sim_setup.get("sim_duration")
        nb_frames = self.sim_setup.get("prod_nb_frames")
        # Ensure we write to exec_path even if basenames are used
        self.sim_setup.get("simulation_config").modify_mdp(sim_time, nb_frames).to_file(exec_path)
        return self

    def _run_prep(self, cmd, cwd=None):
        with subprocess.Popen(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, cwd=cwd) as gmx_process:
            out, err = gmx_process.communicate()
            gmx_out = f"STDOUT:\n{out.decode()}\nSTDERR:\n{err.decode()}"
            gmx_process.kill()

        if gmx_process.returncode == 0:
            return self
        else:
            print_stdout_forced(
                'NON-ZERO EXIT CODE FOR COMMAND:', shlex.join(cmd), '\n\nCOMMAND OUTPUT:\n\n', gmx_out, '\n\n'
            )
            msg = (
                f"Gromacs grompp failed at MD {self.step_name} step.\n"
                f"COMMAND: {shlex.join(cmd)}\n"
                f"OUTPUT:\n{gmx_out}\n"
                f"You may also want to check the parameters of the MDP file provided through "
                f"argument -{self.swarmcg_flag}. If you think this is a bug, please consider opening "
                f"an issue on GitHub at {config.github_url}/issues."
            )
            raise exceptions.ComputationError(msg)

    def _run_md(self, cmd, cwd=None):
        cycles_check, last_log_file_size = 0, 0
        _run_killed = False
        monitor_file = self.sim_setup.get("monitor_file")
        if cwd:
            monitor_file = os.path.join(cwd, monitor_file)

        keep_alive_n_cycles = self.sim_setup.get("keep_alive_n_cycles")
        seconds_between_checks = self.sim_setup.get("seconds_between_checks")
        with tempfile.TemporaryFile() as process_output, subprocess.Popen(
            cmd,
            stdout=process_output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=cwd,
        ) as gmx_process:
            while gmx_process.poll() is None:  # while process is alive
                time.sleep(seconds_between_checks)
                cycles_check += 1

                if cycles_check % keep_alive_n_cycles == 0:
                    
                    if os.path.isfile(monitor_file):
                        log_file_size = os.path.getsize(monitor_file)
                    else:
                        log_file_size = last_log_file_size

                    if log_file_size == last_log_file_size:
                        os.killpg(
                            os.getpgid(gmx_process.pid),
                            signal.SIGKILL
                        )  # kill all processes of process group
                        _run_killed = True
                    else:
                        last_log_file_size = log_file_size

            gmx_process.kill()
            
            gmx_process.wait()
            process_output.seek(0)
            output = process_output.read().decode(errors="replace")
            
        if _run_killed:
            msg = (
                f"MD {self.step_name} run failed (unstable simulation was killed, with unstable "
                f"= NOT writing in log file for {keep_alive_n_cycles * seconds_between_checks} sec)"
            )
            print_stdout_forced(msg)
            raise exceptions.ComputationError(msg)
            
        if gmx_process.returncode != 0:
            gmx_out = output
            print_stdout_forced(
                'NON-ZERO EXIT CODE FOR COMMAND:', shlex.join(cmd), '\n\nCOMMAND OUTPUT:\n\n', gmx_out, '\n\n'
            )
            msg = (
                f"Gromacs mdrun failed at MD {self.step_name} step.\n"
                f"COMMAND: {shlex.join(cmd)}\n"
                f"OUTPUT:\n{gmx_out}\n"
                f"If you think this is a bug, please consider opening an issue on GitHub."
            )
            raise exceptions.ComputationError(msg)

        return gmx_process.returncode

    def run(self, exec_path, aux_command=""):
        """Write the MDP, preprocess it, and execute the simulation stage.

        Args:
            exec_path: Working directory for inputs and outputs.
            aux_command: Additional argument string or sequence appended to
                ``mdrun`` before configured resource arguments.

        Returns:
            GROMACS ``mdrun`` exit code (zero on success).

        Raises:
            ComputationError: If preprocessing, execution, or stall monitoring
                reports a failure.
        """
        prep_cmd = self._prepare_cmd()
        md_cmd = self._run_cmd(aux_command)
        return self._run_setup(exec_path)._run_prep(prep_cmd, cwd=exec_path)._run_md(md_cmd, cwd=exec_path)


def config_to_runner(
    config: SwarmConfig,
    sim_config,
    prev_gro,
    sim_time=None,
    nb_frames=None,
    gmx_args: Sequence[str] = (),
):
    """
    Convert SwarmConfig and SimulationConfig into a runner-compatible dictionary.

    Args:
        config (SwarmConfig): Global configuration.
        sim_config (BaseSimulationConfig): Specific step configuration (Mini/Equi/Prod).
        prev_gro (str): Input structure file for this step.
        sim_time (float, optional): Overwrite simulation duration.
        nb_frames (int, optional): Overwrite number of frames.
        gmx_args: Pre-parsed custom ``mdrun`` arguments. When nonempty these
            replace the configured thread and GPU flags.

    Returns:
        dict: Setup dictionary for SimulationStep.
    """
    if sim_time is not None and sim_time <= 0:
        raise ValueError("simulation duration must be greater than zero")
    if nb_frames is not None and nb_frames <= 0:
        raise ValueError("production frame count must be greater than zero")

    simulation_setup = {
        "exec": config.gromacs.gmx_path,
        "gro": prev_gro,
        "mdp": sim_config.base_name,
        "top": os.path.basename(config.cg_model.top_input_filename),

        "gpu_id": config.gromacs.gpu_id,
        "mpi_tasks": config.gromacs.mpi_tasks,
        "nb_threads": config.gromacs.nb_threads,
        "ntomp": config.gromacs.ntomp,
        "maxwarn": config.gromacs.mini_maxwarn,
        "gmx_args": tuple(gmx_args),

        "swarmcg_flag": sim_config.swarmcg_flag,
        "step_name": sim_config.step_name,
        "md_output": sim_config.md_output,

        "monitor_file": f"{sim_config.md_output}.log",
        "keep_alive_n_cycles": int(config.gromacs.sim_kill_delay / 10),
        "seconds_between_checks": 10,
        "simulation_config": sim_config,
    }
    
    if sim_time is not None:
        simulation_setup["sim_duration"] = sim_time
    if nb_frames is not None:
        simulation_setup["prod_nb_frames"] = nb_frames

    return simulation_setup


class SimulationManager:
    """Manage the minimization, equilibration, and production lifecycle.

    Args:
        config: Validated application configuration. Free-form GROMACS
            arguments are parsed exactly once during initialization.
    """
    
    def __init__(self, config: SwarmConfig):
        self.config = config
        self.gmx_args = tuple(shlex.split(config.gromacs.gmx_args_str))

    def run_simulation(self, working_dir, sim_time=None, nb_frames=None):
        """Run the complete simulation chain in a working directory.

        Args:
            working_dir: Directory containing staged GROMACS inputs.
            sim_time: Optional positive production duration in nanoseconds.
            nb_frames: Optional positive production frame count.

        Returns:
            ``True`` after all three stages complete.

        Raises:
            ValueError: If runtime duration or frame count is not positive.
            ComputationError: If any GROMACS stage fails or stalls.
        """
         
        # We no longer change global CWD
        
        # Initial GRO file (assumed to be in working_dir as basename)
        prev_gro = os.path.basename(self.config.cg_model.gro_input_filename)
        
        steps = ["minimization", "equilibration", "production"]
        
        for step_type in steps:
            sim_config = select_class(step_type, self.config.simulation, base_dir=working_dir)
            simulation_setup = config_to_runner(
                self.config,
                sim_config,
                prev_gro,
                sim_time,
                nb_frames,
                self.gmx_args,
            )
            
            step = SimulationStep(simulation_setup)
            step.run(working_dir) # executing in working_dir via cwd argument
            
            # Update prev_gro for next step
            prev_gro = step.output_gro
            
        return True # Success if we got here (exceptions would raise)
