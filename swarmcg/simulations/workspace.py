
import os
import shutil
import time
from typing import List, Optional
from swarmcg.config_types import SwarmConfig
import swarmcg.config as config
from swarmcg.shared import exceptions

class WorkspaceManager:
    """
    Manages the filesystem workspace for SwarmCG optimizations/evaluations.
    Handles directory creation, file copying, and cleanup.
    """
    def __init__(self, config_obj: SwarmConfig):
        self.config = config_obj
        self.exec_folder = ""
        
    def setup_execution_folder(self, output_folder: str = "") -> str:
        """
        Creates the main execution directory and internal structure.
        Returns the absolute path of the execution folder.
        """
        if output_folder:
            self.exec_folder = output_folder
        else:
            self.exec_folder = time.strftime("MODEL_OPTI__STARTED_%d-%m-%Y_%Hh%Mm%Ss")
            
        if os.path.exists(self.exec_folder):
             msg = (
                "Provided output folder already exists, please delete existing folder "
                "manually or provide another folder name."
            )
             raise exceptions.AvoidOverwritingFolder(msg)
             
        os.mkdir(self.exec_folder)
        os.mkdir(os.path.join(self.exec_folder, ".internal"))
        os.mkdir(os.path.join(self.exec_folder, config.distrib_plots_all_evals_dirname))
        os.mkdir(os.path.join(self.exec_folder, config.log_files_all_evals_dirname))
        
        if self.config.optimization.keep_all_sims:
             os.mkdir(os.path.join(self.exec_folder, config.sim_files_all_evals_dirname))
             
        return self.exec_folder

    def prepare_simulation_input(self, top_includes: List[str]):
        """
        Prepares the input simulation directory by copying necessary files
        and adjusting the topology includes.
        """
        input_sim_dir = os.path.join(self.exec_folder, config.input_sim_files_dirname)
        os.mkdir(input_sim_dir)
        
        # Copy includes
        for include in top_includes:
            shutil.copy(include, input_sim_dir)
            
        # Copy simulation files (MDPs, GRO, TOP, ITP)
        # Note: config objects store filenames, we assume they are valid paths (validated earlier)
        # We need to copy files from their source location.
        # The paths in config might be relative or absolute.
        # We assume the caller has ensured these paths are accessible.
        
        files_to_copy = [
            self.config.cg_model.cg_itp_filename,
            self.config.cg_model.gro_input_filename,
            self.config.cg_model.top_input_filename,
            self.config.simulation.mdp_minimization_filename,
            self.config.simulation.mdp_equi_filename,
            self.config.simulation.mdp_md_filename
        ]
        
        for f in files_to_copy:
            if f and os.path.isfile(f):
                shutil.copy(f, input_sim_dir)
                
        # Modify TOP file includes
        top_basename = os.path.basename(self.config.cg_model.top_input_filename)
        top_path = os.path.join(input_sim_dir, top_basename)
        
        with open(top_path, "r") as fp:
            lines = fp.readlines()
            
        with open(top_path, "w") as fp:
            nb_includes = 0
            for line in lines:
                if line.strip().startswith("#include"):
                     # We assume top_includes list matches the order of includes in the file
                     # This logic mimics the original optimize_model.py logic
                     if nb_includes < len(top_includes):
                         basename = os.path.basename(top_includes[nb_includes])
                         fp.write(f'#include "{basename}"\n')
                         nb_includes += 1
                     else:
                         # Fallback if mismatch (should capture logic error if any)
                         fp.write(line)
                else:
                    fp.write(line)

    def verify_topology_includes(self) -> List[str]:
        """
        Parses the TOP file to verify ITP inclusion and find all included files.
        Returns a list of included file paths.
        """
        top_filename = self.config.cg_model.top_input_filename
        cg_itp_basename = os.path.basename(self.config.cg_model.cg_itp_filename)
        
        top_includes_filenames = []
        with open(top_filename, "r") as fp:
            all_top_lines = fp.read()
            if cg_itp_basename not in all_top_lines:
                msg = "The CG ITP model filename you provided is not included in your TOP file."
                raise exceptions.MDSimulationInputError(msg)

            top_lines = all_top_lines.split("\n")
            top_lines = [top_line.strip().split(";")[0] for top_line in top_lines]
            
            base_dir = os.path.dirname(top_filename) or "."
            
            for top_line in top_lines:
                if top_line.startswith("#include"):
                    # Extract filename: #include "file.itp" or #include 'file.itp'
                    parts = top_line.split(maxsplit=1)
                    if len(parts) > 1:
                        include_name = parts[1].strip("'\"")
                        top_includes_filenames.append(os.path.join(base_dir, include_name))
                        
        return top_includes_filenames

    def cleanup_input_staging(self):
        """Removes the input staging directory."""
        input_sim_dir = os.path.join(self.exec_folder, config.input_sim_files_dirname)
        if os.path.exists(input_sim_dir):
            shutil.rmtree(input_sim_dir)

