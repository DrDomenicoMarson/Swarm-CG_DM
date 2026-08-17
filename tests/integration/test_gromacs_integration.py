
import os
import shutil
import pytest
import tempfile
from pathlib import Path
from swarmcg.config_types import SwarmConfig, SimulationConfig, CGModelConfig, GromacsConfig
from swarmcg.simulations.runner import SimulationManager, SimulationStep, config_to_runner, select_class

# Check for GROMACS
gmx_path = shutil.which("gmx")
GMX_AVAILABLE = gmx_path is not None

@pytest.mark.skipif(not GMX_AVAILABLE, reason="GROMACS not found in PATH")
def test_gromacs_minimization_run():
    """
    Integration test that runs a real GROMACS grompp (and optionally mdrun) 
    command using the project's data.
    """
    # 1. Setup temporary workspace
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # 2. Copy Test Data
        data_dir = Path("tests/data").absolute()
        assert data_dir.exists()
        
        files_to_copy = [
            "start_conf.gro",
            "system.top",
            "cg_model.itp",
            "martini_v2.0_PEO_PS_CNP.itp",
            "martini_v2.0_ions.itp"
        ]
        
        for f in files_to_copy:
            shutil.copy(data_dir / f, tmp_path / f)

        # Copy MDP from swarmcg/data
        mdp_dir = Path("swarmcg/data").absolute()
        shutil.copy(mdp_dir / "mini.mdp", tmp_path / "mini.mdp")
            
        # 3. Configure SwarmConfig
        config = SwarmConfig()
        config.gromacs.gmx_path = gmx_path
        config.gromacs.nb_threads = 1 # Force 1 thread for stability in tests
        
        # IMPORTANT: Use absolute paths for MDP files because SimulationStep validation 
        # checks existence relative to CWD before we might have chdir'd.
        config.cg_model.top_input_filename = "system.top"
        config.cg_model.gro_input_filename = "start_conf.gro"
        config.simulation.mdp_minimization_filename = str(tmp_path / "mini.mdp")
        
        # 4. Prepare SimulationStep for Minimization
        # We need to Select the class (Mini)
        sim_config = select_class("minimization", config.simulation)
        
        # Setup runner dict
        # config_to_runner uses filenames from config passed to it or sim_config?
        # It uses sim_config.base_name for MDP input to grompp.
        # But SimulationStep writes the MDP to exec_path.
        
        setup = config_to_runner(
            config, 
            sim_config, 
            prev_gro="start_conf.gro"
        )
        
        # 5. Run full Minimization Step (Setup -> Prep -> MD)
        # This verifies:
        # 1. MDP modification/writing (_run_setup)
        # 2. Grompp execution (_run_prep)
        # 3. Mdrun execution (_run_md)
        
        step = SimulationStep(setup)
        
        step.run(tmp_path)
            
            # Assert Output Exists
        gro_file = tmp_path / (step.sim_setup['md_output'] + ".gro")
        assert gro_file.exists(), "Minimization output GRO file missing"
            
        tpr_file = tmp_path / (step.sim_setup['md_output'] + ".tpr")
        assert tpr_file.exists(), "TPR file missing"
            
        log_file = tmp_path / step.sim_setup['monitor_file']
        assert log_file.exists(), "Log file missing"

            # 6. Verify Data Extraction
            # We use MDAnalysis to check we can read the generated trajectory
        import MDAnalysis as mda

        u_gro = mda.Universe(str(gro_file))
        assert len(u_gro.atoms) > 0, "GRO file is empty"
        coords = u_gro.atoms.positions
        assert coords.shape == (len(u_gro.atoms), 3)

@pytest.mark.skipif(not GMX_AVAILABLE, reason="GROMACS not found in PATH")
def test_gromacs_full_chain():
    """
    Integration test validating the full SimulationManager workflow:
    Minimization -> Equilibration -> Production
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        data_dir = Path("tests/data").absolute()
        mdp_dir = Path("swarmcg/data").absolute()
        
        # 1. Copy Data
        files_to_copy = [
            "start_conf.gro",
            "system.top",
            "cg_model.itp",
            "martini_v2.0_PEO_PS_CNP.itp",
            "martini_v2.0_ions.itp"
        ]
        for f in files_to_copy:
            shutil.copy(data_dir / f, tmp_path / f)
            
        # Copy ALL MDPs
        for mdp in ["mini.mdp", "equi.mdp", "md.mdp"]:
            shutil.copy(mdp_dir / mdp, tmp_path / mdp)
            
        # PATCH MDPs for Test Speed/Stability
        def patch_mdp(filename, settings):
            with open(filename, 'r') as f:
                lines = f.readlines()
            
            # Remove existing lines for keys we want to set
            keys = settings.keys()
            lines = [l for l in lines if not any(l.strip().startswith(k) for k in keys)]
            
            # Append new settings
            for k, v in settings.items():
                lines.append(f"{k} = {v}\n")
                
            with open(filename, 'w') as f:
                f.writelines(lines)

        # Use 'steep' integrator for all steps to avoid dynamics instability on test data
        # This verifies the software workflow (chaining steps) without needing perfectly equilibrated input
        patch_mdp(tmp_path / "equi.mdp", {
            "nstxout-compressed": 10,
            "integrator": "steep", 
            "nsteps": 50
        })
        
        patch_mdp(tmp_path / "md.mdp", {
            "nstlog": 10,
            "integrator": "steep",
            "nsteps": 50
        })
            
        # 2. Configure
        config = SwarmConfig()
        config.gromacs.gmx_path = gmx_path
        config.gromacs.nb_threads = 1 # Force 1 thread
        
        # Absolute paths for initial input
        config.cg_model.top_input_filename = "system.top"
        config.cg_model.gro_input_filename = "start_conf.gro"
        
        # Absolute paths for MDPs
        config.simulation.mdp_minimization_filename = str(tmp_path / "mini.mdp")
        config.simulation.mdp_equi_filename = str(tmp_path / "equi.mdp")
        config.simulation.mdp_md_filename = str(tmp_path / "md.mdp")
        
        # 3. Initialize Manager
        manager = SimulationManager(config)
        
        # 4. Run Full Simulation Chain
        manager.run_simulation(str(tmp_path), sim_time=0.01, nb_frames=1)

        for stage in ("mini", "equi", "md"):
            assert (tmp_path / f"{stage}.tpr").exists(), f"{stage} TPR missing"
            assert (tmp_path / f"{stage}.gro").exists(), f"{stage} GRO missing"
            assert (tmp_path / f"{stage}.log").exists(), f"{stage} log missing"


@pytest.mark.skipif(not GMX_AVAILABLE, reason="GROMACS not found in PATH")
def test_gromacs_nonzero_restricted_bending_rb_and_cbt():
    """Require GROMACS to preprocess and minimize all newly supported terms."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        data_dir = Path("tests/data").absolute()
        for filename in (
            "start_conf.gro",
            "system.top",
            "cg_model.itp",
            "martini_v2.0_PEO_PS_CNP.itp",
            "martini_v2.0_ions.itp",
        ):
            shutil.copy(data_dir / filename, tmp_path / filename)
        shutil.copy(Path("swarmcg/data/mini.mdp").absolute(), tmp_path / "mini.mdp")

        itp_path = tmp_path / "cg_model.itp"
        topology = itp_path.read_text()
        topology = topology.replace("10       120         0", "10       120        25", 1)
        topology += """

[ dihedrals ]
; i  j  k  l  funct  parameters
5  4  3  1  3   -1.50  1.00  -0.50  0.25  -0.10  0.05
6  5  4  3  11   2.00  0.50  -0.40  0.30  -0.20  0.10
"""
        itp_path.write_text(topology)

        config = SwarmConfig()
        config.gromacs.gmx_path = gmx_path
        config.gromacs.nb_threads = 1
        config.cg_model.top_input_filename = "system.top"
        config.cg_model.gro_input_filename = "start_conf.gro"
        config.simulation.mdp_minimization_filename = str(tmp_path / "mini.mdp")
        sim_config = select_class("minimization", config.simulation)
        step = SimulationStep(config_to_runner(config, sim_config, "start_conf.gro"))
        step.run(tmp_path)

        assert (tmp_path / "mini.tpr").exists()
        assert (tmp_path / "mini.gro").exists()
        assert (tmp_path / "mini.log").exists()
