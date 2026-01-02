
import os
import shutil
import pytest
import tempfile
from pathlib import Path
from swarmcg.config_types import SwarmConfig, SimulationConfig, CGModelConfig, GromacsConfig
from swarmcg.simulations.runner import SimulationManager, SimulationStep, select_class

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
        
        from swarmcg.simulations.runner import config_to_runner
        
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
        
        # Change to tmp dir context
        cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            step.run(tmp_path)
            
            # Assert Output Exists
            gro_file = Path(step.sim_setup['md_output'] + ".gro")
            assert gro_file.exists(), "Minimization output GRO file missing"
            
            tpr_file = Path(step.sim_setup['md_output'] + ".tpr")
            assert tpr_file.exists(), "TPR file missing"
            
            log_file = Path(step.sim_setup['monitor_file'])
            assert log_file.exists(), "Log file missing"

            # 6. Verify Data Extraction
            # We use MDAnalysis to check we can read the generated trajectory
            import MDAnalysis as mda
            
            # Load the universe
            u = mda.Universe(str(gro_file), str(step.sim_setup['md_output'] + ".trr")) 
            # Note: minimisation usually outputs trr/gro, checks mdp nstxout
            # mini.mdp has nstxout=0, so maybe no trr? 
            # Default md output is usually .trr or .xtc
            # SimulationStep sets '-o {md_output}', so it produces default formatted files. 
            # In GROMACS, -o usually produces .trr by default unless specified.
            # But let's check what was produced.
            
            # Actually let's just check the GRO first for data
            u_gro = mda.Universe(str(gro_file))
            assert len(u_gro.atoms) > 0, "GRO file is empty"
            
            # Check we can extract coordinates
            coords = u_gro.atoms.positions
            assert coords.shape == (len(u_gro.atoms), 3)
            
        finally:
            os.chdir(cwd)
