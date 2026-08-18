
import math
import os
import shutil
import subprocess
import pytest
import tempfile
from pathlib import Path

import numpy as np
from MDAnalysis.lib.distances import calc_angles, calc_dihedrals

from swarmcg.config_types import SwarmConfig, SimulationConfig, CGModelConfig, GromacsConfig
from swarmcg.simulations.runner import SimulationManager, SimulationStep, config_to_runner, select_class
from swarmcg.simulations.potentials import (
    gmx_angles_func_10,
    gmx_dihedrals_func_3,
    gmx_dihedrals_func_11,
)

# Check for GROMACS
gmx_path = shutil.which("gmx")
GMX_AVAILABLE = gmx_path is not None


def _single_point_potential(workdir, interaction, coordinates):
    """Evaluate a minimal bonded topology with the installed GROMACS.

    Args:
        workdir: Empty directory used for the GROMACS files.
        interaction: Complete ``[ angles ]`` or ``[ dihedrals ]`` block.
        coordinates: Cartesian coordinates in nanometers.

    Returns:
        GROMACS potential energy in kJ/mol.
    """
    workdir.mkdir()
    atoms = "\n".join(
        f"{index} X 1 MOL X{index} {index} 0.0 1.0"
        for index in range(1, len(coordinates) + 1)
    )
    topology = f"""[ defaults ]
1 1 no 1.0 1.0

[ atomtypes ]
X 1.0 0.0 A 0.0 0.0

[ moleculetype ]
MOL 1

[ atoms ]
{atoms}

{interaction}

[ system ]
bonded form audit

[ molecules ]
MOL 1
"""
    gro_lines = ["bonded form audit", str(len(coordinates))]
    for index, (x, y, z) in enumerate(coordinates, 1):
        gro_lines.append(
            f"{1:5d}{'MOL':<5}{'X':>5}{index:5d}{x:8.3f}{y:8.3f}{z:8.3f}"
        )
    gro_lines.append("   4.00000   4.00000   4.00000")
    mdp = """integrator = md
nsteps = 0
dt = 0.001
continuation = yes
cutoff-scheme = Verlet
nstlist = 10
rlist = 1.0
rvdw = 1.0
rcoulomb = 1.0
coulombtype = Cut-off
vdwtype = Cut-off
pbc = xyz
gen-vel = no
constraints = none
"""
    (workdir / "topol.top").write_text(topology)
    (workdir / "conf.gro").write_text("\n".join(gro_lines) + "\n")
    (workdir / "run.mdp").write_text(mdp)

    subprocess.run(
        [
            gmx_path,
            "grompp",
            "-f",
            "run.mdp",
            "-p",
            "topol.top",
            "-c",
            "conf.gro",
            "-o",
            "topol.tpr",
            "-maxwarn",
            "2",
        ],
        cwd=workdir,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            gmx_path,
            "mdrun",
            "-s",
            "topol.tpr",
            "-rerun",
            "conf.gro",
            "-deffnm",
            "rerun",
        ],
        cwd=workdir,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [gmx_path, "energy", "-f", "rerun.edr", "-o", "potential.xvg"],
        cwd=workdir,
        input="Potential\n0\n",
        check=True,
        capture_output=True,
        text=True,
    )
    values = [
        float(line.split()[1])
        for line in (workdir / "potential.xvg").read_text().splitlines()
        if line and line[0] not in "#@"
    ]
    return values[-1]

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
    """Run the complete manager chain on the stable added-functions fixture."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        data_dir = Path("tests/data").absolute()
        for filename in (
            "added_bonded_forms.gro",
            "added_bonded_forms.top",
            "added_bonded_forms.itp",
            "added_bonded_forms_mini.mdp",
        ):
            shutil.copy(data_dir / filename, tmp_path / filename)
        shutil.copy(
            data_dir / "added_bonded_forms_md.mdp", tmp_path / "equi.mdp"
        )
        shutil.copy(
            data_dir / "added_bonded_forms_md.mdp", tmp_path / "md.mdp"
        )

        config = SwarmConfig()
        config.gromacs.gmx_path = gmx_path
        config.gromacs.nb_threads = 1
        config.cg_model.top_input_filename = "added_bonded_forms.top"
        config.cg_model.gro_input_filename = "added_bonded_forms.gro"
        config.simulation.mdp_minimization_filename = str(
            tmp_path / "added_bonded_forms_mini.mdp"
        )
        config.simulation.mdp_equi_filename = str(tmp_path / "equi.mdp")
        config.simulation.mdp_md_filename = str(tmp_path / "md.mdp")

        manager = SimulationManager(config)

        # 0.0001 ns at 1 fs gives a deterministic 100-step production run.
        manager.run_simulation(str(tmp_path), sim_time=0.0001, nb_frames=10)

        for stage in ("mini", "equi", "md"):
            for suffix in ("tpr", "gro", "log", "edr"):
                assert (tmp_path / f"{stage}.{suffix}").exists(), (
                    f"{stage} {suffix.upper()} missing"
                )

            import MDAnalysis as mda

            universe = mda.Universe(str(tmp_path / f"{stage}.gro"))
            assert np.all(np.isfinite(universe.atoms.positions))

        assert (tmp_path / "md.xtc").exists(), "production trajectory missing"


@pytest.mark.skipif(not GMX_AVAILABLE, reason="GROMACS not found in PATH")
def test_gromacs_nonzero_restricted_bending_rb_and_cbt():
    """Run finite-temperature dynamics with all newly supported terms."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        data_dir = Path("tests/data").absolute()
        for filename in (
            "added_bonded_forms.gro",
            "added_bonded_forms.top",
            "added_bonded_forms.itp",
            "added_bonded_forms_md.mdp",
        ):
            shutil.copy(data_dir / filename, tmp_path / filename)
        subprocess.run(
            [
                gmx_path,
                "grompp",
                "-f",
                "added_bonded_forms_md.mdp",
                "-p",
                "added_bonded_forms.top",
                "-c",
                "added_bonded_forms.gro",
                "-o",
                "added_bonded_forms.tpr",
                "-maxwarn",
                "1",
            ],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                gmx_path,
                "mdrun",
                "-s",
                "added_bonded_forms.tpr",
                "-deffnm",
                "added_bonded_forms",
                "-nt",
                "1",
            ],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )

        for suffix in ("tpr", "gro", "log", "edr", "xtc"):
            assert (tmp_path / f"added_bonded_forms.{suffix}").exists()

        import MDAnalysis as mda

        universe = mda.Universe(
            str(tmp_path / "added_bonded_forms.tpr"),
            str(tmp_path / "added_bonded_forms.xtc"),
        )
        assert len(universe.trajectory) > 0
        for timestep in universe.trajectory:
            assert np.all(np.isfinite(timestep.positions))


@pytest.mark.skipif(not GMX_AVAILABLE, reason="GROMACS not found in PATH")
def test_added_analytical_forms_match_gromacs_single_point_energies(tmp_path):
    """Compare functions 10, 3, and 11 with real GROMACS energies."""
    theta = math.radians(100.0)
    reb_coordinates = np.round(
        np.array(
            [
                [2.0 + 0.2 * math.cos(theta), 2.0 + 0.2 * math.sin(theta), 2.0],
                [2.0, 2.0, 2.0],
                [2.2, 2.0, 2.0],
            ]
        ),
        3,
    )
    actual_theta = float(
        calc_angles(
            reb_coordinates[0:1],
            reb_coordinates[1:2],
            reb_coordinates[2:3],
        )[0]
    )
    reb_gromacs = _single_point_potential(
        tmp_path / "reb",
        "[ angles ]\n1 2 3 10 120.0 25.0",
        reb_coordinates,
    )
    reb_python = float(
        gmx_angles_func_10(actual_theta, 25.0, math.radians(120.0), 0.0)
    )

    theta_previous = math.radians(105.0)
    theta_current = math.radians(125.0)
    rotation = math.radians(70.0)
    r2 = np.array([2.0, 2.0, 2.0])
    r3 = np.array([2.2, 2.0, 2.0])
    r1 = r2 + 0.2 * np.array(
        [math.cos(theta_previous), math.sin(theta_previous), 0.0]
    )
    r4 = r3 + 0.2 * np.array(
        [
            -math.cos(theta_current),
            math.sin(theta_current) * math.cos(rotation),
            math.sin(theta_current) * math.sin(rotation),
        ]
    )
    torsion_coordinates = np.round(np.array([r1, r2, r3, r4]), 3)
    actual_previous = float(
        calc_angles(
            torsion_coordinates[0:1],
            torsion_coordinates[1:2],
            torsion_coordinates[2:3],
        )[0]
    )
    actual_current = float(
        calc_angles(
            torsion_coordinates[1:2],
            torsion_coordinates[2:3],
            torsion_coordinates[3:4],
        )[0]
    )
    actual_phi = float(
        calc_dihedrals(
            torsion_coordinates[0:1],
            torsion_coordinates[1:2],
            torsion_coordinates[2:3],
            torsion_coordinates[3:4],
        )[0]
    )

    rb_parameters = (1.25, -2.0, 0.75, 1.1, -0.4, 0.2)
    rb_gromacs = _single_point_potential(
        tmp_path / "rb",
        "[ dihedrals ]\n1 2 3 4 3 " + " ".join(map(str, rb_parameters)),
        torsion_coordinates,
    )
    rb_python = float(gmx_dihedrals_func_3(actual_phi, *rb_parameters))

    cbt_parameters = (4.0, 0.3, -0.5, 0.2, 0.1, -0.05)
    cbt_gromacs = _single_point_potential(
        tmp_path / "cbt",
        "[ dihedrals ]\n1 2 3 4 11 " + " ".join(map(str, cbt_parameters)),
        torsion_coordinates,
    )
    cbt_python = float(
        gmx_dihedrals_func_11(
            actual_previous,
            actual_current,
            actual_phi,
            *cbt_parameters,
        )
    )

    assert np.isclose(reb_gromacs, reb_python, atol=1e-5)
    assert np.isclose(rb_gromacs, rb_python, atol=1e-5)
    assert np.isclose(cbt_gromacs, cbt_python, atol=1e-5)
