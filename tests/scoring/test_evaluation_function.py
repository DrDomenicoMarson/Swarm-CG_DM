import os
import tempfile
from unittest.mock import patch

from swarmcg import config as config_module
from swarmcg.config_types import SwarmConfig
from swarmcg.context import OptimizationContext


def _make_min_itp():
    return {
        "moleculetype": {"molname": "MOL", "nrexcl": 1},
        "atoms": [
            {
                "bead_id": 0,
                "bead_type": "A",
                "resnr": 1,
                "residue": "RES",
                "atom": "A",
                "cgnr": 1,
                "charge": 0.0,
                "mass": 1.0,
                "vs_type": None,
            }
        ],
        "constraint": [],
        "bond": [],
        "angle": [],
        "dihedral": [],
        "virtual_sites2": {},
        "virtual_sites3": {},
        "virtual_sites4": {},
        "virtual_sitesn": {},
        "exclusion": [],
    }


def test_eval_function_handles_missing_md_and_restores_cwd():
    try:
        from swarmcg.scoring.evaluation_function import eval_function
    except ImportError as exc:
        import pytest
        pytest.skip(f"Skipping due to import error: {exc}")

    config = SwarmConfig()
    ctx = OptimizationContext(config=config)
    ctx.cg_itp = _make_min_itp()
    ctx.out_itp = _make_min_itp()
    ctx.opti_cycle = {
        "nb_cycle": 1,
        "geoms": ["bond"],
        "nb_geoms": {"constraint": 0, "bond": 0, "angle": 0, "dihedral": 0},
    }
    ctx.pso.opti_geoms_all = set()
    ctx.pso.best_fitness = (float("inf"), None)
    ctx.pso.worst_fit_score = 123.0
    ctx.status.nb_eval = 0
    ctx.status.start_opti_ts = 0.0
    ctx.status.total_eval_time = 0.0
    ctx.status.total_gmx_time = 0.0
    ctx.status.total_model_eval_time = 0.0
    # ctx.keep_all_sims = False # Config attribute? No, it was added as logic.
    # Where is keep_all_sims? It's in SwarmConfig.optimization?
    # Or runtime?
    # Let's check context.py def. It's not in OptimizationContext dataclass explicitly.
    # It might be in config.optimization.keep_all_sims.
    
    ctx.files.cg_itp_basename = "cg_model.itp"

    with tempfile.TemporaryDirectory() as tmpdir:
        ctx.files.exec_folder = tmpdir
        internal_dir = os.path.join(tmpdir, ".internal")
        os.makedirs(internal_dir)

        input_dir = os.path.join(tmpdir, config_module.input_sim_files_dirname)
        os.makedirs(input_dir)
        with open(os.path.join(input_dir, "dummy.txt"), "w") as fp:
            fp.write("dummy")

        os.makedirs(os.path.join(tmpdir, config_module.all_evals_files_dirname))

        original_cwd = os.getcwd()
        with patch("swarmcg.scoring.evaluation_function.sim.SimulationManager.run_simulation", return_value=None):
            score = eval_function([], ctx)

        assert score == ctx.pso.worst_fit_score
        assert os.getcwd() == original_cwd

        eval_dir = os.path.join(
            tmpdir,
            f"{config_module.iteration_sim_files_dirname}_eval_step_1",
        )
        assert not os.path.exists(eval_dir)
