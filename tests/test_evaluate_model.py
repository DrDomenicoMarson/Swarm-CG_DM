"""CLI-level evaluation routing tests without trajectory-processing cost."""

from pathlib import Path
from unittest.mock import patch

import pytest

from swarmcg.config_types import SwarmConfig
from swarmcg.evaluate_model import run


@pytest.mark.parametrize("include_cg", [False, True])
def test_evaluate_routes_aa_only_full_sasa_and_plot_paths(tmp_path, include_cg):
    required = {}
    for name in ("aa.tpr", "aa.xtc", "map.ndx", "model.itp"):
        path = tmp_path / name
        path.touch()
        required[name] = str(path)
    config = SwarmConfig()
    config.reference.aa_tpr_filename = required["aa.tpr"]
    config.reference.aa_traj_filename = required["aa.xtc"]
    config.reference.cg_map_filename = required["map.ndx"]
    config.cg_model.cg_itp_filename = required["model.itp"]
    config.output.calculate_sasa = True
    config.output.plot_filename = str(tmp_path / "plot.unsupported")
    if include_cg:
        cg_tpr = tmp_path / "cg.tpr"
        cg_xtc = tmp_path / "cg.xtc"
        cg_tpr.touch()
        cg_xtc.touch()
        config.cg_model.cg_tpr_filename = str(cg_tpr)
        config.cg_model.cg_traj_filename = str(cg_xtc)
    else:
        config.cg_model.cg_tpr_filename = str(tmp_path / "missing.tpr")
        config.cg_model.cg_traj_filename = str(tmp_path / "missing.xtc")

    observed = {}

    def compare(context, **kwargs):
        observed["atom_only"] = context.scoring.atom_only
        observed["calc_sasa"] = kwargs["calc_sasa"]
        observed["plot"] = context.files.plot_filename
        Path(context.files.plot_filename).touch()

    with (
        patch("swarmcg.scoring.evaluator.SwarmEvaluator.initialize"),
        patch("swarmcg.evaluate_model.compare_models", side_effect=compare),
    ):
        run(config)

    assert observed["atom_only"] is (not include_cg)
    assert observed["calc_sasa"] is include_cg
    assert observed["plot"].endswith("plot.unsupported.png")
    assert Path(observed["plot"]).is_file()
