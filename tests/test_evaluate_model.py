"""CLI-level evaluation routing tests without trajectory-processing cost."""

from pathlib import Path
from unittest.mock import patch

import pytest

from swarmcg.config_types import SwarmConfig
from swarmcg.evaluate_model import run
from swarmcg.evaluate_model import _run_requested_sasa
from swarmcg.context import OptimizationContext
from swarmcg.shared.exceptions import ComputationError


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
    config.sasa.enabled = True
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
        observed["plot"] = context.files.plot_filename
        Path(context.files.plot_filename).touch()

    def sasa(context):
        observed["sasa_atom_only"] = context.scoring.atom_only

    with (
        patch("swarmcg.scoring.evaluator.SwarmEvaluator.initialize"),
        patch("swarmcg.evaluate_model.compare_models", side_effect=compare),
        patch("swarmcg.evaluate_model._run_requested_sasa", side_effect=sasa),
    ):
        run(config)

    assert observed["atom_only"] is (not include_cg)
    assert observed["sasa_atom_only"] is (not include_cg)
    assert observed["plot"].endswith("plot.unsupported.png")
    assert Path(observed["plot"]).is_file()


def test_requested_standalone_sasa_failure_is_not_suppressed(tmp_path):
    context = OptimizationContext(config=SwarmConfig())
    context.files.plot_filename = str(tmp_path / "plot.png")
    context.scoring.atom_only = True

    with (
        patch("swarmcg.scoring.sasa.validate_sasa_inputs"),
        patch(
            "swarmcg.scoring.sasa.compute_sasa",
            side_effect=ComputationError("missing AA radius"),
        ),
        pytest.raises(ComputationError, match="missing AA radius"),
    ):
        _run_requested_sasa(context)
