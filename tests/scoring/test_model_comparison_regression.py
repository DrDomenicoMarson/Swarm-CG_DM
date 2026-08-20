from pathlib import Path

import pytest

from swarmcg.config_types import SwarmConfig
from swarmcg.context import OptimizationContext
from swarmcg.scoring.compare import compare_models
from swarmcg.scoring.evaluator import SwarmEvaluator


TEST_DATA = Path(__file__).parents[1] / "data"


def test_bundled_model_comparison_numerical_regression(tmp_path):
    """Preserve the validated AA/CG score decomposition during refactoring."""
    config = SwarmConfig()
    config.reference.aa_tpr_filename = str(TEST_DATA / "aa_topol.tpr")
    config.reference.aa_traj_filename = str(TEST_DATA / "aa_traj.xtc")
    config.reference.cg_map_filename = str(TEST_DATA / "cg_map.ndx")
    config.cg_model.cg_itp_filename = str(TEST_DATA / "cg_model.itp")
    config.cg_model.cg_tpr_filename = str(TEST_DATA / "cg_topol.tpr")
    config.cg_model.cg_traj_filename = str(TEST_DATA / "cg_traj.xtc")

    context = OptimizationContext(config=config)
    context.files.cg_tpr_filename = config.cg_model.cg_tpr_filename
    context.files.cg_traj_filename = config.cg_model.cg_traj_filename
    context.files.plot_filename = str(tmp_path / "comparison.png")
    evaluator = SwarmEvaluator(config)
    context.evaluator = evaluator
    evaluator.initialize(context)
    evaluator.compute_reference_distributions()
    context.scoring.atom_only = False

    result = compare_models(context, manual_mode=False)

    assert result is not None
    assert result[:4] == pytest.approx(
        (8.655850009902657, 3.673883866470647, 4.98196614343201, 0.0),
        rel=1e-12,
        abs=1e-12,
    )
    assert result[5] == pytest.approx(
        {
            "constraints": [],
            "bonds": [
                1.5301459735325402,
                1.9646621856336215,
                1.4123627255754319,
                2.3024790969174775,
            ],
            "angles": [
                1.3395604858379535,
                2.25970775197642,
                2.309753523650249,
                3.376417874488181,
                1.0881756437101713,
            ],
            "dihedrals": [],
        },
        rel=1e-12,
        abs=1e-12,
    )
    assert (
        context.results.gyr_aa_mapped,
        context.results.gyr_aa_mapped_std,
        context.results.gyr_cg,
        context.results.gyr_cg_std,
    ) == pytest.approx((0.995, 0.053, 0.986, 0.054), abs=1e-12)
