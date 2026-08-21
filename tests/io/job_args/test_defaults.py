from swarmcg.io.job_args import defaults
from swarmcg.io.job_args.evaluate_config import get_evaluate_args
from swarmcg.io.job_args.optimize_config import get_optimize_args


def test_defaults():
    assert defaults.bw_bonds.default == 0.01
    assert not defaults.mismatch_ordering.default
    assert defaults.plot_scale.default == 1.0
    assert defaults.plot_scale.metavar == "(1.0)".rjust(25, " ")
    assert defaults.opti_dir.metavar == ""
    assert defaults.user_params.action == "store_true"
    assert "metavar" not in defaults.user_params.args
    assert defaults.cg_mdp_mini.default[-13:] == "data/mini.mdp"
    assert defaults.cg_mdp_equi.default[-13:] == "data/equi.mdp"


def test_sasa_long_option_is_available_for_evaluation_and_optimization():
    evaluate = get_evaluate_args().parse_args(
        [
            "--sasa",
            "--sasa-aa-radii",
            "custom.dat",
            "--sasa-probe-radius",
            "0.2",
            "--sasa-ndots",
            "9600",
        ]
    )
    optimize = get_optimize_args().parse_args(["--sasa"])
    assert evaluate.enabled is True
    assert evaluate.aa_radii_filename == "custom.dat"
    assert evaluate.probe_radius_nm == 0.2
    assert evaluate.sphere_points == 9600
    assert optimize.enabled is True
