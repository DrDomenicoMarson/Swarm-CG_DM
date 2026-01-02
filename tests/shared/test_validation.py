import pytest

from swarmcg.shared import exceptions
from swarmcg.shared.validation import _file_validation, _optimisation_input_validation


def test___file_validation(opt_ctx):
    # All file are missing, here are added one by one so we verify the sequence
    # of errors triggered is correct
    # when:
    args, _ = opt_ctx(
        aa_tpr_filename="",
        aa_traj_filename="",
        cg_map_filename="",
        cg_itp_filename="",
    )

    # then:
    with pytest.raises(exceptions.MissingCoordinateFile):
        _file_validation(args)

    # when:
    filename = "tests/data/aa_topol.tpr"
    args.inputs.aa_tpr_filename = filename

    # then:
    with pytest.raises(exceptions.MissingTrajectoryFile):
        _file_validation(args)

    # when:
    filename = "tests/data/aa_traj.xtc"
    args.inputs.aa_traj_filename = filename

    # then:
    with pytest.raises(exceptions.MissingIndexFile):
        _file_validation(args)

   # when:
    filename = "tests/data/cg_map.ndx"
    args.inputs.cg_map_filename = filename

    # then:
    with pytest.raises(exceptions.MissingItpFile):
        _file_validation(args)

   # when:
    filename = "tests/data/cg_model.itp"
    args.inputs.cg_itp_filename = filename

    # then:
    _file_validation(args)


def test__optimisation_input_validation(opt_ctx):
    # when:
    args, _ = opt_ctx()

    # then:
    _optimisation_input_validation(args)

    # when
    args, _ = opt_ctx(default_max_fct_bonds_opti=-1)

    # then:
    with pytest.raises(exceptions.InputArgumentError):
        _optimisation_input_validation(args)

    # when
    args, _ = opt_ctx(default_max_fct_angles_opti_f1=-1)

    # then:
    with pytest.raises(exceptions.InputArgumentError):
        _optimisation_input_validation(args)

    # when
    args, _ = opt_ctx(default_max_fct_angles_opti_f2=-1)

    # then:
    with pytest.raises(exceptions.InputArgumentError):
        _optimisation_input_validation(args)
