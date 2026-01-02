import os

from swarmcg.shared import exceptions
from swarmcg.shared.styling import header_warning


def _file_validation(args):
    """Check for filename attributes.

    args requires:
        aa_tpr_filename
        aa_traj_filename
        cg_map_filename
        cg_itp_filename
    """
    if not os.path.isfile(args.aa_tpr_filename):
        msg = (
            f"Cannot find topology file of the atomistic simulation at location: {args.aa_tpr_filename}\n"
            f"(TPR or other portable topology formats supported by MDAnalysis)"
        )
        raise exceptions.MissingCoordinateFile(msg)
    if not os.path.isfile(args.aa_traj_filename):
        msg = (
            f"Cannot find trajectory file of the atomistic simulation at location: {args.aa_traj_filename}\n"
            f"(XTC, TRR, or other trajectory formats supported by MDAnalysis)"
        )
        raise exceptions.MissingTrajectoryFile(msg)

    if not os.path.isfile(args.cg_map_filename):
        msg = (
            f"Cannot find CG beads mapping file at location: {args.cg_map_filename}\n"
            f"(NDX-like file format)"
        )
        raise exceptions.MissingIndexFile(msg)

    if not os.path.isfile(args.cg_itp_filename):
        msg = f"Cannot find ITP file of the CG model at location: {args.cg_itp_filename}"
        raise exceptions.MissingItpFile(msg)


def _optimisation_input_validation(args):
    """Check for some attributes required by the optimization.

    args requires:
        default_max_fct_bonds_opti
        default_max_fct_angles_opti_f1
        default_max_fct_angles_opti_f2
        gmx_args_str
        nb_threads
        gpu_id
    """
    # check that force constants limits make sense
    if args.default_max_fct_bonds_opti <= 0:
        msg = f"Please provide a value > 0 for argument -max_fct_bonds_f1."
        raise exceptions.InputArgumentError(msg)
    if args.default_max_fct_angles_opti_f1 <= 0:
        msg = f"Please provide a value > 0 for argument -max_fct_angles_opti_f1."
        raise exceptions.InputArgumentError(msg)
    if args.default_max_fct_angles_opti_f2 <= 0:
        msg = f"Please provide a value > 0 for argument -max_fct_angles_opti_f2."
        raise exceptions.InputArgumentError(msg)

    # check gmx arguments conflicts
    if args.gmx_args_str != "" and (args.nb_threads != 0 or args.gpu_id != ""):
        msg = (
            f"{header_warning}Argument -gmx_args_str is provided together with one of arguments: "
            f"-nb_threads, -gpu_id\nOnly argument -gmx_args_str will be used during this execution"
        )
        print(msg)


def input_parameter_validation(args, config, step=None):
    """Check for filename attributes.

    args requires:
        mapping_type
        bonds_scaling
        min_bonds_length
        bonds_scaling_str

    pass args to:
         _file_validation
        _optimisation_input_validation
    """
    # check the mapping type
    _file_validation(args)

    # check the mapping type
    mapping_type = args.mapping_type.upper()
    if mapping_type != 'COM' and mapping_type != 'COG':
        msg = (
            "Mapping type provided via argument '-mapping' must be either COM or COG (Center of "
            "Mass or Center of Geometry)."
        )
        raise exceptions.InputArgumentError(msg)

    # check bonds scaling arguments conflicts
    if (args.bonds_scaling != config.bonds_scaling and args.min_bonds_length != config.min_bonds_length) or (
            args.bonds_scaling != config.bonds_scaling and args.bonds_scaling_str != config.bonds_scaling_str) or (
            args.min_bonds_length != config.min_bonds_length and args.bonds_scaling_str != config.bonds_scaling_str):
        msg = (
            "Only one of arguments -bonds_scaling, -bonds_scaling_str and -min_bonds_length "
            "can be provided. Please check your parameters"
        )
        raise exceptions.InputArgumentError(msg)

    if step == "optimisation":
        _optimisation_input_validation(args)
