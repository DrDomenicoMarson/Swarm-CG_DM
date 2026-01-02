import os, sys
from shlex import quote as cmd_quote

import numpy as np

import swarmcg.shared.styling
import swarmcg.io as io
import swarmcg.scoring as scores
from swarmcg.engine import comparison, mapping, optimization
from swarmcg.context import SwarmCGArgs, SwarmCGState
from swarmcg.shared import catch_warnings

VISIBLE_DEPRECATION_WARNING = getattr(np, "VisibleDeprecationWarning", Warning)


@catch_warnings(VISIBLE_DEPRECATION_WARNING)  # filter MDAnalysis + numpy deprecation stuff that is annoying
def run(args: SwarmCGArgs, state: SwarmCGState):
    print()
    print(swarmcg.shared.styling.sep_close)
    print("| PRE-PROCESSING                                                                              |")
    print(swarmcg.shared.styling.sep_close)
    print()

    # TODO: make it possible to feed a delta/offset for Rg in case the model has bonds scaling ?

    # get basenames for simulation files
    state.files.cg_itp_basename = os.path.basename(args.inputs.cg_itp_filename)

    # NOTE: some arguments exist only in the scope of optimization (optimize_model.py) or only in the scope of model
    #       evaluation (evaluate_mode.py), but still need to be defined here -> Change this to something less messy
    state.mapping.molname_in = None
    state.model.gyr_aa_mapped, state.model.gyr_aa_mapped_std = None, None
    state.model.sasa_aa_mapped, state.model.sasa_aa_mapped_std = None, None
    args.optimization.aa_rg_offset = 0  # TODO: allow an argument more in evaluate_model, like in optimiwe_model, for adding an offset to Rg
    args.inputs.user_input = False
    args.optimization.default_max_fct_bonds_opti = np.inf
    args.optimization.default_max_fct_angles_opti_f1 = np.inf
    args.optimization.default_max_fct_angles_opti_f2 = np.inf
    args.optimization.default_abs_range_fct_dihedrals_opti_func_with_mult = np.inf
    args.optimization.default_abs_range_fct_dihedrals_opti_func_without_mult = np.inf

    # utils.set_MDA_backend(state)
    state.runtime.mda_backend = "serial"  # actually serial is faster because MDA is not properly parallelized atm

    args.validate()

    # display parameters for function compare_models
    if not os.path.isfile(args.inputs.cg_tpr_filename) or not os.path.isfile(args.inputs.cg_traj_filename):
        # switch to atomistic mapping inspection exclusively (= do NOT plot the CG distributions)
        print("Could not find file(s) for either CG topology or trajectory")
        print("  Going for inspection of AA-mapped distributions exclusively")
        print()
        state.mapping.atom_only = True
    else:
        state.mapping.atom_only = False

    try:
        if not args.paths.plot_filename.split(".")[-1] in ["eps", "pdf", "pgf", "png", "ps", "raw", "rgba", "svg", "svgz"]:
            args.paths.plot_filename = args.paths.plot_filename + ".png"
    except IndexError as e:
        args.paths.plot_filename = args.paths.plot_filename + ".png"

    scores.create_bins_and_dist_matrices(args, state)  # bins for EMD calculations
    mapping.read_ndx_atoms2beads(args, state)  # read mapping, get atoms accurences in beads
    mapping.get_atoms_weights_in_beads(args, state)  # get weights of atoms within beads

    state.model.cg_itp = io.read_cg_itp_file(args)  # load the ITP object and find out geoms grouping
    io.validate_cg_itp(state.model.cg_itp)  # check ITP object is correct
    optimization.process_scaling_str(args, state)  # process the bonds scaling specified by user

    print()
    io.read_aa_traj(args, state)  # create universe and read traj
    mapping.load_aa_data(args, state)  # read atoms attributes
    mapping.make_aa_traj_whole_for_selected_mols(args, state)

    # for each CG bead, create atom groups for trajectory geoms calculation using mass and atom weights across beads
    mapping.get_beads_MDA_atomgroups(args, state)

    print("\nMapping the trajectory from AA to CG representation")
    state.traj.aa2cg_universe = mapping.initialize_cg_traj(state.model.cg_itp)
    mapping.map_aa2cg_traj(args, state)
    print()

    comparison.compare_models(args, state, manual_mode=True, calc_sasa=False)


def main():
    args_parser = io.get_evaluate_args()

    # arguments handling, display command line if help or no arguments provided
    args = SwarmCGArgs.from_namespace(args_parser.parse_args())
    state = SwarmCGState()
    input_cmdline = " ".join(map(cmd_quote, sys.argv))
    print("Working directory:", os.getcwd())
    print("Command line:", input_cmdline)

    run(args, state)


if __name__ == "__main__":
    main()
