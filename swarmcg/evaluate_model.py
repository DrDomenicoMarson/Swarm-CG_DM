import os
import sys
from shlex import quote as cmd_quote

import numpy as np
import matplotlib

import swarmcg.io as io
import swarmcg.shared.styling
from swarmcg.shared import catch_warnings
from swarmcg.shared.logging_utils import get_logger, setup_logging
from swarmcg.scoring.compare import compare_models
from swarmcg.config_types import SwarmConfig
from swarmcg.context import OptimizationContext

matplotlib.use("AGG")  # use the Anti-Grain Geometry non-interactive backend suited for scripted PNG creation

logger = get_logger(__name__)

def run(config_obj: SwarmConfig):
    """
    Main execution logic for model evaluation.
    
    Args:
        config_obj (SwarmConfig): The configuration object containing all runtime parameters.
    """
    # Create context
    ns = OptimizationContext(config=config_obj)
    ns.scoring.mismatch_order = config_obj.output.mismatch_order
    ns.scoring.row_x_scaling = config_obj.output.row_x_scaling
    ns.scoring.row_y_scaling = config_obj.output.row_y_scaling
    ns.scoring.ncols_max = config_obj.output.ncols_max

    logger.info("")
    logger.info(swarmcg.shared.styling.sep_close)
    logger.info("| PRE-PROCESSING                                                                              |")
    logger.info(swarmcg.shared.styling.sep_close)
    logger.info("")

    # TODO: make it possible to feed a delta/offset for Rg in case the model has bonds scaling ?

    # get basenames for simulation files
    ns.cg_itp_basename = os.path.basename(ns.cg_itp_filename)

    # Initialize context variables
    ns.molname_in = None
    ns.gyr_aa_mapped, ns.gyr_aa_mapped_std = None, None
    ns.sasa_aa_mapped, ns.sasa_aa_mapped_std = None, None

    # scg.set_MDA_backend(ns)
    ns.mda_backend = "serial"  # actually serial is faster because MDA is not properly parallelized atm

    # TODO: this eventually will need to be taked out of this function when we can avoid adding new attributed to ns
    # ns.mapping_type = ns.mapping_type.upper() # Handled by Config usually, or we ensure it
    
    # Ideally should use config object directly, but input_parameter_validation is legacy
    # Validate files existence
    try:
        config_obj.validate_files_exist()
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(1)

    # Create Evaluator
    from swarmcg.scoring.evaluator import SwarmEvaluator
    ns.evaluator = SwarmEvaluator(config_obj)

    # display parameters for function compare_models
    if not os.path.isfile(ns.cg_tpr_filename) or not os.path.isfile(ns.cg_traj_filename):
        # switch to atomistic mapping inspection exclusively (= do NOT plot the CG distributions)
        logger.warning("Could not find file(s) for either CG topology or trajectory")
        logger.info("  Going for inspection of AA-mapped distributions exclusively")
        logger.info("")
        ns.atom_only = True
    else:
        ns.atom_only = False

    try:
        if not ns.plot_filename.split(".")[-1] in ["eps", "pdf", "pgf", "png", "ps", "raw", "rgba", "svg", "svgz"]:
            ns.plot_filename = ns.plot_filename + ".png"
    except IndexError as e:
        ns.plot_filename = ns.plot_filename + ".png"

    # Initialize Evaluator (loads AA reference, mapping, maps AA->CG)
    ns.evaluator.initialize(ns)

    # Run comparison
    compare_models(ns, manual_mode=True, calc_sasa=False)


def main():
    module_name = "evaluate"
    setup_logging(module_name=module_name, verbose=("-v" in sys.argv or "--verbose" in sys.argv))
    logger = get_logger(__name__)
    if "--nobanner" in sys.argv or "-nobanner" in sys.argv:
        logger.info(swarmcg.shared.styling.header_simple(module_name))
    else:
        logger.info(swarmcg.shared.styling.header_package("                Module: Model bonded terms assessment\n"))

    args_parser = io.get_evaluate_args()

    # arguments handling, display command line if help or no arguments provided
    ns_args = args_parser.parse_args()
    plot_dir = os.path.dirname(os.path.abspath(ns_args.plot_filename or "distributions.png"))
    setup_logging(module_name=module_name, log_dir=plot_dir, verbose=ns_args.verbose)
    input_cmdline = " ".join(map(cmd_quote, sys.argv))
    logger.info("Working directory: %s", os.getcwd())
    logger.info("Command line: %s", input_cmdline)

    swarm_config = SwarmConfig.from_namespace(ns_args)
    run(swarm_config)


if __name__ == "__main__":
    main()
