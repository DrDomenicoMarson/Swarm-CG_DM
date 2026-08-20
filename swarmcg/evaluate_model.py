import os
import sys
from shlex import quote as cmd_quote

import matplotlib

import swarmcg.io as io
import swarmcg.shared.styling
from swarmcg.shared.logging_utils import get_logger, setup_logging
from swarmcg.scoring.compare import compare_models
from swarmcg.config_types import SwarmConfig
from swarmcg.context import OptimizationContext

matplotlib.use("AGG")  # use the Anti-Grain Geometry non-interactive backend suited for scripted PNG creation

logger = get_logger(__name__)

def run(config_obj: SwarmConfig) -> None:
    """Evaluate an AA-mapped model alone or compare it with a CG trajectory.

    Args:
        config_obj: Validated runtime configuration.

    Raises:
        FileNotFoundError: If a required AA, mapping, or ITP input is missing.
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

    ns.files.cg_itp_basename = os.path.basename(config_obj.cg_model.cg_itp_filename)
    ns.files.cg_tpr_filename = config_obj.cg_model.cg_tpr_filename
    ns.files.cg_traj_filename = config_obj.cg_model.cg_traj_filename
    ns.files.plot_filename = _normalized_plot_filename(config_obj.output.plot_filename)
    ns.scoring.molname_in = None
    ns.scoring.mda_backend = "serial"

    config_obj.validate_files_exist()

    # Create Evaluator
    from swarmcg.scoring.evaluator import SwarmEvaluator
    ns.evaluator = SwarmEvaluator(config_obj)

    # display parameters for function compare_models
    cg_topology_exists = os.path.isfile(ns.files.cg_tpr_filename)
    cg_trajectory_exists = os.path.isfile(ns.files.cg_traj_filename)
    if not cg_topology_exists or not cg_trajectory_exists:
        # switch to atomistic mapping inspection exclusively (= do NOT plot the CG distributions)
        logger.warning("Could not find file(s) for either CG topology or trajectory")
        logger.info("  Going for inspection of AA-mapped distributions exclusively")
        logger.info("")
        ns.scoring.atom_only = True
    else:
        ns.scoring.atom_only = False

    calculate_sasa = config_obj.output.calculate_sasa
    if calculate_sasa and ns.scoring.atom_only:
        logger.warning("SASA requires a CG topology and trajectory; skipping SASA in AA-only mode.")
        calculate_sasa = False

    # Initialize Evaluator (loads AA reference, mapping, maps AA->CG)
    ns.evaluator.initialize(ns)

    # Run comparison
    compare_models(ns, manual_mode=True, calc_sasa=calculate_sasa)


def _normalized_plot_filename(filename: str) -> str:
    """Return an absolute plot filename with a supported extension.

    Args:
        filename: User-supplied plot path.

    Returns:
        Absolute plot path. ``.png`` is appended when the supplied extension
        is absent or unsupported.
    """
    supported = {".eps", ".pdf", ".pgf", ".png", ".ps", ".raw", ".rgba", ".svg", ".svgz"}
    root, extension = os.path.splitext(filename or "distributions.png")
    normalized = filename if extension.lower() in supported else f"{filename or root or 'distributions'}.png"
    return os.path.abspath(normalized)


def main():
    """Parse ``scg_evaluate`` arguments and run model assessment."""
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
