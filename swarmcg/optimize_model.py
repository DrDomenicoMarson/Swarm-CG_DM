import os
import sys
import time
from shlex import quote as cmd_quote

import swarmcg.io as io
from swarmcg.config_types import SwarmConfig
from swarmcg.core.optimization import SwarmOptimizer
from swarmcg.shared import catch_warnings, styling
from swarmcg.shared.logging_utils import get_logger, setup_logging

@catch_warnings(ImportWarning)
@catch_warnings(UserWarning)
def run(config_obj: SwarmConfig):
    """
    Main execution logic for model optimization.

    Args:
        config_obj (SwarmConfig): The configuration object containing all runtime parameters.
    """
    optimizer = SwarmOptimizer(config_obj)
    optimizer.run()


def main():
    module_name = "optimize"
    setup_logging(module_name=module_name, verbose=("-v" in sys.argv or "--verbose" in sys.argv))
    logger = get_logger(__name__)
    if "--nobanner" in sys.argv or "-nobanner" in sys.argv:
        logger.info(styling.header_simple(module_name))
    else:
        logger.info(styling.header_package("                    Module: CG model optimization\n"))

    args_parser = io.get_optimize_args()

    # display help if script was called without arguments
    if len(sys.argv) == 1:
        args_parser.print_help()
        sys.exit()

    # arguments handling, display command line if help or no arguments provided
    ns_args = args_parser.parse_args()
    setup_logging(module_name=module_name, verbose=ns_args.verbose)

    # do NOT display the stack by default
    if not ns_args.verbose:
        sys.tracebacklimit = 0

    input_cmdline = " ".join(map(cmd_quote, sys.argv))

    # Convert to SwarmConfig
    swarm_config = SwarmConfig.from_namespace(ns_args)
    
    # We delay exec_folder creation to run() or handle it here if needed for logging?
    # Original main set ns.exec_folder for print.
    # We can reconstruct it or let run() handle it. 
    # To print it here, we replicate logic
    if swarm_config.output.output_folder != "":
         exec_folder = swarm_config.output.output_folder
    else:
         exec_folder = time.strftime("MODEL_OPTI__STARTED_%d-%m-%Y_%Hh%Mm%Ss")

    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Command line: {input_cmdline}")
    logger.info(f"Results directory: {exec_folder}")

    run(swarm_config)


if __name__ == "__main__":
    main()
