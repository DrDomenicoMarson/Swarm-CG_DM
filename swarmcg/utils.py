import contextlib
import math
import sys

import MDAnalysis as mda
from swarmcg import config
from swarmcg.shared import exceptions
from swarmcg.shared.logging_utils import get_logger

logger = get_logger(__name__)


def print_stdout_forced(*args, **kwargs):
    """Print forced stdout enabled"""
    sep = kwargs.get("sep", " ")
    message = sep.join(str(arg) for arg in args)
    with contextlib.redirect_stdout(sys.__stdout__):
        logger.info(message)


def set_MDA_backend(ns):
    """Set MDAnalysis backend and number of threads

    ns creates:
        mda_backend
    """
    # NOTE: this is not used because MDA is not properly parallelized, in fact with OpenMP backend it's slower than in serial
    if mda.lib.distances.USED_OPENMP:  # if MDAnalysis was compiled with OpenMP support
        ns.scoring.mda_backend = 'OpenMP'
    else:
        ns.scoring.mda_backend = 'serial'


def process_scaling_str(ns):
    """Parse optional per-group bond and constraint target lengths.

    Args:
        ns: Optimization context containing the scaling string and parsed CG
            topology. Its ``scoring.bonds_scaling_specific`` field is updated.

    Returns:
        ``None``.

    Raises:
        InvalidArgument: If the specification is malformed, duplicated,
            references a missing group, or contains a non-positive/non-finite
            target length.
    """
    ns.scoring.bonds_scaling_specific = None
    # We compare against default in config to see if user provided something different
    # ns.bonds_scaling_str comes from config object (context.config.optimization.bonds_scaling_str)
    # config.bonds_scaling_str comes from swarmcg.config (global defaults)
    
    current_val = ns.config.optimization.bonds_scaling_str
    
    if current_val != config.bonds_scaling_str:
        sp_str = current_val.split()
        if len(sp_str) % 2 != 0:
            msg = (
                f"Cannot interpret argument -bonds_scaling_str as provided: {current_val}.\n"
                f"Please check your parameters, or the help (-h) for an example."
            )
            raise exceptions.InvalidArgument(msg)

        ns.scoring.bonds_scaling_specific = dict()
        i = 0
        try:
            while i < len(sp_str):
                geom_id = sp_str[i][1:]
                if sp_str[i][0].upper() == "C":
                    if int(geom_id) > ns.cg_itp["nb_constraints"]:
                        info = "A constraint group id exceeds the number of constraints groups defined in the input CG ITP file."
                        raise exceptions.InvalidArgument("bonds_scaling_str", current_val, info)
                    if not "C" + geom_id in ns.scoring.bonds_scaling_specific:
                        target_length = float(sp_str[i + 1])
                        if not math.isfinite(target_length) or target_length <= 0:
                            info = "Average distribution lengths must be finite and positive."
                            raise exceptions.InvalidArgument("bonds_scaling_str", current_val, info)
                        ns.scoring.bonds_scaling_specific["C" + geom_id] = target_length
                    else:
                        info = f"A constraint group id is provided multiple times (id: {geom_id})"
                        raise exceptions.InvalidArgument("bonds_scaling_str", current_val, info)
                elif sp_str[i][0].upper() == "B":
                    if int(geom_id) > ns.cg_itp["nb_bonds"]:
                        info = "A bond group id exceeds the number of bonds groups defined in the input CG ITP file."
                        raise exceptions.InvalidArgument("bonds_scaling_str", current_val, info)
                    if not "B" + geom_id in ns.scoring.bonds_scaling_specific:
                        target_length = float(sp_str[i + 1])
                        if not math.isfinite(target_length) or target_length <= 0:
                            info = "Average distribution lengths must be finite and positive."
                            raise exceptions.InvalidArgument("bonds_scaling_str", current_val, info)
                        ns.scoring.bonds_scaling_specific["B" + geom_id] = target_length
                    else:
                        info = f"A bond group id is provided multiple times (id: {geom_id})"
                        raise exceptions.InvalidArgument("bonds_scaling_str", current_val, info)
                i += 2
        except ValueError:
            raise exceptions.InvalidArgument("bonds_scaling_str", current_val)
