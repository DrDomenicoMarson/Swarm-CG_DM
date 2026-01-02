import contextlib
import sys

import MDAnalysis as mda

from swarmcg.context import SwarmCGState


def print_stdout_forced(*args, **kwargs):
    """Print forced stdout enabled"""
    with contextlib.redirect_stdout(sys.__stdout__):
        print(*args, **kwargs, flush=True)


def set_MDA_backend(state: SwarmCGState):
    """Set MDAnalysis backend and number of threads

    state creates:
        mda_backend
    """
    # NOTE: this is not used because MDA is not properly parallelized, in fact with OpenMP backend it's slower than in serial
    if mda.lib.distances.USED_OPENMP:  # if MDAnalysis was compiled with OpenMP support
        state.runtime.mda_backend = 'OpenMP'
    else:
        state.runtime.mda_backend = 'serial'
