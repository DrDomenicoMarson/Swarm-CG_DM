"""Trajectory readers used by the Swarm-CG workflows."""

import MDAnalysis as mda

from swarmcg.config_types import ReferenceModelConfig
from swarmcg.shared import catch_warnings
from swarmcg.shared.logging_utils import get_logger

logger = get_logger(__name__)


@catch_warnings(ImportWarning, SyntaxWarning)
def read_aa_traj(config: ReferenceModelConfig) -> mda.Universe:
    """Read the configured all-atom reference trajectory into memory.

    Args:
        config: Reference topology and trajectory configuration.

    Returns:
        Loaded MDAnalysis universe.
    """
    logger.info("Reading All Atom (AA) trajectory")
    universe = mda.Universe(
        config.aa_tpr_filename,
        config.aa_traj_filename,
        in_memory=True,
        refresh_offsets=True,
        guess_bonds=False,
    )
    logger.info("  Found %s frames", len(universe.trajectory))
    return universe
