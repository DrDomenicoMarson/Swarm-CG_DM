import re
import collections
import numpy as np
import MDAnalysis as mda
from swarmcg.shared import exceptions
from swarmcg.config_types import SwarmConfig
from swarmcg.simulations import vs_functions as vsf
from swarmcg.shared.logging_utils import get_logger
from swarmcg.topology import CGTopology

logger = get_logger(__name__)

class Mapping:
    def __init__(self, config: SwarmConfig):
        self.config = config
        self.reference_config = config.reference
        
        # State
        self.all_atoms = dict()
        self.all_aa_mols = []
        self.atoms_occ_total = collections.Counter()
        self.all_beads = dict()
        self.atom_w = dict()
        self.mda_beads_atom_grps = dict()
        self.mda_weights_atom_grps = dict()
        
    def load_aa_data(self, universe: mda.Universe):
        """Read one or more molecules from the AA TPR and trajectory."""
        self.all_atoms = dict()
        self.all_aa_mols = []

        # TODO: Handle molname_in if needed (currently passed as None in original code often)
        # Using simplified logic from original load_aa_data assuming molname_in is None for now
        
        molname_atom_group = universe.atoms[0].fragment
        self.all_aa_mols.append(molname_atom_group)

        for i in range(len(molname_atom_group)):
            atom_id = universe.atoms[i].id
            atom_type = universe.atoms[i].type[0]
            atom_charge = universe.atoms[i].charge
            atom_heavy = True
            if atom_type[0].upper() == "H":
                atom_heavy = False

            self.all_atoms[atom_id] = {
                "conn": set(), "atom_type": atom_type, "atom_charge": atom_charge,
                "heavy": atom_heavy, "beads_ids": set(), "beads_types": set(),
                "residue_names": set()
            }

    def read_ndx_atoms2beads(self):
        """Load CG beads from NDX-like file."""
        with open(self.reference_config.cg_map_filename, "r") as fp:
            ndx_lines = fp.read().split("\n")
            ndx_lines = [ndx_line.strip().split(";")[0] for ndx_line in ndx_lines]

            self.atoms_occ_total = collections.Counter()
            self.all_beads = dict()
            bead_id = 0
            current_section = "Beginning of file"
            
            for i in range(len(ndx_lines)):
                ndx_line = ndx_lines[i]
                if ndx_line != "":
                    if bool(re.search(r"\[.*\]", ndx_line)):
                        current_section = ndx_line
                        self.all_beads[bead_id] = {"atoms_id": [], "section": current_section, "line_nb": i + 1}
                        current_bead_id = bead_id
                        bead_id += 1
                    else:
                        try:
                            bead_atoms_id = [int(atom_id) - 1 for atom_id in ndx_line.split()]
                            self.all_beads[current_bead_id]["atoms_id"].extend(bead_atoms_id)
                            
                            for atom_id in bead_atoms_id:
                                self.atoms_occ_total[atom_id] += 1
                        except NameError:
                            msg = ("The CG beads mapping (NDX) file does NOT seem to contain CG beads "
                                   "sections.\nPlease verify the input mapping. The expected format is Gromacs NDX.")
                            raise exceptions.MissformattedFile(msg)
                        except ValueError:
                            msg = (f"Incorrect reading of the sections content in the CG beads mapping "
                                   f"(NDX) file.\nFound non-integer values for some IDs at line "
                                   f"{str(i + 1)} under section {current_section}.")
                            raise exceptions.MissformattedFile(msg)

        for bead_id in self.all_beads:
            if len(self.all_beads[bead_id]["atoms_id"]) == 0:
                msg = (f"The ITP file contains an empty section named {self.all_beads[bead_id]['section']} "
                       f"starting at line {self.all_beads[bead_id]['line_nb']}. "
                       f"Empty sections are NOT allowed, please fill or delete it.")
                raise exceptions.MissformattedFile(msg)

    def get_atoms_weights_in_beads(self):
        """Calculate weight ratio of atom ID in given CG bead."""
        self.atom_w = dict()
        if self.config.output.verbose:
            logger.info("Calculating atoms weights ratio within mapped CG beads")
            
        for bead_id in self.all_beads:
            self.atom_w[bead_id] = dict()
            beads_atoms_counts = collections.Counter(self.all_beads[bead_id]["atoms_id"])
            for atom_id in beads_atoms_counts:
                self.atom_w[bead_id][atom_id] = round(beads_atoms_counts[atom_id] / self.atoms_occ_total[atom_id], 3)
                if self.config.output.verbose and self.reference_config.mapping_type == "COM":
                    logger.info(
                        "  CG bead ID %s -- Atom ID %s has weight ratio = %s",
                        bead_id + 1,
                        atom_id + 1,
                        self.atom_w[bead_id][atom_id],
                    )
        if self.config.output.verbose:
            logger.info("")

    def get_beads_MDA_atomgroups(self, universe: mda.Universe):
        """For each CG bead, create atom groups for trajectory geoms calculation."""
        self.mda_beads_atom_grps, self.mda_weights_atom_grps = dict(), dict()
        
        for bead_id in self.atom_w:
            try:
                if self.reference_config.mapping_type == "COM":
                    self.mda_beads_atom_grps[bead_id] = mda.AtomGroup([atom_id for atom_id in self.atom_w[bead_id]], universe)
                    self.mda_weights_atom_grps[bead_id] = np.array(
                        [self.atom_w[bead_id][atom_id] * universe.atoms[atom_id].mass for atom_id in self.atom_w[bead_id]])
                elif self.reference_config.mapping_type == "COG":
                    self.mda_beads_atom_grps[bead_id] = mda.AtomGroup([atom_id for atom_id in self.atom_w[bead_id]], universe)
                    self.mda_weights_atom_grps[bead_id] = np.array([1 for _ in self.atom_w[bead_id]])
            except IndexError as e:
                msg = (f"An ID present in your mapping (NDX) file could not be found in the AA trajectory. "
                       f"Please check your mapping (NDX) file.\nSee the error below to understand which "
                       f"ID (here 0-indexed) could not be found:\n\n{str(e)}")
                raise exceptions.MissformattedFile(msg)

    def map_aa2cg_traj(
        self,
        aa_universe: mda.Universe,
        aa2cg_universe: mda.Universe,
        topology: CGTopology,
    ) -> None:
        """Map an atomistic trajectory onto the typed CG topology.

        Args:
            aa_universe: Source atomistic trajectory.
            aa2cg_universe: Destination in-memory CG trajectory.
            topology: Typed topology defining real and virtual beads.

        Returns:
            ``None``. Coordinates are loaded into ``aa2cg_universe`` in place.
        """
        if self.reference_config.mapping_type == "COM":
            logger.info("  Interpretation: Center of Mass (COM)")
        elif self.reference_config.mapping_type == "COG":
            logger.info("  Interpretation: Center of Geometry (COG)")

        # Regular beads
        n_frames = len(aa_universe.trajectory)
        n_beads = len(topology.atoms)
        coord = np.empty((n_frames, n_beads, 3))
        
        # Pre-calculate processing order
        regular_beads_ids = topology.real_bead_ids
        
        logger.info("  Processing %s frames...", n_frames)
        
        # Iterate trajectory once
        for ts in aa_universe.trajectory:
            frame_idx = ts.frame
            for bead_id in regular_beads_ids:
                coord[frame_idx, bead_id] = self.mda_beads_atom_grps[bead_id].center(
                    self.mda_weights_atom_grps[bead_id], compound="group"
                )

        aa2cg_universe.load_new(coord, format=mda.coordinates.memory.MemoryReader)

        dispatch = {
            (2, 1): lambda site, traj: vsf.vs2_func_1(
                aa2cg_universe, traj, site.defining_beads, site.parameters[0]
            ),
            (2, 2): lambda site, traj: vsf.vs2_func_2(
                aa2cg_universe, traj, site.defining_beads, site.parameters[0]
            ),
            (3, 1): lambda site, traj: vsf.vs3_func_1(
                aa2cg_universe, traj, site.defining_beads, site.parameters
            ),
            (3, 2): lambda site, traj: vsf.vs3_func_2(
                aa2cg_universe, traj, site.defining_beads, site.parameters
            ),
            (3, 3): lambda site, traj: vsf.vs3_func_3(
                aa2cg_universe, traj, site.defining_beads, site.parameters
            ),
            (3, 4): lambda site, traj: vsf.vs3_func_4(
                aa2cg_universe, traj, site.defining_beads, site.parameters
            ),
            (4, 2): lambda site, traj: vsf.vs4_func_2(
                aa2cg_universe, traj, site.defining_beads, site.parameters
            ),
            ("n", 1): lambda site, traj: vsf.vsn_func_1(
                aa2cg_universe, traj, site.defining_beads
            ),
            ("n", 2): lambda site, traj: vsf.vsn_func_2(
                aa2cg_universe,
                traj,
                site.defining_beads,
                site.bead_id,
                topology=topology,
            ),
            ("n", 3): lambda site, traj: vsf.vsn_func_3(
                aa2cg_universe, traj, site.defining_beads, site.parameters
            ),
        }
        for site in topology.virtual_sites:
            traj = np.empty((len(aa2cg_universe.trajectory), 3))
            dispatch[(site.kind, site.function)](site, traj)
            coord[:, site.bead_id, :] = traj
        
        aa2cg_universe.load_new(coord, format=mda.coordinates.memory.MemoryReader)

def initialize_cg_traj(topology: CGTopology):
    """Initialize an in-memory trajectory universe for a typed CG topology.

    Args:
        topology: Typed coarse-grained topology.

    Returns:
        Empty MDAnalysis universe with atom metadata and one trajectory slot.
    """
    masses = np.array([atom.mass for atom in topology.atoms])
    names = np.array([atom.atom_name for atom in topology.atoms])
    resnames = np.array([atom.residue_name for atom in topology.atoms])
    resid = np.array([atom.residue_number for atom in topology.atoms])
    nr = len(set(resid))

    aa2cg_universe = mda.Universe.empty(len(topology.atoms), n_residues=nr, atom_resindex=resid, n_segments=1,
                                        residue_segindex=np.ones(nr), trajectory=True)
    aa2cg_universe.add_TopologyAttr("masses")
    aa2cg_universe._topology.masses.values = np.array(masses)
    aa2cg_universe.add_TopologyAttr("names")
    aa2cg_universe._topology.names.values = names
    aa2cg_universe.add_TopologyAttr("resnames")
    aa2cg_universe._topology.resnames.values = resnames
    return aa2cg_universe

def make_aa_traj_whole_for_selected_mols(aa_universe, all_aa_mols):
    """Use selected whole molecules as MDA atomgroups and make their coordinates whole, inplace."""
    for _ in aa_universe.trajectory:
        for aa_mol in all_aa_mols:
            mda.lib.mdamath.make_whole(aa_mol, inplace=True)
