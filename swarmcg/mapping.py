import re
import collections
import numpy as np
import MDAnalysis as mda
from swarmcg import config
from swarmcg.shared import exceptions
from swarmcg.config_types import SwarmConfig
from swarmcg.simulations import vs_functions as vsf

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
            print("Calculating atoms weights ratio within mapped CG beads")
            
        for bead_id in self.all_beads:
            self.atom_w[bead_id] = dict()
            beads_atoms_counts = collections.Counter(self.all_beads[bead_id]["atoms_id"])
            for atom_id in beads_atoms_counts:
                self.atom_w[bead_id][atom_id] = round(beads_atoms_counts[atom_id] / self.atoms_occ_total[atom_id], 3)
                if self.config.output.verbose and self.reference_config.mapping_type == "COM":
                    print("  CG bead ID", bead_id + 1, "-- Atom ID", atom_id + 1, "has weight ratio =",
                          self.atom_w[bead_id][atom_id])
        if self.config.output.verbose:
            print()

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

    def map_aa2cg_traj(self, aa_universe: mda.Universe, aa2cg_universe: mda.Universe, cg_itp: dict):
        """Map AA trajectory to CG trajectory."""
        if self.reference_config.mapping_type == "COM":
            print("  Interpretation: Center of Mass (COM)")
        elif self.reference_config.mapping_type == "COG":
            print("  Interpretation: Center of Geometry (COG)")

        # Regular beads
        coord = np.empty((len(aa_universe.trajectory), len(cg_itp["atoms"]), 3))
        for bead_id in range(len(cg_itp["atoms"])):
            if not cg_itp["atoms"][bead_id]["bead_type"].startswith("v"):
                traj = np.empty((len(aa_universe.trajectory), 3))
                for ts in aa_universe.trajectory:
                    traj[ts.frame] = self.mda_beads_atom_grps[bead_id].center(
                        self.mda_weights_atom_grps[bead_id], compound="group"
                    )
                coord[:, bead_id, :] = traj

        aa2cg_universe.load_new(coord, format=mda.coordinates.memory.MemoryReader)

        # Virtual sites
        for bead_id in range(len(cg_itp["atoms"])):
            if cg_itp["atoms"][bead_id]["bead_type"].startswith("v"):
                
                traj = np.empty((len(aa2cg_universe.trajectory), 3))
                
                # VS Type 2
                if cg_itp["atoms"][bead_id]["vs_type"] == 2:
                    vs_def_beads_ids = cg_itp["virtual_sites2"][bead_id]["vs_def_beads_ids"]
                    vs_params = cg_itp["virtual_sites2"][bead_id]["vs_params"]

                    if cg_itp["virtual_sites2"][bead_id]["func"] == 1:
                        vsf.vs2_func_1(aa2cg_universe, traj, vs_def_beads_ids, vs_params)
                    elif cg_itp["virtual_sites2"][bead_id]["func"] == 2:
                        vsf.vs2_func_2(aa2cg_universe, traj, vs_def_beads_ids, vs_params)

                # VS Type 3
                if cg_itp["atoms"][bead_id]["vs_type"] == 3:
                    vs_def_beads_ids = cg_itp["virtual_sites3"][bead_id]["vs_def_beads_ids"]
                    vs_params = cg_itp["virtual_sites3"][bead_id]["vs_params"]

                    if cg_itp["virtual_sites3"][bead_id]["func"] == 1:
                        vsf.vs3_func_1(aa2cg_universe, traj, vs_def_beads_ids, vs_params)
                    elif cg_itp["virtual_sites3"][bead_id]["func"] == 2:
                        vsf.vs3_func_2(aa2cg_universe, traj, vs_def_beads_ids, vs_params)
                    elif cg_itp["virtual_sites3"][bead_id]["func"] == 3:
                        vsf.vs3_func_3(aa2cg_universe, traj, vs_def_beads_ids, vs_params)
                    elif cg_itp["virtual_sites3"][bead_id]["func"] == 4:
                        vsf.vs3_func_4(aa2cg_universe, traj, vs_def_beads_ids, vs_params)

                # VS Type 4
                if cg_itp["atoms"][bead_id]["vs_type"] == 4:
                    vs_def_beads_ids = cg_itp["virtual_sites4"][bead_id]["vs_def_beads_ids"]
                    vs_params = cg_itp["virtual_sites4"][bead_id]["vs_params"]

                    if cg_itp["virtual_sites4"][bead_id]["func"] == 2:
                        vsf.vs4_func_2(aa2cg_universe, traj, vs_def_beads_ids, vs_params)

                # VS Type n
                if cg_itp["atoms"][bead_id]["vs_type"] == "n":
                    vs_def_beads_ids = cg_itp["virtual_sitesn"][bead_id]["vs_def_beads_ids"]
                    vs_params = cg_itp["virtual_sitesn"][bead_id]["vs_params"]

                    if cg_itp["virtual_sitesn"][bead_id]["func"] == 1:
                        vsf.vsn_func_1(aa2cg_universe, traj, vs_def_beads_ids)
                    elif cg_itp["virtual_sitesn"][bead_id]["func"] == 2:
                        vsf.vsn_func_2(aa2cg_universe, traj, vs_def_beads_ids, bead_id, cg_itp=cg_itp)
                    elif cg_itp["virtual_sitesn"][bead_id]["func"] == 3:
                        vsf.vsn_func_3(aa2cg_universe, traj, vs_def_beads_ids, vs_params)

                coord[:, bead_id, :] = traj
        
        aa2cg_universe.load_new(coord, format=mda.coordinates.memory.MemoryReader)

def initialize_cg_traj(cg_itp):
    """Initialize cg trajectory universe object."""
    masses = np.array([val["mass"] for val in cg_itp["atoms"]])
    names = np.array([val["atom"] for val in cg_itp["atoms"]])
    resnames = np.array([val["residue"] for val in cg_itp["atoms"]])
    resid = np.array([val["resnr"] for val in cg_itp["atoms"]])
    nr = len(set([val["resnr"] for val in cg_itp["atoms"]]))

    aa2cg_universe = mda.Universe.empty(len(cg_itp["atoms"]), n_residues=nr, atom_resindex=resid, n_segments=1,
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
