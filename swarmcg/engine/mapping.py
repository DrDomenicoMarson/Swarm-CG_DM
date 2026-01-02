import re
import collections

import numpy as np
import MDAnalysis as mda

import swarmcg.simulations.vs_functions as vsf
from swarmcg.context import SwarmCGArgs, SwarmCGState
from swarmcg.shared import exceptions


def load_aa_data(args: SwarmCGArgs, state: SwarmCGState):
    """Read one or more molecules from the AA TPR and trajectory.

    state requires:
        traj.aa_universe

    state creates:
        mapping.all_atoms
        mapping.all_aa_mols
    """
    state.mapping.all_atoms = dict()  # atom centered connectivity + atom type + heavy atom boolean + bead(s) to which the atom belongs (can belong to multiple beads depending on mapping)
    state.mapping.all_aa_mols = []  # atom groups for each molecule of interest, in case we use several and average the distributions across many molecules, as we would do for membranes analysis

    if state.mapping.molname_in == None:

        molname_atom_group = state.traj.aa_universe.atoms[
            0].fragment  # select the AA connected graph for the first moltype found in TPR
        state.mapping.all_aa_mols.append(molname_atom_group)

        # atoms and their attributes
        for atom in molname_atom_group:
            atom_id = atom.id
            atom_type = atom.type
            atom_charge = atom.charge
            atom_heavy = not atom_type.upper().startswith("H")

            state.mapping.all_atoms[atom_id] = {
                "conn": set(),
                "atom_type": atom_type,
                "atom_charge": atom_charge,
                "heavy": atom_heavy,
                "beads_ids": set(),
                "beads_types": set(),
                "residue_names": set(),
            }

    # TODO: allow reading multiple instances of a molecule to build the reference distributions,
    #       for extended usage with NOT just one flexible molecule in solvent
    else:
        pass


def read_ndx_atoms2beads(args: SwarmCGArgs, state: SwarmCGState):
    """Load CG beads from NDX-like file.

    args requires:
        inputs.cg_map_filename

    state creates:
        mapping.atoms_occ_total
        mapping.all_beads
    """
    with open(args.inputs.cg_map_filename, "r") as fp:

        ndx_lines = fp.read().split("\n")
        ndx_lines = [ndx_line.strip().split(";")[0] for ndx_line in ndx_lines]  # split for comments

        state.mapping.atoms_occ_total = collections.Counter()
        state.mapping.all_beads = dict()  # atoms id mapped to each bead
        bead_id = 0
        current_section = "Beginning of file"

        for i in range(len(ndx_lines)):
            ndx_line = ndx_lines[i]
            if ndx_line != "":

                if bool(re.search("\[.*\]", ndx_line)):
                    current_section = ndx_line
                    state.mapping.all_beads[bead_id] = {"atoms_id": [], "section": current_section, "line_nb": i + 1}
                    current_bead_id = bead_id
                    bead_id += 1

                else:
                    try:
                        bead_atoms_id = [int(atom_id) - 1 for atom_id in
                                         ndx_line.split()]  # retrieve indexing from 0 for atoms IDs for MDAnalysis
                        state.mapping.all_beads[current_bead_id]["atoms_id"].extend(
                            bead_atoms_id)  # all atoms included in current bead

                        for atom_id in bead_atoms_id:  # bead to which each atom belongs (one atom can belong to multiple beads if there is split-mapping)
                            state.mapping.atoms_occ_total[atom_id] += 1

                    except NameError:
                        msg = (
                            "The CG beads mapping (NDX) file does NOT seem to contain CG beads "
                            "sections.\nPlease verify the input mapping. The expected format is "
                            "Gromacs NDX."
                        )
                        raise exceptions.MissformattedFile(msg)

                    except ValueError:  # non-integer atom ID provided
                        msg = (
                            f"Incorrect reading of the sections content in the CG beads mapping "
                            f"(NDX) file.\nFound non-integer values for some IDs at line "
                            f"{str(i + 1)} under section {current_section}."
                        )
                        raise exceptions.MissformattedFile(msg)

    for bead_id in state.mapping.all_beads:
        if len(state.mapping.all_beads[bead_id]["atoms_id"]) == 0:
            msg = (
                f"The ITP file contains an empty section named {state.mapping.all_beads[bead_id]['section']} starting at line {state.mapping.all_beads[bead_id]['line_nb']}."
                f"Empty sections are NOT allowed, please fill or delete it."
            )
            raise exceptions.MissformattedFile(msg)


def get_atoms_weights_in_beads(args: SwarmCGArgs, state: SwarmCGState):
    """Calculate weight ratio of atom ID in given CG bead.

    This is for splitting atom weight in case an atom is mapped to several CG beads.

    args requires:
        runtime.verbose
        inputs.mapping_type

    state requires:
        mapping.all_beads
        mapping.atoms_occ_total

    state creates:
        mapping.atom_w
    """
    state.mapping.atom_w = dict()
    if args.runtime.verbose:
        print("Calculating atoms weights ratio within mapped CG beads")
    for bead_id in state.mapping.all_beads:
        # print("Weighting bead_id", bead_id)
        state.mapping.atom_w[bead_id] = dict()
        beads_atoms_counts = collections.Counter(state.mapping.all_beads[bead_id]["atoms_id"])
        for atom_id in beads_atoms_counts:
            state.mapping.atom_w[bead_id][atom_id] = round(beads_atoms_counts[atom_id] / state.mapping.atoms_occ_total[atom_id], 3)
            if args.runtime.verbose and args.inputs.mapping_type == "COM":
                print("  CG bead ID", bead_id + 1, "-- Atom ID", atom_id + 1, "has weight ratio =",
                      state.mapping.atom_w[bead_id][atom_id])
    if args.runtime.verbose:
        print()


def get_beads_MDA_atomgroups(args: SwarmCGArgs, state: SwarmCGState):
    """For each CG bead, create atom groups for trajectory geoms calculation using mass and atom
    weights across beads.

    args requires:
        inputs.mapping_type

    state requires:
        mapping.atom_w
        traj.aa_universe

    state creates:
        mapping.mda_beads_atom_grps
        mapping.mda_weights_atom_grps
    """
    state.mapping.mda_beads_atom_grps, state.mapping.mda_weights_atom_grps = dict(), dict()
    for bead_id in state.mapping.atom_w:
        try:
            # print("Created bead_id", bead_id, "using atoms", [atom_id for atom_id in state.mapping.atom_w[bead_id]])
            if args.inputs.mapping_type == "COM":
                state.mapping.mda_beads_atom_grps[bead_id] = mda.AtomGroup([atom_id for atom_id in state.mapping.atom_w[bead_id]],
                                                                state.traj.aa_universe)
                state.mapping.mda_weights_atom_grps[bead_id] = np.array(
                    [state.mapping.atom_w[bead_id][atom_id] * state.traj.aa_universe.atoms[atom_id].mass for atom_id in
                     state.mapping.atom_w[bead_id]])
            elif args.inputs.mapping_type == "COG":
                state.mapping.mda_beads_atom_grps[bead_id] = mda.AtomGroup([atom_id for atom_id in state.mapping.atom_w[bead_id]],
                                                                state.traj.aa_universe)
                state.mapping.mda_weights_atom_grps[bead_id] = np.array([1 for _ in state.mapping.atom_w[bead_id]])

        except IndexError as e:
            msg = (
                f"An ID present in your mapping (NDX) file could not be found in the AA trajectory. "
                f"Please check your mapping (NDX) file.\nSee the error below to understand which "
                f"ID (here 0-indexed) could not be found:\n\n{str(e)}"
            )
            raise exceptions.MissformattedFile(msg)


def initialize_cg_traj(cg_itp):
    """Initialize cg trajectory universe object."""
    masses = np.array([val["mass"] for val in cg_itp["atoms"]])
    names = np.array([val["atom"] for val in cg_itp["atoms"]])
    resnames = np.array([val["residue"] for val in cg_itp["atoms"]])
    resid = np.array([val["resnr"] for val in cg_itp["atoms"]])
    nr = len(set([val["resnr"] for val in cg_itp["atoms"]]))

    aa2cg_universe = mda.Universe.empty(
        len(cg_itp["atoms"]),
        n_residues=nr,
        atom_resindex=resid,
        n_segments=1,
        residue_segindex=np.ones(nr, dtype=int),
        trajectory=True,
    )
    aa2cg_universe.add_TopologyAttr("masses")
    aa2cg_universe._topology.masses.values = np.array(masses)
    aa2cg_universe.add_TopologyAttr("names")
    aa2cg_universe._topology.names.values = names
    aa2cg_universe.add_TopologyAttr("resnames")
    aa2cg_universe._topology.resnames.values = resnames
    return aa2cg_universe


def map_aa2cg_traj(args: SwarmCGArgs, state: SwarmCGState):
    """Initialize cg trajectory universe object.

    args requires:
        inputs.mapping_type

    state requires:
        model.cg_itp
        mapping.mda_beads_atom_grps
        mapping.mda_weights_atom_grps
        traj.aa_universe
        traj.aa2cg_universe (edited inplace)

    state creates:
        traj.aa2cg_universe

    pass state to:
        vsn_func_1
        vsn_func_2
        vsn_func_3
        vs2_func_1
        vs2_func_2
        vs3_func_1
        vs3_func_2
        vs3_func_3
        vs3_func_4
        vs4_func_2
    """
    if args.inputs.mapping_type == "COM":
        print("  Interpretation: Center of Mass (COM)")
    elif args.inputs.mapping_type == "COG":
        print("  Interpretation: Center of Geometry (COG)")

    # regular beads are mapped using center of mass of groups of atoms
    coord = np.empty((len(state.traj.aa_universe.trajectory), len(state.model.cg_itp["atoms"]), 3))
    for bead_id in range(len(state.model.cg_itp["atoms"])):
        if not state.model.cg_itp["atoms"][bead_id]["bead_type"].startswith("v"):  # bead is NOT a virtual site
            traj = np.empty((len(state.traj.aa_universe.trajectory), 3))
            for ts in state.traj.aa_universe.trajectory:
                traj[ts.frame] = state.mapping.mda_beads_atom_grps[bead_id].center(
                    state.mapping.mda_weights_atom_grps[bead_id], pbc=None, compound="group"
                )  # no need for PBC handling, trajectories were made wholes for the molecule
            coord[:, bead_id, :] = traj

    state.traj.aa2cg_universe.load_new(coord, format=mda.coordinates.memory.MemoryReader)

    # virtual sites are mapped using previously defined regular beads positions and appropriate virtual sites functions
    # it is also possible to use a VS for defining another VS position, if the VS used for definition is defined before
    # no need to check if the functions used for VS definition are correct here, this has been done already
    for bead_id in range(len(state.model.cg_itp["atoms"])):
        if state.model.cg_itp["atoms"][bead_id]["bead_type"].startswith("v"):

            traj = np.empty((len(state.traj.aa2cg_universe.trajectory), 3))

            if state.model.cg_itp["atoms"][bead_id]["vs_type"] == 2:
                vs_def_beads_ids = state.model.cg_itp["virtual_sites2"][bead_id]["vs_def_beads_ids"]
                vs_params = state.model.cg_itp["virtual_sites2"][bead_id]["vs_params"]

                if state.model.cg_itp["virtual_sites2"][bead_id]["func"] == 1:
                    vsf.vs2_func_1(state, traj, vs_def_beads_ids, vs_params)
                elif state.model.cg_itp["virtual_sites2"][bead_id]["func"] == 2:
                    vsf.vs2_func_2(state, traj, vs_def_beads_ids, vs_params)

            if state.model.cg_itp["atoms"][bead_id]["vs_type"] == 3:
                vs_def_beads_ids = state.model.cg_itp["virtual_sites3"][bead_id]["vs_def_beads_ids"]
                vs_params = state.model.cg_itp["virtual_sites3"][bead_id]["vs_params"]

                if state.model.cg_itp["virtual_sites3"][bead_id]["func"] == 1:
                    vsf.vs3_func_1(state, traj, vs_def_beads_ids, vs_params)
                elif state.model.cg_itp["virtual_sites3"][bead_id]["func"] == 2:
                    vsf.vs3_func_2(state, traj, vs_def_beads_ids, vs_params)
                elif state.model.cg_itp["virtual_sites3"][bead_id]["func"] == 3:
                    vsf.vs3_func_3(state, traj, vs_def_beads_ids, vs_params)
                elif state.model.cg_itp["virtual_sites3"][bead_id]["func"] == 4:
                    vsf.vs3_func_4(state, traj, vs_def_beads_ids, vs_params)

            # here it"s normal there is only function 2, that"s the only one that exists in gromacs for some reason
            if state.model.cg_itp["atoms"][bead_id]["vs_type"] == 4:
                vs_def_beads_ids = state.model.cg_itp["virtual_sites4"][bead_id]["vs_def_beads_ids"]
                vs_params = state.model.cg_itp["virtual_sites4"][bead_id]["vs_params"]

                if state.model.cg_itp["virtual_sites4"][bead_id]["func"] == 2:
                    vsf.vs4_func_2(state, traj, vs_def_beads_ids, vs_params)

            if state.model.cg_itp["atoms"][bead_id]["vs_type"] == "n":
                vs_def_beads_ids = state.model.cg_itp["virtual_sitesn"][bead_id]["vs_def_beads_ids"]
                vs_params = state.model.cg_itp["virtual_sitesn"][bead_id]["vs_params"]

                if state.model.cg_itp["virtual_sitesn"][bead_id]["func"] == 1:
                    vsf.vsn_func_1(state, traj, vs_def_beads_ids)
                elif state.model.cg_itp["virtual_sitesn"][bead_id]["func"] == 2:
                    vsf.vsn_func_2(state, traj, vs_def_beads_ids, bead_id)
                elif state.model.cg_itp["virtual_sitesn"][bead_id]["func"] == 3:
                    vsf.vsn_func_3(state, traj, vs_def_beads_ids, vs_params)

            coord[:, bead_id, :] = traj

    state.traj.aa2cg_universe.load_new(coord, format=mda.coordinates.memory.MemoryReader)


def make_aa_traj_whole_for_selected_mols(args: SwarmCGArgs, state: SwarmCGState):
    """Use selected whole molecules as MDA atomgroups and make their coordinates whole, inplace,
    across the complete AA trajectory

    state requires:
        traj.aa_universe (edited inplace)
    """
    # TODO: add an option to NOT read the PBC in case user would feed a trajectory that is already unwrapped for
    #       molecule and their trajectory does NOT contain box dimensions (universe.dimensions)
    #       (this was an issue I encountered with Davide B3T traj GRO)
    for _ in state.traj.aa_universe.trajectory:
        for aa_mol in state.mapping.all_aa_mols:
            mda.lib.mdamath.make_whole(aa_mol, inplace=True)
