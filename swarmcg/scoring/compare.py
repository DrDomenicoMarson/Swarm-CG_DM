import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import MDAnalysis as mda
from pyemd import emd

import swarmcg.scoring as scores
from swarmcg.config_types import SwarmConfig
from swarmcg.context import OptimizationContext
from swarmcg import config
from swarmcg.shared import styling

# Use the Anti-Grain Geometry non-interactive backend suited for scripted PNG creation
matplotlib.use("AGG")

def compare_models(context: OptimizationContext, manual_mode: bool = True, ignore_dihedrals: bool = False, 
                  calc_sasa: bool = False, record_best_indep_params: bool = False):
    """
    Compare 2 models -- atomistic and CG models with plotting.
    
    Args:
        context (OptimizationContext): The optimization context containing configuration and state.
        manual_mode (bool): Whether running in manual mode (scg_evaluate) or optimization mode.
        ignore_dihedrals (bool): If True, exclude dihedrals from scoring.
        calc_sasa (bool): Whether to calculate and compare SASA.
        record_best_indep_params (bool): Use independent parameter recording (optimization feature).
    """
    ns = context # Alias for backward compatibility during refactoring
    
    # graphical parameters
    plt.rcParams["grid.color"] = "k"  # plt grid appearance settings
    plt.rcParams["grid.linestyle"] = ":"
    plt.rcParams["grid.linewidth"] = 0.5

    row_wise_ranges = {}
    row_wise_ranges["max_range_constraints"] = 0 
    row_wise_ranges["max_range_bonds"] = 0
    row_wise_ranges["max_range_angles"] = 0
    row_wise_ranges["max_range_dihedrals"] = 0

    if ns.scoring.atom_only:
        # scores.compute_Rg(ns, traj_type="AA")
        ns.scoring.gyr_aa, ns.scoring.gyr_aa_std = scores.compute_Rg(ns.scoring.aa_universe, ns.scoring.aa_universe.atoms[:len(ns.scoring.all_atoms)], backend=ns.scoring.mda_backend)
        print("Radius of gyration (AA reference, NOT CG-mapped):", ns.scoring.gyr_aa, "nm")

    # proceed with CG data
    if not ns.scoring.atom_only:
        config_obj = ns.config if ns.config else SwarmConfig.from_namespace(ns) # Fallback for safety

        print("Reading CG trajectory")
        ns.scoring.cg_universe = mda.Universe(ns.files.cg_tpr_filename, ns.files.cg_traj_filename, in_memory=True, refresh_offsets=True,
                                      guess_bonds=False)
        print("  Found", len(ns.scoring.cg_universe.trajectory), "frames")

        if manual_mode:
            # here we read the CG beads masses + actualize the mapped trajectory object
            for bead_id in range(len(ns.cg_itp["atoms"])):
                ns.cg_itp["atoms"][bead_id]["mass"] = ns.scoring.cg_universe.atoms[bead_id].mass
            masses = np.array([val["mass"] for val in ns.cg_itp["atoms"]])
            ns.scoring.aa2cg_universe._topology.masses.values = np.array(masses)

        # create fake bonds in the CG MDA universe, that will be used only for making the molecule whole
        # we make bonds between each VS and their beads definition, so we retrieve the connectivity
        # iteratively towards the real CG beads, that are all connected
        if len(ns.cg_itp["vs_beads_ids"]) > 0:
            fake_bonds = []
            for vs_type in ["2", "3", "4", "n"]:
                try:
                    for bead_id in ns.cg_itp["virtual_sites" + vs_type]:
                        for vs_def_bead_id in ns.cg_itp["virtual_sites" + vs_type][bead_id]["vs_def_beads_ids"]:
                            fake_bonds.append([bead_id, vs_def_bead_id])
                except (IndexError, ValueError):
                    pass
            ns.scoring.cg_universe.add_bonds(fake_bonds, guessed=False)

        # select the whole molecule as an MDA atomgroup and make its coordinates whole, inplace, across the complete trajectory
        ag_mol = mda.AtomGroup([bead_id for bead_id in range(len(ns.cg_itp["atoms"]))], ns.scoring.cg_universe)
        for _ in ns.scoring.cg_universe.trajectory:
            mda.lib.mdamath.make_whole(ag_mol, inplace=True)

        # this requires CG data for mapping -- especially, masses are taken from the CG TPR but the CG ITP is also used atm
        if ns.results.gyr_aa_mapped is None:
            # scores.compute_Rg(ns, traj_type="AA_mapped")
            ns.results.gyr_aa_mapped, ns.results.gyr_aa_mapped_std = scores.compute_Rg(ns.scoring.aa2cg_universe, ns.scoring.aa2cg_universe.atoms[:len(ns.cg_itp["atoms"])], backend=ns.scoring.mda_backend, offset=ns.config.reference.aa_rg_offset)
            print()
            print("Radius of gyration (AA reference, CG-mapped, no bonds scaling):", ns.results.gyr_aa_mapped, "+/-",
                  ns.results.gyr_aa_mapped_std, "nm")

        # scores.compute_Rg(ns, traj_type="CG")
        ns.results.gyr_cg, ns.results.gyr_cg_std = scores.compute_Rg(ns.scoring.cg_universe, ns.scoring.cg_universe.atoms[:len(ns.cg_itp["atoms"])], backend=ns.scoring.mda_backend)
        print("Radius of gyration (CG model):", ns.results.gyr_cg, "+/-", ns.results.gyr_cg_std, "nm")



        if calc_sasa:
            if ns.results.sasa_aa_mapped is None:
                # scores.compute_SASA(ns, traj_type="AA_mapped")
                ns.results.sasa_aa_mapped, ns.results.sasa_aa_mapped_std = scores.compute_SASA(config_obj, ns.cg_itp, traj_type="AA_mapped")

            # scores.compute_SASA(ns, traj_type="CG")
            ns.results.sasa_cg, ns.results.sasa_cg_std = scores.compute_SASA(config_obj, ns.cg_itp, traj_type="CG")
            print()

            # this line checks that gmx trjconv could read the md.xtc trajectory from the opti
            if ns.results.sasa_cg is None:
                return 0, 0, 0, 0, 0, None  # ns.sasa_cg == None will be checked in eval_function and worst score will be attributed

    print()
    print(styling.sep_close, flush=True)
    print("| SCORING AND PLOTTING                                                                        |", flush=True)
    print(styling.sep_close, flush=True)
    print()

    # constraints
    print("Processing constraints ...", flush=True)
    diff_ordered_grp_constraints = list(range(ns.cg_itp["nb_constraints"]))
    avg_diff_grp_constraints, row_wise_ranges["constraints"] = [], {}
    constraints = {}

    for grp_constraint in range(ns.cg_itp["nb_constraints"]):

        constraints[grp_constraint] = {"AA": {"x": [], "y": []}, "CG": {"x": [], "y": []}}

        if manual_mode:
            constraint_avg, constraint_hist, _ = scores.get_AA_bonds_distrib(ns.scoring.aa2cg_universe, beads_ids=ns.cg_itp["constraint"][grp_constraint]["beads"], grp_type="constraints group", grp_nb=grp_constraint, config=config_obj if 'config_obj' in locals() else SwarmConfig.from_namespace(ns), bins=ns.scoring.bins_constraints, bandwidth=ns.bw_constraints)
            constraints[grp_constraint]["AA"]["avg"] = constraint_avg
            constraints[grp_constraint]["AA"]["hist"] = constraint_hist
        else:  # use atomistic reference that was loaded by the optimization routines
            constraints[grp_constraint]["AA"]["avg"] = ns.cg_itp["constraint"][grp_constraint]["avg"]
            constraints[grp_constraint]["AA"]["hist"] = ns.cg_itp["constraint"][grp_constraint]["hist"]

        for i in range(1, len(constraints[grp_constraint]["AA"]["hist"]) - 1):
            if constraints[grp_constraint]["AA"]["hist"][i - 1] > 0 or constraints[grp_constraint]["AA"]["hist"][
                i] > 0 or constraints[grp_constraint]["AA"]["hist"][i + 1] > 0:
                constraints[grp_constraint]["AA"]["x"].append(np.mean(ns.scoring.bins_constraints[i:i + 2]))
                constraints[grp_constraint]["AA"]["y"].append(constraints[grp_constraint]["AA"]["hist"][i])

        if not ns.scoring.atom_only:
            try:
                constraint_avg, constraint_hist, _ = scores.get_CG_bonds_distrib(ns.scoring.cg_universe, beads_ids=ns.cg_itp["constraint"][grp_constraint]["beads"], grp_type="constraint", bins=ns.scoring.bins_constraints, bandwidth=ns.bw_constraints)
                constraints[grp_constraint]["CG"]["avg"] = constraint_avg
                constraints[grp_constraint]["CG"]["hist"] = constraint_hist

                for i in range(1, len(constraint_hist) - 1):
                    if constraint_hist[i - 1] > 0 or constraint_hist[i] > 0 or constraint_hist[
                        i + 1] > 0: 
                        constraints[grp_constraint]["CG"]["x"].append(np.mean(ns.scoring.bins_constraints[i:i + 2]))
                        constraints[grp_constraint]["CG"]["y"].append(constraint_hist[i])

                domain_min = min(constraints[grp_constraint]["AA"]["x"][0], constraints[grp_constraint]["CG"]["x"][0])
                domain_max = max(constraints[grp_constraint]["AA"]["x"][-1], constraints[grp_constraint]["CG"]["x"][-1])
                avg_diff_grp_constraints.append(
                    emd(constraints[grp_constraint]["AA"]["hist"], constraints[grp_constraint]["CG"]["hist"],
                        ns.scoring.bins_constraints_dist_matrix) * ns.bonds2angles_scoring_factor)
            except IndexError:
                msg = (
                    f"Most probably because you have bonds or constraints that "
                    f"exceed {ns.bonded_max_range} nm.\nIncrease bins range for bonds and "
                    f"constraints and retry!\nSee argument -bonds_max_range."
                )
                raise ValueError(msg)
        else:
            avg_diff_grp_constraints.append(constraints[grp_constraint]["AA"]["avg"])

        if ns.scoring.row_x_scaling:
            if ns.scoring.atom_only:
                row_wise_ranges["constraints"][grp_constraint] = [constraints[grp_constraint]["AA"]["x"][0],
                                                                  constraints[grp_constraint]["AA"]["x"][-1]]
            else:
                row_wise_ranges["constraints"][grp_constraint] = [domain_min, domain_max]
            if row_wise_ranges["constraints"][grp_constraint][1] - row_wise_ranges["constraints"][grp_constraint][0] > \
                    row_wise_ranges["max_range_constraints"]:
                row_wise_ranges["max_range_constraints"] = row_wise_ranges["constraints"][grp_constraint][1] - \
                                                           row_wise_ranges["constraints"][grp_constraint][0]

    # constraint groups ordered by mean difference between atomistic-mapped and CG models
    if ns.scoring.mismatch_order and not ns.scoring.atom_only:
        diff_ordered_grp_constraints = [x for _, x in
                                        sorted(zip(avg_diff_grp_constraints, diff_ordered_grp_constraints),
                                               key=lambda pair: pair[0], reverse=True)]

    # bonds
    print("Processing bonds ...", flush=True)
    diff_ordered_grp_bonds = list(range(ns.cg_itp["nb_bonds"]))
    avg_diff_grp_bonds, row_wise_ranges["bonds"] = [], {}
    bonds = {}

    for grp_bond in range(ns.cg_itp["nb_bonds"]):

        bonds[grp_bond] = {"AA": {"x": [], "y": []}, "CG": {"x": [], "y": []}}

        if manual_mode:
            bond_avg, bond_hist, _ = scores.get_AA_bonds_distrib(ns.scoring.aa2cg_universe, beads_ids=ns.cg_itp["bond"][grp_bond]["beads"], grp_type="bonds group", grp_nb=grp_bond, config=config_obj if 'config_obj' in locals() else SwarmConfig.from_namespace(ns), bins=ns.scoring.bins_bonds, bandwidth=ns.bw_bonds)
            bonds[grp_bond]["AA"]["avg"] = bond_avg
            bonds[grp_bond]["AA"]["hist"] = bond_hist
        else:  # use atomistic reference that was loaded by the optimization routines
            bonds[grp_bond]["AA"]["avg"] = ns.cg_itp["bond"][grp_bond]["avg"]
            bonds[grp_bond]["AA"]["hist"] = ns.cg_itp["bond"][grp_bond]["hist"]

        for i in range(1, len(bonds[grp_bond]["AA"]["hist"]) - 1):
            if bonds[grp_bond]["AA"]["hist"][i - 1] > 0 or bonds[grp_bond]["AA"]["hist"][i] > 0 or \
                    bonds[grp_bond]["AA"]["hist"][i + 1] > 0:
                bonds[grp_bond]["AA"]["x"].append(np.mean(ns.scoring.bins_bonds[i:i + 2]))
                bonds[grp_bond]["AA"]["y"].append(bonds[grp_bond]["AA"]["hist"][i])

        if not ns.scoring.atom_only:
            try:
                bond_avg, bond_hist, _ = scores.get_CG_bonds_distrib(ns.scoring.cg_universe, beads_ids=ns.cg_itp["bond"][grp_bond]["beads"], grp_type="bond", bins=ns.scoring.bins_bonds, bandwidth=ns.bw_bonds)
                bonds[grp_bond]["CG"]["avg"] = bond_avg
                bonds[grp_bond]["CG"]["hist"] = bond_hist

                for i in range(1, len(bond_hist) - 1):
                    if bond_hist[i - 1] > 0 or bond_hist[i] > 0 or bond_hist[i + 1] > 0:
                        bonds[grp_bond]["CG"]["x"].append(np.mean(ns.scoring.bins_bonds[i:i + 2]))
                        bonds[grp_bond]["CG"]["y"].append(bond_hist[i])

                domain_min = min(bonds[grp_bond]["AA"]["x"][0], bonds[grp_bond]["CG"]["x"][0])
                domain_max = max(bonds[grp_bond]["AA"]["x"][-1], bonds[grp_bond]["CG"]["x"][-1])
                avg_diff_grp_bonds.append(emd(bonds[grp_bond]["AA"]["hist"], bonds[grp_bond]["CG"]["hist"],
                                              ns.scoring.bins_bonds_dist_matrix) * ns.bonds2angles_scoring_factor)
            except IndexError:
                msg = (
                    f"Most probably because you have bonds or constraints that "
                    f"exceed {ns.bonded_max_range} nm.\nIncrease bins range for bonds and "
                    f"constraints and retry!\nSee argument -bonds_max_range."
                )
                raise ValueError(msg)
        else:
            avg_diff_grp_bonds.append(bonds[grp_bond]["AA"]["avg"])

        if ns.scoring.row_x_scaling:
            if ns.scoring.atom_only:
                row_wise_ranges["bonds"][grp_bond] = [bonds[grp_bond]["AA"]["x"][0], bonds[grp_bond]["AA"]["x"][-1]]
            else:
                row_wise_ranges["bonds"][grp_bond] = [domain_min, domain_max]
            if row_wise_ranges["bonds"][grp_bond][1] - row_wise_ranges["bonds"][grp_bond][0] > row_wise_ranges[
                "max_range_bonds"]:
                row_wise_ranges["max_range_bonds"] = row_wise_ranges["bonds"][grp_bond][1] - \
                                                     row_wise_ranges["bonds"][grp_bond][0]

    # bond groups ordered by mean difference between atomistic-mapped and CG models
    if ns.scoring.mismatch_order and not ns.scoring.atom_only:
        diff_ordered_grp_bonds = [x for _, x in
                                  sorted(zip(avg_diff_grp_bonds, diff_ordered_grp_bonds), key=lambda pair: pair[0],
                                         reverse=True)]

    # angles
    print("Processing angles ...", flush=True)
    diff_ordered_grp_angles = list(range(ns.cg_itp["nb_angles"]))
    avg_diff_grp_angles, row_wise_ranges["angles"] = [], {}
    angles = {}

    for grp_angle in range(ns.cg_itp["nb_angles"]):

        angles[grp_angle] = {"AA": {"x": [], "y": []}, "CG": {"x": [], "y": []}}

        if manual_mode:
            angle_avg, angle_hist, _, _ = scores.get_AA_angles_distrib(ns.scoring.aa2cg_universe, beads_ids=ns.cg_itp["angle"][grp_angle]["beads"], bins=ns.scoring.bins_angles, bandwidth=ns.bw_angles)
            angles[grp_angle]["AA"]["avg"] = angle_avg
            angles[grp_angle]["AA"]["hist"] = angle_hist
        else:  # use atomistic reference that was loaded by the optimization routines
            angles[grp_angle]["AA"]["avg"] = ns.cg_itp["angle"][grp_angle]["avg"]
            angles[grp_angle]["AA"]["hist"] = ns.cg_itp["angle"][grp_angle]["hist"]

        for i in range(1, len(angles[grp_angle]["AA"]["hist"]) - 1):
            if angles[grp_angle]["AA"]["hist"][i - 1] > 0 or angles[grp_angle]["AA"]["hist"][i] > 0 or \
                    angles[grp_angle]["AA"]["hist"][i + 1] > 0:
                angles[grp_angle]["AA"]["x"].append(np.mean(ns.scoring.bins_angles[i:i + 2]))
                angles[grp_angle]["AA"]["y"].append(angles[grp_angle]["AA"]["hist"][i])

        if not ns.scoring.atom_only:
            angle_avg, angle_hist, _, _ = scores.get_CG_angles_distrib(ns.scoring.cg_universe, beads_ids=ns.cg_itp["angle"][grp_angle]["beads"], bins=ns.scoring.bins_angles, bandwidth=ns.bw_angles)
            angles[grp_angle]["CG"]["avg"] = angle_avg
            angles[grp_angle]["CG"]["hist"] = angle_hist

            for i in range(1, len(angle_hist) - 1):
                if angle_hist[i - 1] > 0 or angle_hist[i] > 0 or angle_hist[i + 1] > 0:
                    angles[grp_angle]["CG"]["x"].append(np.mean(ns.scoring.bins_angles[i:i + 2]))
                    angles[grp_angle]["CG"]["y"].append(angle_hist[i])

            domain_min = min(angles[grp_angle]["AA"]["x"][0], angles[grp_angle]["CG"]["x"][0])
            domain_max = max(angles[grp_angle]["AA"]["x"][-1], angles[grp_angle]["CG"]["x"][-1])
            avg_diff_grp_angles.append(
                emd(angles[grp_angle]["AA"]["hist"], angles[grp_angle]["CG"]["hist"], ns.scoring.bins_angles_dist_matrix))
        else:
            avg_diff_grp_angles.append(angles[grp_angle]["AA"]["avg"])

        if ns.scoring.row_x_scaling:
            if ns.scoring.atom_only:
                row_wise_ranges["angles"][grp_angle] = [angles[grp_angle]["AA"]["x"][0],
                                                        angles[grp_angle]["AA"]["x"][-1]]
            else:
                row_wise_ranges["angles"][grp_angle] = [domain_min, domain_max]
            if row_wise_ranges["angles"][grp_angle][1] - row_wise_ranges["angles"][grp_angle][0] > row_wise_ranges[
                "max_range_angles"]:
                row_wise_ranges["max_range_angles"] = row_wise_ranges["angles"][grp_angle][1] - \
                                                      row_wise_ranges["angles"][grp_angle][0]

    # angle groups ordered by mean difference between atomistic-mapped and CG models
    if ns.scoring.mismatch_order and not ns.scoring.atom_only:
        diff_ordered_grp_angles = [x for _, x in
                                   sorted(zip(avg_diff_grp_angles, diff_ordered_grp_angles), key=lambda pair: pair[0],
                                          reverse=True)]

    # dihedrals
    print("Processing dihedrals ...", flush=True)
    diff_ordered_grp_dihedrals = list(range(ns.cg_itp["nb_dihedrals"]))
    avg_diff_grp_dihedrals, row_wise_ranges["dihedrals"] = [], {}
    dihedrals = {}

    for grp_dihedral in range(ns.cg_itp["nb_dihedrals"]):

        dihedrals[grp_dihedral] = {"AA": {"x": [], "y": []}, "CG": {"x": [], "y": []}}

        if manual_mode:
            dihedral_avg, dihedral_hist, _, _ = scores.get_AA_dihedrals_distrib(ns.scoring.aa2cg_universe, beads_ids=ns.cg_itp["dihedral"][grp_dihedral]["beads"], bins=ns.scoring.bins_dihedrals, bandwidth=ns.bw_dihedrals)
            dihedrals[grp_dihedral]["AA"]["avg"] = dihedral_avg
            dihedrals[grp_dihedral]["AA"]["hist"] = dihedral_hist
        else:  # use atomistic reference that was loaded by the optimization routines
            dihedrals[grp_dihedral]["AA"]["avg"] = ns.cg_itp["dihedral"][grp_dihedral]["avg"]
            dihedrals[grp_dihedral]["AA"]["hist"] = ns.cg_itp["dihedral"][grp_dihedral]["hist"]

        for i in range(1, len(dihedrals[grp_dihedral]["AA"]["hist"]) - 1):
            if dihedrals[grp_dihedral]["AA"]["hist"][i - 1] > 0 or dihedrals[grp_dihedral]["AA"]["hist"][i] > 0 or \
                    dihedrals[grp_dihedral]["AA"]["hist"][i + 1] > 0:
                dihedrals[grp_dihedral]["AA"]["x"].append(np.mean(ns.scoring.bins_dihedrals[i:i + 2]))
                dihedrals[grp_dihedral]["AA"]["y"].append(dihedrals[grp_dihedral]["AA"]["hist"][i])

        if not ns.scoring.atom_only:
            dihedral_avg, dihedral_hist, _, _ = scores.get_CG_dihedrals_distrib(ns.scoring.cg_universe, beads_ids=ns.cg_itp["dihedral"][grp_dihedral]["beads"], bins=ns.scoring.bins_dihedrals, bandwidth=ns.bw_dihedrals)
            dihedrals[grp_dihedral]["CG"]["avg"] = dihedral_avg
            dihedrals[grp_dihedral]["CG"]["hist"] = dihedral_hist

            for i in range(1, len(dihedral_hist) - 1):
                if dihedral_hist[i - 1] > 0 or dihedral_hist[i] > 0 or dihedral_hist[i + 1] > 0:
                    dihedrals[grp_dihedral]["CG"]["x"].append(np.mean(ns.scoring.bins_dihedrals[i:i + 2]))
                    dihedrals[grp_dihedral]["CG"]["y"].append(dihedral_hist[i])

            domain_min = min(dihedrals[grp_dihedral]["AA"]["x"][0], dihedrals[grp_dihedral]["CG"]["x"][0])
            domain_max = max(dihedrals[grp_dihedral]["AA"]["x"][-1], dihedrals[grp_dihedral]["CG"]["x"][-1])
            avg_diff_grp_dihedrals.append(
                emd(dihedrals[grp_dihedral]["AA"]["hist"], dihedrals[grp_dihedral]["CG"]["hist"],
                    ns.scoring.bins_dihedrals_dist_matrix))
        else:
            avg_diff_grp_dihedrals.append(dihedrals[grp_dihedral]["AA"]["avg"])

        if ns.scoring.row_x_scaling:
            if ns.scoring.atom_only:
                row_wise_ranges["dihedrals"][grp_dihedral] = [dihedrals[grp_dihedral]["AA"]["x"][0],
                                                              dihedrals[grp_dihedral]["AA"]["x"][-1]]
            else:
                row_wise_ranges["dihedrals"][grp_dihedral] = [domain_min, domain_max]
            if row_wise_ranges["dihedrals"][grp_dihedral][1] - row_wise_ranges["dihedrals"][grp_dihedral][0] > \
                    row_wise_ranges["max_range_dihedrals"]:
                row_wise_ranges["max_range_dihedrals"] = row_wise_ranges["dihedrals"][grp_dihedral][1] - \
                                                         row_wise_ranges["dihedrals"][grp_dihedral][0]

    # dihedral groups ordered by mean difference between atomistic-mapped and CG models
    if ns.scoring.mismatch_order and not ns.scoring.atom_only:
        diff_ordered_grp_dihedrals = [x for _, x in sorted(zip(avg_diff_grp_dihedrals, diff_ordered_grp_dihedrals),
                                                           key=lambda pair: pair[0], reverse=True)]

    ###############################
    # DISPLAY DISTRIBUTIONS PLOTS #
    ###############################

    larger_group = max(ns.cg_itp["nb_constraints"], ns.cg_itp["nb_bonds"], ns.cg_itp["nb_angles"],
                       ns.cg_itp["nb_dihedrals"])
    nrow, nrows, ncols = -1, 4, min(ns.scoring.ncols_max, larger_group)
    if ns.scoring.ncols_max == 0:
        ncols = larger_group
    if larger_group > ncols:
        hidden_cols = larger_group - ncols
        if ns.scoring.atom_only:
            print(
                f"Displaying max {ncols} distributions per row using the CG ITP file ordering of distributions groups ({hidden_cols} more are hidden)")
        else:
            if not ns.scoring.mismatch_order:
                print(
                    f"{styling.header_warning}Displaying max {ncols} distributions groups per row and this can be MISLEADING because ordering by pairwise AA-mapped vs. CG distributions mismatch is DISABLED ({hidden_cols} more are hidden)")
            else:
                print(
                    f"Displaying max {ncols} distributions groups per row ordered by pairwise AA-mapped vs. CG distributions difference ({hidden_cols} more are hidden)")
    else:
        print()
        if not ns.scoring.mismatch_order:
            print("Distributions groups will be displayed using the CG ITP file groups ordering")
        else:
            print(
                "Distributions groups will be displayed using ranked mismatch score between pairwise AA-mapped and CG distributions")
    nrows -= sum([ns.cg_itp["nb_constraints"] == 0, ns.cg_itp["nb_bonds"] == 0, ns.cg_itp["nb_angles"] == 0,
                  ns.cg_itp["nb_dihedrals"] == 0])

    fig = plt.figure(figsize=(ncols * 3, nrows * 3))
    ax = fig.subplots(nrows=nrows, ncols=ncols, squeeze=False)

    # record the min/max y for each geom type
    constraints_min_y, bonds_min_y, angles_min_y, dihedrals_min_y = 10, 10, 10, 10
    constraints_max_y, bonds_max_y, angles_max_y, dihedrals_max_y = 0, 0, 0, 0

    # constraints
    if ns.cg_itp["nb_constraints"] != 0:
        print()
        nrow += 1
        for i in range(ncols):
            if i < ns.cg_itp["nb_constraints"]:
                grp_constraint = diff_ordered_grp_constraints[i]
                if config.use_hists:
                    ax[nrow][i].step(constraints[grp_constraint]["AA"]["x"], constraints[grp_constraint]["AA"]["y"],
                                     label="AA-mapped", color=config.atom_color, where="mid", alpha=config.line_alpha)
                    ax[nrow][i].fill_between(constraints[grp_constraint]["AA"]["x"],
                                             constraints[grp_constraint]["AA"]["y"], color=config.atom_color,
                                             step="mid", alpha=config.fill_alpha)
                else:
                    ax[nrow][i].plot(constraints[grp_constraint]["AA"]["x"], constraints[grp_constraint]["AA"]["y"],
                                     label="AA-mapped", color=config.atom_color, alpha=config.line_alpha)
                    ax[nrow][i].fill_between(constraints[grp_constraint]["AA"]["x"],
                                             constraints[grp_constraint]["AA"]["y"], color=config.atom_color,
                                             alpha=config.fill_alpha)
                ax[nrow][i].plot(constraints[grp_constraint]["AA"]["avg"], 0, color=config.atom_color, marker="D")

                if not ns.scoring.atom_only:
                    ax[nrow][i].set_title(
                        f"Constraint grp {grp_constraint + 1} - EMD Δ {round(avg_diff_grp_constraints[grp_constraint], 3)}")
                    if config.use_hists:
                        ax[nrow][i].step(constraints[grp_constraint]["CG"]["x"], constraints[grp_constraint]["CG"]["y"],
                                         label="CG", color=config.cg_color, where="mid", alpha=config.line_alpha)
                        ax[nrow][i].fill_between(constraints[grp_constraint]["CG"]["x"],
                                                 constraints[grp_constraint]["CG"]["y"], color=config.cg_color,
                                                 step="mid", alpha=config.fill_alpha)
                    else:
                        ax[nrow][i].plot(constraints[grp_constraint]["CG"]["x"], constraints[grp_constraint]["CG"]["y"],
                                         label="CG", color=config.cg_color, alpha=config.line_alpha)
                        ax[nrow][i].fill_between(constraints[grp_constraint]["CG"]["x"],
                                                 constraints[grp_constraint]["CG"]["y"], color=config.cg_color,
                                                 alpha=config.fill_alpha)
                    ax[nrow][i].plot(constraints[grp_constraint]["CG"]["avg"], 0, color=config.cg_color, marker="D")
                    print(
                        f"Constraint {grp_constraint + 1} -- AA Avg: {round(constraints[grp_constraint]['AA']['avg'], 3)} nm -- CG Avg: {round(constraints[grp_constraint]['CG']['avg'], 3)}")
                else:
                    ax[nrow][i].set_title(
                        f"Constraint grp {grp_constraint + 1} - Avg {round(avg_diff_grp_constraints[grp_constraint], 3)} nm")
                    print(
                        f"Constraint {grp_constraint + 1} -- AA Avg: {round(constraints[grp_constraint]['AA']['avg'], 3)}")
                ax[nrow][i].grid(zorder=0.5)
                if ns.scoring.row_x_scaling:
                    ax[nrow][i].set_xlim(np.mean(row_wise_ranges["constraints"][grp_constraint]) - row_wise_ranges[
                        "max_range_constraints"] / 2 * 1.1,
                                         np.mean(row_wise_ranges["constraints"][grp_constraint]) + row_wise_ranges[
                                             "max_range_constraints"] / 2 * 1.1)
                if i % 2 == 0:
                    ax[nrow][i].legend(loc="upper left")
                if ax[nrow][i].get_ylim()[0] < constraints_min_y:
                    constraints_min_y = ax[nrow][i].get_ylim()[0]
                if ax[nrow][i].get_ylim()[1] > constraints_max_y:
                    constraints_max_y = ax[nrow][i].get_ylim()[1]
            else:
                ax[nrow][i].set_visible(False)

    # bonds
    if ns.cg_itp["nb_bonds"] != 0:
        print()
        nrow += 1
        for i in range(ncols):
            if i < ns.cg_itp["nb_bonds"]:
                grp_bond = diff_ordered_grp_bonds[i]
                if config.use_hists:
                    ax[nrow][i].step(bonds[grp_bond]["AA"]["x"], bonds[grp_bond]["AA"]["y"], label="AA-mapped",
                                     color=config.atom_color, where="mid", alpha=config.line_alpha)
                    ax[nrow][i].fill_between(bonds[grp_bond]["AA"]["x"], bonds[grp_bond]["AA"]["y"],
                                             color=config.atom_color, step="mid", alpha=config.fill_alpha)
                else:
                    ax[nrow][i].plot(bonds[grp_bond]["AA"]["x"], bonds[grp_bond]["AA"]["y"], label="AA-mapped",
                                     color=config.atom_color, alpha=config.line_alpha)
                    ax[nrow][i].fill_between(bonds[grp_bond]["AA"]["x"], bonds[grp_bond]["AA"]["y"],
                                             color=config.atom_color, alpha=config.fill_alpha)
                ax[nrow][i].plot(bonds[grp_bond]["AA"]["avg"], 0, color=config.atom_color, marker="D")

                if not ns.scoring.atom_only:
                    ax[nrow][i].set_title(f"Bond grp {grp_bond + 1} - EMD Δ {round(avg_diff_grp_bonds[grp_bond], 3)}")
                    if config.use_hists:
                        ax[nrow][i].step(bonds[grp_bond]["CG"]["x"], bonds[grp_bond]["CG"]["y"], label="CG",
                                         color=config.cg_color, where="mid", alpha=config.line_alpha)
                        ax[nrow][i].fill_between(bonds[grp_bond]["CG"]["x"], bonds[grp_bond]["CG"]["y"],
                                                 color=config.cg_color, step="mid", alpha=config.fill_alpha)
                    else:
                        ax[nrow][i].plot(bonds[grp_bond]["CG"]["x"], bonds[grp_bond]["CG"]["y"], label="CG",
                                         color=config.cg_color, alpha=config.line_alpha)
                        ax[nrow][i].fill_between(bonds[grp_bond]["CG"]["x"], bonds[grp_bond]["CG"]["y"],
                                                 color=config.cg_color, alpha=config.fill_alpha)
                    ax[nrow][i].plot(bonds[grp_bond]["CG"]["avg"], 0, color=config.cg_color, marker="D")
                    print(
                        f"Bond {grp_bond + 1} -- AA Avg: {round(bonds[grp_bond]['AA']['avg'], 3)} nm -- CG Avg: {round(bonds[grp_bond]['CG']['avg'], 3)} nm")
                else:
                    ax[nrow][i].set_title(f"Bond grp {grp_bond + 1} - Avg {round(avg_diff_grp_bonds[grp_bond], 3)} nm")
                    print(f"Bond {grp_bond + 1} -- AA Avg: {round(bonds[grp_bond]['AA']['avg'], 3)}")
                ax[nrow][i].grid(zorder=0.5)
                if ns.scoring.row_x_scaling:
                    ax[nrow][i].set_xlim(
                        np.mean(row_wise_ranges["bonds"][grp_bond]) - row_wise_ranges["max_range_bonds"] / 2 * 1.1,
                        np.mean(row_wise_ranges["bonds"][grp_bond]) + row_wise_ranges["max_range_bonds"] / 2 * 1.1)
                if i % 2 == 0:
                    ax[nrow][i].legend(loc="upper left")
                if ax[nrow][i].get_ylim()[0] < bonds_min_y:
                    bonds_min_y = ax[nrow][i].get_ylim()[0]
                if ax[nrow][i].get_ylim()[1] > bonds_max_y:
                    bonds_max_y = ax[nrow][i].get_ylim()[1]
            else:
                ax[nrow][i].set_visible(False)

    # angles
    if ns.cg_itp["nb_angles"] != 0:
        print()
        nrow += 1
        for i in range(ncols):
            if i < ns.cg_itp["nb_angles"]:
                grp_angle = diff_ordered_grp_angles[i]
                if config.use_hists:
                    ax[nrow][i].step(angles[grp_angle]["AA"]["x"], angles[grp_angle]["AA"]["y"], label="AA-mapped",
                                     color=config.atom_color, where="mid", alpha=config.line_alpha)
                    ax[nrow][i].fill_between(angles[grp_angle]["AA"]["x"], angles[grp_angle]["AA"]["y"],
                                             color=config.atom_color, step="mid", alpha=config.fill_alpha)
                else:
                    ax[nrow][i].plot(angles[grp_angle]["AA"]["x"], angles[grp_angle]["AA"]["y"], label="AA-mapped",
                                     color=config.atom_color, alpha=config.line_alpha)
                    ax[nrow][i].fill_between(angles[grp_angle]["AA"]["x"], angles[grp_angle]["AA"]["y"],
                                             color=config.atom_color, alpha=config.fill_alpha)
                ax[nrow][i].plot(angles[grp_angle]["AA"]["avg"], 0, color=config.atom_color, marker="D")

                if not ns.scoring.atom_only:
                    ax[nrow][i].set_title(
                        f"Angle grp {grp_angle + 1} - EMD Δ {round(avg_diff_grp_angles[grp_angle], 3)}")
                    if config.use_hists:
                        ax[nrow][i].step(angles[grp_angle]["CG"]["x"], angles[grp_angle]["CG"]["y"], label="CG",
                                         color=config.cg_color, where="mid", alpha=config.line_alpha)
                        ax[nrow][i].fill_between(angles[grp_angle]["CG"]["x"], angles[grp_angle]["CG"]["y"],
                                                 color=config.cg_color, step="mid", alpha=config.fill_alpha)
                    else:
                        ax[nrow][i].plot(angles[grp_angle]["CG"]["x"], angles[grp_angle]["CG"]["y"], label="CG",
                                         color=config.cg_color, alpha=config.line_alpha)
                        ax[nrow][i].fill_between(angles[grp_angle]["CG"]["x"], angles[grp_angle]["CG"]["y"],
                                                 color=config.cg_color, alpha=config.fill_alpha)
                    ax[nrow][i].plot(angles[grp_angle]["CG"]["avg"], 0, color=config.cg_color, marker="D")
                    print(
                        f"Angle {grp_angle + 1} -- AA Avg: {round(angles[grp_angle]['AA']['avg'], 1)}° -- CG Avg: {round(angles[grp_angle]['CG']['avg'], 1)}°")
                else:
                    ax[nrow][i].set_title(
                        f"Angle grp {grp_angle + 1} - Avg {round(avg_diff_grp_angles[grp_angle], 1)}°")
                    print(f"Angle {grp_angle + 1} -- AA Avg: {round(angles[grp_angle]['AA']['avg'], 1)}")
                ax[nrow][i].grid(zorder=0.5)
                if ns.scoring.row_x_scaling:
                    ax[nrow][i].set_xlim(
                        np.mean(row_wise_ranges["angles"][grp_angle]) - row_wise_ranges["max_range_angles"] / 2 * 1.1,
                        np.mean(row_wise_ranges["angles"][grp_angle]) + row_wise_ranges["max_range_angles"] / 2 * 1.1)
                if i % 2 == 0:
                    ax[nrow][i].legend(loc="upper left")
                if ax[nrow][i].get_ylim()[0] < angles_min_y:
                    angles_min_y = ax[nrow][i].get_ylim()[0]
                if ax[nrow][i].get_ylim()[1] > angles_max_y:
                    angles_max_y = ax[nrow][i].get_ylim()[1]
            else:
                ax[nrow][i].set_visible(False)

    # dihedrals
    if ns.cg_itp["nb_dihedrals"] != 0:
        print()
        nrow += 1
        for i in range(ncols):
            if i < ns.cg_itp["nb_dihedrals"]:
                grp_dihedral = diff_ordered_grp_dihedrals[i]
                if config.use_hists:
                    ax[nrow][i].step(dihedrals[grp_dihedral]["AA"]["x"], dihedrals[grp_dihedral]["AA"]["y"],
                                     label="AA-mapped", color=config.atom_color, where="mid", alpha=config.line_alpha)
                    ax[nrow][i].fill_between(dihedrals[grp_dihedral]["AA"]["x"], dihedrals[grp_dihedral]["AA"]["y"],
                                             color=config.atom_color, step="mid", alpha=config.fill_alpha)
                else:
                    ax[nrow][i].plot(dihedrals[grp_dihedral]["AA"]["x"], dihedrals[grp_dihedral]["AA"]["y"],
                                     label="AA-mapped", color=config.atom_color, alpha=config.line_alpha)
                    ax[nrow][i].fill_between(dihedrals[grp_dihedral]["AA"]["x"], dihedrals[grp_dihedral]["AA"]["y"],
                                             color=config.atom_color, alpha=config.fill_alpha)
                ax[nrow][i].plot(dihedrals[grp_dihedral]["AA"]["avg"], 0, color=config.atom_color, marker="D")

                if not ns.scoring.atom_only:
                    ax[nrow][i].set_title(
                        f"Dihedral grp {grp_dihedral + 1} - EMD Δ {round(avg_diff_grp_dihedrals[grp_dihedral], 3)}")
                    if config.use_hists:
                        ax[nrow][i].step(dihedrals[grp_dihedral]["CG"]["x"], dihedrals[grp_dihedral]["CG"]["y"],
                                         label="CG", color=config.cg_color, where="mid", alpha=config.line_alpha)
                        ax[nrow][i].fill_between(dihedrals[grp_dihedral]["CG"]["x"], dihedrals[grp_dihedral]["CG"]["y"],
                                                 color=config.cg_color, step="mid", alpha=config.fill_alpha)
                    else:
                        ax[nrow][i].plot(dihedrals[grp_dihedral]["CG"]["x"], dihedrals[grp_dihedral]["CG"]["y"],
                                         label="CG", color=config.cg_color, alpha=config.line_alpha)
                        ax[nrow][i].fill_between(dihedrals[grp_dihedral]["CG"]["x"], dihedrals[grp_dihedral]["CG"]["y"],
                                                 color=config.cg_color, alpha=config.fill_alpha)
                    ax[nrow][i].plot(dihedrals[grp_dihedral]["CG"]["avg"], 0, color=config.cg_color, marker="D")
                    print(
                        f"Dihedral {grp_dihedral + 1} -- AA Avg: {round(dihedrals[grp_dihedral]['AA']['avg'], 1)}° -- CG Avg: {round(dihedrals[grp_dihedral]['CG']['avg'], 1)}°")
                else:
                    ax[nrow][i].set_title(
                        f"Dihedral grp {grp_dihedral + 1} - Avg {round(avg_diff_grp_dihedrals[grp_dihedral], 1)}°")
                    print(f"Dihedral {grp_dihedral + 1} -- AA Avg: {round(dihedrals[grp_dihedral]['AA']['avg'], 1)}")
                ax[nrow][i].grid(zorder=0.5)
                if ns.scoring.row_x_scaling:
                    ax[nrow][i].set_xlim(np.mean(row_wise_ranges["dihedrals"][grp_dihedral]) - row_wise_ranges[
                        "max_range_dihedrals"] / 2 * 1.1,
                                         np.mean(row_wise_ranges["dihedrals"][grp_dihedral]) + row_wise_ranges[
                                             "max_range_dihedrals"] / 2 * 1.1)
                if i % 2 == 0:
                    ax[nrow][i].legend(loc="upper left")
                if ax[nrow][i].get_ylim()[0] < dihedrals_min_y:
                    dihedrals_min_y = ax[nrow][i].get_ylim()[0]
                if ax[nrow][i].get_ylim()[1] > dihedrals_max_y:
                    dihedrals_max_y = ax[nrow][i].get_ylim()[1]
            else:
                ax[nrow][i].set_visible(False)

    # now we have all the ylims, so make them all consistent
    if ns.scoring.row_y_scaling:
        nrow = -1
        if ns.cg_itp["nb_constraints"] != 0:
            nrow += 1
            for i in range(ns.cg_itp["nb_constraints"]):
                ax[nrow][i].set_ylim(bottom=constraints_min_y, top=constraints_max_y)
        if ns.cg_itp["nb_bonds"] != 0:
            nrow += 1
            for i in range(ns.cg_itp["nb_bonds"]):
                ax[nrow][i].set_ylim(bottom=bonds_min_y, top=bonds_max_y)
        if ns.cg_itp["nb_angles"] != 0:
            nrow += 1
            for i in range(ns.cg_itp["nb_angles"]):
                ax[nrow][i].set_ylim(bottom=angles_min_y, top=angles_max_y)
        if ns.cg_itp["nb_dihedrals"] != 0:
            nrow += 1
            for i in range(ns.cg_itp["nb_dihedrals"]):
                ax[nrow][i].set_ylim(bottom=dihedrals_min_y, top=dihedrals_max_y)

    # calculate global fitness score and contributions from each geom type
    all_dist_pairwise = ""  # for global optimization plotting
    all_emd_dist_geoms = {"constraints": [], "bonds": [], "angles": [], "dihedrals": []}

    if not ns.scoring.atom_only:
        fit_score_total, fit_score_constraints_bonds, fit_score_angles, fit_score_dihedrals = 0, 0, 0, 0

        for i in range(ns.cg_itp["nb_constraints"]):
            dist_pairwise = avg_diff_grp_constraints[diff_ordered_grp_constraints[i]]
            all_dist_pairwise += str(dist_pairwise) + " "
            all_emd_dist_geoms["constraints"].append(dist_pairwise)

            # keep track of independent best parameters
            if record_best_indep_params:
                if dist_pairwise < ns.pso.all_best_emd_dist_geoms["constraints"][i]:
                    ns.pso.all_best_emd_dist_geoms["constraints"][i] = dist_pairwise
                    ns.pso.all_best_params_dist_geoms["constraints"][i]["params"] = [ns.out_itp["constraint"][i]["value"]]

            dist_pairwise = dist_pairwise ** 2
            fit_score_constraints_bonds += dist_pairwise

        for i in range(ns.cg_itp["nb_bonds"]):
            dist_pairwise = avg_diff_grp_bonds[diff_ordered_grp_bonds[i]]
            all_dist_pairwise += str(dist_pairwise) + " "
            all_emd_dist_geoms["bonds"].append(dist_pairwise)

            # keep track of independent best parameters
            if record_best_indep_params:
                if dist_pairwise < ns.pso.all_best_emd_dist_geoms["bonds"][i]:
                    ns.pso.all_best_emd_dist_geoms["bonds"][i] = dist_pairwise
                    ns.pso.all_best_params_dist_geoms["bonds"][i]["params"] = [ns.out_itp["bond"][i]["value"],
                                                                           ns.out_itp["bond"][i]["fct"]]

            dist_pairwise = dist_pairwise ** 2
            fit_score_constraints_bonds += dist_pairwise

        for i in range(ns.cg_itp["nb_angles"]):
            dist_pairwise = avg_diff_grp_angles[diff_ordered_grp_angles[i]]
            all_dist_pairwise += str(dist_pairwise) + " "
            all_emd_dist_geoms["angles"].append(dist_pairwise)

            # keep track of independent best parameters
            if record_best_indep_params:
                if dist_pairwise < ns.pso.all_best_emd_dist_geoms["angles"][i]:
                    ns.pso.all_best_emd_dist_geoms["angles"][i] = dist_pairwise
                    ns.pso.all_best_params_dist_geoms["angles"][i]["params"] = [ns.out_itp["angle"][i]["value"],
                                                                            ns.out_itp["angle"][i]["fct"]]

            dist_pairwise = dist_pairwise ** 2
            fit_score_angles += dist_pairwise

        # dihedrals_dist_pairwise = 0
        for i in range(ns.cg_itp["nb_dihedrals"]):
            dist_pairwise = avg_diff_grp_dihedrals[diff_ordered_grp_dihedrals[i]]
            all_dist_pairwise += str(dist_pairwise) + " "
            all_emd_dist_geoms["dihedrals"].append(dist_pairwise)

            # keep track of independent best parameters
            if record_best_indep_params and not ignore_dihedrals:
                if dist_pairwise < ns.pso.all_best_emd_dist_geoms["dihedrals"][i]:
                    ns.pso.all_best_emd_dist_geoms["dihedrals"][i] = dist_pairwise
                    ns.pso.all_best_params_dist_geoms["dihedrals"][i]["params"] = [ns.out_itp["dihedral"][i]["value"],
                                                                               ns.out_itp["dihedral"][i]["fct"]]

            dist_pairwise = dist_pairwise ** 2
            fit_score_dihedrals += dist_pairwise

        fit_score_constraints_bonds = np.sqrt(fit_score_constraints_bonds)
        fit_score_angles = np.sqrt(fit_score_angles)
        fit_score_dihedrals = np.sqrt(fit_score_dihedrals)

        fit_score_total = fit_score_constraints_bonds + fit_score_angles + fit_score_dihedrals

        fit_score_total, fit_score_constraints_bonds, fit_score_angles, fit_score_dihedrals = round(fit_score_total,
                                                                                                    3), round(
            fit_score_constraints_bonds, 3), round(fit_score_angles, 3), round(fit_score_dihedrals, 3)
        all_dist_pairwise += "\n"
        print()
        print("Using bonds to angles/dihedrals (C) scoring constant:", ns.bonds2angles_scoring_factor)
        print()
        print("Global fitness score:", fit_score_total, "(lower is better)", flush=True)
        print("  Bonds/Constraints constribution to fitness score:", fit_score_constraints_bonds, flush=True)
        print("  Angles constribution to fitness score:", fit_score_angles, flush=True)
        print("  Dihedrals constribution to fitness score:", fit_score_dihedrals, flush=True)

        plt.tight_layout(rect=[0, 0, 1, 0.9])
        eval_score = fit_score_total
        if ignore_dihedrals and ns.cg_itp["nb_dihedrals"] > 0:
            eval_score -= fit_score_dihedrals
        sup_title = f"FITNESS SCORE\nTotal: {round(eval_score, 3)} -- Constraints/Bonds: {fit_score_constraints_bonds} -- Angles: {fit_score_angles} -- Dihedrals: {fit_score_dihedrals}"
        if ignore_dihedrals and ns.cg_itp["nb_dihedrals"] > 0:
            sup_title += " (ignored)"
        plt.suptitle(sup_title)
    else:
        plt.tight_layout()

    # here we close everything we can close because there was a memory leak from plotting
    plt.savefig(ns.files.plot_filename)
    plt.close(fig)
    print()
    print("Distributions plot written at location:\n ", ns.files.plot_filename, flush=True)
    print()

    if not manual_mode and not ns.scoring.atom_only:
        return fit_score_total, fit_score_constraints_bonds, fit_score_angles, fit_score_dihedrals, all_dist_pairwise, all_emd_dist_geoms
    else:
        return
