import numpy as np
import matplotlib
matplotlib.use("AGG")  # use the Anti-Grain Geometry non-interactive backend suited for scripted PNG creation
import matplotlib.pyplot as plt
import MDAnalysis as mda
from pyemd import emd

import swarmcg.scoring as scores
from swarmcg import config
from swarmcg.context import SwarmCGArgs, SwarmCGState
from swarmcg.shared import styling


def _configure_plotting():
    plt.rcParams["grid.color"] = "k"  # plt grid appearance settings
    plt.rcParams["grid.linestyle"] = ":"
    plt.rcParams["grid.linewidth"] = 0.5


def _init_row_wise_ranges():
    return {
        "constraints": {},
        "bonds": {},
        "angles": {},
        "dihedrals": {},
        "max_range_constraints": 0,
        "max_range_bonds": 0,
        "max_range_angles": 0,
        "max_range_dihedrals": 0,
    }


def _histogram_to_xy(hist, bins):
    x_vals, y_vals = [], []
    for i in range(1, len(hist) - 1):
        if hist[i - 1] > 0 or hist[i] > 0 or hist[i + 1] > 0:
            x_vals.append(np.mean(bins[i:i + 2]))
            y_vals.append(hist[i])
    return x_vals, y_vals


def _update_row_range(row_wise_ranges, max_key, row_range):
    span = row_range[1] - row_range[0]
    if span > row_wise_ranges[max_key]:
        row_wise_ranges[max_key] = span


def _report_atom_only_rg(args: SwarmCGArgs, state: SwarmCGState):
    if state.mapping.atom_only:
        scores.compute_Rg(args, state, traj_type="AA")
        print("Radius of gyration (AA reference, NOT CG-mapped):", state.model.gyr_aa, "nm")


def _prepare_cg_universe(args: SwarmCGArgs, state: SwarmCGState, manual_mode, calc_sasa):
    print("Reading CG trajectory")
    state.traj.cg_universe = mda.Universe(
        args.inputs.cg_tpr_filename,
        args.inputs.cg_traj_filename,
        in_memory=True,
        refresh_offsets=True,
        guess_bonds=False,
    )
    print("  Found", len(state.traj.cg_universe.trajectory), "frames")

    if manual_mode:
        # here we read the CG beads masses + actualize the mapped trajectory object
        for bead_id in range(len(state.model.cg_itp["atoms"])):
            state.model.cg_itp["atoms"][bead_id]["mass"] = state.traj.cg_universe.atoms[bead_id].mass
        masses = np.array([val["mass"] for val in state.model.cg_itp["atoms"]])
        state.traj.aa2cg_universe._topology.masses.values = np.array(masses)

    # create fake bonds in the CG MDA universe, that will be used only for making the molecule whole
    # we make bonds between each VS and their beads definition, so we retrieve the connectivity
    # iteratively towards the real CG beads, that are all connected
    if len(state.model.cg_itp["vs_beads_ids"]) > 0:
        fake_bonds = []
        for vs_type in ["2", "3", "4", "n"]:
            try:
                for bead_id in state.model.cg_itp["virtual_sites" + vs_type]:
                    for vs_def_bead_id in state.model.cg_itp["virtual_sites" + vs_type][bead_id]["vs_def_beads_ids"]:
                        fake_bonds.append([bead_id, vs_def_bead_id])
            except (IndexError, ValueError):
                pass
        state.traj.cg_universe.add_bonds(fake_bonds, guessed=False)

    # select the whole molecule as an MDA atomgroup and make its coordinates whole, inplace, across the complete trajectory
    ag_mol = mda.AtomGroup([bead_id for bead_id in range(len(state.model.cg_itp["atoms"]))], state.traj.cg_universe)
    for _ in state.traj.cg_universe.trajectory:
        mda.lib.mdamath.make_whole(ag_mol, inplace=True)

    # this requires CG data for mapping -- especially, masses are taken from the CG TPR but the CG ITP is also used atm
    if state.model.gyr_aa_mapped is None:
        scores.compute_Rg(args, state, traj_type="AA_mapped")
        print()
        print("Radius of gyration (AA reference, CG-mapped, no bonds scaling):", state.model.gyr_aa_mapped, "+/-",
              state.model.gyr_aa_mapped_std, "nm")

    scores.compute_Rg(args, state, traj_type="CG")
    print("Radius of gyration (CG model):", state.model.gyr_cg, "+/-", state.model.gyr_cg_std, "nm")

    if calc_sasa:
        if state.model.sasa_aa_mapped is None:
            scores.compute_SASA(args, state, traj_type="AA_mapped")

        scores.compute_SASA(args, state, traj_type="CG")
        print()

        # this line checks that gmx trjconv could read the md.xtc trajectory from the opti
        # this is to catch bugged simulation that actually finished and produced the files,
        # but the .gro is a 2D bugged file for example, or trjactory is unreadable by gmx
        if state.model.sasa_cg is None:
            return False

    return True


def _compute_constraints(args: SwarmCGArgs, state: SwarmCGState, manual_mode, row_wise_ranges):
    print("Processing constraints ...", flush=True)
    diff_ordered = list(range(state.model.cg_itp["nb_constraints"]))
    avg_diff = []
    constraints = {}
    row_wise_ranges["constraints"] = {}

    for grp_constraint in range(state.model.cg_itp["nb_constraints"]):
        constraints[grp_constraint] = {"AA": {"x": [], "y": []}, "CG": {"x": [], "y": []}}

        if manual_mode:
            constraint_avg, constraint_hist, _ = scores.get_AA_bonds_distrib(
                args,
                state,
                beads_ids=state.model.cg_itp["constraint"][grp_constraint]["beads"],
                grp_type="constraints group",
                grp_nb=grp_constraint,
            )
            constraints[grp_constraint]["AA"]["avg"] = constraint_avg
            constraints[grp_constraint]["AA"]["hist"] = constraint_hist
        else:  # use atomistic reference that was loaded by the optimization routines
            constraints[grp_constraint]["AA"]["avg"] = state.model.cg_itp["constraint"][grp_constraint]["avg"]
            constraints[grp_constraint]["AA"]["hist"] = state.model.cg_itp["constraint"][grp_constraint]["hist"]

        constraints[grp_constraint]["AA"]["x"], constraints[grp_constraint]["AA"]["y"] = _histogram_to_xy(
            constraints[grp_constraint]["AA"]["hist"],
            state.bins.bins_constraints,
        )

        if not state.mapping.atom_only:
            try:
                constraint_avg, constraint_hist, _ = scores.get_CG_bonds_distrib(
                    args,
                    state,
                    beads_ids=state.model.cg_itp["constraint"][grp_constraint]["beads"],
                    grp_type="constraint",
                )
                constraints[grp_constraint]["CG"]["avg"] = constraint_avg
                constraints[grp_constraint]["CG"]["hist"] = constraint_hist
                constraints[grp_constraint]["CG"]["x"], constraints[grp_constraint]["CG"]["y"] = _histogram_to_xy(
                    constraint_hist,
                    state.bins.bins_constraints,
                )

                domain_min = min(constraints[grp_constraint]["AA"]["x"][0], constraints[grp_constraint]["CG"]["x"][0])
                domain_max = max(constraints[grp_constraint]["AA"]["x"][-1], constraints[grp_constraint]["CG"]["x"][-1])
                avg_diff.append(
                    emd(constraints[grp_constraint]["AA"]["hist"], constraints[grp_constraint]["CG"]["hist"],
                        state.bins.bins_constraints_dist_matrix) * args.optimization.bonds2angles_scoring_factor
                )
            except IndexError:
                msg = (
                    f"Most probably because you have bonds or constraints that "
                    f"exceed {args.optimization.bonded_max_range} nm.\nIncrease bins range for bonds and "
                    f"constraints and retry!\nSee argument -bonds_max_range."
                )
                raise ValueError(msg)
        else:
            avg_diff.append(constraints[grp_constraint]["AA"]["avg"])

        if args.plotting.row_x_scaling:
            if state.mapping.atom_only:
                row_wise_ranges["constraints"][grp_constraint] = [
                    constraints[grp_constraint]["AA"]["x"][0],
                    constraints[grp_constraint]["AA"]["x"][-1],
                ]
            else:
                row_wise_ranges["constraints"][grp_constraint] = [domain_min, domain_max]
            _update_row_range(row_wise_ranges, "max_range_constraints", row_wise_ranges["constraints"][grp_constraint])

    if args.plotting.mismatch_order and not state.mapping.atom_only:
        diff_ordered = [
            x
            for _, x in sorted(zip(avg_diff, diff_ordered), key=lambda pair: pair[0], reverse=True)
        ]

    return constraints, avg_diff, diff_ordered


def _compute_bonds(args: SwarmCGArgs, state: SwarmCGState, manual_mode, row_wise_ranges):
    print("Processing bonds ...", flush=True)
    diff_ordered = list(range(state.model.cg_itp["nb_bonds"]))
    avg_diff = []
    bonds = {}
    row_wise_ranges["bonds"] = {}

    for grp_bond in range(state.model.cg_itp["nb_bonds"]):
        bonds[grp_bond] = {"AA": {"x": [], "y": []}, "CG": {"x": [], "y": []}}

        if manual_mode:
            bond_avg, bond_hist, _ = scores.get_AA_bonds_distrib(
                args,
                state,
                beads_ids=state.model.cg_itp["bond"][grp_bond]["beads"],
                grp_type="bonds group",
                grp_nb=grp_bond,
            )
            bonds[grp_bond]["AA"]["avg"] = bond_avg
            bonds[grp_bond]["AA"]["hist"] = bond_hist
        else:  # use atomistic reference that was loaded by the optimization routines
            bonds[grp_bond]["AA"]["avg"] = state.model.cg_itp["bond"][grp_bond]["avg"]
            bonds[grp_bond]["AA"]["hist"] = state.model.cg_itp["bond"][grp_bond]["hist"]

        bonds[grp_bond]["AA"]["x"], bonds[grp_bond]["AA"]["y"] = _histogram_to_xy(
            bonds[grp_bond]["AA"]["hist"],
            state.bins.bins_bonds,
        )

        if not state.mapping.atom_only:
            try:
                bond_avg, bond_hist, _ = scores.get_CG_bonds_distrib(
                    args,
                    state,
                    beads_ids=state.model.cg_itp["bond"][grp_bond]["beads"],
                    grp_type="bond",
                )
                bonds[grp_bond]["CG"]["avg"] = bond_avg
                bonds[grp_bond]["CG"]["hist"] = bond_hist
                bonds[grp_bond]["CG"]["x"], bonds[grp_bond]["CG"]["y"] = _histogram_to_xy(
                    bond_hist,
                    state.bins.bins_bonds,
                )

                domain_min = min(bonds[grp_bond]["AA"]["x"][0], bonds[grp_bond]["CG"]["x"][0])
                domain_max = max(bonds[grp_bond]["AA"]["x"][-1], bonds[grp_bond]["CG"]["x"][-1])
                avg_diff.append(
                    emd(bonds[grp_bond]["AA"]["hist"], bonds[grp_bond]["CG"]["hist"],
                        state.bins.bins_bonds_dist_matrix) * args.optimization.bonds2angles_scoring_factor
                )
            except IndexError:
                msg = (
                    f"Most probably because you have bonds or constraints that "
                    f"exceed {args.optimization.bonded_max_range} nm.\nIncrease bins range for bonds and "
                    f"constraints and retry!\nSee argument -bonds_max_range."
                )
                raise ValueError(msg)
        else:
            avg_diff.append(bonds[grp_bond]["AA"]["avg"])

        if args.plotting.row_x_scaling:
            if state.mapping.atom_only:
                row_wise_ranges["bonds"][grp_bond] = [bonds[grp_bond]["AA"]["x"][0], bonds[grp_bond]["AA"]["x"][-1]]
            else:
                row_wise_ranges["bonds"][grp_bond] = [domain_min, domain_max]
            _update_row_range(row_wise_ranges, "max_range_bonds", row_wise_ranges["bonds"][grp_bond])

    if args.plotting.mismatch_order and not state.mapping.atom_only:
        diff_ordered = [
            x
            for _, x in sorted(zip(avg_diff, diff_ordered), key=lambda pair: pair[0], reverse=True)
        ]

    return bonds, avg_diff, diff_ordered


def _compute_angles(args: SwarmCGArgs, state: SwarmCGState, manual_mode, row_wise_ranges):
    print("Processing angles ...", flush=True)
    diff_ordered = list(range(state.model.cg_itp["nb_angles"]))
    avg_diff = []
    angles = {}
    row_wise_ranges["angles"] = {}

    for grp_angle in range(state.model.cg_itp["nb_angles"]):
        angles[grp_angle] = {"AA": {"x": [], "y": []}, "CG": {"x": [], "y": []}}

        if manual_mode:
            angle_avg, angle_hist, _, _ = scores.get_AA_angles_distrib(
                args,
                state,
                beads_ids=state.model.cg_itp["angle"][grp_angle]["beads"],
            )
            angles[grp_angle]["AA"]["avg"] = angle_avg
            angles[grp_angle]["AA"]["hist"] = angle_hist
        else:  # use atomistic reference that was loaded by the optimization routines
            angles[grp_angle]["AA"]["avg"] = state.model.cg_itp["angle"][grp_angle]["avg"]
            angles[grp_angle]["AA"]["hist"] = state.model.cg_itp["angle"][grp_angle]["hist"]

        angles[grp_angle]["AA"]["x"], angles[grp_angle]["AA"]["y"] = _histogram_to_xy(
            angles[grp_angle]["AA"]["hist"],
            state.bins.bins_angles,
        )

        if not state.mapping.atom_only:
            angle_avg, angle_hist, _, _ = scores.get_CG_angles_distrib(
                args,
                state,
                beads_ids=state.model.cg_itp["angle"][grp_angle]["beads"],
            )
            angles[grp_angle]["CG"]["avg"] = angle_avg
            angles[grp_angle]["CG"]["hist"] = angle_hist
            angles[grp_angle]["CG"]["x"], angles[grp_angle]["CG"]["y"] = _histogram_to_xy(
                angle_hist,
                state.bins.bins_angles,
            )

            domain_min = min(angles[grp_angle]["AA"]["x"][0], angles[grp_angle]["CG"]["x"][0])
            domain_max = max(angles[grp_angle]["AA"]["x"][-1], angles[grp_angle]["CG"]["x"][-1])
            avg_diff.append(
                emd(angles[grp_angle]["AA"]["hist"], angles[grp_angle]["CG"]["hist"], state.bins.bins_angles_dist_matrix)
            )
        else:
            avg_diff.append(angles[grp_angle]["AA"]["avg"])

        if args.plotting.row_x_scaling:
            if state.mapping.atom_only:
                row_wise_ranges["angles"][grp_angle] = [angles[grp_angle]["AA"]["x"][0], angles[grp_angle]["AA"]["x"][-1]]
            else:
                row_wise_ranges["angles"][grp_angle] = [domain_min, domain_max]
            _update_row_range(row_wise_ranges, "max_range_angles", row_wise_ranges["angles"][grp_angle])

    if args.plotting.mismatch_order and not state.mapping.atom_only:
        diff_ordered = [
            x
            for _, x in sorted(zip(avg_diff, diff_ordered), key=lambda pair: pair[0], reverse=True)
        ]

    return angles, avg_diff, diff_ordered


def _compute_dihedrals(args: SwarmCGArgs, state: SwarmCGState, manual_mode, row_wise_ranges):
    print("Processing dihedrals ...", flush=True)
    diff_ordered = list(range(state.model.cg_itp["nb_dihedrals"]))
    avg_diff = []
    dihedrals = {}
    row_wise_ranges["dihedrals"] = {}

    for grp_dihedral in range(state.model.cg_itp["nb_dihedrals"]):
        dihedrals[grp_dihedral] = {"AA": {"x": [], "y": []}, "CG": {"x": [], "y": []}}

        if manual_mode:
            dihedral_avg, dihedral_hist, _, _ = scores.get_AA_dihedrals_distrib(
                args,
                state,
                beads_ids=state.model.cg_itp["dihedral"][grp_dihedral]["beads"],
            )
            dihedrals[grp_dihedral]["AA"]["avg"] = dihedral_avg
            dihedrals[grp_dihedral]["AA"]["hist"] = dihedral_hist
        else:  # use atomistic reference that was loaded by the optimization routines
            dihedrals[grp_dihedral]["AA"]["avg"] = state.model.cg_itp["dihedral"][grp_dihedral]["avg"]
            dihedrals[grp_dihedral]["AA"]["hist"] = state.model.cg_itp["dihedral"][grp_dihedral]["hist"]

        dihedrals[grp_dihedral]["AA"]["x"], dihedrals[grp_dihedral]["AA"]["y"] = _histogram_to_xy(
            dihedrals[grp_dihedral]["AA"]["hist"],
            state.bins.bins_dihedrals,
        )

        if not state.mapping.atom_only:
            dihedral_avg, dihedral_hist, _, _ = scores.get_CG_dihedrals_distrib(
                args,
                state,
                beads_ids=state.model.cg_itp["dihedral"][grp_dihedral]["beads"],
            )
            dihedrals[grp_dihedral]["CG"]["avg"] = dihedral_avg
            dihedrals[grp_dihedral]["CG"]["hist"] = dihedral_hist
            dihedrals[grp_dihedral]["CG"]["x"], dihedrals[grp_dihedral]["CG"]["y"] = _histogram_to_xy(
                dihedral_hist,
                state.bins.bins_dihedrals,
            )

            domain_min = min(dihedrals[grp_dihedral]["AA"]["x"][0], dihedrals[grp_dihedral]["CG"]["x"][0])
            domain_max = max(dihedrals[grp_dihedral]["AA"]["x"][-1], dihedrals[grp_dihedral]["CG"]["x"][-1])
            avg_diff.append(
                emd(dihedrals[grp_dihedral]["AA"]["hist"], dihedrals[grp_dihedral]["CG"]["hist"],
                    state.bins.bins_dihedrals_dist_matrix)
            )
        else:
            avg_diff.append(dihedrals[grp_dihedral]["AA"]["avg"])

        if args.plotting.row_x_scaling:
            if state.mapping.atom_only:
                row_wise_ranges["dihedrals"][grp_dihedral] = [
                    dihedrals[grp_dihedral]["AA"]["x"][0],
                    dihedrals[grp_dihedral]["AA"]["x"][-1],
                ]
            else:
                row_wise_ranges["dihedrals"][grp_dihedral] = [domain_min, domain_max]
            _update_row_range(row_wise_ranges, "max_range_dihedrals", row_wise_ranges["dihedrals"][grp_dihedral])

    if args.plotting.mismatch_order and not state.mapping.atom_only:
        diff_ordered = [
            x
            for _, x in sorted(zip(avg_diff, diff_ordered), key=lambda pair: pair[0], reverse=True)
        ]

    return dihedrals, avg_diff, diff_ordered


def _resolve_plot_layout(args: SwarmCGArgs, state: SwarmCGState):
    larger_group = max(
        state.model.cg_itp["nb_constraints"],
        state.model.cg_itp["nb_bonds"],
        state.model.cg_itp["nb_angles"],
        state.model.cg_itp["nb_dihedrals"],
    )
    nrows = 4
    ncols = min(args.plotting.ncols_max, larger_group)
    if args.plotting.ncols_max == 0:
        ncols = larger_group

    if larger_group > ncols:
        hidden_cols = larger_group - ncols
        if state.mapping.atom_only:
            print(
                f"Displaying max {ncols} distributions per row using the CG ITP file ordering of distributions groups ({hidden_cols} more are hidden)"
            )
        else:
            if not args.plotting.mismatch_order:
                print(
                    f"{styling.header_warning}Displaying max {ncols} distributions groups per row and this can be MISLEADING because ordering by pairwise AA-mapped vs. CG distributions mismatch is DISABLED ({hidden_cols} more are hidden)"
                )
            else:
                print(
                    f"Displaying max {ncols} distributions groups per row ordered by pairwise AA-mapped vs. CG distributions difference ({hidden_cols} more are hidden)"
                )
    else:
        print()
        if not args.plotting.mismatch_order:
            print("Distributions groups will be displayed using the CG ITP file groups ordering")
        else:
            print("Distributions groups will be displayed using ranked mismatch score between pairwise AA-mapped and CG distributions")

    nrows -= sum([
        state.model.cg_itp["nb_constraints"] == 0,
        state.model.cg_itp["nb_bonds"] == 0,
        state.model.cg_itp["nb_angles"] == 0,
        state.model.cg_itp["nb_dihedrals"] == 0,
    ])

    return nrows, ncols


def _plot_constraints_row(ax, nrow, ncols, args, state, constraints, diff_ordered, avg_diff, row_wise_ranges):
    constraints_min_y, constraints_max_y = 10, 0
    if state.model.cg_itp["nb_constraints"] != 0:
        print()
        nrow += 1
        for i in range(ncols):
            if i < state.model.cg_itp["nb_constraints"]:
                grp_constraint = diff_ordered[i]

                if config.use_hists:
                    ax[nrow][i].step(
                        constraints[grp_constraint]["AA"]["x"],
                        constraints[grp_constraint]["AA"]["y"],
                        label="AA-mapped",
                        color=config.atom_color,
                        where="mid",
                        alpha=config.line_alpha,
                    )
                    ax[nrow][i].fill_between(
                        constraints[grp_constraint]["AA"]["x"],
                        constraints[grp_constraint]["AA"]["y"],
                        color=config.atom_color,
                        step="mid",
                        alpha=config.fill_alpha,
                    )
                else:
                    ax[nrow][i].plot(
                        constraints[grp_constraint]["AA"]["x"],
                        constraints[grp_constraint]["AA"]["y"],
                        label="AA-mapped",
                        color=config.atom_color,
                        alpha=config.line_alpha,
                    )
                    ax[nrow][i].fill_between(
                        constraints[grp_constraint]["AA"]["x"],
                        constraints[grp_constraint]["AA"]["y"],
                        color=config.atom_color,
                        alpha=config.fill_alpha,
                    )
                ax[nrow][i].plot(constraints[grp_constraint]["AA"]["avg"], 0, color=config.atom_color, marker="D")

                if not state.mapping.atom_only:
                    ax[nrow][i].set_title(
                        f"Constraint grp {grp_constraint + 1} - EMD Δ {round(avg_diff[grp_constraint], 3)}"
                    )
                    if config.use_hists:
                        ax[nrow][i].step(
                            constraints[grp_constraint]["CG"]["x"],
                            constraints[grp_constraint]["CG"]["y"],
                            label="CG",
                            color=config.cg_color,
                            where="mid",
                            alpha=config.line_alpha,
                        )
                        ax[nrow][i].fill_between(
                            constraints[grp_constraint]["CG"]["x"],
                            constraints[grp_constraint]["CG"]["y"],
                            color=config.cg_color,
                            step="mid",
                            alpha=config.fill_alpha,
                        )
                    else:
                        ax[nrow][i].plot(
                            constraints[grp_constraint]["CG"]["x"],
                            constraints[grp_constraint]["CG"]["y"],
                            label="CG",
                            color=config.cg_color,
                            alpha=config.line_alpha,
                        )
                        ax[nrow][i].fill_between(
                            constraints[grp_constraint]["CG"]["x"],
                            constraints[grp_constraint]["CG"]["y"],
                            color=config.cg_color,
                            alpha=config.fill_alpha,
                        )
                    ax[nrow][i].plot(constraints[grp_constraint]["CG"]["avg"], 0, color=config.cg_color, marker="D")
                    print(
                        f"Constraint {grp_constraint + 1} -- AA Avg: {round(constraints[grp_constraint]['AA']['avg'], 3)} nm -- CG Avg: {round(constraints[grp_constraint]['CG']['avg'], 3)}"
                    )
                else:
                    ax[nrow][i].set_title(
                        f"Constraint grp {grp_constraint + 1} - Avg {round(avg_diff[grp_constraint], 3)} nm"
                    )
                    print(f"Constraint {grp_constraint + 1} -- AA Avg: {round(constraints[grp_constraint]['AA']['avg'], 3)}")
                ax[nrow][i].grid(zorder=0.5)
                if args.plotting.row_x_scaling:
                    ax[nrow][i].set_xlim(
                        np.mean(row_wise_ranges["constraints"][grp_constraint]) -
                        row_wise_ranges["max_range_constraints"] / 2 * 1.1,
                        np.mean(row_wise_ranges["constraints"][grp_constraint]) +
                        row_wise_ranges["max_range_constraints"] / 2 * 1.1,
                    )
                if i % 2 == 0:
                    ax[nrow][i].legend(loc="upper left")
                if ax[nrow][i].get_ylim()[0] < constraints_min_y:
                    constraints_min_y = ax[nrow][i].get_ylim()[0]
                if ax[nrow][i].get_ylim()[1] > constraints_max_y:
                    constraints_max_y = ax[nrow][i].get_ylim()[1]
            else:
                ax[nrow][i].set_visible(False)

    return nrow, constraints_min_y, constraints_max_y


def _plot_bonds_row(ax, nrow, ncols, args, state, bonds, diff_ordered, avg_diff, row_wise_ranges):
    bonds_min_y, bonds_max_y = 10, 0
    if state.model.cg_itp["nb_bonds"] != 0:
        print()
        nrow += 1
        for i in range(ncols):
            if i < state.model.cg_itp["nb_bonds"]:
                grp_bond = diff_ordered[i]

                if config.use_hists:
                    ax[nrow][i].step(
                        bonds[grp_bond]["AA"]["x"],
                        bonds[grp_bond]["AA"]["y"],
                        label="AA-mapped",
                        color=config.atom_color,
                        where="mid",
                        alpha=config.line_alpha,
                    )
                    ax[nrow][i].fill_between(
                        bonds[grp_bond]["AA"]["x"],
                        bonds[grp_bond]["AA"]["y"],
                        color=config.atom_color,
                        step="mid",
                        alpha=config.fill_alpha,
                    )
                else:
                    ax[nrow][i].plot(
                        bonds[grp_bond]["AA"]["x"],
                        bonds[grp_bond]["AA"]["y"],
                        label="AA-mapped",
                        color=config.atom_color,
                        alpha=config.line_alpha,
                    )
                    ax[nrow][i].fill_between(
                        bonds[grp_bond]["AA"]["x"],
                        bonds[grp_bond]["AA"]["y"],
                        color=config.atom_color,
                        alpha=config.fill_alpha,
                    )
                ax[nrow][i].plot(bonds[grp_bond]["AA"]["avg"], 0, color=config.atom_color, marker="D")

                if not state.mapping.atom_only:
                    ax[nrow][i].set_title(f"Bond grp {grp_bond + 1} - EMD Δ {round(avg_diff[grp_bond], 3)}")
                    if config.use_hists:
                        ax[nrow][i].step(
                            bonds[grp_bond]["CG"]["x"],
                            bonds[grp_bond]["CG"]["y"],
                            label="CG",
                            color=config.cg_color,
                            where="mid",
                            alpha=config.line_alpha,
                        )
                        ax[nrow][i].fill_between(
                            bonds[grp_bond]["CG"]["x"],
                            bonds[grp_bond]["CG"]["y"],
                            color=config.cg_color,
                            step="mid",
                            alpha=config.fill_alpha,
                        )
                    else:
                        ax[nrow][i].plot(
                            bonds[grp_bond]["CG"]["x"],
                            bonds[grp_bond]["CG"]["y"],
                            label="CG",
                            color=config.cg_color,
                            alpha=config.line_alpha,
                        )
                        ax[nrow][i].fill_between(
                            bonds[grp_bond]["CG"]["x"],
                            bonds[grp_bond]["CG"]["y"],
                            color=config.cg_color,
                            alpha=config.fill_alpha,
                        )
                    ax[nrow][i].plot(bonds[grp_bond]["CG"]["avg"], 0, color=config.cg_color, marker="D")
                    print(
                        f"Bond {grp_bond + 1} -- AA Avg: {round(bonds[grp_bond]['AA']['avg'], 3)} nm -- CG Avg: {round(bonds[grp_bond]['CG']['avg'], 3)} nm"
                    )
                else:
                    ax[nrow][i].set_title(f"Bond grp {grp_bond + 1} - Avg {round(avg_diff[grp_bond], 3)} nm")
                    print(f"Bond {grp_bond + 1} -- AA Avg: {round(bonds[grp_bond]['AA']['avg'], 3)}")
                ax[nrow][i].grid(zorder=0.5)
                if args.plotting.row_x_scaling:
                    ax[nrow][i].set_xlim(
                        np.mean(row_wise_ranges["bonds"][grp_bond]) - row_wise_ranges["max_range_bonds"] / 2 * 1.1,
                        np.mean(row_wise_ranges["bonds"][grp_bond]) + row_wise_ranges["max_range_bonds"] / 2 * 1.1,
                    )
                if i % 2 == 0:
                    ax[nrow][i].legend(loc="upper left")
                if ax[nrow][i].get_ylim()[0] < bonds_min_y:
                    bonds_min_y = ax[nrow][i].get_ylim()[0]
                if ax[nrow][i].get_ylim()[1] > bonds_max_y:
                    bonds_max_y = ax[nrow][i].get_ylim()[1]
            else:
                ax[nrow][i].set_visible(False)

    return nrow, bonds_min_y, bonds_max_y


def _plot_angles_row(ax, nrow, ncols, args, state, angles, diff_ordered, avg_diff, row_wise_ranges):
    angles_min_y, angles_max_y = 10, 0
    if state.model.cg_itp["nb_angles"] != 0:
        print()
        nrow += 1
        for i in range(ncols):
            if i < state.model.cg_itp["nb_angles"]:
                grp_angle = diff_ordered[i]

                if config.use_hists:
                    ax[nrow][i].step(
                        angles[grp_angle]["AA"]["x"],
                        angles[grp_angle]["AA"]["y"],
                        label="AA-mapped",
                        color=config.atom_color,
                        where="mid",
                        alpha=config.line_alpha,
                    )
                    ax[nrow][i].fill_between(
                        angles[grp_angle]["AA"]["x"],
                        angles[grp_angle]["AA"]["y"],
                        color=config.atom_color,
                        step="mid",
                        alpha=config.fill_alpha,
                    )
                else:
                    ax[nrow][i].plot(
                        angles[grp_angle]["AA"]["x"],
                        angles[grp_angle]["AA"]["y"],
                        label="AA-mapped",
                        color=config.atom_color,
                        alpha=config.line_alpha,
                    )
                    ax[nrow][i].fill_between(
                        angles[grp_angle]["AA"]["x"],
                        angles[grp_angle]["AA"]["y"],
                        color=config.atom_color,
                        alpha=config.fill_alpha,
                    )
                ax[nrow][i].plot(angles[grp_angle]["AA"]["avg"], 0, color=config.atom_color, marker="D")

                if not state.mapping.atom_only:
                    ax[nrow][i].set_title(f"Angle grp {grp_angle + 1} - EMD Δ {round(avg_diff[grp_angle], 3)}")
                    if config.use_hists:
                        ax[nrow][i].step(
                            angles[grp_angle]["CG"]["x"],
                            angles[grp_angle]["CG"]["y"],
                            label="CG",
                            color=config.cg_color,
                            where="mid",
                            alpha=config.line_alpha,
                        )
                        ax[nrow][i].fill_between(
                            angles[grp_angle]["CG"]["x"],
                            angles[grp_angle]["CG"]["y"],
                            color=config.cg_color,
                            step="mid",
                            alpha=config.fill_alpha,
                        )
                    else:
                        ax[nrow][i].plot(
                            angles[grp_angle]["CG"]["x"],
                            angles[grp_angle]["CG"]["y"],
                            label="CG",
                            color=config.cg_color,
                            alpha=config.line_alpha,
                        )
                        ax[nrow][i].fill_between(
                            angles[grp_angle]["CG"]["x"],
                            angles[grp_angle]["CG"]["y"],
                            color=config.cg_color,
                            alpha=config.fill_alpha,
                        )
                    ax[nrow][i].plot(angles[grp_angle]["CG"]["avg"], 0, color=config.cg_color, marker="D")
                    print(
                        f"Angle {grp_angle + 1} -- AA Avg: {round(angles[grp_angle]['AA']['avg'], 1)}° -- CG Avg: {round(angles[grp_angle]['CG']['avg'], 1)}°"
                    )
                else:
                    ax[nrow][i].set_title(f"Angle grp {grp_angle + 1} - Avg {round(avg_diff[grp_angle], 1)}°")
                    print(f"Angle {grp_angle + 1} -- AA Avg: {round(angles[grp_angle]['AA']['avg'], 1)}")
                ax[nrow][i].grid(zorder=0.5)
                if args.plotting.row_x_scaling:
                    ax[nrow][i].set_xlim(
                        np.mean(row_wise_ranges["angles"][grp_angle]) - row_wise_ranges["max_range_angles"] / 2 * 1.1,
                        np.mean(row_wise_ranges["angles"][grp_angle]) + row_wise_ranges["max_range_angles"] / 2 * 1.1,
                    )
                if i % 2 == 0:
                    ax[nrow][i].legend(loc="upper left")
                if ax[nrow][i].get_ylim()[0] < angles_min_y:
                    angles_min_y = ax[nrow][i].get_ylim()[0]
                if ax[nrow][i].get_ylim()[1] > angles_max_y:
                    angles_max_y = ax[nrow][i].get_ylim()[1]
            else:
                ax[nrow][i].set_visible(False)

    return nrow, angles_min_y, angles_max_y


def _plot_dihedrals_row(ax, nrow, ncols, args, state, dihedrals, diff_ordered, avg_diff, row_wise_ranges):
    dihedrals_min_y, dihedrals_max_y = 10, 0
    if state.model.cg_itp["nb_dihedrals"] != 0:
        print()
        nrow += 1
        for i in range(ncols):
            if i < state.model.cg_itp["nb_dihedrals"]:
                grp_dihedral = diff_ordered[i]

                if config.use_hists:
                    ax[nrow][i].step(
                        dihedrals[grp_dihedral]["AA"]["x"],
                        dihedrals[grp_dihedral]["AA"]["y"],
                        label="AA-mapped",
                        color=config.atom_color,
                        where="mid",
                        alpha=config.line_alpha,
                    )
                    ax[nrow][i].fill_between(
                        dihedrals[grp_dihedral]["AA"]["x"],
                        dihedrals[grp_dihedral]["AA"]["y"],
                        color=config.atom_color,
                        step="mid",
                        alpha=config.fill_alpha,
                    )
                else:
                    ax[nrow][i].plot(
                        dihedrals[grp_dihedral]["AA"]["x"],
                        dihedrals[grp_dihedral]["AA"]["y"],
                        label="AA-mapped",
                        color=config.atom_color,
                        alpha=config.line_alpha,
                    )
                    ax[nrow][i].fill_between(
                        dihedrals[grp_dihedral]["AA"]["x"],
                        dihedrals[grp_dihedral]["AA"]["y"],
                        color=config.atom_color,
                        alpha=config.fill_alpha,
                    )
                ax[nrow][i].plot(dihedrals[grp_dihedral]["AA"]["avg"], 0, color=config.atom_color, marker="D")

                if not state.mapping.atom_only:
                    ax[nrow][i].set_title(
                        f"Dihedral grp {grp_dihedral + 1} - EMD Δ {round(avg_diff[grp_dihedral], 3)}"
                    )
                    if config.use_hists:
                        ax[nrow][i].step(
                            dihedrals[grp_dihedral]["CG"]["x"],
                            dihedrals[grp_dihedral]["CG"]["y"],
                            label="CG",
                            color=config.cg_color,
                            where="mid",
                            alpha=config.line_alpha,
                        )
                        ax[nrow][i].fill_between(
                            dihedrals[grp_dihedral]["CG"]["x"],
                            dihedrals[grp_dihedral]["CG"]["y"],
                            color=config.cg_color,
                            step="mid",
                            alpha=config.fill_alpha,
                        )
                    else:
                        ax[nrow][i].plot(
                            dihedrals[grp_dihedral]["CG"]["x"],
                            dihedrals[grp_dihedral]["CG"]["y"],
                            label="CG",
                            color=config.cg_color,
                            alpha=config.line_alpha,
                        )
                        ax[nrow][i].fill_between(
                            dihedrals[grp_dihedral]["CG"]["x"],
                            dihedrals[grp_dihedral]["CG"]["y"],
                            color=config.cg_color,
                            alpha=config.fill_alpha,
                        )
                    ax[nrow][i].plot(dihedrals[grp_dihedral]["CG"]["avg"], 0, color=config.cg_color, marker="D")
                    print(
                        f"Dihedral {grp_dihedral + 1} -- AA Avg: {round(dihedrals[grp_dihedral]['AA']['avg'], 1)}° -- CG Avg: {round(dihedrals[grp_dihedral]['CG']['avg'], 1)}°"
                    )
                else:
                    ax[nrow][i].set_title(f"Dihedral grp {grp_dihedral + 1} - Avg {round(avg_diff[grp_dihedral], 1)}°")
                    print(f"Dihedral {grp_dihedral + 1} -- AA Avg: {round(dihedrals[grp_dihedral]['AA']['avg'], 1)}")
                ax[nrow][i].grid(zorder=0.5)
                if args.plotting.row_x_scaling:
                    ax[nrow][i].set_xlim(
                        np.mean(row_wise_ranges["dihedrals"][grp_dihedral]) -
                        row_wise_ranges["max_range_dihedrals"] / 2 * 1.1,
                        np.mean(row_wise_ranges["dihedrals"][grp_dihedral]) +
                        row_wise_ranges["max_range_dihedrals"] / 2 * 1.1,
                    )
                if i % 2 == 0:
                    ax[nrow][i].legend(loc="upper left")
                if ax[nrow][i].get_ylim()[0] < dihedrals_min_y:
                    dihedrals_min_y = ax[nrow][i].get_ylim()[0]
                if ax[nrow][i].get_ylim()[1] > dihedrals_max_y:
                    dihedrals_max_y = ax[nrow][i].get_ylim()[1]
            else:
                ax[nrow][i].set_visible(False)

    return nrow, dihedrals_min_y, dihedrals_max_y


def _apply_row_y_scaling(args, state, ax, constraints_bounds, bonds_bounds, angles_bounds, dihedrals_bounds):
    if not args.plotting.row_y_scaling:
        return
    nrow = -1
    if state.model.cg_itp["nb_constraints"] != 0:
        nrow += 1
        for i in range(state.model.cg_itp["nb_constraints"]):
            ax[nrow][i].set_ylim(bottom=constraints_bounds[0], top=constraints_bounds[1])
    if state.model.cg_itp["nb_bonds"] != 0:
        nrow += 1
        for i in range(state.model.cg_itp["nb_bonds"]):
            ax[nrow][i].set_ylim(bottom=bonds_bounds[0], top=bonds_bounds[1])
    if state.model.cg_itp["nb_angles"] != 0:
        nrow += 1
        for i in range(state.model.cg_itp["nb_angles"]):
            ax[nrow][i].set_ylim(bottom=angles_bounds[0], top=angles_bounds[1])
    if state.model.cg_itp["nb_dihedrals"] != 0:
        nrow += 1
        for i in range(state.model.cg_itp["nb_dihedrals"]):
            ax[nrow][i].set_ylim(bottom=dihedrals_bounds[0], top=dihedrals_bounds[1])


def _calculate_fit_scores(
    args: SwarmCGArgs,
    state: SwarmCGState,
    diff_ordered_grp_constraints,
    avg_diff_grp_constraints,
    diff_ordered_grp_bonds,
    avg_diff_grp_bonds,
    diff_ordered_grp_angles,
    avg_diff_grp_angles,
    diff_ordered_grp_dihedrals,
    avg_diff_grp_dihedrals,
    ignore_dihedrals,
    record_best_indep_params,
):
    all_dist_pairwise = ""  # for global optimization plotting
    all_emd_dist_geoms = {"constraints": [], "bonds": [], "angles": [], "dihedrals": []}

    if state.mapping.atom_only:
        return None, None, None, None, all_dist_pairwise, all_emd_dist_geoms, None

    fit_score_total = 0
    fit_score_constraints_bonds = 0
    fit_score_angles = 0
    fit_score_dihedrals = 0

    for i in range(state.model.cg_itp["nb_constraints"]):
        dist_pairwise = avg_diff_grp_constraints[diff_ordered_grp_constraints[i]]
        all_dist_pairwise += str(dist_pairwise) + " "
        all_emd_dist_geoms["constraints"].append(dist_pairwise)

        if record_best_indep_params:
            if dist_pairwise < state.opti.all_best_emd_dist_geoms["constraints"][i]:
                state.opti.all_best_emd_dist_geoms["constraints"][i] = dist_pairwise
                state.opti.all_best_params_dist_geoms["constraints"][i]["params"] = [
                    state.opti.out_itp["constraint"][i]["value"]
                ]

        dist_pairwise = dist_pairwise ** 2
        fit_score_constraints_bonds += dist_pairwise

    for i in range(state.model.cg_itp["nb_bonds"]):
        dist_pairwise = avg_diff_grp_bonds[diff_ordered_grp_bonds[i]]
        all_dist_pairwise += str(dist_pairwise) + " "
        all_emd_dist_geoms["bonds"].append(dist_pairwise)

        if record_best_indep_params:
            if dist_pairwise < state.opti.all_best_emd_dist_geoms["bonds"][i]:
                state.opti.all_best_emd_dist_geoms["bonds"][i] = dist_pairwise
                state.opti.all_best_params_dist_geoms["bonds"][i]["params"] = [
                    state.opti.out_itp["bond"][i]["value"],
                    state.opti.out_itp["bond"][i]["fct"],
                ]

        dist_pairwise = dist_pairwise ** 2
        fit_score_constraints_bonds += dist_pairwise

    for i in range(state.model.cg_itp["nb_angles"]):
        dist_pairwise = avg_diff_grp_angles[diff_ordered_grp_angles[i]]
        all_dist_pairwise += str(dist_pairwise) + " "
        all_emd_dist_geoms["angles"].append(dist_pairwise)

        if record_best_indep_params:
            if dist_pairwise < state.opti.all_best_emd_dist_geoms["angles"][i]:
                state.opti.all_best_emd_dist_geoms["angles"][i] = dist_pairwise
                state.opti.all_best_params_dist_geoms["angles"][i]["params"] = [
                    state.opti.out_itp["angle"][i]["value"],
                    state.opti.out_itp["angle"][i]["fct"],
                ]

        dist_pairwise = dist_pairwise ** 2
        fit_score_angles += dist_pairwise

    for i in range(state.model.cg_itp["nb_dihedrals"]):
        dist_pairwise = avg_diff_grp_dihedrals[diff_ordered_grp_dihedrals[i]]
        all_dist_pairwise += str(dist_pairwise) + " "
        all_emd_dist_geoms["dihedrals"].append(dist_pairwise)

        if record_best_indep_params and not ignore_dihedrals:
            if dist_pairwise < state.opti.all_best_emd_dist_geoms["dihedrals"][i]:
                state.opti.all_best_emd_dist_geoms["dihedrals"][i] = dist_pairwise
                state.opti.all_best_params_dist_geoms["dihedrals"][i]["params"] = [
                    state.opti.out_itp["dihedral"][i]["value"],
                    state.opti.out_itp["dihedral"][i]["fct"],
                ]

        dist_pairwise = dist_pairwise ** 2
        fit_score_dihedrals += dist_pairwise

    fit_score_constraints_bonds = np.sqrt(fit_score_constraints_bonds)
    fit_score_angles = np.sqrt(fit_score_angles)
    fit_score_dihedrals = np.sqrt(fit_score_dihedrals)

    fit_score_total = fit_score_constraints_bonds + fit_score_angles + fit_score_dihedrals

    fit_score_total = round(fit_score_total, 3)
    fit_score_constraints_bonds = round(fit_score_constraints_bonds, 3)
    fit_score_angles = round(fit_score_angles, 3)
    fit_score_dihedrals = round(fit_score_dihedrals, 3)
    all_dist_pairwise += "\n"

    print()
    print("Using bonds to angles/dihedrals (C) scoring constant:", args.optimization.bonds2angles_scoring_factor)
    print()
    print("Global fitness score:", fit_score_total, "(lower is better)", flush=True)
    print("  Bonds/Constraints constribution to fitness score:", fit_score_constraints_bonds, flush=True)
    print("  Angles constribution to fitness score:", fit_score_angles, flush=True)
    print("  Dihedrals constribution to fitness score:", fit_score_dihedrals, flush=True)

    eval_score = fit_score_total
    if ignore_dihedrals and state.model.cg_itp["nb_dihedrals"] > 0:
        eval_score -= fit_score_dihedrals
    sup_title = (
        "FITNESS SCORE\nTotal: "
        f"{round(eval_score, 3)} -- Constraints/Bonds: {fit_score_constraints_bonds} -- "
        f"Angles: {fit_score_angles} -- Dihedrals: {fit_score_dihedrals}"
    )
    if ignore_dihedrals and state.model.cg_itp["nb_dihedrals"] > 0:
        sup_title += " (ignored)"

    return (
        fit_score_total,
        fit_score_constraints_bonds,
        fit_score_angles,
        fit_score_dihedrals,
        all_dist_pairwise,
        all_emd_dist_geoms,
        sup_title,
    )


def compare_models(args: SwarmCGArgs, state: SwarmCGState, manual_mode=True, ignore_dihedrals=False, calc_sasa=False, record_best_indep_params=False):
    """Compare 2 models -- atomistic and CG models with plotting.

    args requires:
        inputs.cg_tpr_filename
        inputs.cg_traj_filename
        plotting.mismatch_order
        plotting.row_x_scaling
        plotting.row_y_scaling
        plotting.ncols_max
        paths.plot_filename

    state requires:
        opti.all_best_emd_dist_geoms (edited inplace)
        opti.all_best_params_dist_geoms (edited inplace)
        mapping.atom_only
        model.cg_itp
        traj.aa2cg_universe

    state creates:
        traj.cg_universe

    pass args/state to:
        compute_Rg
        compute_SASA
        get_AA_bonds_distrib
        get_CG_bonds_distrib
        get_AA_angles_distrib
        get_CG_angles_distrib
        get_AA_dihedrals_distrib
        get_CG_dihedrals_distrib
    """
    _configure_plotting()
    row_wise_ranges = _init_row_wise_ranges()

    _report_atom_only_rg(args, state)

    if not state.mapping.atom_only:
        if not _prepare_cg_universe(args, state, manual_mode, calc_sasa):
            return 0, 0, 0, 0, 0, None

    print()
    print(styling.sep_close, flush=True)
    print("| SCORING AND PLOTTING                                                                        |", flush=True)
    print(styling.sep_close, flush=True)
    print()

    constraints, avg_diff_grp_constraints, diff_ordered_grp_constraints = _compute_constraints(
        args, state, manual_mode, row_wise_ranges
    )
    bonds, avg_diff_grp_bonds, diff_ordered_grp_bonds = _compute_bonds(
        args, state, manual_mode, row_wise_ranges
    )
    angles, avg_diff_grp_angles, diff_ordered_grp_angles = _compute_angles(
        args, state, manual_mode, row_wise_ranges
    )
    dihedrals, avg_diff_grp_dihedrals, diff_ordered_grp_dihedrals = _compute_dihedrals(
        args, state, manual_mode, row_wise_ranges
    )

    nrow = -1
    nrows, ncols = _resolve_plot_layout(args, state)
    fig = plt.figure(figsize=(ncols * 3, nrows * 3))
    ax = fig.subplots(nrows=nrows, ncols=ncols, squeeze=False)

    nrow, constraints_min_y, constraints_max_y = _plot_constraints_row(
        ax, nrow, ncols, args, state, constraints, diff_ordered_grp_constraints, avg_diff_grp_constraints, row_wise_ranges
    )
    nrow, bonds_min_y, bonds_max_y = _plot_bonds_row(
        ax, nrow, ncols, args, state, bonds, diff_ordered_grp_bonds, avg_diff_grp_bonds, row_wise_ranges
    )
    nrow, angles_min_y, angles_max_y = _plot_angles_row(
        ax, nrow, ncols, args, state, angles, diff_ordered_grp_angles, avg_diff_grp_angles, row_wise_ranges
    )
    nrow, dihedrals_min_y, dihedrals_max_y = _plot_dihedrals_row(
        ax, nrow, ncols, args, state, dihedrals, diff_ordered_grp_dihedrals, avg_diff_grp_dihedrals, row_wise_ranges
    )

    _apply_row_y_scaling(
        args,
        state,
        ax,
        (constraints_min_y, constraints_max_y),
        (bonds_min_y, bonds_max_y),
        (angles_min_y, angles_max_y),
        (dihedrals_min_y, dihedrals_max_y),
    )

    fit_score_total, fit_score_constraints_bonds, fit_score_angles, fit_score_dihedrals, all_dist_pairwise, all_emd_dist_geoms, sup_title = _calculate_fit_scores(
        args,
        state,
        diff_ordered_grp_constraints,
        avg_diff_grp_constraints,
        diff_ordered_grp_bonds,
        avg_diff_grp_bonds,
        diff_ordered_grp_angles,
        avg_diff_grp_angles,
        diff_ordered_grp_dihedrals,
        avg_diff_grp_dihedrals,
        ignore_dihedrals,
        record_best_indep_params,
    )

    if state.mapping.atom_only:
        plt.tight_layout()
    else:
        plt.tight_layout(rect=[0, 0, 1, 0.9])
        plt.suptitle(sup_title)

    # here we close everything we can close because there was a memory leak from plotting
    plt.savefig(args.paths.plot_filename)
    plt.close(fig)
    print()
    print("Distributions plot written at location:\n ", args.paths.plot_filename, flush=True)
    print()

    if not manual_mode and not state.mapping.atom_only:
        return fit_score_total, fit_score_constraints_bonds, fit_score_angles, fit_score_dihedrals, all_dist_pairwise, all_emd_dist_geoms
    return
