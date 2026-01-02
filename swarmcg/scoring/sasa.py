import os
from pathlib import Path

import numpy as np

from swarmcg import config
from swarmcg.shared import exceptions
from swarmcg.io import read_xvg_col
from swarmcg.simulations.runner import exec_gmx
from swarmcg.context import SwarmCGArgs, SwarmCGState

PROBE_RADIUS = 0.26  # nm


def compute_SASA(args: SwarmCGArgs, state: SwarmCGState, traj_type):
    """Compute average SASA.

    args/state requires:
        cg_itp
        gmx_path
        aa_tpr_filename
        aa_traj_filename
        cg_ndx_filename
        aa_mapped_tpr_sasa_filename
        cg_tpr_filename
        cg_traj_filename
        cg_sasa_filename

    state creates:
        cg_ndx_filename
        aa_traj_whole_filename
        aa_mapped_traj_whole_filename
        aa_mapped_sasa_filename
        aa_mapped_tpr_sasa_filename
        sasa_aa_mapped
        sasa_aa_mapped_std
        cg_traj_whole_filename
        cg_sasa_filename
        sasa_cg
        sasa_cg_std
    """
    # NOTE: currently this is just COM mappping via GMX to get the SASA, so it's approximative but that's OK
    #       this works with calls to GMX because only library MDTraj can compute SASA (not MDAnalysis)
    # TODO: MDA is working on it, keep an eye on this: https://github.com/MDAnalysis/mdanalysis/issues/2439
    base_dir = Path(args.inputs.cg_tpr_filename).resolve().parent

    if traj_type == 'AA':
        raise exceptions.InputArgumentError("Compute_SASA not implemented for AA atm")

    elif traj_type == 'AA_mapped':

        # NOTE: here we assume the VS all come after the real beads in the ITP [ atoms ] field
        #       we generate a new truncated TPR so that we can use GMX sasa, this is shit but no choice atm
        nb_beads_real = len(state.model.cg_itp["real_beads_ids"])

        # generate an cg_map.ndx file with the number of beads,
        # so we can call SASA on this group and we will have exactly the content we want
        state.files.cg_ndx_filename = str(base_dir / "cg_index.ndx")
        with open(state.files.cg_ndx_filename, 'w') as fp:
            beads_ids_str = ' '.join(
                map(str, list(range(1, nb_beads_real + 1))))  # includes VS if present
            fp.write('[' + state.model.cg_itp['moleculetype']['molname'] + ' ]\n' + beads_ids_str + '\n')

        # TODO: all these paths need to be fixed to allow for SASA calculation within evaluate_model.py
        #       that's why it's disabled atm

        state.files.aa_traj_whole_filename = str(base_dir / "aa_traj_whole.xtc")
        state.files.aa_mapped_traj_whole_filename = str(base_dir / "aa_mapped_traj_whole.xtc")
        state.files.aa_mapped_sasa_filename = str(base_dir / "aa_mapped_sasa.xvg")
        state.files.aa_mapped_tpr_sasa_filename = str(base_dir / "aa_mapped_tpr_sasa.tpr")

        aa_tpr_path = Path(args.inputs.aa_tpr_filename).resolve()
        aa_traj_path = Path(args.inputs.aa_traj_filename).resolve()
        cg_map_path = Path(args.inputs.cg_map_filename).resolve()

        non_zero_return_code = False

        # first make traj whole
        gmx_cmd = f'seq 0 1 | {args.runtime.gmx_path} trjconv -s {aa_tpr_path} -f {aa_traj_path} -pbc mol -o {state.files.aa_traj_whole_filename}'
        return_code = exec_gmx(gmx_cmd, workdir=base_dir)
        if return_code != 0:
            non_zero_return_code = True

        # then map AA traj
        if not non_zero_return_code:
            gmx_cmd = f'seq 0 {nb_beads_real - 1} | {args.runtime.gmx_path} traj -f {state.files.aa_traj_whole_filename} -s {aa_tpr_path} -oxt {state.files.aa_mapped_traj_whole_filename} -n {cg_map_path} -com -ng {nb_beads_real}'
            return_code = exec_gmx(gmx_cmd, workdir=base_dir)
            if return_code != 0:
                non_zero_return_code = True

        # truncate the CG TPR to get only real beads
        if not non_zero_return_code:
            gmx_cmd = f'{args.runtime.gmx_path} convert-tpr -s {args.inputs.cg_tpr_filename} -n {state.files.cg_ndx_filename} -o {state.files.aa_mapped_tpr_sasa_filename}'
            return_code = exec_gmx(gmx_cmd, workdir=base_dir)
            if return_code != 0:
                non_zero_return_code = True

        # finally get sasa
        if not non_zero_return_code:
            gmx_cmd = f'{args.runtime.gmx_path} sasa -s {state.files.aa_mapped_tpr_sasa_filename} -f {state.files.aa_mapped_traj_whole_filename} -n {state.files.cg_ndx_filename} -surface 0 -o {state.files.aa_mapped_sasa_filename} -probe {PROBE_RADIUS}'
            return_code = exec_gmx(gmx_cmd, workdir=base_dir)
            if return_code != 0:
                non_zero_return_code = True

        if non_zero_return_code:
            msg = (
                "There were some errors while calculating SASA for AA-mapped trajectory.\n"
                "Please check the error messages displayed above."
            )
            raise exceptions.ComputationError(msg)
        else:
            sasa_aa_mapped_per_frame = read_xvg_col(state.files.aa_mapped_sasa_filename, 1)
            state.model.sasa_aa_mapped = round(np.mean(sasa_aa_mapped_per_frame), 2)
            state.model.sasa_aa_mapped_std = round(np.std(sasa_aa_mapped_per_frame), 2)

    elif traj_type == 'CG':

        state.files.cg_traj_whole_filename = str(base_dir / "md_whole.xtc")
        state.files.cg_sasa_filename = str(base_dir / "cg_sasa.xvg")
        non_zero_return_code = False

        if state.files.cg_ndx_filename is None:
            nb_beads_real = len(state.model.cg_itp["real_beads_ids"])
            state.files.cg_ndx_filename = str(base_dir / "cg_index.ndx")
            with open(state.files.cg_ndx_filename, 'w') as fp:
                beads_ids_str = ' '.join(map(str, list(range(1, nb_beads_real + 1))))
                fp.write('[' + state.model.cg_itp['moleculetype']['molname'] + ' ]\n' + beads_ids_str + '\n')

        # first make traj whole
        gmx_cmd = f'seq 0 1 | {args.runtime.gmx_path} trjconv -s {args.inputs.cg_tpr_filename} -f {args.inputs.cg_traj_filename} -pbc mol -o {state.files.cg_traj_whole_filename}'
        return_code = exec_gmx(gmx_cmd, workdir=base_dir)
        if return_code != 0:
            non_zero_return_code = True

        # then compute SASA
        if not non_zero_return_code:
            # surface to choose the index group, 2 is the molecule even when there are ions (0 and 1 are System and Others)
            gmx_cmd = f'{args.runtime.gmx_path} sasa -s {args.inputs.cg_tpr_filename} -f {state.files.cg_traj_whole_filename} -n {state.files.cg_ndx_filename} -surface 0 -o {state.files.cg_sasa_filename} -probe {PROBE_RADIUS}'
            return_code = exec_gmx(gmx_cmd, workdir=base_dir)
            if return_code != 0:
                non_zero_return_code = True

        if non_zero_return_code or not os.path.isfile(state.files.cg_sasa_filename):  # extra security
            state.model.sasa_cg, state.model.sasa_cg_std = None, None
        else:
            sasa_cg_per_frame = read_xvg_col(state.files.cg_sasa_filename, 1)
            state.model.sasa_cg = round(np.mean(sasa_cg_per_frame), 2)
            state.model.sasa_cg_std = round(np.std(sasa_cg_per_frame), 2)

    else:
        raise exceptions.ComputationError('Code error compute SASA')
