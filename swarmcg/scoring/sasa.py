import os
import numpy as np

from swarmcg import config as global_config
from swarmcg.shared import exceptions
from swarmcg.io import read_xvg_col
from swarmcg.simulations.runner import exec_gmx
from swarmcg.config_types import SwarmConfig

PROBE_RADIUS = 0.26  # nm

def compute_SASA(config: SwarmConfig, cg_itp, traj_type):
    """Compute average SASA.
    Returns: (sasa_avg, sasa_std) in nm
    """
    gmx_path = config.gromacs.gmx_path
    
    # NOTE: currently this is just COM mappping via GMX to get the SASA, so it's approximative but that's OK
    #       this works with calls to GMX because only library MDTraj can compute SASA (not MDAnalysis)

    if traj_type == 'AA':
        raise exceptions.InvalidArgument('Compute_SASA not implemented for AA atm')

    elif traj_type == 'AA_mapped':

        # NOTE: here we assume the VS all come after the real beads in the ITP [ atoms ] field
        nb_beads_real = len(cg_itp["real_beads_ids"])

        # generate an cg_map.ndx file with the number of beads
        cg_ndx_filename = '../' + global_config.input_sim_files_dirname + '/cg_index.ndx'
        with open(cg_ndx_filename, 'w') as fp:
            beads_ids_str = ' '.join(
                map(str, list(range(1, nb_beads_real + 1))))  # includes VS if present
            fp.write('[' + cg_itp['moleculetype']['molname'] + ' ]\n' + beads_ids_str + '\n')

        # TODO: all these paths need to be fixed to allow for SASA calculation within evaluate_model.py
        
        # We need to access filenames from config. 
        # In the original code, ns.aa_tpr_filename etc were used.
        # These are in config.reference.aa_tpr_filename
        
        aa_tpr_filename = config.reference.aa_tpr_filename
        aa_traj_filename = config.reference.aa_traj_filename
        cg_map_filename = config.reference.cg_map_filename  # Mapping file (ndx)

        aa_traj_whole_filename = '../' + global_config.input_sim_files_dirname + '/aa_traj_whole.xtc'
        aa_mapped_traj_whole_filename = '../' + global_config.input_sim_files_dirname + '/aa_mapped_traj_whole.xtc'
        aa_mapped_sasa_filename = '../' + global_config.input_sim_files_dirname + '/aa_mapped_sasa.xvg'
        aa_mapped_tpr_sasa_filename = '../' + global_config.input_sim_files_dirname + '/aa_mapped_tpr_sasa.tpr'

        non_zero_return_code = False

        # first make traj whole
        gmx_cmd = f'seq 0 1 | {gmx_path} trjconv -s ../../{aa_tpr_filename} -f ../../{aa_traj_filename} -pbc mol -o {aa_traj_whole_filename}'
        return_code = exec_gmx(gmx_cmd)
        if return_code != 0:
            non_zero_return_code = True

        # then map AA traj
        if not non_zero_return_code:
            gmx_cmd = f'seq 0 {nb_beads_real - 1} | {gmx_path} traj -f {aa_traj_whole_filename} -s ../../{aa_tpr_filename} -oxt {aa_mapped_traj_whole_filename} -n ../../{cg_map_filename} -com -ng {nb_beads_real}'
            return_code = exec_gmx(gmx_cmd)
            if return_code != 0:
                print("Failed GMX Command:", gmx_cmd) # Debug help
                non_zero_return_code = True

        # truncate the CG TPR to get only real beads
        if not non_zero_return_code:
            gmx_cmd = f'{gmx_path} convert-tpr -s md.tpr -n {cg_ndx_filename} -o {aa_mapped_tpr_sasa_filename}'
            return_code = exec_gmx(gmx_cmd)
            if return_code != 0:
                print("Failed GMX Command:", gmx_cmd)
                non_zero_return_code = True

        # finally get sasa
        if not non_zero_return_code:
            gmx_cmd = f'{gmx_path} sasa -s {aa_mapped_tpr_sasa_filename} -f {aa_mapped_traj_whole_filename} -n {cg_ndx_filename} -surface 0 -o {aa_mapped_sasa_filename} -probe {PROBE_RADIUS}'
            return_code = exec_gmx(gmx_cmd)
            if return_code != 0:
                print("Failed GMX Command:", gmx_cmd)
                non_zero_return_code = True

        if non_zero_return_code:
            msg = (
                "There were some errors while calculating SASA for AA-mapped trajectory.\n"
                "Please check the error messages displayed above."
            )
            raise exceptions.ComputationError(msg)
        else:
            sasa_aa_mapped_per_frame = read_xvg_col(aa_mapped_sasa_filename, 1)
            sasa_aa_mapped = round(np.mean(sasa_aa_mapped_per_frame), 2)
            sasa_aa_mapped_std = round(np.std(sasa_aa_mapped_per_frame), 2)
            
            return sasa_aa_mapped, sasa_aa_mapped_std

    elif traj_type == 'CG':
        
        cg_tpr_filename = 'md.tpr' # Assumed standard output name in exec folder
        cg_traj_filename = 'md.xtc'
        
        cg_traj_whole_filename = 'md_whole.xtc'
        cg_sasa_filename = 'cg_sasa.xvg'
        cg_ndx_filename = 'cg_index_sasa.ndx' # We need this for CG too? Original used ns.cg_ndx_filename which was set in AA_mapped block primarily? 
        
        # Original code reused ns.cg_ndx_filename or expected it to be present?
        # In evaluate_model/optimize_model, AA_mapped is usually calculated first, so ns.cg_ndx_filename is set.
        # We need to recreate it if it doesn't exist or specific to CG step.
        # But wait, logic: "generate an cg_map.ndx file... so we can call SASA".
        # If we are in CG mode, we are in the execution folder.
        
        nb_beads_real = len(cg_itp["real_beads_ids"])
        with open(cg_ndx_filename, 'w') as fp:
             beads_ids_str = ' '.join(map(str, list(range(1, nb_beads_real + 1))))
             fp.write('[' + cg_itp['moleculetype']['molname'] + ' ]\n' + beads_ids_str + '\n')

        non_zero_return_code = False

        # first make traj whole
        gmx_cmd = f'seq 0 1 | {gmx_path} trjconv -s {cg_tpr_filename} -f {cg_traj_filename} -pbc mol -o {cg_traj_whole_filename}'
        return_code = exec_gmx(gmx_cmd)
        if return_code != 0:
            non_zero_return_code = True

        # then compute SASA
        if not non_zero_return_code:
            gmx_cmd = f'{gmx_path} sasa -s {cg_tpr_filename} -f {cg_traj_whole_filename} -n {cg_ndx_filename} -surface 0 -o {cg_sasa_filename} -probe {PROBE_RADIUS}'
            return_code = exec_gmx(gmx_cmd)
            if return_code != 0:
                non_zero_return_code = True

        if non_zero_return_code or not os.path.isfile(cg_sasa_filename):
            return None, None
        else:
            sasa_cg_per_frame = read_xvg_col(cg_sasa_filename, 1)
            sasa_cg = round(np.mean(sasa_cg_per_frame), 2)
            sasa_cg_std = round(np.std(sasa_cg_per_frame), 2)
            return sasa_cg, sasa_cg_std

    else:
        raise exceptions.ComputationError('Code error compute SASA')
