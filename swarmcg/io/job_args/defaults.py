from types import SimpleNamespace

from swarmcg.config_types import SwarmConfig


DEFAULTS = SwarmConfig()


class BaseField(SimpleNamespace):

    @property
    def metavar(self):
        if "default" in vars(self):
            return f"({str(self.default)})".rjust(25, " ")
        else:
            return ""

    @property
    def args(self):
        attributes = vars(self)
        if "action" not in attributes:
            return {**attributes, "metavar": self.metavar}
        else:
            return attributes


# EXECUTION MODE
exec_mode = BaseField(
    dest="exec_mode",
    type=int,
    default=DEFAULTS.optimization.exec_mode,
    help="MODE 1: Tune both bonds/angles/dihedrals equilibrium values\n        and their force constants\nMODE 2: Tune only bonds/angles/dihedrals force constants\n        with FIXED equilibrium values from the prelim. CG ITP",
)
sim_type = BaseField(
    dest="sim_type",
    type=str,
    default=DEFAULTS.optimization.sim_type,
    help="Simulation type setting",
)

# GROMACS SETTINGS
gmx = BaseField(
    dest="gmx_path",
    type=str,
    default=DEFAULTS.gromacs.gmx_path,
    help="Your Gromacs alias/path",
)
nt = BaseField(
    dest="nb_threads",
    type=int,
    default=DEFAULTS.gromacs.nb_threads,
    help="Nb of threads to use, forwarded to 'gmx mdrun -nt'",
)
ntomp = BaseField(
    dest="ntomp",
    type=int,
    default=DEFAULTS.gromacs.ntomp,
    help="Nb of OpenMP threads per MPI rank, forwarded to 'gmx mdrun -ntomp'",
)
mpi = BaseField(
    dest="mpi_tasks",
    type=int,
    default=DEFAULTS.gromacs.mpi_tasks,
    help="Nb of mpi programs (X), triggers 'mpirun -np X gmx'",
)
gpu_id = BaseField(
    dest="gpu_id",
    type=str,
    default=DEFAULTS.gromacs.gpu_id,
    help="String (use quotes) space-separated list of GPU device IDs",
)
gmx_args_str = BaseField(
    dest="gmx_args_str",
    type=str,
    default=DEFAULTS.gromacs.gmx_args_str,
    help="String (use quotes) of arguments to forward to gmx mdrun\nIf provided, arguments -nt, -ntomp and -gpu_id are ignored",
)
mini_maxwarn = BaseField(
    dest="mini_maxwarn",
    type=int,
    default=DEFAULTS.gromacs.mini_maxwarn,
    help="Max. number of warnings to ignore, forwarded to gmx\ngrompp -maxwarn at each minimization step",
)
sim_kill_delay = BaseField(
    dest="sim_kill_delay",
    type=int,
    default=DEFAULTS.gromacs.sim_kill_delay,
    help="Time (s) after which to kill a simulation that has not been\nwriting into its log file, in case a simulation gets stuck",
)

# REFERENCE AA MODEL
aa_tpr = BaseField(
    dest="aa_tpr_filename",
    type=str,
    default=DEFAULTS.reference.aa_tpr_filename,
    help="Topology binary file of your reference AA simulation (TPR)",
)
aa_traj = BaseField(
    dest="aa_traj_filename",
    type=str,
    default=DEFAULTS.reference.aa_traj_filename,
    help="Trajectory file of the reference AA simulation (XTC, TRR)\nPBC are handled internally if trajectory contains box dimensions",
)
cg_map = BaseField(
    dest="cg_map_filename",
    type=str,
    default=DEFAULTS.reference.cg_map_filename,
    help="Mapping file of the atoms to CG beads (NDX-like file format)",
)
mapping = BaseField(
    dest="mapping_type",
    type=str,
    default=DEFAULTS.reference.mapping_type,
    help="Center Of Mass (COM) or Center Of Geometry (COG), for\ninterpreting the mapping file",
)

# CG MODEL
cg_itp = BaseField(
    dest="cg_itp_filename",
    type=str,
    default=DEFAULTS.cg_model.cg_itp_filename,
    help="ITP file of the CG model to optimize",
)
user_params = BaseField(
    dest="user_input",
    default=DEFAULTS.cg_model.user_input,
    help="If absent, only the BI is used as starting point for parametrization\nIf present, parameters in the input ITP files are considered",
    action="store_true"
)
cg_gro = BaseField(
    dest="gro_input_filename",
    type=str,
    default=DEFAULTS.cg_model.gro_input_filename,
    help="Starting GRO file used for iterative simulation\nWill be minimized and relaxed before each MD run",
)
cg_top = BaseField(
    dest="top_input_filename",
    type=str,
    default=DEFAULTS.cg_model.top_input_filename,
    help="TOP file used for iterative simulation",
)
cg_tpr = BaseField(
    dest="cg_tpr_filename",
    type=str,
    default=DEFAULTS.cg_model.cg_tpr_filename,
    help="TPR file of your CG simulation (omit for solo AA inspection)",
)
cg_traj = BaseField(
    dest="cg_traj_filename",
    type=str,
    default=DEFAULTS.cg_model.cg_traj_filename,
    help="XTC file of your CG trajectory (omit for solo AA inspection)",
)
cg_mdp_mini = BaseField(
    dest="mdp_minimization_filename",
    type=str,
    default=DEFAULTS.simulation.mdp_minimization_filename,
    help="MDP file used for minimization runs",
)
cg_mdp_equi = BaseField(
    dest="mdp_equi_filename",
    type=str,
    default=DEFAULTS.simulation.mdp_equi_filename,
    help="MDP file used for equilibration runs",
)
cg_mdp_md = BaseField(
    dest="mdp_md_filename",
    type=str,
    default=DEFAULTS.simulation.mdp_md_filename,
    help="MDP file used for the MD runs analyzed for optimization",
)

# FILES HANDLING
in_dir = BaseField(
    dest="input_folder",
    type=str,
    default="",
    help="Additional prefix path used to find argument-provided files\nIf ambiguous, files found without prefix are preferred",
)
out_dir = BaseField(
    dest="output_folder",
    type=str,
    default="",
    help="Directory where to store all outputs of this program\nDefault -out_dir is named after timestamp",
)
opti_dir = BaseField(
    dest="opti_dirname",
    type=str,
    help="Directory created by module 'scg_optimize' that contains all files\ngenerated during the optimization procedure",
)
o_an = BaseField(
    dest="plot_filename",
    type=str,
    default="opti_summary.png",
    help="Filename for the output plot, produced in directory -opti_dir.\nExtension/format can be one of: eps, pdf, pgf, png, ps, raw, rgba,\nsvg, svgz",
)
o_ev = BaseField(
    dest="plot_filename",
    type=str,
    default="distributions.png",
    help="Filename for the output plot (extension/format can be one of:\neps, pdf, pgf, png, ps, raw, rgba, svg, svgz)",
)

# CG MODEL FORCE CONSTANTS
max_fct_bonds_f1 = BaseField(
    dest="default_max_fct_bonds_opti",
    type=float,
    default=DEFAULTS.optimization.default_max_fct_bonds_opti,
    help="Max. force constants for bonds function 1 (kJ.mol⁻¹.nm⁻²)",
)
max_fct_angles_f1 = BaseField(
    dest="default_max_fct_angles_opti_f1",
    type=float,
    default=DEFAULTS.optimization.default_max_fct_angles_opti_f1,
    help="Max. force ct. for angles function 1 (kJ.mol⁻¹.rad⁻²)",
)
max_fct_angles_f2 = BaseField(
    dest="default_max_fct_angles_opti_f2",
    type=float,
    default=DEFAULTS.optimization.default_max_fct_angles_opti_f2,
    help="Max. force ct. for angles function 2 (kJ.mol⁻¹)",
)
max_fct_angles_f10 = BaseField(
    dest="default_max_fct_angles_opti_f10",
    type=float,
    default=DEFAULTS.optimization.default_max_fct_angles_opti_f10,
    help="Max. force ct. for angles function 10 (kJ.mol⁻¹)",
)
max_fct_dihedrals_f149 = BaseField(
    dest="default_abs_range_fct_dihedrals_opti_func_with_mult",
    type=float,
    default=DEFAULTS.optimization.default_abs_range_fct_dihedrals_opti_func_with_mult,
    help="Upper bound for canonical nonnegative force constants of dihedral\nfunctions 1 and 4 (kJ.mol⁻¹)",
)
max_fct_dihedrals_f2 = BaseField(
    dest="default_abs_range_fct_dihedrals_opti_func_without_mult",
    type=float,
    default=DEFAULTS.optimization.default_abs_range_fct_dihedrals_opti_func_without_mult,
    help="Max. force ct. for dihedrals function 2 (abs. kJ.mol⁻¹.rad⁻²)",
)
max_rb_coeff = BaseField(
    dest="max_abs_rb_coefficient",
    type=float,
    default=DEFAULTS.optimization.max_abs_rb_coefficient,
    help="Optional absolute bound for independent RB C1..C5 coefficients\n(kJ.mol⁻¹; default is derived from the target PMF)",
)
max_cbt_coeff = BaseField(
    dest="max_abs_cbt_effective_coefficient",
    type=float,
    default=DEFAULTS.optimization.max_abs_cbt_effective_coefficient,
    help="Optional absolute bound for effective CBT coefficients k_phi*a_i\n(kJ.mol⁻¹; default is derived from the target PMF)",
)
# MODEL SCORING
cg_time_short = BaseField(
    dest="sim_duration_short",
    type=float,
    default=DEFAULTS.simulation.sim_duration_short,
    help="Simulation time (ns) of the MD runs analyzed for optimization\nIn opti. cycles 1 and 2, this will modify MDP file for the MD runs",
)
cg_time_long = BaseField(
    dest="sim_duration_long",
    type=float,
    default=DEFAULTS.simulation.sim_duration_long,
    help="Simulation time (ns) of the MD runs analyzed for optimization\nIn opti. cycle 3, this will modify MDP file for the MD runs",
)
b2a_score_fact = BaseField(
    dest="bonds2angles_scoring_factor",
    type=float,
    default=DEFAULTS.optimization.bonds2angles_scoring_factor,
    help="Weight of bonds vs. angles/dihedrals (constant C in the paper)\nAt 500, bonds mismatch 0.4 Å == angles/dihedrals mismatch 20°\nDecreasing would linearly increase the weight of bonds",
)
bw_constraints = BaseField(
    dest="bw_constraints",
    type=float,
    default=DEFAULTS.optimization.bw_constraints,
    help="Bandwidth for constraints distributions processing (nm)",
)
bw_bonds = BaseField(
    dest="bw_bonds",
    type=float,
    default=DEFAULTS.optimization.bw_bonds,
    help="Bandwidth for bonds distributions processing (nm)",
)
bw_angles = BaseField(
    dest="bw_angles",
    type=float,
    default=DEFAULTS.optimization.bw_angles,
    help="Bandwidth for angles distributions processing (degrees)",
)
bw_dihedrals = BaseField(
    dest="bw_dihedrals",
    type=float,
    default=DEFAULTS.optimization.bw_dihedrals,
    help="Bandwidth for dihedrals distributions processing (degrees)",
)
disable_x_scaling = BaseField(
    dest="row_x_scaling",
    default=True,
    help="Disable auto-scaling of X axis across each row of the plot",
    action="store_false",
)
disable_y_scaling = BaseField(
    dest="row_y_scaling",
    default=True,
    help="Disable auto-scaling of Y axis across each row of the plot",
    action="store_false",
)
bonds_max_range = BaseField(
    dest="bonded_max_range",
    type=float,
    default=DEFAULTS.optimization.bonded_max_range,
    help="Max. range of grid for bonds/constraints distributions (nm)",
)

sasa = BaseField(
    dest="calculate_sasa",
    default=DEFAULTS.output.calculate_sasa,
    help="Calculate SASA as an optional diagnostic; SASA never affects fitness",
    action="store_true",
)

# MODEL SCALING
aa_rg_offset = BaseField(
    dest="aa_rg_offset",
    type=float,
    default=DEFAULTS.reference.aa_rg_offset,
    help="Radius of gyration offset (nm) to be applied to AA data\naccording to your potential bonds rescaling (for display only)",
)
bonds_scaling = BaseField(
    dest="bonds_scaling",
    type=float,
    default=DEFAULTS.optimization.bonds_scaling,
    help="Scaling factor for ALL AA-mapped bonds/constraints lengths\nOnly one of arguments -bonds_scaling, -bonds_scaling_str\nand -min_bonds_length can be provided",
)
bonds_scaling_str = BaseField(
    dest="bonds_scaling_str",
    type=str,
    default=DEFAULTS.optimization.bonds_scaling_str,
    # constraints and bonds ids + their required target AA-mapped distributions rescaled averages
    help="String (use quotes) for providing SPECIFIC bonds/constraints\ngroups ids and their required lengths (nm, rescaled\ndistributions avg to use as target for optimization)\nEx: \'C1 0.23 B5 0.27\' will modify distributions of constraints\ngrp 1 and bonds grp 5 to averages 0.23 and 0.27 nm",
)
min_bonds_length = BaseField(
    dest="min_bonds_length",
    type=float,
    default=DEFAULTS.optimization.min_bonds_length,
    help="Required minimum length of a bond or constraint between 2 CG\nbeads (distributions avg in nm) used both as:\n1. Threshold to identify ALL short AA-mapped bonds/constraints\n2. Target avg to rescale ALL those bonds/constraints",
)

# FIGURE DISPLAY
mismatch_ordering = BaseField(
    dest="mismatch_order",
    default=DEFAULTS.output.mismatch_order,
    help="Enables ordering of bonds/angles/dihedrals by mismatch score\nbetween pairwise AA-mapped/CG distributions (can help diagnosis)",
    action="store_true",
)
ncols = BaseField(
    dest="ncols_max",
    type=int,
    default=DEFAULTS.output.ncols_max,
    help="Max. nb of columns displayed in figure",
)  # TODO: make this a line return in plot instead of ignoring groups
plot_scale = BaseField(
    dest="plot_scale",
    type=float,
    default=DEFAULTS.output.plot_scale,
    help="Scale factor of the plot",
)

# OTHERS
temp = BaseField(
    dest="temp",
    type=float,
    default=DEFAULTS.simulation.temp,
    help="Temperature used to perform Boltzmann inversion (K)",
)
keep_all_sims = BaseField(
    dest="keep_all_sims",
    default=DEFAULTS.optimization.keep_all_sims,
    help="Store all gmx files for all simulations, may use disk space",
    action="store_true",
)
verbose = BaseField(
    dest="verbose",
    default=DEFAULTS.output.verbose,
    help="Display more processing details & error traceback",
    action="store_true"
)
nobanner = BaseField(
    dest="no_banner",
    default=False,
    help="Suppress the ASCII banner header and print a single-line header instead",
    action="store_true",
)
help = BaseField(
    help="Show this help message and exit",
    action="help"
)
