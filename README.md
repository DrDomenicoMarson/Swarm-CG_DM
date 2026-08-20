# Swarm-CG

Swarm-CG is designed for automatically optimizing the bonded terms of a MARTINI-based coarse-grained (CG) molecular model, in explicit or implicit solvent, with respect to a reference all-atom (AA) trajectory and starting from a preliminary CG model (topology and non-bonded parameters). The package is designed for usage with Gromacs (which you whould install) and contains 3 modules for:

1. Evaluating the bonded parametrization of a CG model
2. Optimizing bonded terms of a CG model
3. Monitoring an optimization procedure

![Swarm-CG](https://raw.githubusercontent.com/GMPavanLab/Swarm-CG/master/images/TOC_Swarm-CG_paper.png)

Swarm-CG works with MARTINI version 2 or 3. The AA-to-CG mapping can be interpreted as center of mass (COM) or center of geometry (COG). Virtual sites handling is under development and will be available soon.

### Publication

> Empereur-mot, C.; Pesce, L.; Bochicchio, D.; Capelli, R.; Perego, C.; Pavan, G.M. (2020) Swarm-CG: Automatic Parametrization of Bonded Terms in MARTINI-based Coarse-Grained Models of Simple to Complex Molecules via Fuzzy Self-Tuning Particle Swarm Optimization. [ACS Omega](https://pubs.acs.org/doi/10.1021/acsomega.0c05469)

### Installation & Usage

This fork supports Python 3.13+ and GROMACS 2025+. Older Python and GROMACS
releases are outside the supported platform.

	python -m pip install .          # creates the 3 entrypoints below
	
	scg_evaluate -h                  # see point 1
	scg_optimize -h                  # see point 2
	scg_monitor -h                   # see point 3

To better handle sampling in symmetrical molecules you can form groups of bonds/angles/dihedrals that Swarm-CG will consider identical, using line returns and/or comments in the topology (ITP) file. AA-mapped distributions will be averaged within groups to create the references used for evaluation (see point 1) or as target of the optimization procedure (see point 2). For optimization, identical parameters will be used for the bonds/angles/dihedrals within each group.

Here is an ITP file extract from the demonstration data of [PAMAM G1](https://github.com/GMPavanLab/Swarm-CG/tree/master/G1_DATA/cg_model.itp):

	[ bonds ]
	;   i     j   funct   length   force.c.   
	; bond group 1
	    1     2       1        0         0           ; B1
	; bond group 2
	    1     3       1        0         0           ; B2
	    1     9       1        0         0           ; B2
	; bond group 3
	    3     4       1        0         0           ; B3
	    9    10       1        0         0           ; B3

### 1. Evaluate bonded parametrization of a CG model

The module `scg_evaluate` enables quick evaluation of the fit of bond, angle and dihedral distributions between a CG model trajectory and a reference AA model trajectory of an identical molecule, by producing a single comprehensive figure.

	scg_evaluate -aa_tpr G1_DATA/aa_topol.tpr -aa_traj G1_DATA/aa_traj.xtc -cg_map G1_DATA/cg_map.ndx -cg_itp G1_DATA/cg_model.itp -cg_tpr G1_DATA/cg_topol.tpr -cg_traj G1_DATA/cg_traj.xtc

It can also be used for inspecting AA-mapped distributions exclusively.

	scg_evaluate -aa_tpr G1_DATA/aa_topol.tpr -aa_traj G1_DATA/aa_traj.xtc -cg_map G1_DATA/cg_map.ndx -cg_itp G1_DATA/cg_model.itp

This module is particularly useful to assess the need to run an optimization procedure (assuming one already has a CG model). It is also suited to the assessment of geometrical changes triggered by a modification of CG beads types (defining non-bonded parameters) or after manually editing bonded parameters while working on a model. This command also provides publication-quality figures to support the parametrization of your models (also in vectorized formats). Radius of gyration (Rg) is always calculated. SASA is an optional diagnostic enabled with `-sasa`/`--sasa`; a SASA failure is reported but never changes a fitness score or model selection.

### 2. Optimize bonded terms of a CG model

The module `scg_optimize` allows to automatically optimize the bonded parameters of a CG model according to a reference AA trajectory. To this end, several simulations will be run to explore and evaluate the relevance of different sets of bonded parameters, using 3 optimization cycles.

For example, using demonstration data of [PAMAM G1](https://github.com/GMPavanLab/Swarm-CG/tree/master/G1_DATA):

	scg_optimize -in_dir G1_DATA/ -gmx gmx

Which will use all default filenames of the software and is *exactly identical* to this command:

	scg_optimize -aa_tpr G1_DATA/aa_topol.tpr -aa_traj G1_DATA/aa_traj.xtc -cg_map G1_DATA/cg_map.ndx -cg_itp G1_DATA/cg_model.itp -cg_gro G1_DATA/start_conf.gro -cg_top G1_DATA/system.top -cg_mdp_mini G1_DATA/mini.mdp -cg_mdp_equi G1_DATA/equi.mdp -cg_mdp_md G1_DATA/md.mdp -gmx gmx

We recommend to first prepare files in a directory to be fed to Swarm-CG via argument `-in_dir`.

The input is composed of:

1. An AA reference trajectory (TPR + XTC/TRR)
2. The AA to CG mapping (NDX)
3. A preliminary CG model (ITP, equilibrium values and force constants can be initialized arbitrarily to e.g. 0)
4. A CG configuration used as starting point of each iterative optimization run (GRO file, from a mapped AA frame and solvated if necessary)
5. Other simulation files (TOP and MDP, notably with your barostat and thermostat choices)

At all times during execution, the best parametrized model is accessible in the optimization output folder at `out_dir/optimized_CG_model/cg_model.itp`. The approximate Boltzmann-initialized parameters are also available at `out_dir/boltzmann_inv_CG_model/cg_model.itp`. Initialization fits the normalized marginal PMF `U = -kBT ln(p/pmax)` using only occupied histogram bins and a range derived from the complete finite sample extrema, so duplicating trajectory frames does not change the seed. Equilibrium values remain fixed to the reference-derived value in execution mode 1 and to the input ITP value in mode 2. As in the paper, this is an approximate reference particle rather than a replacement for swarm optimization; no radial or angular Jacobian correction is applied.

Reference histograms are strict: every expected AA geometry sample must be finite, fall inside its scoring domain, and the resulting probability masses must sum to one. CG histograms retain the same expected denominator. A missing, non-finite, underflowing, or overflowing CG sample therefore remains missing mass rather than disappearing through renormalization; EMD charges that mass the largest transport cost in the active grid and plots annotate the missing fraction. A completely undefined CG geometry receives the grid maximum, while a true evaluation failure remains strictly worse than every finite histogram score.

The AA trajectory is mapped on-the-fly (if atoms are mapped to multiple CG beads, atom masses are split accordingly). The AA trajectory must contain box information for PBC handling, otherwise it is assumed the molecule is "unwrapped" already. Only the MDP file provided via arg `-cg_mdp_md` will be modified to adjust `nsteps` according to arguments `-cg_time_short` and `cg_time_long`, taking into account the timestep `ts` you provided. To minimize the execution time of `scg_optimize`, equilibration should stay short (e.g. 50-500 fs) and so should the optimization cycles 1 and 2 (via arg `-cg_time_short` e.g. 10-20 ns). To maximize the precision of `scg_optimize`, optimization cycle 3 must always use longer simulation times (via arg `-cg_time_long` e.g. 25-100 ns). Execution times should vary between 4h to 24h according to parameters and hardware used.

For information about the different execution modes, please see paper sections 2.4 and 4 and command help (-h).

The paper's class-wise L2 composition and bond conversion factor of 500 are
retained. Production (`OPTIMAL`) swarm sizes and iteration limits are based on
the actual number `D` of free parameters: `max(3, round(2 + sqrt(D)))`
particles and `round(6 + sqrt(D))` iterations. Each optimization cycle starts
from the preceding cycle's optimum; global-best selection remains a separate
whole-model decision.

Supported added bonded terms include restricted bending (angle function 10),
Ryckaert--Bellemans (dihedral function 3), and combined bending--torsion
(dihedral function 11). Function-10 equilibrium angles must lie in 10--170°.
Before optimization, every function-10 angle in the starting CG GRO file is
checked against the same interval using minimum-image PBC when box data are
available; the GRO must contain the modeled molecule as its first ITP-sized
atom block. Evaluation-only workflows do not require this starting structure.
RB is optimized through the five force-relevant coefficients C1--C5 and is
written with `C0 = -sum(C1..C5)`. CBT is optimized through the five effective
products `B_i = k_phi*a_i`; output uses `k_phi = max(abs(B_i))` and
`a_i = B_i/k_phi` (or all zeros for the zero potential). Consequently,
canonical output parameters can differ from input while preserving RB forces
or the full CBT energy. Default RB/CBT bounds are derived from the target PMF;
use `-max_rb_coeff` or `-max_cbt_coeff` when a deliberately larger input range
is required. RB initialization requires six independent columns (a free PMF
intercept plus C1--C5); sparse targets retain the canonical input seed and
emit a warning. CBT retains its canonical input seed because its angular
coupling is not identifiable from a one-dimensional torsional marginal. On
the first cycle where an RB or CBT group is active, all other polynomial
particles explore the full coefficient bounds independently; later cycles
refine locally around the staged optimum.

RB and CBT torsional marginals are mirror-symmetric by construction. Swarm-CG
reports their histogram-to-mirror total-variation distance and warns above
0.10, because an asymmetric target cannot be reproduced directly by either
polynomial form. RB-related 1--4 exclusions and `nrexcl` choices are
force-field-specific; Swarm-CG deliberately neither infers nor modifies them.

Dihedral summaries use circular statistics. Symmetric targets can have an
undefined first circular moment: RB and CBT continue normally and plots report
the mean as unavailable. In execution mode 1, periodic functions 1 and 4 with
multiplicity `n` use the direction of the order-`n` moment
`<exp(i*n*phi)>`, without dividing that direction by `n`, and search the full
360° phase interval around the resulting GROMACS phase. Function 2 continues
to use the first moment. If the required moment is undefined, improve sampling
or use execution mode 2 to retain the ITP phase.

Periodic functions 1 and 4 are canonicalized to `k >= 0`. A negative input
record is read as `k' = abs(k)` and `delta' = wrap(delta + 180°)`; this
preserves forces and changes the potential only by a permissible constant
energy offset. Written phases are normalized and written force constants are
always nonnegative. Zero-force records retain their normalized input phase.

### 3. Monitor an ongoing CG model optimization

Optimization procedures can be monitored at any point during execution. The module `scg_monitor` produces a visual summary (see paper Fig. 3) of the progress of an optimization procedure started with module `scg_optimize`. The plot will be produced in the directory provided via arg `-opti_dir`.

	scg_monitor -opti_dir MODEL_OPTI__STARTED_03-07-2020_10h_12m_15s

Each completed evaluation is flushed as one strict schema-version-1 JSON record
in `.internal/optimization_history.jsonl`. This is the monitor's only supported
input format; optimization directories created by older Swarm-CG versions and
their whitespace recap files are intentionally unsupported. While an
optimization is active, `scg_monitor` tolerates a syntactically incomplete
final line but rejects malformed earlier records.

See the help (`-h` or `--help`) for a complete description of `scg_monitor`
output. Rg and requested SASA values may be rough estimates because they come
from short optimization simulations and should be validated with longer
trajectories. The monitor marks stalled, crashed, and scoring-failed
evaluations explicitly and omits SASA summaries and panels when SASA was not
requested or no result is available. Using `scg_evaluate` can be helpful to
this end.

### Extended usage (untested)

In principle, Swarm-CG workflow is general and can be applied also for tuning bonded terms in coarser CG models (by mapping more than 3-5 atoms to each CG bead and providing adequate non-bonded parameters). To this end, it is possible to use an AA trajectory as reference for optimization, but also instead a high resolution CG trajectory (fine grain) for tuning the coarser CG model (see paper section 4 for a more detailed discussion about crossing CG scales).

Another possible use case would be the tuning of elastic networks in CG models of proteins, although this still requires a well sampled AA or fine CG reference trajectory.

Please feel free to open an [Issue](https://github.com/DrDomenicoMarson/Swarm-CG_DM/issues) if you are interested in extended usages and need help.

### Credits

Swarm-CG makes extensive use of [FST-PSO](https://doi.org/10.1016/j.swevo.2017.09.001) and [MDAnalysis](https://doi.org/10.1002/jcc.21787). We thank [Marco S. Nobile](http://msnobile.it/personal/) for his valuable insights.
