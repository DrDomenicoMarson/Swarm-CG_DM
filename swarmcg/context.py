from dataclasses import dataclass, fields
from typing import Any


@dataclass(slots=True)
class SwarmCGArgs:
    exec_mode: int | None = None
    sim_type: str | None = None
    gmx_path: str | None = None
    nb_threads: int | None = None
    mpi_tasks: int | None = None
    gpu_id: str | None = None
    gmx_args_str: str | None = None
    mini_maxwarn: int | None = None
    sim_kill_delay: int | None = None
    aa_tpr_filename: str | None = None
    aa_traj_filename: str | None = None
    cg_map_filename: str | None = None
    mapping_type: str | None = None
    cg_itp_filename: str | None = None
    user_input: bool | None = None
    gro_input_filename: str | None = None
    top_input_filename: str | None = None
    cg_tpr_filename: str | None = None
    cg_traj_filename: str | None = None
    mdp_minimization_filename: str | None = None
    mdp_equi_filename: str | None = None
    mdp_md_filename: str | None = None
    input_folder: str | None = None
    output_folder: str | None = None
    opti_dirname: str | None = None
    plot_filename: str | None = None
    default_max_fct_bonds_opti: float | None = None
    default_max_fct_angles_opti_f1: float | None = None
    default_max_fct_angles_opti_f2: float | None = None
    default_abs_range_fct_dihedrals_opti_func_with_mult: float | None = None
    default_abs_range_fct_dihedrals_opti_func_without_mult: float | None = None
    sim_duration_short: float | None = None
    sim_duration_long: float | None = None
    bonds2angles_scoring_factor: float | None = None
    bw_constraints: float | None = None
    bw_bonds: float | None = None
    bw_angles: float | None = None
    bw_dihedrals: float | None = None
    row_x_scaling: bool | None = None
    row_y_scaling: bool | None = None
    bonded_max_range: float | None = None
    aa_rg_offset: float | None = None
    bonds_scaling: float | None = None
    bonds_scaling_str: str | None = None
    min_bonds_length: float | None = None
    mismatch_order: bool | None = None
    ncols_max: int | None = None
    plot_scale: float | None = None
    temp: float | None = None
    keep_all_sims: bool | None = None
    verbose: bool | None = None

    def normalize(self) -> "SwarmCGArgs":
        if self.mapping_type:
            self.mapping_type = self.mapping_type.upper()
        return self

    def validate(self, step: str | None = None) -> "SwarmCGArgs":
        from swarmcg import config
        from swarmcg.shared.validation import input_parameter_validation

        self.normalize()
        input_parameter_validation(self, config, step=step)
        return self

    def as_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_namespace(cls, namespace: Any) -> "SwarmCGArgs":
        data = {f.name: getattr(namespace, f.name, None) for f in fields(cls)}
        return cls(**data)


@dataclass(slots=True)
class SwarmCGState:
    cg_itp_basename: str | None = None
    gro_input_basename: str | None = None
    top_input_basename: str | None = None
    mdp_minimization_basename: str | None = None
    mdp_equi_basename: str | None = None
    mdp_md_basename: str | None = None
    exec_folder: str | None = None
    process_alive_time_sleep: int | None = None
    process_alive_nb_cycles_dead: int | None = None

    eval_nb_geoms: dict[str, int] | None = None
    opti_cycle: dict[str, Any] | None = None
    opti_geoms_all: list[str] | None = None
    opti_itp: dict[str, Any] | None = None
    out_itp: dict[str, Any] | None = None
    domains_val: dict[str, Any] | None = None
    data_BI: dict[str, Any] | None = None
    performed_init_BI: dict[str, bool] | None = None
    fct_guess_fact: float | None = None
    val_guess_fact: float | None = None
    max_swarm_iter: int | None = None
    max_swarm_iter_without_new_global_best: int | None = None
    prod_sim_time: float | None = None
    prod_nb_frames: int | None = None
    nb_eval: int | None = None
    best_fitness: tuple[float, int] | None = None
    worst_fit_score: float | None = None
    start_opti_ts: float | None = None
    total_eval_time: float | None = None
    total_gmx_time: float | None = None
    total_model_eval_time: float | None = None

    atom_only: bool | None = None
    molname_in: str | None = None
    mda_backend: str | None = None
    all_atoms: dict[int, Any] | None = None
    all_aa_mols: list[Any] | None = None
    all_beads: dict[int, Any] | None = None
    atoms_occ_total: dict[int, int] | None = None
    atom_w: dict[int, Any] | None = None
    mda_beads_atom_grps: dict[int, Any] | None = None
    mda_weights_atom_grps: dict[int, Any] | None = None
    bonds_rescaling_performed: bool | None = None
    bonds_scaling_specific: dict[str, float] | None = None

    aa_universe: Any | None = None
    aa2cg_universe: Any | None = None
    cg_universe: Any | None = None

    bins_constraints: Any | None = None
    bins_bonds: Any | None = None
    bins_angles: Any | None = None
    bins_dihedrals: Any | None = None
    bins_constraints_dist_matrix: Any | None = None
    bins_bonds_dist_matrix: Any | None = None
    bins_angles_dist_matrix: Any | None = None
    bins_dihedrals_dist_matrix: Any | None = None

    cg_itp: dict[str, Any] | None = None
    all_emd_dist_geoms: dict[str, Any] | None = None
    all_best_emd_dist_geoms: dict[str, Any] | None = None
    all_best_params_dist_geoms: dict[str, Any] | None = None
    gyr_aa: float | None = None
    gyr_aa_std: float | None = None
    gyr_aa_mapped: float | None = None
    gyr_aa_mapped_std: float | None = None
    gyr_cg: float | None = None
    gyr_cg_std: float | None = None
    sasa_aa_mapped: float | None = None
    sasa_aa_mapped_std: float | None = None
    sasa_cg: float | None = None
    sasa_cg_std: float | None = None

    cg_ndx_filename: str | None = None
    aa_traj_whole_filename: str | None = None
    aa_mapped_traj_whole_filename: str | None = None
    aa_mapped_sasa_filename: str | None = None
    aa_mapped_tpr_sasa_filename: str | None = None
    cg_traj_whole_filename: str | None = None
    cg_sasa_filename: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}
