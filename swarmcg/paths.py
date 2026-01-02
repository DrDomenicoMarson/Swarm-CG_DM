from dataclasses import dataclass
from pathlib import Path

from swarmcg import config


@dataclass(slots=True)
class PathManager:
    exec_dir: Path
    internal_dir: Path
    distrib_plots_dir: Path
    log_files_dir: Path
    sim_files_dir: Path
    input_sim_dir: Path

    @classmethod
    def from_exec_dir(cls, exec_dir: str | Path) -> "PathManager":
        exec_dir = Path(exec_dir)
        return cls(
            exec_dir=exec_dir,
            internal_dir=exec_dir / ".internal",
            distrib_plots_dir=exec_dir / config.distrib_plots_all_evals_dirname,
            log_files_dir=exec_dir / config.log_files_all_evals_dirname,
            sim_files_dir=exec_dir / config.sim_files_all_evals_dirname,
            input_sim_dir=exec_dir / config.input_sim_files_dirname,
        )

    def ensure_dirs(self, keep_all_sims: bool) -> None:
        self.exec_dir.mkdir(parents=True, exist_ok=False)
        self.internal_dir.mkdir()
        self.distrib_plots_dir.mkdir()
        self.log_files_dir.mkdir()
        if keep_all_sims:
            self.sim_files_dir.mkdir()
        self.input_sim_dir.mkdir()

    def eval_dir(self, eval_nb: int) -> Path:
        return self.exec_dir / f"{config.iteration_sim_files_dirname}_eval_step_{eval_nb}"

    @property
    def best_distrib_plot(self) -> Path:
        return self.exec_dir / config.best_distrib_plots

    @property
    def opti_perf_recap(self) -> Path:
        return self.exec_dir / config.opti_perf_recap_file

    @property
    def opti_pairwise_distances(self) -> Path:
        return self.exec_dir / config.opti_pairwise_distances_file

    @property
    def ref_distrib_plot(self) -> Path:
        return self.exec_dir / config.ref_distrib_plots
