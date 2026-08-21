
import math
import os
from importlib import resources
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from swarmcg import config as config_module
from swarmcg.shared.logging_utils import get_logger

logger = get_logger(__name__)


def _package_data_path(filename: str) -> str:
    """Return the installed path of a bundled Swarm-CG data file.

    Args:
        filename: Basename inside the package ``data`` directory.

    Returns:
        Absolute or installation-relative filesystem path to the resource.
    """
    return str(resources.files("swarmcg").joinpath("data", filename))


def _section_from_namespace(model_type, defaults, namespace):
    """Build one Pydantic configuration section from an argparse namespace.

    Args:
        model_type: Pydantic model class to construct.
        defaults: Default model instance providing missing namespace values.
        namespace: Parsed argparse namespace or namespace-like object.

    Returns:
        A validated instance of ``model_type``.
    """
    values = {
        name: getattr(namespace, name, getattr(defaults, name))
        for name in model_type.model_fields
    }
    return model_type(**values)

class GromacsConfig(BaseModel):
    """Configuration for GROMACS preprocessing and simulation commands."""

    gmx_path: str = "gmx"
    nb_threads: int = 0
    ntomp: int = 0
    mpi_tasks: int = 0
    gpu_id: str = ""
    gmx_args_str: str = ""
    mini_maxwarn: int = 1
    sim_kill_delay: int = 60

    @field_validator("nb_threads", "ntomp", "mpi_tasks", "mini_maxwarn")
    @classmethod
    def check_non_negative_integer(cls, value: int, info: Any) -> int:
        """Require non-negative GROMACS integer options.

        Args:
            value: Value supplied for the field.
            info: Pydantic validation metadata.

        Returns:
            The validated value.

        Raises:
            ValueError: If the value is negative.
        """
        if value < 0:
            raise ValueError(f"{info.field_name} must be greater than or equal to zero.")
        return value

    @field_validator("sim_kill_delay")
    @classmethod
    def check_stall_timeout(cls, value: int) -> int:
        """Require at least one ten-second simulation monitoring interval.

        Args:
            value: Requested stall timeout in seconds.

        Returns:
            Validated timeout.

        Raises:
            ValueError: If the timeout is shorter than one interval.
        """
        if value < 10:
            raise ValueError("sim_kill_delay must be at least 10 seconds.")
        return value

    @model_validator(mode='after')
    def check_gmx_args_conflict(self) -> "GromacsConfig":
        """Warn when a free-form mdrun argument string takes precedence."""
        if self.gmx_args_str != "" and (self.nb_threads != 0 or self.ntomp != 0 or self.gpu_id != ""):
            logger.warning(
                "Argument -gmx_args_str is provided together with -nb_threads, -ntomp or -gpu_id; "
                "only -gmx_args_str will be used."
            )
        return self

class ReferenceModelConfig(BaseModel):
    """Input files and mapping rules for the atomistic reference model."""

    aa_tpr_filename: str = "aa_topol.tpr"
    aa_traj_filename: str = "aa_traj.xtc"
    cg_map_filename: str = "cg_map.ndx"
    mapping_type: str = "COM"
    aa_rg_offset: float = 0.0

    @field_validator("aa_rg_offset")
    @classmethod
    def check_finite_rg_offset(cls, value: float) -> float:
        """Require a finite atomistic radius-of-gyration offset.

        Args:
            value: Requested offset in nanometers.

        Returns:
            Validated finite offset.

        Raises:
            ValueError: If the offset is ``NaN`` or infinite.
        """
        if not math.isfinite(value):
            raise ValueError("aa_rg_offset must be finite.")
        return value

    @field_validator('mapping_type')
    @classmethod
    def check_mapping_type(cls, v: str) -> str:
        if v.upper() not in ('COM', 'COG'):
            raise ValueError(
                "Mapping type provided via argument '-mapping' must be either COM or COG "
                "(Center of Mass or Center of Geometry)."
            )
        return v.upper()

class CGModelConfig(BaseModel):
    """Input files and parameter-source policy for the CG model."""

    cg_itp_filename: str = "cg_model.itp"
    gro_input_filename: str = "start_conf.gro"
    top_input_filename: str = "system.top"
    cg_tpr_filename: str = "cg_topol.tpr"
    cg_traj_filename: str = "cg_traj.xtc"
    user_input: bool = False

class SimulationConfig(BaseModel):
    """MDP files and physical conditions for optimization simulations."""

    # MDP files
    mdp_minimization_filename: str = Field(
        default_factory=lambda: _package_data_path("mini.mdp")
    )
    mdp_equi_filename: str = Field(
        default_factory=lambda: _package_data_path("equi.mdp")
    )
    mdp_md_filename: str = Field(
        default_factory=lambda: _package_data_path("md.mdp")
    )
    
    # Simulation Parameters
    sim_duration_short: float = 10.0 # ns
    sim_duration_long: float = 25.0 # ns
    temp: float = 300.0 # Kelvin

    @field_validator("sim_duration_short", "sim_duration_long", "temp")
    @classmethod
    def check_positive_simulation_value(cls, value: float, info: Any) -> float:
        """Require positive simulation durations and temperature.

        Args:
            value: Duration or temperature value.
            info: Pydantic field metadata.

        Returns:
            Validated positive value.

        Raises:
            ValueError: If *value* is not positive.
        """
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{info.field_name} must be finite and greater than zero.")
        return value

class OptimizationConfig(BaseModel):
    """Search-space and scoring configuration for bonded optimization."""

    exec_mode: int = 1
    sim_type: str = "OPTIMAL"
    
    # Force constants limits
    default_max_fct_bonds_opti: float = 18000.0
    default_max_fct_angles_opti_f1: float = 1700.0
    default_max_fct_angles_opti_f2: float = 1700.0
    default_max_fct_angles_opti_f10: float = 1700.0
    default_abs_range_fct_dihedrals_opti_func_with_mult: float = Field(
        default=15.0,
        description=(
            "Upper bound for canonical nonnegative periodic-dihedral force "
            "constants in kJ/mol."
        ),
    )
    default_abs_range_fct_dihedrals_opti_func_without_mult: float = 1500.0
    max_abs_rb_coefficient: Optional[float] = Field(
        default=None,
        description="Optional absolute bound for independent RB C1--C5 coefficients in kJ/mol.",
    )
    max_abs_cbt_effective_coefficient: Optional[float] = Field(
        default=None,
        description="Optional absolute bound for effective CBT k_phi*a_i coefficients in kJ/mol.",
    )
    
    # Scoring
    bonds2angles_scoring_factor: float = 500.0
    bw_constraints: float = 0.002
    bw_bonds: float = 0.01
    bw_angles: float = 2.5
    bw_dihedrals: float = 2.5
    bonded_max_range: float = 15.0
    
    # Scaling
    bonds_scaling: float = 1.0
    bonds_scaling_str: str = ""
    min_bonds_length: float = 0.0
    
    # Other
    keep_all_sims: bool = False
    bi_nb_bins: int = 50
    bond_dist_guess_variation: float = config_module.bond_dist_guess_variation
    angle_value_guess_variation: float = config_module.angle_value_guess_variation
    dihedral_value_guess_variation: float = config_module.dihedral_value_guess_variation
    fct_guess_min_flat_diff_bonds: float = config_module.fct_guess_min_flat_diff_bonds
    fct_guess_min_flat_diff_angles: float = config_module.fct_guess_min_flat_diff_angles
    fct_guess_min_flat_diff_dihedrals_without_mult: float = config_module.fct_guess_min_flat_diff_dihedrals_without_mult
    fct_guess_min_flat_diff_dihedrals_with_mult: float = config_module.fct_guess_min_flat_diff_dihedrals_with_mult
    @field_validator("exec_mode")
    @classmethod
    def check_exec_mode(cls, value: int) -> int:
        """Require one of the two documented optimization modes.

        Args:
            value: Requested execution mode.

        Returns:
            Validated mode.

        Raises:
            ValueError: If the mode is not 1 or 2.
        """
        if value not in (1, 2):
            raise ValueError("exec_mode must be either 1 or 2.")
        return value

    @field_validator("sim_type")
    @classmethod
    def normalize_sim_type(cls, value: str) -> str:
        """Normalize and validate the optimization strategy name.

        Args:
            value: Requested strategy name.

        Returns:
            Uppercase validated strategy.

        Raises:
            ValueError: If the strategy is unsupported.
        """
        normalized = value.upper()
        if normalized not in {"OPTIMAL", "FAST", "TEST"}:
            raise ValueError("sim_type must be OPTIMAL, FAST, or TEST.")
        return normalized

    @field_validator('default_max_fct_bonds_opti', 'default_max_fct_angles_opti_f1',
                     'default_max_fct_angles_opti_f2', 'default_max_fct_angles_opti_f10',
                     'default_abs_range_fct_dihedrals_opti_func_with_mult',
                     'default_abs_range_fct_dihedrals_opti_func_without_mult',
                     'bonds2angles_scoring_factor', 'bw_constraints', 'bw_bonds',
                     'bw_angles', 'bw_dihedrals', 'bonded_max_range', 'bi_nb_bins')
    @classmethod
    def check_positive(cls, v: float, info: Any) -> float:
        """Require positive force, scoring, histogram, and bin-count values.

        Args:
            v: Numeric value to validate.
            info: Pydantic field metadata.

        Returns:
            Validated positive value.

        Raises:
            ValueError: If *v* is not positive.
        """
        if not math.isfinite(v) or v <= 0:
            raise ValueError(
                f"Please provide a finite value > 0 for argument corresponding to {info.field_name}."
            )
        return v

    @field_validator("max_abs_rb_coefficient", "max_abs_cbt_effective_coefficient")
    @classmethod
    def check_optional_positive(cls, value: Optional[float], info: Any) -> Optional[float]:
        """Validate optional explicit polynomial coefficient bounds.

        Args:
            value: Optional absolute bound.
            info: Pydantic field metadata.

        Returns:
            ``None`` or the validated positive bound.

        Raises:
            ValueError: If a supplied bound is not positive.
        """
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(
                f"{info.field_name} must be finite and greater than zero when provided."
            )
        return value

    @field_validator(
        "bond_dist_guess_variation",
        "angle_value_guess_variation",
        "dihedral_value_guess_variation",
        "fct_guess_min_flat_diff_bonds",
        "fct_guess_min_flat_diff_angles",
        "fct_guess_min_flat_diff_dihedrals_without_mult",
        "fct_guess_min_flat_diff_dihedrals_with_mult",
    )
    @classmethod
    def check_positive_initialization_scale(cls, value: float, info: Any) -> float:
        """Require finite positive particle-initialization scales.

        Args:
            value: Exploration width or minimum force-constant displacement.
            info: Pydantic validation metadata.

        Returns:
            Validated positive finite value.

        Raises:
            ValueError: If the value is non-finite or non-positive.
        """
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{info.field_name} must be finite and greater than zero.")
        return value

    @field_validator("bonds_scaling")
    @classmethod
    def check_positive_bond_scaling(cls, value: float) -> float:
        """Require a finite positive global bond-scaling factor.

        Args:
            value: Requested multiplicative scaling factor.

        Returns:
            Validated positive finite factor.

        Raises:
            ValueError: If the factor is non-finite or non-positive.
        """
        if not math.isfinite(value) or value <= 0:
            raise ValueError("bonds_scaling must be finite and greater than zero.")
        return value

    @field_validator("min_bonds_length")
    @classmethod
    def check_nonnegative_minimum_bond_length(cls, value: float) -> float:
        """Require a finite nonnegative minimum rescaled bond length.

        Args:
            value: Requested minimum length in nanometers; zero disables it.

        Returns:
            Validated finite length.

        Raises:
            ValueError: If the length is non-finite or negative.
        """
        if not math.isfinite(value) or value < 0:
            raise ValueError("min_bonds_length must be finite and nonnegative.")
        return value
    
    @model_validator(mode='after')
    def check_bonds_scaling_conflicts(self) -> "OptimizationConfig":
        """Require exactly one active bond-rescaling policy at most.

        Returns:
            The validated configuration.

        Raises:
            ValueError: If multiple scaling policies are active.
        """
        set_count = sum(
            (
                self.bonds_scaling != 1.0,
                self.bonds_scaling_str != "",
                self.min_bonds_length != 0.0,
            )
        )
        if set_count > 1:
            raise ValueError(
                "Only one of arguments -bonds_scaling, -bonds_scaling_str and -min_bonds_length "
                "can be provided. Please check your parameters"
            )
        return self

class OutputConfig(BaseModel):
    """Output and plotting settings."""

    input_folder: str = ""
    output_folder: str = ""
    opti_dirname: str = ""
    plot_filename: str = "distributions.png"
    mismatch_order: bool = False
    row_x_scaling: bool = True
    row_y_scaling: bool = True
    ncols_max: int = 0
    plot_scale: float = 1.0
    verbose: bool = False
    @field_validator("ncols_max")
    @classmethod
    def check_nonnegative_columns(cls, value: int) -> int:
        """Require a nonnegative optional plot-column limit.

        Args:
            value: Requested maximum column count.

        Returns:
            Validated column count.

        Raises:
            ValueError: If *value* is negative.
        """
        if value < 0:
            raise ValueError("ncols_max must be greater than or equal to zero.")
        return value

    @field_validator("plot_scale")
    @classmethod
    def check_positive_plot_scale(cls, value: float) -> float:
        """Require a positive plot scaling factor.

        Args:
            value: Requested plot scale.

        Returns:
            Validated plot scale.

        Raises:
            ValueError: If *value* is not positive.
        """
        if not math.isfinite(value) or value <= 0:
            raise ValueError("plot_scale must be finite and greater than zero.")
        return value


class SasaConfig(BaseModel):
    """Validated Martini 3 solvent-accessible surface-area protocol.

    Args:
        enabled: Enable SASA validation without adding it to optimization
            fitness.
        probe_radius_nm: Solvent-probe radius in nanometres.
        sphere_points: Number of surface dots per sphere used by GROMACS.
        aa_radii_filename: Optional GROMACS ``vdwradii.dat``-format AA radii
            override. The packaged Rowland--Taylor profile is used otherwise.
    """

    enabled: bool = False
    probe_radius_nm: float = 0.191
    sphere_points: int = 4800
    aa_radii_filename: str | None = None

    @field_validator("probe_radius_nm")
    @classmethod
    def check_positive_probe_radius(cls, value: float) -> float:
        """Require a finite positive solvent-probe radius.

        Args:
            value: Probe radius in nanometres.

        Returns:
            Validated probe radius.

        Raises:
            ValueError: If *value* is non-finite or not positive.
        """
        if not math.isfinite(value) or value <= 0:
            raise ValueError("probe_radius_nm must be finite and greater than zero.")
        return value

    @field_validator("sphere_points")
    @classmethod
    def check_positive_sphere_points(cls, value: int) -> int:
        """Require a positive GROMACS surface-dot count.

        Args:
            value: Requested number of dots per sphere.

        Returns:
            Validated dot count.

        Raises:
            ValueError: If *value* is not positive.
        """
        if value <= 0:
            raise ValueError("sphere_points must be greater than zero.")
        return value

    @field_validator("aa_radii_filename")
    @classmethod
    def normalize_optional_radii_path(cls, value: str | None) -> str | None:
        """Normalize an empty radii override to ``None``.

        Args:
            value: Optional user-supplied filename.

        Returns:
            The filename, or ``None`` for an empty value.
        """
        return value or None


class SwarmConfig(BaseModel):
    """Complete validated configuration for a Swarm-CG command."""

    gromacs: GromacsConfig = Field(default_factory=GromacsConfig)
    reference: ReferenceModelConfig = Field(default_factory=ReferenceModelConfig)
    cg_model: CGModelConfig = Field(default_factory=CGModelConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    sasa: SasaConfig = Field(default_factory=SasaConfig)
    
    # We can add a method to validation files existence if desired, 
    # but Pydantic validators usually validate *data*, not external state (like file existence).
    # However, for a CLI tool, validating file existence at configuration load is good practice.
    
    def validate_files_exist(self, require_cg_trajectory: bool = False) -> None:
        """Check that required model and optional CG trajectory files exist.

        Args:
            require_cg_trajectory: Require both the CG topology and trajectory.

        Raises:
            FileNotFoundError: If a required input file cannot be found.
        """
        # Reference files
        if not os.path.isfile(self.reference.aa_tpr_filename):
            raise FileNotFoundError(
                f"Cannot find topology file of the atomistic simulation at location: {self.reference.aa_tpr_filename}\n"
                f"(TPR or other portable topology formats supported by MDAnalysis)"
            )
        if not os.path.isfile(self.reference.aa_traj_filename):
            raise FileNotFoundError(
                f"Cannot find trajectory file of the atomistic simulation at location: {self.reference.aa_traj_filename}\n"
                f"(XTC, TRR, or other trajectory formats supported by MDAnalysis)"
            )
        if not os.path.isfile(self.reference.cg_map_filename):
            raise FileNotFoundError(
                f"Cannot find CG beads mapping file at location: {self.reference.cg_map_filename}\n"
                f"(NDX-like file format)"
            )
            
        # CG Model file
        if not os.path.isfile(self.cg_model.cg_itp_filename):
            raise FileNotFoundError(f"Cannot find ITP file of the CG model at location: {self.cg_model.cg_itp_filename}")
        if require_cg_trajectory:
            if not os.path.isfile(self.cg_model.cg_tpr_filename):
                raise FileNotFoundError(
                    f"Cannot find topology file of the CG simulation at location: {self.cg_model.cg_tpr_filename}"
                )
            if not os.path.isfile(self.cg_model.cg_traj_filename):
                raise FileNotFoundError(
                    f"Cannot find trajectory file of the CG simulation at location: {self.cg_model.cg_traj_filename}"
                )

    @classmethod
    def from_namespace(cls, ns) -> 'SwarmConfig':
        """Convert a flat argparse namespace into structured configuration.

        Args:
            ns: Namespace produced by a Swarm-CG command parser.

        Returns:
            Fully validated nested configuration.

        Raises:
            ValidationError: If a numeric, strategy, mapping, or cross-field
                constraint is invalid.
        """
        defaults = cls()
        return cls(
            gromacs=_section_from_namespace(GromacsConfig, defaults.gromacs, ns),
            reference=_section_from_namespace(
                ReferenceModelConfig, defaults.reference, ns
            ),
            cg_model=_section_from_namespace(CGModelConfig, defaults.cg_model, ns),
            simulation=_section_from_namespace(
                SimulationConfig, defaults.simulation, ns
            ),
            optimization=_section_from_namespace(
                OptimizationConfig, defaults.optimization, ns
            ),
            output=_section_from_namespace(OutputConfig, defaults.output, ns),
            sasa=_section_from_namespace(SasaConfig, defaults.sasa, ns),
        )
