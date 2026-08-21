"""Typed records for the reproducible Martini 3 SASA protocol."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class SasaRepresentation(str, Enum):
    """Molecular representation used for one SASA measurement."""

    AA = "aa"
    AA_MAPPED = "aa_mapped"
    CG = "cg"


@dataclass(frozen=True)
class SasaProtocol:
    """Protocol metadata required to reproduce one SASA measurement.

    Args:
        probe_radius_nm: Solvent-probe radius in nanometres.
        sphere_points: Number of surface dots per sphere.
        radii_source: Human-readable source of the staged radii.
        radii_sha256: SHA-256 hash of the exact staged ``vdwradii.dat``.
    """

    probe_radius_nm: float
    sphere_points: int
    radii_source: str
    radii_sha256: str


@dataclass(frozen=True)
class SasaMeasurement:
    """Full-precision statistics for one completed SASA calculation.

    Args:
        representation: Atomistic, mapped-reference, or CG representation.
        mean: Arithmetic mean in square nanometres.
        standard_deviation: Population standard deviation in square
            nanometres.
        frame_count: Number of finite frames contributing to the statistics.
        protocol: Exact radii and numerical protocol metadata.
    """

    representation: SasaRepresentation
    mean: float
    standard_deviation: float
    frame_count: int
    protocol: SasaProtocol


SasaDiagnosticStatus = Literal["not_scheduled", "success", "failed"]


@dataclass(frozen=True)
class SasaDiagnostic:
    """Scheduling and outcome state for one SASA representation.

    Args:
        status: ``not_scheduled``, ``success``, or ``failed``.
        measurement: Completed measurement for a successful diagnostic.
        error: Failure details for a failed diagnostic.
    """

    status: SasaDiagnosticStatus = "not_scheduled"
    measurement: SasaMeasurement | None = None
    error: str | None = None

    @classmethod
    def success(cls, measurement: SasaMeasurement) -> "SasaDiagnostic":
        """Build a successful diagnostic.

        Args:
            measurement: Completed measurement.

        Returns:
            Successful diagnostic record.
        """
        return cls(status="success", measurement=measurement)

    @classmethod
    def failed(cls, error: Exception | str) -> "SasaDiagnostic":
        """Build a failed diagnostic without changing optimization fitness.

        Args:
            error: Captured error or error text.

        Returns:
            Failed diagnostic record.
        """
        return cls(status="failed", error=str(error))
