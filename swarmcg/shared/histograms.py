"""Validated histogram observations shared by scoring and initialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class HistogramObservation:
    """Frame-normalized histogram masses and sample-coverage diagnostics.

    Args:
        probabilities: Per-bin masses divided by the expected sample count.
        expected_count: Total number of supplied samples.
        binned_count: Number of finite samples inside the inclusive edge range.
        nonfinite_count: Number of ``NaN`` or infinite samples.
        underflow_count: Number of finite samples below the first edge.
        overflow_count: Number of finite samples above the last edge.

    Raises:
        ValueError: If counts are inconsistent or probabilities are invalid.
    """

    probabilities: np.ndarray
    expected_count: int
    binned_count: int
    nonfinite_count: int
    underflow_count: int
    overflow_count: int

    def __post_init__(self) -> None:
        """Validate counts and freeze a private copy of the mass vector."""
        probabilities = np.array(self.probabilities, dtype=float, copy=True)
        counts = (
            self.expected_count,
            self.binned_count,
            self.nonfinite_count,
            self.underflow_count,
            self.overflow_count,
        )
        if probabilities.ndim != 1 or not np.all(np.isfinite(probabilities)):
            raise ValueError("Histogram probabilities must be a finite one-dimensional vector.")
        if np.any(probabilities < 0) or any(count < 0 for count in counts):
            raise ValueError("Histogram probabilities and sample counts cannot be negative.")
        classified = (
            self.binned_count
            + self.nonfinite_count
            + self.underflow_count
            + self.overflow_count
        )
        if classified != self.expected_count:
            raise ValueError(
                f"Histogram classified {classified} samples but expected {self.expected_count}."
            )
        expected_mass = self.coverage
        if not np.isclose(
            float(probabilities.sum()), expected_mass, rtol=1e-12, atol=1e-14
        ):
            raise ValueError(
                "Histogram mass must equal its sampled coverage: "
                f"received {probabilities.sum()} for coverage {expected_mass}."
            )
        probabilities.setflags(write=False)
        object.__setattr__(self, "probabilities", probabilities)

    @property
    def missing_count(self) -> int:
        """Return the number of samples not represented by a regular bin."""
        return self.expected_count - self.binned_count

    @property
    def coverage(self) -> float:
        """Return the fraction of expected samples represented by bins."""
        if self.expected_count == 0:
            return 0.0
        return self.binned_count / self.expected_count

    def coverage_message(self) -> str:
        """Return a compact description of missing-sample causes."""
        return (
            f"coverage={100.0 * self.coverage:.2f}% "
            f"({self.binned_count}/{self.expected_count}); "
            f"non-finite={self.nonfinite_count}, underflow={self.underflow_count}, "
            f"overflow={self.overflow_count}"
        )


def observe_histogram(
    values: Sequence[float] | np.ndarray,
    edges: Sequence[float] | np.ndarray,
) -> HistogramObservation:
    """Bin samples without renormalizing away invalid or out-of-range values.

    The returned masses are divided by the total number of supplied samples.
    Consequently, their sum is one for complete data and is the observed
    coverage for incomplete data.

    Args:
        values: Scalar samples to classify and bin.
        edges: Strictly increasing finite histogram edges.

    Returns:
        Frozen histogram masses and coverage counts.

    Raises:
        ValueError: If the edge definition is invalid.
    """
    samples = np.asarray(values, dtype=float).reshape(-1)
    bin_edges = np.asarray(edges, dtype=float)
    if (
        bin_edges.ndim != 1
        or bin_edges.size < 2
        or not np.all(np.isfinite(bin_edges))
        or np.any(np.diff(bin_edges) <= 0)
    ):
        raise ValueError("Histogram edges must be a finite, strictly increasing vector.")

    finite = np.isfinite(samples)
    underflow = finite & (samples < bin_edges[0])
    overflow = finite & (samples > bin_edges[-1])
    in_range = finite & ~underflow & ~overflow
    counts = np.histogram(samples[in_range], bin_edges, density=False)[0]
    expected_count = int(samples.size)
    probabilities = (
        counts.astype(float) / expected_count
        if expected_count
        else np.zeros(bin_edges.size - 1, dtype=float)
    )
    return HistogramObservation(
        probabilities=probabilities,
        expected_count=expected_count,
        binned_count=int(counts.sum()),
        nonfinite_count=int(np.count_nonzero(~finite)),
        underflow_count=int(np.count_nonzero(underflow)),
        overflow_count=int(np.count_nonzero(overflow)),
    )
