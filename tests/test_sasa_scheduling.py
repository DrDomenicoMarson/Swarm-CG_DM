"""Optimization scheduling tests for non-fitness SASA diagnostics."""

from unittest.mock import patch

from swarmcg.config_types import SwarmConfig
from swarmcg.core.optimization import SwarmOptimizer
from swarmcg.sasa_types import (
    SasaMeasurement,
    SasaProtocol,
    SasaRepresentation,
)


def _measurement(representation):
    return SasaMeasurement(
        representation=representation,
        mean=2.0,
        standard_deviation=0.1,
        frame_count=10,
        protocol=SasaProtocol(0.191, 4800, "synthetic", "0" * 64),
    )


def test_optimizer_computes_both_reference_representations_once(tmp_path):
    optimizer = SwarmOptimizer(SwarmConfig())
    optimizer.ns.files.exec_folder = str(tmp_path)

    def compute(context, representation, output_dir):
        return _measurement(representation)

    with (
        patch("swarmcg.scoring.sasa.validate_sasa_inputs") as validate,
        patch("swarmcg.scoring.sasa.compute_sasa", side_effect=compute) as compute_mock,
    ):
        optimizer._initialize_sasa_references()

    validate.assert_called_once_with(optimizer.ns)
    assert [entry.args[1] for entry in compute_mock.call_args_list] == [
        SasaRepresentation.AA,
        SasaRepresentation.AA_MAPPED,
    ]
    assert optimizer.ns.results.sasa_aa.status == "success"
    assert optimizer.ns.results.sasa_aa_mapped.status == "success"
    assert optimizer.ns.results.sasa_cg.status == "not_scheduled"
