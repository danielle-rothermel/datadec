from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from repro.datasets import CLAIM_RESULTS_SCHEMA, read_claim_results, write_claim_results
from repro.results import (
    ClaimEvidence,
    ClaimStatus,
    PredictionMeasurement,
    TaskMeasurement,
)


@pytest.fixture
def measured_result():
    return ClaimEvidence(
        claim_id="DD-0014",
        status=ClaimStatus.SUPPORTED,
        evidence_ids=(7,),
        unavailable_ids=(),
        judgment="Configured bound.",
        supporting_tables=("curves.csv",),
        measurements=(
            TaskMeasurement(
                task="mmlu",
                available_comparisons=1,
                unavailable_comparisons=0,
                best=PredictionMeasurement(
                    evidence_id=7,
                    predictor_size="4M",
                    predictor_step=5,
                    metric="correct_prob_per_char",
                    compute=100.0,
                    compute_ratio=0.0001,
                    decision_accuracy=0.81,
                    decision_accuracy_std=0.01,
                    seed_accuracies=(0.80, 0.81, 0.82),
                ),
                accuracy_gt=0.8,
                passes_bound=True,
            ),
        ),
    )


def test_claim_results_round_trip_with_and_without_measurements(
    tmp_path, measured_result
):
    unavailable = replace(
        measured_result,
        claim_id="DD-0016",
        status=ClaimStatus.INSUFFICIENT_DATA,
        evidence_ids=(),
        unavailable_ids=(8,),
        measurements=(
            TaskMeasurement(
                task="hellaswag",
                available_comparisons=0,
                unavailable_comparisons=1,
                best=None,
                accuracy_gt=0.8,
                passes_bound=None,
            ),
        ),
    )
    path = tmp_path / "results.parquet"
    write_claim_results((measured_result, unavailable), path)
    assert read_claim_results(path) == (measured_result, unavailable)


def test_claim_dataset_pins_persisted_keys_and_numbers(tmp_path, measured_result):
    path = tmp_path / "results.parquet"
    write_claim_results((measured_result,), path)
    assert pq.read_table(path).to_pylist() == [
        {
            "claim_id": "DD-0014",
            "status": "supported",
            "evidence_ids": [7],
            "unavailable_ids": [],
            "judgment": "Configured bound.",
            "supporting_tables": ["curves.csv"],
            "measurements": [
                {
                    "task": "mmlu",
                    "available_comparisons": 1,
                    "unavailable_comparisons": 0,
                    "accuracy_gt": 0.8,
                    "passes_bound": True,
                    "best": {
                        "evidence_id": 7,
                        "predictor_size": "4M",
                        "predictor_step": 5,
                        "metric": "correct_prob_per_char",
                        "compute": 100.0,
                        "compute_ratio": 0.0001,
                        "decision_accuracy": 0.81,
                        "decision_accuracy_std": 0.01,
                        "seed_accuracies": [0.80, 0.81, 0.82],
                    },
                }
            ],
        }
    ]
    assert ClaimStatus.NOT_SUPPORTED.value == "not_supported"
    assert ClaimStatus.REQUIRES_JUDGMENT.value == "requires_judgment"
    assert ClaimStatus.INSUFFICIENT_DATA.value == "insufficient_data"


def test_unknown_persisted_verdict_is_rejected(tmp_path, measured_result):
    path = tmp_path / "results.parquet"
    write_claim_results((measured_result,), path)
    rows = pq.read_table(path).to_pylist()
    rows[0]["status"] = "typo"
    pq.write_table(pa.Table.from_pylist(rows, schema=CLAIM_RESULTS_SCHEMA), path)
    with pytest.raises(ValidationError):
        read_claim_results(path)


def test_claim_dataset_schema_drift_is_rejected(tmp_path):
    path = tmp_path / "results.parquet"
    pq.write_table(pa.table({"claim_id": ["DD-0014"]}), path)
    with pytest.raises(ValueError, match="schema"):
        read_claim_results(path)
