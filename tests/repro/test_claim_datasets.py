from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from repro.datasets import (
    read_claim_evidence,
    write_claim_evidence,
)
from repro.results import (
    ClaimEvidence,
    PredictionMeasurement,
    TaskMeasurement,
)


@pytest.fixture
def measured_result():
    return ClaimEvidence(
        claim_id="DD-0014",
        evidence_ids=(7,),
        unavailable_ids=(),
        related_tables=("curves.csv",),
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
            ),
        ),
    )


def test_claim_evidence_round_trip_with_and_without_measurements(
    tmp_path, measured_result
):
    unavailable = replace(
        measured_result,
        claim_id="DD-0016",
        evidence_ids=(),
        unavailable_ids=(8,),
        measurements=(
            TaskMeasurement(
                task="hellaswag",
                available_comparisons=0,
                unavailable_comparisons=1,
                best=None,
            ),
        ),
    )
    path = tmp_path / "results.parquet"
    write_claim_evidence((measured_result, unavailable), path)
    assert read_claim_evidence(path) == (measured_result, unavailable)


def test_claim_dataset_pins_persisted_keys_and_numbers(tmp_path, measured_result):
    path = tmp_path / "results.parquet"
    write_claim_evidence((measured_result,), path)
    assert pq.read_table(path).to_pylist() == [
        {
            "claim_id": "DD-0014",
            "evidence_ids": [7],
            "unavailable_ids": [],
            "related_tables": ["curves.csv"],
            "measurements": [
                {
                    "task": "mmlu",
                    "available_comparisons": 1,
                    "unavailable_comparisons": 0,
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


def test_unexpected_text_column_is_rejected(tmp_path, measured_result):
    path = tmp_path / "evidence.parquet"
    write_claim_evidence((measured_result,), path)
    table = pq.read_table(path).append_column("summary", pa.array(["Authored text"]))
    pq.write_table(table, path)
    with pytest.raises(ValueError, match="schema"):
        read_claim_evidence(path)


def test_claim_dataset_schema_drift_is_rejected(tmp_path):
    path = tmp_path / "results.parquet"
    pq.write_table(pa.table({"claim_id": ["DD-0014"]}), path)
    with pytest.raises(ValueError, match="schema"):
        read_claim_evidence(path)
