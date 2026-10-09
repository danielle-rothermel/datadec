from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from repro.datasets import read_claim_evidence, write_claim_evidence
from repro.results import ClaimEvidence, TaskEvidence


@pytest.fixture
def measured_result():
    return ClaimEvidence(
        claim_id="DD-0014",
        related_tables=("curves.csv",),
        tasks=(TaskEvidence("mmlu", (7,), (), 7),),
    )


def test_claim_evidence_round_trip_with_and_without_measurements(
    tmp_path, measured_result
):
    unavailable = replace(
        measured_result,
        claim_id="DD-0016",
        tasks=(TaskEvidence("hellaswag", (), (8,), None),),
    )
    path = tmp_path / "results.parquet"
    write_claim_evidence((measured_result, unavailable), path)
    assert read_claim_evidence(path) == (measured_result, unavailable)
    assert measured_result.evidence_ids == (7,)
    assert unavailable.unavailable_ids == (8,)
    assert measured_result.tasks[0].available_comparisons == 1
    assert unavailable.tasks[0].unavailable_comparisons == 1


def test_claim_dataset_pins_persisted_links_without_copied_measurements(
    tmp_path, measured_result
):
    path = tmp_path / "results.parquet"
    write_claim_evidence((measured_result,), path)
    assert pq.read_table(path).to_pylist() == [
        {
            "claim_id": "DD-0014",
            "related_tables": ["curves.csv"],
            "tasks": [
                {
                    "task": "mmlu",
                    "evidence_ids": [7],
                    "unavailable_ids": [],
                    "best_evidence_id": 7,
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


@pytest.mark.parametrize("available,best", [((7,), 8), ((7,), None), ((), 7)])
def test_best_row_must_be_an_available_link(available, best):
    with pytest.raises(ValueError, match="best evidence ID"):
        TaskEvidence("mmlu", available, (), best)
