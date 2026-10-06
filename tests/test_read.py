from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

from datadec.data.artifacts import DataArtifacts
from datadec.data.read import ProcessedTable, read_processed_table


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_processed_table_values_are_stable() -> None:
    assert tuple(table.value for table in ProcessedTable) == (
        "ppl",
        "olmes",
        "scaling-law/evaluations",
        "scaling-law/checkpoint-losses",
        "published-results/transformed",
        "published-results/prediction_model_scale",
        "published-results/processed_ladder",
        "published-results/cheap_decisions",
        "published-results/new_eval_decision_accuracy",
        "published-results/new_eval_means",
        "published-results/target_pairs",
        "olmes-details/tasks",
        "olmes-details/instances",
        "olmes-details/choices",
    )


def test_read_processed_table_projects_columns_and_filters_rows(
    tmp_path: Path,
) -> None:
    artifacts = DataArtifacts(tmp_path)
    _write_rows(
        artifacts.get_path("ppl_processed"),
        [
            {"params": "4M", "step": 1, "loss": 3.0},
            {"params": "6M", "step": 2, "loss": 2.0},
        ],
    )

    table = read_processed_table(
        artifacts,
        "ppl",
        filters=ds.field("step") == 2,
        columns=["params", "loss"],
    )

    assert table.column_names == ["params", "loss"]
    assert table.to_pylist() == [{"params": "6M", "loss": 2.0}]


def test_read_processed_table_filters_published_result_provenance(
    tmp_path: Path,
) -> None:
    artifacts = DataArtifacts(tmp_path)
    _write_rows(
        artifacts.published_result_table_path("target_pairs"),
        [
            {
                "source_unit": "outputs2",
                "source_file": "outputs2/0_target_pairs.json",
                "pair_index": 0,
            },
            {
                "source_unit": "per-task-mmlu",
                "source_file": "per_task_out/mmlu/0_target_pairs.json",
                "pair_index": 1,
            },
        ],
    )

    table = read_processed_table(
        artifacts,
        "published-results/target_pairs",
        filters=(ds.field("source_unit") == "outputs2")
        & (ds.field("source_file") == "outputs2/0_target_pairs.json"),
        columns=["source_file", "pair_index"],
    )

    assert table.to_pylist() == [
        {"source_file": "outputs2/0_target_pairs.json", "pair_index": 0}
    ]


@pytest.mark.parametrize("table", ["olmes-details/instances", "olmes-details/choices"])
def test_read_processed_table_filters_recipe_across_detail_shards(
    tmp_path: Path,
    table: str,
) -> None:
    artifacts = DataArtifacts(tmp_path)
    table_directory = tmp_path / "processed" / Path(table)
    _write_rows(
        table_directory / "c4.parquet",
        [{"recipe": "c4", "task": "arc_easy", "value": 1}],
    )
    _write_rows(
        table_directory / "dolma1.7.parquet",
        [{"recipe": "dolma1.7", "task": "boolq", "value": 2}],
    )

    result = read_processed_table(
        artifacts,
        table,
        filters=ds.field("recipe") == "dolma1.7",
        columns=["recipe", "task"],
    )

    assert result.to_pylist() == [{"recipe": "dolma1.7", "task": "boolq"}]


def test_read_processed_table_rejects_unknown_table(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown processed table 'mystery'"):
        read_processed_table(DataArtifacts(tmp_path), "mystery")


def test_read_processed_table_reports_missing_known_table(tmp_path: Path) -> None:
    with pytest.raises(
        FileNotFoundError,
        match="processed table 'scaling-law/evaluations' does not exist",
    ):
        read_processed_table(DataArtifacts(tmp_path), "scaling-law/evaluations")
