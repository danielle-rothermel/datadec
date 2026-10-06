from __future__ import annotations

from enum import UNIQUE, StrEnum, verify
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.dataset as ds

from datadec.config import PUBLISHED_RESULT_SCHEMAS, PublishedResultSchema
from datadec.data.artifacts import DataArtifacts


@verify(UNIQUE)
class ProcessedTable(StrEnum):
    PPL = "ppl"
    OLMES = "olmes"
    SCALING_LAW_EVALUATIONS = "scaling-law/evaluations"
    SCALING_LAW_CHECKPOINT_LOSSES = "scaling-law/checkpoint-losses"
    PUBLISHED_RESULTS_TRANSFORMED = "published-results/transformed"
    PUBLISHED_RESULTS_PREDICTION_MODEL_SCALE = (
        "published-results/prediction_model_scale"
    )
    PUBLISHED_RESULTS_PROCESSED_LADDER = "published-results/processed_ladder"
    PUBLISHED_RESULTS_CHEAP_DECISIONS = "published-results/cheap_decisions"
    PUBLISHED_RESULTS_NEW_EVAL_DECISION_ACCURACY = (
        "published-results/new_eval_decision_accuracy"
    )
    PUBLISHED_RESULTS_NEW_EVAL_MEANS = "published-results/new_eval_means"
    PUBLISHED_RESULTS_TARGET_PAIRS = "published-results/target_pairs"
    OLMES_DETAILS_TASKS = "olmes-details/tasks"
    OLMES_DETAILS_INSTANCES = "olmes-details/instances"
    OLMES_DETAILS_CHOICES = "olmes-details/choices"


_PUBLISHED_RESULT_PREFIX = "published-results/"
_KNOWN_TABLES = tuple(table.value for table in ProcessedTable)


def _processed_table_path(
    artifacts: DataArtifacts,
    table: ProcessedTable | str,
) -> Path:
    try:
        table_name = ProcessedTable(table).value
    except ValueError as error:
        available = ", ".join(_KNOWN_TABLES)
        raise ValueError(
            f"Unknown processed table {table!r}. Available: {available}"
        ) from error

    table = table_name
    if table == "ppl":
        return artifacts.get_path("ppl_processed")
    if table == "olmes":
        return artifacts.get_path("olmes_processed")
    if table == "scaling-law/evaluations":
        return artifacts.scaling_law_evaluations_path()
    if table == "scaling-law/checkpoint-losses":
        return artifacts.scaling_law_checkpoint_losses_path()
    if table == "olmes-details/tasks":
        return artifacts.olmes_details_tasks_path()
    if table == "olmes-details/instances":
        return artifacts.data_dir / "processed" / "olmes-details" / "instances"
    if table == "olmes-details/choices":
        return artifacts.data_dir / "processed" / "olmes-details" / "choices"
    if table.startswith(_PUBLISHED_RESULT_PREFIX):
        schema = table.removeprefix(_PUBLISHED_RESULT_PREFIX)
        if schema in PUBLISHED_RESULT_SCHEMAS:
            return artifacts.published_result_table_path(
                cast(PublishedResultSchema, schema)
            )

    raise AssertionError(f"unhandled processed table: {table}")


def read_processed_table(
    artifacts: DataArtifacts,
    table: ProcessedTable | str,
    *,
    filters: ds.Expression | None = None,
    columns: list[str] | None = None,
) -> pa.Table:
    """Read a known local processed table with optional Arrow pushdown filters."""
    path = _processed_table_path(artifacts, table)
    if not path.exists():
        raise FileNotFoundError(f"processed table {table!r} does not exist: {path}")

    dataset = ds.dataset(path, format="parquet")
    return dataset.to_table(filter=filters, columns=columns)


__all__ = ["ProcessedTable", "read_processed_table"]
