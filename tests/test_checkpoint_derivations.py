from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from datadec.data.artifacts import DataArtifacts
from datadec.data.model_utils import checkpoint_enrichment
from datadec.data.selection import DatasetSelection
from datadec.data.verify.derivations import verify_preprocessed_derivations
from datadec.data.verify.olmes_details import verify_olmes_details


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _detail_selection(*recipes: str) -> DatasetSelection:
    return DatasetSelection(
        ppl=False,
        olmes=False,
        olmes_details=recipes,
        scaling_law=False,
        published_results=(),
        published_figures=False,
        all_data=False,
    )


def _derivation_row(
    recipe: str, *, model: str = "fixture-1B-5xC-2"
) -> dict[str, object]:
    return {
        "recipe": recipe,
        "params": "1B",
        "step": 1,
        **checkpoint_enrichment("1B", 1),
        "model_config": json.dumps(
            {
                "max_length": 2048,
                "model": model,
                "revision": "step1-unsharded-hf",
            }
        ),
    }


def test_derivation_verification_checks_selected_shared_task_rows_once(
    tmp_path: Path,
) -> None:
    artifacts = DataArtifacts(tmp_path)
    _write_parquet(
        artifacts.olmes_details_tasks_path(),
        [
            _derivation_row("selected"),
            _derivation_row("unselected", model="wrong-model"),
        ],
    )

    result = verify_preprocessed_derivations(
        artifacts,
        _detail_selection("selected"),
    )

    assert len(result.processed_outputs) == 1
    assert result.processed_outputs[0].name == "OLMES detail tasks"
    assert result.processed_outputs[0].row_count == 1
    assert result.processed_outputs[0].contradiction_count == 0
    assert len(result.detail_tasks) == 1
    assert result.detail_tasks[0].row_count == 1
    assert result.detail_tasks[0].contradiction_count == 0


def test_olmes_verification_filters_shared_and_sharded_tables_by_recipe(
    tmp_path: Path,
) -> None:
    artifacts = DataArtifacts(tmp_path)
    task_key = {
        "params": "1B",
        "seed_value": 2,
        "step": 1,
        "task": "arc_easy",
    }
    selected_task = {
        "recipe": "selected",
        **task_key,
        "num_instances": 2,
        "primary_metric": "acc_raw",
        "primary_score": 0.5,
        "acc_raw": 0.5,
    }
    unrelated_task = {
        "recipe": "unselected",
        **task_key,
        "num_instances": 999,
        "primary_metric": "acc_raw",
        "primary_score": 0.0,
        "acc_raw": 0.0,
    }
    _write_parquet(
        artifacts.olmes_details_tasks_path(),
        [selected_task, unrelated_task, unrelated_task],
    )

    selected_instances = [
        {
            "recipe": "selected",
            **task_key,
            "doc_id": doc_id,
            "acc_raw": score,
        }
        for doc_id, score in ((0, 1.0), (1, 0.0))
    ]
    unrelated_instance = {
        "recipe": "unselected",
        **task_key,
        "doc_id": 0,
        "acc_raw": 0.0,
    }
    _write_parquet(
        artifacts.olmes_details_instances_path("selected"),
        selected_instances,
    )
    _write_parquet(
        artifacts.olmes_details_instances_path("unselected"),
        [unrelated_instance, unrelated_instance],
    )

    selected_choices = [
        {
            "recipe": "selected",
            **task_key,
            "doc_id": doc_id,
            "choice_index": 0,
        }
        for doc_id in (0, 1)
    ]
    unrelated_choice = {
        "recipe": "unselected",
        **task_key,
        "doc_id": 0,
        "choice_index": 0,
    }
    _write_parquet(
        artifacts.olmes_details_choices_path("selected"),
        selected_choices,
    )
    _write_parquet(
        artifacts.olmes_details_choices_path("unselected"),
        [unrelated_choice, unrelated_choice],
    )

    result = verify_olmes_details(
        recipe="selected",
        paths=artifacts,
        detail_archive=None,
        aggregate_path=None,
    )

    assert result.reconstructed_task_count == 1
    assert result.overlapping_checkpoint_count is None
    assert result.parity_row_count is None
