from __future__ import annotations

import csv
import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from datadec.config import load_scaling_law_contract
from datadec.data.constants import (
    FLOPS_PER_TOKEN_PER_PARAMETER,
    MAX_SEQ_LEN,
    NOMINAL_PARAMETER_COUNTS,
    TRAINING_PARAMETER_COUNTS,
)
from datadec.data.model_utils import calc_batch_size
from datadec.data.artifacts import DataArtifacts
from datadec.data.preprocess import scaling_law as scaling_law_module
from datadec.data.preprocess.scaling_law import RAW_COLUMNS, preprocess_scaling_law


def _metrics(**overrides: object) -> str:
    values: dict[str, object] = {
        "correct_choice": 1,
        "acc_raw": 0.25,
        "acc_per_char": 0.5,
        "predicted_index_raw": 2,
        "predicted_index_per_token": 2,
        "predicted_index_per_char": 2,
        "predicted_index_per_byte": None,
        "predicted_index_uncond": 2,
        "primary_metric": 999.0,
    }
    values.update(overrides)
    return repr(values)


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "group": "c4",
        "model": "4M",
        "task": "boolq",
        "chinchilla": "5xC",
        "step": "100",
        "metrics": _metrics(),
        "eval/c4_en-validation/CrossEntropyLoss": "2.1",
        "eval/dolma_common-crawl-validation/CrossEntropyLoss": "2.2",
        "eval/pile-validation/CrossEntropyLoss": "2.3",
        "eval/wikitext_103-validation/CrossEntropyLoss": "2.4",
        "train/CrossEntropyLoss": "1.9",
        "throughput/total_tokens": "204800.0",
        "seed": "2.0",
    }
    row.update(overrides)
    model = str(row["model"])
    schedule_model = model if model in TRAINING_PARAMETER_COUNTS else "4M"
    step = int(float(str(row["step"])))
    tokens = step * calc_batch_size(schedule_model) * MAX_SEQ_LEN
    if "tokens" not in overrides:
        row["tokens"] = str(tokens)
    if "compute" not in overrides:
        row["compute"] = str(
            tokens
            * NOMINAL_PARAMETER_COUNTS[schedule_model]
            * FLOPS_PER_TOKEN_PER_PARAMETER
        )
    return row


def _write_sources(
    tmp_path: Path,
    rows_by_source: tuple[list[dict[str, object]], ...],
) -> DataArtifacts:
    paths = DataArtifacts(tmp_path)
    raw_paths = paths.scaling_law_raw_paths()
    assert len(rows_by_source) == len(raw_paths)
    for path, rows in zip(raw_paths, rows_by_source, strict=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=RAW_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    return paths


def test_preprocess_resolves_precedence_normalizes_and_writes_typed_sorted_outputs(
    tmp_path: Path,
) -> None:
    source_zero = [
        _row(task="arc_easy", metrics=_metrics(acc_per_char=0.4)),
        _row(task="boolq", metrics=_metrics(acc_raw=0.1)),
        _row(
            group="baseline",
            model="6M",
            task="csqa",
            metrics=_metrics(acc_raw=0.6),
        ),
        _row(group="DCLM-baseline-25p", metrics="not parsed"),
        _row(seed="", group="retired", metrics="not parsed"),
        _row(seed="6198", model="retired", metrics="not parsed"),
    ]
    source_one = [
        _row(task="boolq", metrics=_metrics(acc_raw=0.2)),
        _row(
            group="DCLM-baseline",
            model="6M",
            task="mmlu_world_religions",
            step="0.0",
            seed="14.0",
            tokens="",
            compute="",
            metrics=_metrics(acc_raw=0.7),
            **{
                raw_field: ""
                for raw_field in tuple(
                    column
                    for column in RAW_COLUMNS
                    if column.startswith(("eval/", "train/", "throughput/"))
                )
            },
        ),
    ]
    source_two = [
        _row(
            task="boolq",
            tokens="",
            compute="",
            metrics=_metrics(acc_raw=0.3),
            **{
                raw_field: ""
                for raw_field in tuple(
                    column
                    for column in RAW_COLUMNS
                    if column.startswith(("eval/", "train/", "throughput/"))
                )
            },
        )
    ]
    paths = _write_sources(tmp_path, (source_zero, source_one, source_two))

    result = preprocess_scaling_law(paths)

    evaluations = pd.read_parquet(result.evaluations_output_path)
    checkpoints = pd.read_parquet(result.checkpoint_losses_output_path)
    contract = load_scaling_law_contract()
    assert result.input_row_count == 9
    assert result.clean_row_count == 6
    assert result.excluded_row_count == 3
    assert result.superseded_row_count == 2
    assert result.evaluation_count == 4
    assert result.checkpoint_count == 3
    assert tuple(evaluations.columns) == tuple(
        column.name for column in contract.tables.evaluations.columns
    )
    assert tuple(checkpoints.columns) == tuple(
        column.name for column in contract.tables.checkpoint_losses.columns
    )
    assert pq.read_schema(result.evaluations_output_path).types == [
        {
            "string": pa.string(),
            "int64": pa.int64(),
            "float64": pa.float64(),
            "bool": pa.bool_(),
        }[column.logical_type]
        for column in contract.tables.evaluations.columns
    ]
    assert list(
        evaluations.loc[
            :, ["recipe", "params", "seed_value", "step", "task"]
        ].itertuples(index=False, name=None)
    ) == [
        ("c4", "4M", 2, 100, "arc_easy"),
        ("c4", "4M", 2, 100, "boolq"),
        ("dclm-baseline", "6M", 14, 0, "mmlu_world_religions"),
        ("dolma1.7", "6M", 2, 100, "csqa"),
    ]
    boolq = evaluations[evaluations["task"] == "boolq"].iloc[0]
    assert boolq["source_file"] == paths.scaling_law_raw_paths()[2].name
    assert boolq["seed"] == "default"
    assert boolq["acc_raw"] == 0.3
    assert boolq["primary_metric"] == 0.3
    assert boolq["tokens"] == 6_553_600
    assert boolq["compute"] == 147_252_785_971_200.0
    arc_easy = evaluations[evaluations["task"] == "arc_easy"].iloc[0]
    assert arc_easy["primary_metric"] == 0.4
    step_zero = evaluations[evaluations["step"] == 0].iloc[0]
    assert step_zero["recipe"] == "dclm-baseline"
    assert step_zero["data"] == "DCLM-Baseline"
    assert step_zero["seed"] == "small aux 2"
    assert step_zero["tokens"] == 0
    assert step_zero["compute"] == 0.0
    assert step_zero["primary_metric"] == 0.7
    baseline_alias = evaluations[evaluations["task"] == "csqa"].iloc[0]
    assert baseline_alias["recipe"] == "dolma1.7"
    assert baseline_alias["data"] == "Dolma1.7"
    c4_checkpoint = checkpoints[checkpoints["recipe"] == "c4"].iloc[0]
    assert c4_checkpoint["source_file"] == paths.scaling_law_raw_paths()[1].name
    assert c4_checkpoint["c4_en_validation_cross_entropy"] == 2.1
    assert not any(tmp_path.rglob(".*.tmp"))


def test_missing_inputs_are_reported_together_without_creating_outputs(
    tmp_path: Path,
) -> None:
    paths = DataArtifacts(tmp_path)

    with pytest.raises(FileNotFoundError, match="missing required") as exc:
        preprocess_scaling_law(paths)

    for path in paths.scaling_law_raw_paths():
        assert str(path) in str(exc.value)
    assert not paths.scaling_law_evaluations_path().exists()
    assert not paths.scaling_law_checkpoint_losses_path().exists()


def test_exact_header_is_required_for_every_source(tmp_path: Path) -> None:
    paths = _write_sources(tmp_path, ([], [], []))
    bad_path = paths.scaling_law_raw_paths()[1]
    with bad_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow((*RAW_COLUMNS[:-1], "seed_value"))

    with pytest.raises(ValueError, match="invalid scaling-law CSV header") as exc:
        preprocess_scaling_law(paths)

    assert str(bad_path) in str(exc.value)


def test_header_only_inputs_write_empty_contract_typed_outputs(tmp_path: Path) -> None:
    paths = _write_sources(tmp_path, ([], [], []))

    result = preprocess_scaling_law(paths)

    contract = load_scaling_law_contract()
    assert result.input_row_count == 0
    assert result.clean_row_count == 0
    assert result.evaluation_count == 0
    assert result.checkpoint_count == 0
    assert pq.read_schema(result.evaluations_output_path).names == [
        column.name for column in contract.tables.evaluations.columns
    ]
    assert pq.read_schema(result.checkpoint_losses_output_path).names == [
        column.name for column in contract.tables.checkpoint_losses.columns
    ]


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("group", "not-a-group", "unknown scaling-law group"),
        ("model", "5M", "unknown scaling-law model"),
        ("task", "", "invalid scaling-law task"),
        ("task", " boolq", "invalid scaling-law task"),
        ("chinchilla", "", "invalid scaling-law chinchilla"),
        ("step", "1.5", "invalid scaling-law step"),
        ("tokens", "nan", "invalid scaling-law tokens"),
        ("compute", "inf", "invalid scaling-law compute"),
        (
            "eval/c4_en-validation/CrossEntropyLoss",
            "bad",
            "invalid scaling-law eval/c4_en-validation/CrossEntropyLoss",
        ),
        ("seed", "3", "unknown scaling-law seed"),
        ("seed", "bad", "invalid scaling-law seed"),
    ],
)
def test_malformed_clean_fields_are_never_silently_dropped(
    tmp_path: Path,
    field: str,
    value: object,
    error: str,
) -> None:
    paths = _write_sources(tmp_path, ([_row(**{field: value})], [], []))

    with pytest.raises(Exception, match=error):
        preprocess_scaling_law(paths)


@pytest.mark.parametrize(
    ("metrics", "error"),
    [
        ("not a dict", "metrics syntax"),
        ("[('acc_raw', 1.0)]", "expected a Python dict"),
        ("{'acc_raw': '1.0'}", "expected a finite number or None"),
        ("{'acc_raw': True}", "expected a finite number or None"),
        ("{'acc_raw': 1e309}", "expected a finite number or None"),
        ("{'acc_raw': 10**1000}", "metrics value"),
        ("{'new_metric': 1.0}", "unknown scaling-law metrics key"),
        ("{'acc_raw': 1.0, 'acc_raw': 2.0}", "duplicate scaling-law metrics key"),
    ],
)
def test_metrics_payload_syntax_types_and_keys_are_validated(
    tmp_path: Path,
    metrics: str,
    error: str,
) -> None:
    paths = _write_sources(tmp_path, ([_row(metrics=metrics)], [], []))

    with pytest.raises(Exception, match=error):
        preprocess_scaling_law(paths)


def test_duplicate_same_priority_evaluation_is_rejected(tmp_path: Path) -> None:
    paths = _write_sources(
        tmp_path,
        ([_row(), _row(metrics=_metrics(acc_raw=0.9))], [], []),
    )

    with pytest.raises(ValueError, match="duplicate same-priority"):
        preprocess_scaling_law(paths)


def test_selected_checkpoint_rejects_conflicting_repeated_task_values(
    tmp_path: Path,
) -> None:
    paths = _write_sources(
        tmp_path,
        (
            [
                _row(task="boolq", **{"train/CrossEntropyLoss": "1.8"}),
                _row(task="arc_easy", **{"train/CrossEntropyLoss": "1.9"}),
            ],
            [],
            [],
        ),
    )

    with pytest.raises(ValueError, match="conflicting checkpoint values") as exc:
        preprocess_scaling_law(paths)

    assert "train_cross_entropy" in str(exc.value)


def test_missing_checkpoint_tokens_and_compute_are_derived(tmp_path: Path) -> None:
    paths = _write_sources(
        tmp_path,
        ([_row(tokens="", compute="")], [], []),
    )

    result = preprocess_scaling_law(paths)

    checkpoint = pd.read_parquet(result.checkpoint_losses_output_path).iloc[0]
    assert checkpoint["tokens"] == 6_553_600
    assert checkpoint["compute"] == 147_252_785_971_200.0


def test_compute_uses_exact_parameter_count_instead_of_raw_value(
    tmp_path: Path,
) -> None:
    paths = _write_sources(
        tmp_path,
        ([_row(model="1B", step="100", compute="1")], [], []),
    )

    result = preprocess_scaling_law(paths)

    checkpoint = pd.read_parquet(result.checkpoint_losses_output_path).iloc[0]
    assert checkpoint["tokens"] == 144_179_200
    assert checkpoint["compute"] == 1_018_048_177_766_400_000.0


def test_present_token_schedule_values_must_match_catalog(tmp_path: Path) -> None:
    paths = _write_sources(tmp_path, ([_row(tokens="100")], [], []))

    with pytest.raises(ValueError, match="checkpoint schedule mismatch"):
        preprocess_scaling_law(paths)


def test_failure_preserves_both_existing_outputs_and_cleans_owned_temps(
    tmp_path: Path,
) -> None:
    paths = _write_sources(
        tmp_path,
        ([_row(), _row()], [], []),
    )
    evaluations_path = paths.scaling_law_evaluations_path()
    checkpoints_path = paths.scaling_law_checkpoint_losses_path()
    evaluations_path.parent.mkdir(parents=True)
    evaluations_path.write_bytes(b"old evaluations")
    checkpoints_path.write_bytes(b"old checkpoints")

    with pytest.raises(ValueError, match="duplicate same-priority"):
        preprocess_scaling_law(paths)

    assert evaluations_path.read_bytes() == b"old evaluations"
    assert checkpoints_path.read_bytes() == b"old checkpoints"
    assert not evaluations_path.with_name(f".{evaluations_path.name}.tmp").exists()
    assert not checkpoints_path.with_name(f".{checkpoints_path.name}.tmp").exists()


def test_second_export_failure_preserves_outputs_and_cleans_first_temp(
    tmp_path: Path,
) -> None:
    paths = _write_sources(tmp_path, ([_row()], [], []))
    evaluations_path = paths.scaling_law_evaluations_path()
    checkpoints_path = paths.scaling_law_checkpoint_losses_path()
    evaluations_path.parent.mkdir(parents=True)
    evaluations_path.write_bytes(b"old evaluations")
    checkpoints_path.write_bytes(b"old checkpoints")
    real_prepare = scaling_law_module.prepare_parquet_export
    call_count = 0

    def fail_second_export(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("second export failed")
        return real_prepare(*args, **kwargs)

    with (
        patch.object(
            scaling_law_module,
            "prepare_parquet_export",
            side_effect=fail_second_export,
        ),
        pytest.raises(RuntimeError, match="second export failed"),
    ):
        preprocess_scaling_law(paths)

    assert evaluations_path.read_bytes() == b"old evaluations"
    assert checkpoints_path.read_bytes() == b"old checkpoints"
    assert not evaluations_path.with_name(f".{evaluations_path.name}.tmp").exists()
    assert not checkpoints_path.with_name(f".{checkpoints_path.name}.tmp").exists()


def test_second_replace_failure_rolls_back_both_outputs_and_cleans_backups(
    tmp_path: Path,
) -> None:
    paths = _write_sources(tmp_path, ([_row()], [], []))
    evaluations_path = paths.scaling_law_evaluations_path()
    checkpoints_path = paths.scaling_law_checkpoint_losses_path()
    evaluations_path.parent.mkdir(parents=True)
    evaluations_path.write_bytes(b"old evaluations")
    checkpoints_path.write_bytes(b"old checkpoints")
    real_replace = os.replace
    call_count = 0

    def fail_second_replace(source, destination):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise OSError("second replace failed")
        return real_replace(source, destination)

    with (
        patch.object(os, "replace", side_effect=fail_second_replace),
        pytest.raises(OSError, match="second replace failed"),
    ):
        preprocess_scaling_law(paths)

    assert evaluations_path.read_bytes() == b"old evaluations"
    assert checkpoints_path.read_bytes() == b"old checkpoints"
    assert not evaluations_path.with_name(f".{evaluations_path.name}.tmp").exists()
    assert not checkpoints_path.with_name(f".{checkpoints_path.name}.tmp").exists()
    assert not evaluations_path.with_name(
        f".{evaluations_path.name}.backup.tmp"
    ).exists()
    assert not checkpoints_path.with_name(
        f".{checkpoints_path.name}.backup.tmp"
    ).exists()


def test_preprocessing_does_not_download_or_upload(tmp_path: Path) -> None:
    paths = _write_sources(tmp_path, ([_row()], [], []))

    with (
        patch("datadec.data.download.download_sources") as download_sources,
        patch("huggingface_hub.HfApi.upload_file") as upload_file,
    ):
        preprocess_scaling_law(paths)

    download_sources.assert_not_called()
    upload_file.assert_not_called()
