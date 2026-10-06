from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from datadec.config import (
    PUBLISHED_RESULT_SCHEMAS,
    PublishedResultFile,
    PublishedResultSchema,
    PublishedResultsManifest,
)
from datadec.data.artifacts import DataArtifacts
from datadec.data.selection import (
    published_result_units,
    resolve_published_result_units,
)
from datadec.data.preprocess.published_results import preprocess_published_results

FOLDER_URL = "https://drive.google.com/drive/folders/1weYlEOlHrA_fzT2OsRa40uLc4EKTGz1D"

EXPECTED_SCHEMAS = {
    "transformed": (
        ("model", "string", False),
        ("group", "string", False),
        ("seed", "int64", False),
        ("metric", "string", False),
        ("models", "string", False),
        ("compute_latest", "float64", False),
        ("token_latest", "float64", False),
        ("raw_values", "string", False),
        ("value", "float64", False),
    ),
    "prediction_model_scale": (
        ("binary_accuracy", "float64", True),
        ("magnitude_correlation", "float64", True),
        ("pearson_correlation", "float64", True),
        ("weighted_pearson_correlation", "float64", True),
        ("NDCG", "float64", True),
        ("correct_count", "float64", True),
        ("incorrect_count", "float64", True),
        ("abstain_count", "float64", True),
        ("total_count", "float64", True),
        ("primary_abstain", "float64", True),
        ("mix1_better", "float64", True),
        ("mix2_better", "float64", True),
        ("actual_mix1_better", "float64", True),
        ("actual_mix2_better", "float64", True),
        ("mix_pairs_incorrect", "string", False),
        ("mix_pairs_correct", "string", False),
        ("metric", "string", False),
        ("model", "string", False),
        ("seed", "int64", False),
        ("compute_limit", "float64", True),
        ("compute_latest", "float64", True),
        ("proportion", "float64", False),
        ("tokens", "float64", False),
        ("three_way_accuracy", "float64", True),
        ("compute", "float64", True),
        ("proportion_target", "float64", True),
    ),
    "processed_ladder": (
        ("model", "string", False),
        ("group", "string", False),
        ("task", "string", False),
        ("step", "int64", False),
        ("seed", "int64", False),
        ("chinchilla", "string", False),
        ("tokens", "int64", False),
        ("compute", "float64", False),
        ("metrics", "string", False),
    ),
    "cheap_decisions": (
        ("task", "string", False),
        ("mix", "string", False),
        ("metric", "string", False),
        ("setup", "string", False),
        ("step_1_y", "float64", False),
        ("step_2_y", "float64", False),
        ("stacked_y", "float64", False),
        ("step_1_pred", "float64", False),
        ("step_2_pred", "float64", False),
        ("stacked_pred", "float64", False),
        ("abs_error_step_1", "float64", False),
        ("abs_error_step_2", "float64", False),
        ("abs_error_stacked", "float64", False),
        ("rel_error_stacked", "float64", False),
    ),
    "new_eval_decision_accuracy": (
        ("size", "string", False),
        ("task", "string", False),
        ("target_ranking", "string", False),
        ("logits_per_byte_corr", "float64", False),
        ("logits_per_char_corr", "float64", False),
        ("primary_score", "float64", False),
    ),
    "new_eval_means": (
        ("size", "string", False),
        ("task", "string", False),
        ("primary_score", "float64", False),
        ("logits_per_byte_corr", "float64", False),
        ("logits_per_char_corr", "float64", False),
    ),
    "target_pairs": (
        ("pair_index", "int64", False),
        ("model_1", "string", False),
        ("model_2", "string", False),
    ),
}


def _source(
    path: str,
    *,
    unit: str = "outputs2",
    schema: str = "transformed",
    file_id: str = "file-id",
) -> PublishedResultFile:
    return PublishedResultFile.model_validate(
        {
            "id": file_id,
            "path": path,
            "expected_size": 1,
            "sha256": "0" * 64,
            "category": "published_results",
            "publication_unit": unit,
            "schema": schema,
        }
    )


def _manifest(*sources: PublishedResultFile) -> PublishedResultsManifest:
    return PublishedResultsManifest(folder_url=FOLDER_URL, files=sources)


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(header)
        writer.writerows(rows)


def _valid_row(schema_name: PublishedResultSchema, **overrides: str) -> list[str]:
    schema = PUBLISHED_RESULT_SCHEMAS[schema_name]
    values = {
        column.name: (
            "text"
            if column.logical_type == "string"
            else "1"
            if column.logical_type == "int64"
            else "1.25"
        )
        for column in schema.columns
    }
    values.update(overrides)
    return [values[column.name] for column in schema.columns]


def _write_source_csv(
    paths: DataArtifacts,
    source: PublishedResultFile,
    rows: list[list[str]],
) -> None:
    assert source.schema is not None
    schema = PUBLISHED_RESULT_SCHEMAS[source.schema]
    _write_csv(
        paths.published_result_source_path(source),
        [column.name for column in schema.columns],
        rows,
    )


def test_schema_contracts_pin_exact_column_order_types_and_nullability() -> None:
    assert {
        name: tuple(
            (column.name, column.logical_type, column.nullable)
            for column in schema.columns
        )
        for name, schema in PUBLISHED_RESULT_SCHEMAS.items()
    } == EXPECTED_SCHEMAS


def test_manifest_units_follow_stable_source_order_and_all_semantics() -> None:
    manifest = _manifest(
        _source("outputs2/1_metric_transformed.csv", file_id="output"),
        _source(
            "per_task_out/arc_easy_out/1_metric_transformed.csv",
            unit="per-task-arc-easy",
            file_id="arc-easy",
        ),
        _source("outputs2/1_primary_transformed.csv", file_id="output-2"),
    )

    assert published_result_units(manifest) == ("outputs2", "per-task-arc-easy")
    assert resolve_published_result_units((), manifest) == (
        "outputs2",
        "per-task-arc-easy",
    )
    assert resolve_published_result_units(["all", "outputs2"], manifest) == (
        "outputs2",
        "per-task-arc-easy",
    )
    assert resolve_published_result_units(
        ["per-task-arc-easy", "outputs2", "outputs2"], manifest
    ) == ("outputs2", "per-task-arc-easy")


def test_manifest_rejects_wrong_path_membership() -> None:
    with pytest.raises(
        ValidationError, match="publication_unit does not match its source path"
    ):
        _source(
            "per_task_out/arc_easy_out/1_metric_transformed.csv",
            unit="outputs2",
        )
    with pytest.raises(ValidationError, match="schema does not match its source path"):
        _source(
            "outputs2/davidh_new_evals_means_df.csv",
            schema="new_eval_means",
        )


def test_selected_unit_consolidates_complete_schema_family_with_provenance(
    tmp_path: Path,
) -> None:
    sources = (
        _source(
            "per_task_out/arc_easy_out/1_metric_transformed.csv",
            unit="per-task-arc-easy",
            file_id="arc",
        ),
        _source("outputs2/1_metric_transformed.csv", file_id="output"),
    )
    paths = DataArtifacts(tmp_path)
    duplicate = _valid_row(
        "transformed",
        model="m-a",
        group="g",
        seed="7",
        metric="acc",
        models="['a', 'b']",
        raw_values="[1, 2]",
    )
    _write_source_csv(paths, sources[0], [duplicate])
    _write_source_csv(
        paths,
        sources[1],
        [_valid_row("transformed", model="m-z"), duplicate, duplicate],
    )

    results = preprocess_published_results(
        paths,
        units=("outputs2",),
        manifest=_manifest(*sources),
    )

    output = paths.published_result_table_path("transformed")
    table = pq.read_table(output)
    schema = PUBLISHED_RESULT_SCHEMAS["transformed"]
    assert output == tmp_path / "processed/published-results/transformed.parquet"
    assert table.schema.names == [
        "source_file",
        "source_unit",
        *(column.name for column in schema.columns),
    ]
    assert table.schema.types == [
        pa.string(),
        pa.string(),
        pa.string(),
        pa.string(),
        pa.int64(),
        pa.string(),
        pa.string(),
        pa.float64(),
        pa.float64(),
        pa.string(),
        pa.float64(),
    ]
    assert table.num_rows == 4
    assert table.column("source_unit").to_pylist() == [
        "outputs2",
        "outputs2",
        "outputs2",
        "per-task-arc-easy",
    ]
    assert table.column("model").to_pylist() == ["m-a", "m-a", "m-z", "m-a"]
    assert table.column("models").to_pylist()[:2] == ["['a', 'b']"] * 2
    assert table.column("raw_values").to_pylist()[:2] == ["[1, 2]"] * 2
    assert not (
        tmp_path / "processed/published-results/outputs2/1_metric_transformed.parquet"
    ).exists()
    assert len(results) == 1
    assert results[0].schema == "transformed"
    assert results[0].output_path == output
    assert results[0].row_count == 4
    assert [(file.source.id, file.row_count) for file in results[0].files] == [
        ("arc", 1),
        ("output", 3),
    ]


def test_prediction_nullable_numeric_blanks_become_null(tmp_path: Path) -> None:
    source = _source(
        "outputs2/2_prediction_model_scale.csv",
        schema="prediction_model_scale",
    )
    paths = DataArtifacts(tmp_path)
    schema = PUBLISHED_RESULT_SCHEMAS["prediction_model_scale"]
    values = []
    for column in schema.columns:
        if column.nullable:
            values.append("")
        elif column.logical_type == "string":
            values.append("['kept', 'as', 'text']")
        elif column.logical_type == "int64":
            values.append("3")
        else:
            values.append("1.25")
    _write_source_csv(paths, source, [values])

    preprocess_published_results(paths, manifest=_manifest(source))

    table = pq.read_table(paths.published_result_table_path("prediction_model_scale"))
    assert table.column("source_file").to_pylist() == [source.path]
    assert table.column("source_unit").to_pylist() == ["outputs2"]
    assert table.column("three_way_accuracy").to_pylist() == [None]
    assert table.column("compute").to_pylist() == [None]
    assert table.column("proportion_target").to_pylist() == [None]
    assert table.column("mix_pairs_correct").to_pylist() == ["['kept', 'as', 'text']"]


def test_required_numeric_blank_is_rejected_without_replacing_family(
    tmp_path: Path,
) -> None:
    source = _source("outputs2/1_metric_transformed.csv")
    paths = DataArtifacts(tmp_path)
    _write_source_csv(paths, source, [_valid_row("transformed", seed="")])
    output = paths.published_result_table_path("transformed")
    output.parent.mkdir(parents=True)
    output.write_bytes(b"existing")

    with pytest.raises(ValueError, match="seed"):
        preprocess_published_results(paths, manifest=_manifest(source))

    assert output.read_bytes() == b"existing"
    assert not output.with_name(f".{output.name}.tmp").exists()


def test_wrong_header_order_and_malformed_rows_are_rejected(tmp_path: Path) -> None:
    source = _source("outputs2/1_metric_transformed.csv")
    paths = DataArtifacts(tmp_path)
    input_path = paths.published_result_source_path(source)
    schema = PUBLISHED_RESULT_SCHEMAS["transformed"]
    header = [column.name for column in schema.columns]
    _write_csv(input_path, [header[1], header[0], *header[2:]], [["x"] * len(header)])
    with pytest.raises(ValueError, match="header mismatch"):
        preprocess_published_results(paths, manifest=_manifest(source))

    _write_csv(input_path, header, [["x"] * (len(header) - 1)])
    with pytest.raises(Exception, match="column"):
        preprocess_published_results(paths, manifest=_manifest(source))


def test_target_pair_families_preserve_index_orientation_and_provenance(
    tmp_path: Path,
) -> None:
    sources = (
        _source(
            "per_task_out/arc_easy_out/0_target_pairs.json",
            unit="per-task-arc-easy",
            schema="target_pairs",
            file_id="arc-pairs",
        ),
        _source(
            "outputs2/0_target_pairs.json",
            schema="target_pairs",
            file_id="output-pairs",
        ),
    )
    paths = DataArtifacts(tmp_path)
    for prefix, source in zip(("arc", "output"), sources, strict=True):
        input_path = paths.published_result_source_path(source)
        input_path.parent.mkdir(parents=True, exist_ok=True)
        pairs = [
            [f"{prefix}-left-{index}", f"{prefix}-right-{index}"]
            for index in range(300)
        ]
        input_path.write_text(json.dumps(pairs), encoding="utf-8")

    preprocess_published_results(
        paths,
        units=("outputs2",),
        manifest=_manifest(*sources),
    )

    table = pq.read_table(paths.published_result_table_path("target_pairs"))
    assert table.schema.names == [
        "source_file",
        "source_unit",
        "pair_index",
        "model_1",
        "model_2",
    ]
    assert table.schema.types == [
        pa.string(),
        pa.string(),
        pa.int64(),
        pa.string(),
        pa.string(),
    ]
    assert table.num_rows == 600
    assert table.column("source_unit").to_pylist()[:2] == ["outputs2"] * 2
    assert table.column("pair_index").to_pylist()[:3] == [0, 1, 2]
    assert table.column("model_1").to_pylist()[:2] == [
        "output-left-0",
        "output-left-1",
    ]
    assert table.column("model_2").to_pylist()[-2:] == [
        "arc-right-298",
        "arc-right-299",
    ]


@pytest.mark.parametrize(
    "bad_pair",
    [
        ["", "right"],
        ["left", ""],
        ["left"],
        ["left", 2],
    ],
)
def test_target_pairs_reject_invalid_or_empty_entries(
    tmp_path: Path, bad_pair: object
) -> None:
    source = _source(
        "outputs2/0_target_pairs.json", schema="target_pairs", file_id="pairs"
    )
    paths = DataArtifacts(tmp_path)
    input_path = paths.published_result_source_path(source)
    input_path.parent.mkdir(parents=True)
    pairs: list[object] = [[f"left-{index}", f"right-{index}"] for index in range(300)]
    pairs[17] = bad_pair
    input_path.write_text(json.dumps(pairs), encoding="utf-8")

    with pytest.raises(ValueError, match="two-string arrays"):
        preprocess_published_results(paths, manifest=_manifest(source))


def test_missing_expanded_family_source_preserves_existing_output(
    tmp_path: Path,
) -> None:
    sources = (
        _source("outputs2/1_metric_transformed.csv", file_id="output"),
        _source(
            "per_task_out/arc_easy_out/1_metric_transformed.csv",
            unit="per-task-arc-easy",
            file_id="arc",
        ),
    )
    paths = DataArtifacts(tmp_path)
    _write_source_csv(paths, sources[0], [_valid_row("transformed")])
    output = paths.published_result_table_path("transformed")
    output.parent.mkdir(parents=True)
    output.write_bytes(b"existing-family")

    with pytest.raises(FileNotFoundError, match="arc_easy_out"):
        preprocess_published_results(
            paths,
            units=("outputs2",),
            manifest=_manifest(*sources),
        )

    assert output.read_bytes() == b"existing-family"
    assert not output.with_name(f".{output.name}.tmp").exists()


def test_second_family_replacement_failure_rolls_back_every_family(
    tmp_path: Path,
) -> None:
    sources = (
        _source("outputs2/1_metric_transformed.csv", file_id="metric"),
        _source(
            "cheap_decisions_stacked_rc_pred_all.csv",
            unit="cheap-decisions",
            schema="cheap_decisions",
            file_id="cheap",
        ),
    )
    paths = DataArtifacts(tmp_path)
    for source in sources:
        assert source.schema is not None
        _write_source_csv(paths, source, [_valid_row(source.schema)])

    outputs = {
        schema: paths.published_result_table_path(schema)
        for schema in ("transformed", "cheap_decisions")
    }
    for schema, output in outputs.items():
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(f"old-{schema}".encode())

    real_replace = os.replace
    output_destinations = set(outputs.values())
    replacement_count = 0

    def fail_second_output(source: Path, destination: Path) -> None:
        nonlocal replacement_count
        if Path(destination) in output_destinations:
            replacement_count += 1
            if replacement_count == 2:
                raise OSError("second replacement failed")
        real_replace(source, destination)

    with (
        patch(
            "datadec.data.preprocess.duckdb.os.replace",
            side_effect=fail_second_output,
        ),
        pytest.raises(OSError, match="second replacement failed"),
    ):
        preprocess_published_results(paths, manifest=_manifest(*sources))

    for schema, output in outputs.items():
        assert output.read_bytes() == f"old-{schema}".encode()
        assert not output.with_name(f".{output.name}.tmp").exists()
        assert not output.with_name(f".{output.name}.backup.tmp").exists()
    for source in sources:
        assert paths.published_result_source_path(source).is_file()


def test_preprocessing_uses_no_network_clients(tmp_path: Path) -> None:
    source = _source("outputs2/1_metric_transformed.csv")
    paths = DataArtifacts(tmp_path)
    _write_source_csv(paths, source, [_valid_row("transformed")])
    with (
        patch("datadec.data.download.urlopen") as urlopen,
        patch("datadec.data.download.load_dataset") as load_dataset,
        patch("datadec.data.download.hf_hub_download") as hf_hub_download,
    ):
        preprocess_published_results(paths, manifest=_manifest(source))

    urlopen.assert_not_called()
    load_dataset.assert_not_called()
    hf_hub_download.assert_not_called()
