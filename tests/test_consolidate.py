from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from datadec.config import (
    PUBLISHED_RESULT_SCHEMAS,
    PublishedResultFile,
    PublishedResultsManifest,
    load_olmes_contract,
)
from datadec.data.artifacts import DataArtifacts
from datadec.data.consolidate import (
    consolidate_processed_outputs,
    verify_consolidated_outputs,
)

FOLDER_URL = "https://drive.google.com/drive/folders/1weYlEOlHrA_fzT2OsRa40uLc4EKTGz1D"
RECIPES = ("c4", "dolma1.7")


def _source(path: str, *, unit: str, file_id: str) -> PublishedResultFile:
    return PublishedResultFile.model_validate(
        {
            "id": file_id,
            "path": path,
            "expected_size": 1,
            "sha256": "0" * 64,
            "category": "published_results",
            "publication_unit": unit,
            "schema": "target_pairs",
        }
    )


def _manifest() -> PublishedResultsManifest:
    return PublishedResultsManifest(
        folder_url=FOLDER_URL,
        files=(
            _source(
                "outputs2/0_target_pairs.json",
                unit="outputs2",
                file_id="outputs-pairs",
            ),
            _source(
                "per_task_out/arc_easy_out/0_target_pairs.json",
                unit="per-task-arc-easy",
                file_id="arc-easy-pairs",
            ),
        ),
    )


def _arrow_type(logical_type: str) -> pa.DataType:
    return {
        "string": pa.string(),
        "int64": pa.int64(),
        "float64": pa.float64(),
        "bool": pa.bool_(),
    }[logical_type]


def _write_table(path: Path, rows: list[dict[str, object]], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def _target_pairs_schema() -> pa.Schema:
    return pa.schema(
        pa.field(
            column.name, _arrow_type(column.logical_type), nullable=column.nullable
        )
        for column in PUBLISHED_RESULT_SCHEMAS["target_pairs"].columns
    )


def _task_schema() -> pa.Schema:
    return pa.schema(
        pa.field(
            column.name, _arrow_type(column.logical_type), nullable=column.nullable
        )
        for column in load_olmes_contract().tables.detailed_tasks.columns
    )


def _task_row(recipe: str, marker: int) -> dict[str, object]:
    values: dict[str, object] = {}
    for column in load_olmes_contract().tables.detailed_tasks.columns:
        values[column.name] = {
            "string": f"{column.name}-{marker}",
            "int64": marker,
            "float64": marker + 0.25,
            "bool": marker % 2 == 0,
        }[column.logical_type]
    values["recipe"] = recipe
    values["acc_uncond"] = None
    return values


def _legacy_path(artifacts: DataArtifacts, source: PublishedResultFile) -> Path:
    return (
        artifacts.data_dir
        / "processed"
        / "published-results"
        / source.parquet_relative_path()
    )


def _legacy_task_path(artifacts: DataArtifacts, recipe: str) -> Path:
    return artifacts.data_dir / "processed" / "olmes-details" / recipe / "tasks.parquet"


def _write_legacy_inputs(
    artifacts: DataArtifacts, manifest: PublishedResultsManifest
) -> dict[Path, bytes]:
    sources = manifest.files
    duplicate = {"pair_index": 0, "model_1": "a", "model_2": "b"}
    _write_table(
        _legacy_path(artifacts, sources[0]),
        [duplicate, duplicate, {"pair_index": 1, "model_1": "c", "model_2": "d"}],
        _target_pairs_schema(),
    )
    _write_table(
        _legacy_path(artifacts, sources[1]),
        [{"pair_index": 0, "model_1": "left", "model_2": "right"}],
        _target_pairs_schema(),
    )
    for marker, recipe in enumerate(RECIPES, start=1):
        _write_table(
            _legacy_task_path(artifacts, recipe),
            [_task_row(recipe, marker)],
            _task_schema(),
        )
    return {
        path: path.read_bytes()
        for path in (
            *(_legacy_path(artifacts, source) for source in sources),
            *(_legacy_task_path(artifacts, recipe) for recipe in RECIPES),
        )
    }


def _row_multiset(
    path: Path,
    columns: tuple[str, ...],
    *,
    filters: Callable[[dict[str, object]], bool],
) -> Counter[tuple[object, ...]]:
    return Counter(
        tuple(row[column] for column in columns)
        for row in pq.read_table(path).to_pylist()
        if filters(row)
    )


def test_consolidation_preserves_multisets_provenance_recipes_and_inputs(
    tmp_path: Path,
) -> None:
    artifacts = DataArtifacts(tmp_path)
    manifest = _manifest()
    original_inputs = _write_legacy_inputs(artifacts, manifest)

    checks = consolidate_processed_outputs(
        artifacts, manifest=manifest, recipes=RECIPES
    )

    family_path = artifacts.published_result_table_path("target_pairs")
    tasks_path = artifacts.olmes_details_tasks_path()
    assert tuple(
        (check.source_path, check.output_path, check.row_count) for check in checks
    ) == (
        (_legacy_path(artifacts, manifest.files[0]), family_path, 3),
        (_legacy_path(artifacts, manifest.files[1]), family_path, 1),
        (_legacy_task_path(artifacts, "c4"), tasks_path, 1),
        (_legacy_task_path(artifacts, "dolma1.7"), tasks_path, 1),
    )

    pair_columns = tuple(
        column.name for column in PUBLISHED_RESULT_SCHEMAS["target_pairs"].columns
    )
    for source in manifest.files:
        expected = _row_multiset(
            _legacy_path(artifacts, source), pair_columns, filters=lambda row: True
        )
        actual = _row_multiset(
            family_path,
            pair_columns,
            filters=lambda row, source=source: (
                row["source_file"] == source.path
                and row["source_unit"] == source.publication_unit
            ),
        )
        assert actual == expected

    task_columns = tuple(
        column.name for column in load_olmes_contract().tables.detailed_tasks.columns
    )
    nullable_metric_index = task_columns.index("acc_uncond")
    for recipe in RECIPES:
        expected = _row_multiset(
            _legacy_task_path(artifacts, recipe),
            task_columns,
            filters=lambda row: True,
        )
        actual = _row_multiset(
            tasks_path,
            task_columns,
            filters=lambda row, recipe=recipe: row["recipe"] == recipe,
        )
        assert actual == expected
        assert all(row[nullable_metric_index] is None for row in actual)

    assert {path: path.read_bytes() for path in original_inputs} == original_inputs


def _mutate_family_output(artifacts: DataArtifacts, mutation: str) -> None:
    path = artifacts.published_result_table_path("target_pairs")
    table = pq.read_table(path)
    rows = table.to_pylist()
    outputs_index = next(
        index
        for index, row in enumerate(rows)
        if row["source_file"] == "outputs2/0_target_pairs.json"
    )
    if mutation == "dropped duplicate":
        rows.pop(outputs_index)
    elif mutation == "altered value":
        rows[outputs_index]["model_2"] = "altered"
    elif mutation == "wrong source unit":
        rows[outputs_index]["source_unit"] = "per-task-arc-easy"
    elif mutation == "extra unexpected row":
        unexpected = dict(rows[outputs_index])
        unexpected["source_file"] = "unexpected/0_target_pairs.json"
        rows.append(unexpected)
    else:  # pragma: no cover - the parametrization owns this closed set
        raise AssertionError(f"unknown test mutation: {mutation}")
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("dropped duplicate", "consolidation changed 1 rows"),
        ("altered value", "consolidation changed 1 rows"),
        ("wrong source unit", "consolidation changed 1 rows"),
        ("extra unexpected row", "unexpected rows in consolidated table"),
    ],
)
def test_verification_rejects_non_preserving_family_outputs(
    tmp_path: Path, mutation: str, message: str
) -> None:
    artifacts = DataArtifacts(tmp_path)
    manifest = _manifest()
    _write_legacy_inputs(artifacts, manifest)
    consolidate_processed_outputs(artifacts, manifest=manifest, recipes=RECIPES)
    _mutate_family_output(artifacts, mutation)

    with pytest.raises(AssertionError, match=message):
        verify_consolidated_outputs(artifacts, manifest=manifest, recipes=RECIPES)


def test_missing_legacy_input_prevents_any_output_rewrite(tmp_path: Path) -> None:
    artifacts = DataArtifacts(tmp_path)
    manifest = _manifest()
    original_inputs = _write_legacy_inputs(artifacts, manifest)
    missing = _legacy_task_path(artifacts, RECIPES[-1])
    missing.unlink()
    original_inputs.pop(missing)

    outputs = (
        artifacts.published_result_table_path("target_pairs"),
        artifacts.olmes_details_tasks_path(),
    )
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(f"existing:{output.name}".encode())
    original_outputs = {path: path.read_bytes() for path in outputs}

    with pytest.raises(FileNotFoundError, match="missing legacy consolidation inputs"):
        consolidate_processed_outputs(artifacts, manifest=manifest, recipes=RECIPES)

    assert {path: path.read_bytes() for path in original_inputs} == original_inputs
    assert {path: path.read_bytes() for path in outputs} == original_outputs
    assert all(
        not output.with_name(f".{output.name}.tmp").exists() for output in outputs
    )
