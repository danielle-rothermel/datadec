from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import duckdb
import pyarrow as pa

from datadec.config import (
    PUBLISHED_RESULT_SCHEMAS,
    PublishedResultFile,
    PublishedResultTableSchema,
    PublishedResultUnit,
    PublishedResultsManifest,
    load_published_results_manifest,
)
from datadec.data.paths import DataDecidePaths
from datadec.data.preprocess.duckdb import (
    PendingParquetExport,
    duckdb_type,
    prepare_parquet_export,
    quote_identifier,
    remove_owned_file,
    replace_parquet_exports,
    sql_literal,
)


@dataclass(frozen=True, slots=True)
class PublishedResultPreprocessFile:
    source: PublishedResultFile
    source_path: Path
    output_path: Path
    row_count: int


@dataclass(frozen=True, slots=True)
class PublishedResultsPreprocessResult:
    publication_unit: PublishedResultUnit
    files: tuple[PublishedResultPreprocessFile, ...]


def published_result_units(
    manifest: PublishedResultsManifest,
) -> tuple[PublishedResultUnit, ...]:
    units: list[PublishedResultUnit] = []
    for source in manifest.files:
        unit = source.publication_unit
        if source.category == "published_results" and unit is not None:
            if unit not in units:
                units.append(unit)
    return tuple(units)


def resolve_published_result_units(
    requested: Sequence[str], manifest: PublishedResultsManifest
) -> tuple[PublishedResultUnit, ...]:
    available = published_result_units(manifest)
    allowed = set(available)
    unknown = [unit for unit in requested if unit != "all" and unit not in allowed]
    if unknown:
        names = ", ".join(dict.fromkeys(unknown))
        raise ValueError(f"unknown published-results unit: {names}")
    if not requested or "all" in requested:
        return available
    requested_set = set(requested)
    return tuple(unit for unit in available if unit in requested_set)


def _validate_csv_header(path: Path, schema: PublishedResultTableSchema) -> None:
    with path.open(newline="", encoding="utf-8") as input_file:
        try:
            header = next(csv.reader(input_file, strict=True))
        except StopIteration:
            raise ValueError(f"published result CSV is empty: {path}") from None
        except csv.Error as error:
            raise ValueError(
                f"malformed published result CSV header: {path}"
            ) from error
    expected = [column.name for column in schema.columns]
    if header != expected:
        raise ValueError(
            f"published result CSV header mismatch for {path}: "
            f"expected {expected!r}, found {header!r}"
        )


def _csv_select_sql(path: Path, schema: PublishedResultTableSchema) -> str:
    columns = ", ".join(
        f"{sql_literal(column.name)}: {sql_literal(duckdb_type(column.logical_type))}"
        for column in schema.columns
    )
    string_columns = ", ".join(
        sql_literal(column.name)
        for column in schema.columns
        if column.logical_type == "string"
    )
    options = [
        "header = true",
        "auto_detect = false",
        f"columns = {{{columns}}}",
        "strict_mode = true",
        "ignore_errors = false",
        "nullstr = ''",
    ]
    if string_columns:
        options.append(f"force_not_null = [{string_columns}]")
    projected = ", ".join(quote_identifier(column.name) for column in schema.columns)
    return (
        f"SELECT {projected} FROM read_csv({sql_literal(path)}, {', '.join(options)})"
    )


def _validate_non_nullable_columns(
    connection: duckdb.DuckDBPyConnection,
    *,
    select_sql: str,
    schema: PublishedResultTableSchema,
    source_path: Path,
) -> int:
    non_nullable = [column for column in schema.columns if not column.nullable]
    checks = ", ".join(
        f"count(*) FILTER (WHERE {quote_identifier(column.name)} IS NULL)"
        for column in non_nullable
    )
    row = connection.execute(
        f"SELECT count(*), {checks} FROM ({select_sql})"
    ).fetchone()
    assert row is not None
    row_count = int(row[0])
    if row_count == 0:
        raise ValueError(f"published result contains no rows: {source_path}")
    missing = [
        column.name
        for column, null_count in zip(non_nullable, row[1:], strict=True)
        if null_count
    ]
    if missing:
        raise ValueError(
            f"published result has null values in required columns for {source_path}: "
            + ", ".join(missing)
        )
    return row_count


def _prepare_csv(
    connection: duckdb.DuckDBPyConnection,
    *,
    source_path: Path,
    output_path: Path,
    schema: PublishedResultTableSchema,
) -> PendingParquetExport:
    _validate_csv_header(source_path, schema)
    select_sql = _csv_select_sql(source_path, schema)
    export = prepare_parquet_export(
        connection, select_sql=select_sql, output_path=output_path
    )
    try:
        validated_count = _validate_non_nullable_columns(
            connection,
            select_sql=(
                f"SELECT * FROM read_parquet({sql_literal(export.temporary_path)})"
            ),
            schema=schema,
            source_path=source_path,
        )
    except BaseException:
        remove_owned_file(export.temporary_path)
        raise
    if export.row_count != validated_count:
        remove_owned_file(export.temporary_path)
        raise RuntimeError(
            f"published result row count changed during export: {source_path}"
        )
    return export


def _load_target_pairs(path: Path) -> list[tuple[int, str, str]]:
    try:
        with path.open("rb") as input_file:
            value = json.load(input_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid published target-pairs JSON: {path}") from error
    if not isinstance(value, list) or len(value) != 300:
        raise ValueError(
            f"published target-pairs JSON must contain exactly 300 pairs: {path}"
        )
    rows: list[tuple[int, str, str]] = []
    for pair_index, pair in enumerate(value):
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or not all(isinstance(model, str) and model for model in pair)
        ):
            raise ValueError(
                "published target-pairs JSON entries must be two-string arrays: "
                f"{path} at index {pair_index}"
            )
        rows.append((pair_index, pair[0], pair[1]))
    return rows


def _prepare_target_pairs(
    connection: duckdb.DuckDBPyConnection,
    *,
    source_path: Path,
    output_path: Path,
) -> PendingParquetExport:
    rows = _load_target_pairs(source_path)
    table = pa.table(
        {
            "pair_index": pa.array((row[0] for row in rows), type=pa.int64()),
            "model_1": pa.array((row[1] for row in rows), type=pa.string()),
            "model_2": pa.array((row[2] for row in rows), type=pa.string()),
        }
    )
    connection.register("_published_target_pairs", table)
    try:
        return prepare_parquet_export(
            connection,
            select_sql=(
                "SELECT pair_index, model_1, model_2 "
                "FROM _published_target_pairs ORDER BY pair_index"
            ),
            output_path=output_path,
        )
    finally:
        connection.unregister("_published_target_pairs")


def preprocess_published_results(
    paths: DataDecidePaths,
    *,
    units: Sequence[str] = (),
    manifest: PublishedResultsManifest | None = None,
    verbose: bool = False,
) -> tuple[PublishedResultsPreprocessResult, ...]:
    manifest = manifest or load_published_results_manifest()
    selected_units = resolve_published_result_units(units, manifest)
    selected = tuple(
        source
        for source in manifest.files
        if source.category == "published_results"
        and source.publication_unit in selected_units
    )
    source_paths = tuple(
        paths.published_result_source_path(source) for source in selected
    )
    missing = [source_path for source_path in source_paths if not source_path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing structured published result sources: "
            + ", ".join(str(path) for path in missing)
        )

    exports: list[PendingParquetExport] = []
    converted: list[PublishedResultPreprocessFile] = []
    replaced = False
    connection = duckdb.connect()
    try:
        for source, source_path in zip(selected, source_paths, strict=True):
            assert source.schema is not None
            schema = PUBLISHED_RESULT_SCHEMAS[source.schema]
            output_path = paths.published_result_output_path(source)
            if source.schema == "target_pairs":
                export = _prepare_target_pairs(
                    connection, source_path=source_path, output_path=output_path
                )
            else:
                export = _prepare_csv(
                    connection,
                    source_path=source_path,
                    output_path=output_path,
                    schema=schema,
                )
            exports.append(export)
            converted.append(
                PublishedResultPreprocessFile(
                    source=source,
                    source_path=source_path,
                    output_path=output_path,
                    row_count=export.row_count,
                )
            )
        replace_parquet_exports(tuple(exports))
        replaced = True
    finally:
        connection.close()
        if not replaced:
            for export in exports:
                remove_owned_file(export.temporary_path)

    results = tuple(
        PublishedResultsPreprocessResult(
            publication_unit=unit,
            files=tuple(
                result for result in converted if result.source.publication_unit == unit
            ),
        )
        for unit in selected_units
    )
    if verbose:
        for result in results:
            for file in result.files:
                print(
                    f"{result.publication_unit}: {file.row_count} rows -> "
                    f"{file.output_path}"
                )
    return results


__all__ = [
    "PublishedResultPreprocessFile",
    "PublishedResultsPreprocessResult",
    "preprocess_published_results",
    "published_result_units",
    "resolve_published_result_units",
]
