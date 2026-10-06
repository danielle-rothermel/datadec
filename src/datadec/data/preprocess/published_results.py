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
    PublishedResultSchema,
    PublishedResultTableSchema,
    PublishedResultsManifest,
    load_published_results_manifest,
)
from datadec.data.artifacts import DataArtifacts
from datadec.data.selection import selected_published_result_sources
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
    row_count: int


@dataclass(frozen=True, slots=True)
class PublishedResultsPreprocessResult:
    schema: PublishedResultSchema
    output_path: Path
    row_count: int
    files: tuple[PublishedResultPreprocessFile, ...]


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


def _target_pairs_select_sql(
    connection: duckdb.DuckDBPyConnection,
    *,
    source_path: Path,
    relation_name: str,
) -> str:
    rows = _load_target_pairs(source_path)
    table = pa.table(
        {
            "pair_index": pa.array((row[0] for row in rows), type=pa.int64()),
            "model_1": pa.array((row[1] for row in rows), type=pa.string()),
            "model_2": pa.array((row[2] for row in rows), type=pa.string()),
        }
    )
    connection.register(relation_name, table)
    return f"SELECT pair_index, model_1, model_2 FROM {quote_identifier(relation_name)}"


def _with_provenance(select_sql: str, source: PublishedResultFile) -> str:
    assert source.publication_unit is not None
    return (
        f"SELECT {sql_literal(source.path)}::VARCHAR AS source_file, "
        f"{sql_literal(source.publication_unit)}::VARCHAR AS source_unit, source.* "
        f"FROM ({select_sql}) AS source"
    )


def _family_sort_columns(schema: PublishedResultTableSchema) -> tuple[str, ...]:
    names = {column.name for column in schema.columns}
    useful_columns = ("model", "group", "seed", "metric", "task", "step")
    if "pair_index" in names:
        useful_columns = ("pair_index",)
    return ("source_unit", "source_file") + tuple(
        column for column in useful_columns if column in names
    )


def _prepare_family(
    connection: duckdb.DuckDBPyConnection,
    *,
    schema_name: PublishedResultSchema,
    sources: tuple[PublishedResultFile, ...],
    paths: DataArtifacts,
) -> tuple[PendingParquetExport, tuple[PublishedResultPreprocessFile, ...]]:
    schema = PUBLISHED_RESULT_SCHEMAS[schema_name]
    source_selects: list[str] = []
    files: list[PublishedResultPreprocessFile] = []

    for source_index, source in enumerate(sources):
        source_path = paths.published_result_source_path(source)
        if schema_name == "target_pairs":
            relation_name = f"_published_target_pairs_{source_index}"
            select_sql = _target_pairs_select_sql(
                connection,
                source_path=source_path,
                relation_name=relation_name,
            )
        else:
            _validate_csv_header(source_path, schema)
            select_sql = _csv_select_sql(source_path, schema)

        row_count = _validate_non_nullable_columns(
            connection,
            select_sql=select_sql,
            schema=schema,
            source_path=source_path,
        )
        files.append(
            PublishedResultPreprocessFile(
                source=source,
                source_path=source_path,
                row_count=row_count,
            )
        )
        source_selects.append(_with_provenance(select_sql, source))

    output_path = paths.published_result_table_path(schema_name)
    union_sql = " UNION ALL ".join(f"({select_sql})" for select_sql in source_selects)
    sort_sql = ", ".join(
        quote_identifier(column) for column in _family_sort_columns(schema)
    )
    expected_count = sum(file.row_count for file in files)
    try:
        export = prepare_parquet_export(
            connection,
            select_sql=f"SELECT * FROM ({union_sql}) ORDER BY {sort_sql}",
            output_path=output_path,
        )
    except BaseException:
        remove_owned_file(output_path.with_name(f".{output_path.name}.tmp"))
        raise
    if export.row_count != expected_count:
        remove_owned_file(export.temporary_path)
        raise RuntimeError(
            f"published result row count changed during {schema_name} family export"
        )
    return export, tuple(files)


def preprocess_published_results(
    paths: DataArtifacts,
    *,
    units: Sequence[str] = (),
    manifest: PublishedResultsManifest | None = None,
    verbose: bool = False,
) -> tuple[PublishedResultsPreprocessResult, ...]:
    manifest = manifest or load_published_results_manifest()
    selected = selected_published_result_sources(units, manifest)
    source_paths = tuple(
        paths.published_result_source_path(source) for source in selected
    )
    missing = [source_path for source_path in source_paths if not source_path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing structured published result family sources: "
            + ", ".join(str(path) for path in missing)
        )

    selected_schemas = tuple(
        schema_name
        for schema_name in PUBLISHED_RESULT_SCHEMAS
        if any(source.schema == schema_name for source in selected)
    )
    exports: list[PendingParquetExport] = []
    results: list[PublishedResultsPreprocessResult] = []
    replaced = False
    connection = duckdb.connect()
    try:
        for schema_name in selected_schemas:
            family_sources = tuple(
                source for source in selected if source.schema == schema_name
            )
            export, files = _prepare_family(
                connection,
                schema_name=schema_name,
                sources=family_sources,
                paths=paths,
            )
            exports.append(export)
            results.append(
                PublishedResultsPreprocessResult(
                    schema=schema_name,
                    output_path=export.output_path,
                    row_count=export.row_count,
                    files=files,
                )
            )
        replace_parquet_exports(tuple(exports))
        replaced = True
    finally:
        connection.close()
        if not replaced:
            for export in exports:
                remove_owned_file(export.temporary_path)

    final_results = tuple(results)
    if verbose:
        for result in final_results:
            print(
                f"{result.schema}: {result.row_count} rows from "
                f"{len(result.files)} sources -> {result.output_path}"
            )
    return final_results


__all__ = [
    "PublishedResultPreprocessFile",
    "PublishedResultsPreprocessResult",
    "preprocess_published_results",
]
