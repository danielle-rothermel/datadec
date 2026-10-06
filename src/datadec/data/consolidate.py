"""One-time migration of source-shaped processed exports to canonical tables.

This module never deletes a source or publishes remotely. Preservation checks
must pass before the caller retires any legacy artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb

from datadec.config import (
    PUBLISHED_RESULT_SCHEMAS,
    PublishedResultsManifest,
    load_olmes_contract,
    load_published_results_manifest,
    load_source_manifest,
)
from datadec.data.artifacts import DataArtifacts
from datadec.data.preprocess.duckdb import (
    prepare_parquet_export,
    quote_identifier,
    remove_owned_file,
    replace_parquet_exports,
    sql_literal,
)


@dataclass(frozen=True, slots=True)
class PreservationCheck:
    source_path: Path
    output_path: Path
    row_count: int


def _assert_same_rows(
    connection: duckdb.DuckDBPyConnection,
    source_sql: str,
    output_sql: str,
    *,
    source_path: Path,
) -> int:
    # EXCEPT ALL compares the multiset, preserving duplicate multiplicity and nulls.
    for left, right in ((source_sql, output_sql), (output_sql, source_sql)):
        mismatch = connection.execute(
            f"SELECT count(*) FROM (({left}) EXCEPT ALL ({right}))"
        ).fetchone()
        assert mismatch is not None
        if mismatch[0]:
            raise AssertionError(
                f"consolidation changed {mismatch[0]} rows from {source_path}"
            )
    count = connection.execute(f"SELECT count(*) FROM ({source_sql})").fetchone()
    assert count is not None
    return int(count[0])


def verify_consolidated_outputs(
    artifacts: DataArtifacts,
    *,
    manifest: PublishedResultsManifest | None = None,
    recipes: tuple[str, ...] | None = None,
) -> tuple[PreservationCheck, ...]:
    """Prove exact row preservation for every legacy source and recipe."""
    manifest = manifest or load_published_results_manifest()
    recipes = (
        recipes if recipes is not None else load_source_manifest().olmes_details.recipes
    )
    checks: list[PreservationCheck] = []
    connection = duckdb.connect()
    try:
        for source in manifest.files:
            if source.category != "published_results":
                continue
            assert source.schema is not None
            old = (
                artifacts.data_dir
                / "processed"
                / "published-results"
                / source.parquet_relative_path()
            )
            new = artifacts.published_result_output_path(source)
            columns = ", ".join(
                quote_identifier(c.name)
                for c in PUBLISHED_RESULT_SCHEMAS[source.schema].columns
            )
            count = _assert_same_rows(
                connection,
                f"SELECT {columns} FROM read_parquet({sql_literal(old)})",
                f"SELECT {columns} FROM read_parquet({sql_literal(new)}) WHERE source_file={sql_literal(source.path)} AND source_unit={sql_literal(source.publication_unit or '')}",
                source_path=old,
            )
            checks.append(PreservationCheck(old, new, count))
        tasks = artifacts.olmes_details_tasks_path()
        columns = ", ".join(
            quote_identifier(c.name)
            for c in load_olmes_contract().tables.detailed_tasks.columns
        )
        for recipe in recipes:
            old = (
                artifacts.data_dir
                / "processed"
                / "olmes-details"
                / recipe
                / "tasks.parquet"
            )
            count = _assert_same_rows(
                connection,
                f"SELECT {columns} FROM read_parquet({sql_literal(old)})",
                f"SELECT {columns} FROM read_parquet({sql_literal(tasks)}) WHERE recipe={sql_literal(recipe)}",
                source_path=old,
            )
            checks.append(PreservationCheck(old, tasks, count))
        # Per-source comparisons alone would not reject unexpected provenance rows.
        expected_tables: dict[Path, int] = {}
        for check in checks:
            expected_tables[check.output_path] = (
                expected_tables.get(check.output_path, 0) + check.row_count
            )
        for path, expected_count in expected_tables.items():
            actual = connection.execute(
                "SELECT count(*) FROM read_parquet(?)", [str(path)]
            ).fetchone()
            assert actual is not None
            if actual[0] != expected_count:
                raise AssertionError(f"unexpected rows in consolidated table {path}")
    finally:
        connection.close()
    return tuple(checks)


def consolidate_processed_outputs(
    artifacts: DataArtifacts,
    *,
    manifest: PublishedResultsManifest | None = None,
    recipes: tuple[str, ...] | None = None,
) -> tuple[PreservationCheck, ...]:
    """Consolidate legacy tables locally, then prove their data was preserved."""
    manifest = manifest or load_published_results_manifest()
    recipes = (
        recipes if recipes is not None else load_source_manifest().olmes_details.recipes
    )
    sources = tuple(
        source for source in manifest.files if source.category == "published_results"
    )
    old_paths = [
        artifacts.data_dir
        / "processed"
        / "published-results"
        / source.parquet_relative_path()
        for source in sources
    ]
    old_paths.extend(
        artifacts.data_dir / "processed" / "olmes-details" / recipe / "tasks.parquet"
        for recipe in recipes
    )
    missing = [path for path in old_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing legacy consolidation inputs: {missing}")
    exports = []
    connection = duckdb.connect()
    try:
        for schema_name, schema in PUBLISHED_RESULT_SCHEMAS.items():
            family = [source for source in sources if source.schema == schema_name]
            if not family:
                continue
            parts = []
            for source in family:
                old = (
                    artifacts.data_dir
                    / "processed"
                    / "published-results"
                    / source.parquet_relative_path()
                )
                parts.append(
                    f"SELECT {sql_literal(source.path)} AS source_file, {sql_literal(source.publication_unit or '')} AS source_unit, * FROM read_parquet({sql_literal(old)})"
                )
            keys = [
                "source_unit",
                "source_file",
                *(
                    column.name
                    for column in schema.columns
                    if column.name
                    in {
                        "model",
                        "group",
                        "seed",
                        "metric",
                        "task",
                        "step",
                        "pair_index",
                    }
                ),
            ]
            order = ", ".join(quote_identifier(key) for key in keys)
            exports.append(
                prepare_parquet_export(
                    connection,
                    select_sql=f"SELECT * FROM ({' UNION ALL '.join(parts)}) ORDER BY {order}",
                    output_path=artifacts.published_result_table_path(schema_name),
                )
            )
        if recipes:
            inputs = ", ".join(
                sql_literal(
                    artifacts.data_dir
                    / "processed"
                    / "olmes-details"
                    / recipe
                    / "tasks.parquet"
                )
                for recipe in recipes
            )
            order = ", ".join(
                quote_identifier(key)
                for key in load_olmes_contract().tables.detailed_tasks.sort_key
            )
            exports.append(
                prepare_parquet_export(
                    connection,
                    select_sql=f"SELECT * FROM read_parquet([{inputs}]) ORDER BY {order}",
                    output_path=artifacts.olmes_details_tasks_path(),
                )
            )
        replace_parquet_exports(tuple(exports))
    finally:
        connection.close()
        for export in exports:
            remove_owned_file(export.temporary_path)
    return verify_consolidated_outputs(artifacts, manifest=manifest, recipes=recipes)


__all__ = [
    "PreservationCheck",
    "consolidate_processed_outputs",
    "verify_consolidated_outputs",
]
