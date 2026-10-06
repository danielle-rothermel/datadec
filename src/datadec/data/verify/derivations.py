from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb

from datadec.config import (
    CHECKPOINT_ENRICHMENT_COLUMNS,
    MODEL_DETAIL_TYPES,
    load_catalog,
)
from datadec.data.artifacts import DataArtifacts
from datadec.data.model_utils import create_model_schedules
from datadec.data.preprocess.duckdb import sql_literal
from datadec.data.preprocess.model_enrichment import create_model_enrichment_table
from datadec.data.selection import DatasetSelection

_RELATIVE_TOLERANCE = 1e-12
_ABSOLUTE_TOLERANCE = 1e-6


@dataclass(frozen=True, slots=True)
class ScheduleTableVerification:
    name: str
    path: Path
    row_count: int
    unknown_model_count: int
    negative_step_count: int
    token_evidence_count: int
    token_mismatch_count: int
    compute_evidence_count: int
    exact_compute_mismatch_count: int
    model_detail_evidence_count: int
    model_detail_mismatch_count: int
    lr_evidence_count: int
    lr_mismatch_count: int

    @property
    def contradiction_count(self) -> int:
        return (
            self.unknown_model_count
            + self.negative_step_count
            + self.token_mismatch_count
            + self.exact_compute_mismatch_count
            + self.model_detail_mismatch_count
            + self.lr_mismatch_count
        )


@dataclass(frozen=True, slots=True)
class ScalingRawVerification:
    paths: tuple[Path, ...]
    row_count: int
    unknown_model_count: int
    invalid_step_count: int
    token_evidence_count: int
    token_mismatch_count: int
    compute_evidence_count: int
    exact_compute_mismatch_count: int
    nominal_compute_mismatch_count: int

    @property
    def contradiction_count(self) -> int:
        return (
            self.unknown_model_count
            + self.invalid_step_count
            + self.token_mismatch_count
            + self.exact_compute_mismatch_count
        )


@dataclass(frozen=True, slots=True)
class DetailTasksVerification:
    path: Path
    row_count: int
    unknown_model_count: int
    negative_step_count: int
    max_length_mismatch_count: int
    model_identity_mismatch_count: int
    revision_step_mismatch_count: int

    @property
    def contradiction_count(self) -> int:
        return (
            self.unknown_model_count
            + self.negative_step_count
            + self.max_length_mismatch_count
            + self.model_identity_mismatch_count
            + self.revision_step_mismatch_count
        )


@dataclass(frozen=True, slots=True)
class DerivationVerificationResult:
    processed_outputs: tuple[ScheduleTableVerification, ...]
    raw_olmes: ScheduleTableVerification | None
    raw_scaling_law: ScalingRawVerification | None
    detail_tasks: tuple[DetailTasksVerification, ...]
    lr_raw_evidence_count: int = 0

    @property
    def contradiction_count(self) -> int:
        return sum(
            verification.contradiction_count
            for verification in (
                *self.processed_outputs,
                *((self.raw_olmes,) if self.raw_olmes is not None else ()),
                *((self.raw_scaling_law,) if self.raw_scaling_law is not None else ()),
                *self.detail_tasks,
            )
        )


def _register_model_schedule(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        CREATE TEMP TABLE _model_schedule (
            params VARCHAR PRIMARY KEY,
            nominal_parameter_count BIGINT NOT NULL,
            training_parameter_count BIGINT NOT NULL,
            exact_parameter_count BIGINT NOT NULL,
            tokens_per_step BIGINT NOT NULL,
            flops_per_token_per_parameter BIGINT NOT NULL
        )
        """
    )
    connection.executemany(
        "INSERT INTO _model_schedule VALUES (?, ?, ?, ?, ?, ?)",
        [
            (
                schedule.params,
                schedule.nominal_parameter_count,
                schedule.training_parameter_count,
                schedule.exact_parameter_count,
                schedule.tokens_per_step,
                schedule.flops_per_token_per_parameter,
            )
            for schedule in create_model_schedules()
        ],
    )


def _table_columns(
    connection: duckdb.DuckDBPyConnection,
    *,
    path: Path,
    source_sql: str | None = None,
) -> set[str]:
    source_sql = source_sql or f"SELECT * FROM read_parquet({sql_literal(path)})"
    return {
        str(row[0]) for row in connection.execute(f"DESCRIBE {source_sql}").fetchall()
    }


def _compute_mismatch_sql(actual: str, expected: str) -> str:
    return (
        f"abs({actual} - ({expected})) > "
        f"greatest({_ABSOLUTE_TOLERANCE}, "
        f"{_RELATIVE_TOLERANCE} * abs({expected}))"
    )


def _verify_schedule_table(
    connection: duckdb.DuckDBPyConnection,
    *,
    name: str,
    path: Path,
    expect_full_enrichment: bool,
    source_sql: str | None = None,
) -> ScheduleTableVerification:
    columns = _table_columns(connection, path=path, source_sql=source_sql)
    required = {"params", "step"}
    missing = required.difference(columns)
    if missing:
        raise ValueError(
            f"{name} is missing schedule identity columns: {sorted(missing)!r}"
        )

    enrichment_columns = set(CHECKPOINT_ENRICHMENT_COLUMNS)
    present_enrichment = enrichment_columns.intersection(columns)
    if expect_full_enrichment and present_enrichment != enrichment_columns:
        missing_enrichment = sorted(enrichment_columns.difference(columns))
        raise ValueError(
            f"{name} is missing checkpoint enrichment columns: {missing_enrichment!r}"
        )

    token_evidence = "count(x.tokens)" if "tokens" in columns else "0"
    token_mismatches = (
        "count(*) FILTER (WHERE x.tokens IS NULL OR x.tokens <> enrichment.tokens)"
        if expect_full_enrichment
        else "count(*) FILTER (WHERE x.tokens IS NOT NULL AND x.tokens <> enrichment.tokens)"
        if "tokens" in columns
        else "0"
    )
    expected_compute = "enrichment.compute"
    compute_evidence = "count(x.compute)" if "compute" in columns else "0"
    compute_mismatches = (
        "count(*) FILTER (WHERE x.compute IS NULL OR "
        + _compute_mismatch_sql("x.compute::DOUBLE", expected_compute)
        + ")"
        if expect_full_enrichment
        else "count(*) FILTER (WHERE x.compute IS NOT NULL AND "
        + _compute_mismatch_sql("x.compute::DOUBLE", expected_compute)
        + ")"
        if "compute" in columns
        else "0"
    )
    model_mismatch_conditions = []
    for field, logical_type in MODEL_DETAIL_TYPES:
        if logical_type == "float64":
            model_mismatch_conditions.append(
                f"x.{field} IS NULL OR "
                + _compute_mismatch_sql(
                    f"x.{field}::DOUBLE",
                    f"enrichment.{field}::DOUBLE",
                )
            )
        else:
            model_mismatch_conditions.append(
                f"x.{field} IS DISTINCT FROM enrichment.{field}"
            )
    model_detail_evidence = "count(*)" if expect_full_enrichment else "0"
    model_detail_mismatches = (
        "count(*) FILTER (WHERE " + " OR ".join(model_mismatch_conditions) + ")"
        if expect_full_enrichment
        else "0"
    )
    lr_mismatch_conditions = " OR ".join(
        f"x.{field} IS NULL OR "
        + _compute_mismatch_sql(
            f"x.{field}::DOUBLE",
            f"enrichment.{field}::DOUBLE",
        )
        for field in ("lr_at_step", "cumulative_lr")
    )
    lr_evidence = "count(*)" if expect_full_enrichment else "0"
    lr_mismatches = (
        f"count(*) FILTER (WHERE {lr_mismatch_conditions})"
        if expect_full_enrichment
        else "0"
    )
    source_sql = source_sql or f"SELECT * FROM read_parquet({sql_literal(path)})"
    row = connection.execute(
        f"""
        SELECT
            count(*),
            count(*) FILTER (WHERE enrichment.params IS NULL),
            count(*) FILTER (WHERE x.step < 0),
            {token_evidence},
            {token_mismatches},
            {compute_evidence},
            {compute_mismatches},
            {model_detail_evidence},
            {model_detail_mismatches},
            {lr_evidence},
            {lr_mismatches}
        FROM ({source_sql}) AS x
        LEFT JOIN _model_enrichment AS enrichment USING (params, step)
        """
    ).fetchone()
    assert row is not None
    return ScheduleTableVerification(name, path, *map(int, row))


def _verify_scaling_raw(
    connection: duckdb.DuckDBPyConnection,
    *,
    paths: tuple[Path, ...],
) -> ScalingRawVerification:
    if not paths:
        raise ValueError("no scaling-law raw CSV paths are configured")
    files_sql = ", ".join(sql_literal(path) for path in paths)
    expected_tokens = "step_value * schedule.tokens_per_step"
    expected_exact_compute = (
        "step_value::DOUBLE * schedule.tokens_per_step::DOUBLE "
        "* schedule.exact_parameter_count::DOUBLE "
        "* schedule.flops_per_token_per_parameter::DOUBLE"
    )
    expected_nominal_compute = (
        "step_value::DOUBLE * schedule.tokens_per_step::DOUBLE "
        "* schedule.nominal_parameter_count::DOUBLE "
        "* schedule.flops_per_token_per_parameter::DOUBLE"
    )
    exact_mismatch = _compute_mismatch_sql("compute_value", expected_exact_compute)
    nominal_mismatch = _compute_mismatch_sql("compute_value", expected_nominal_compute)
    row = connection.execute(
        f"""
        WITH raw AS (
            SELECT
                model AS params,
                try_cast(step AS BIGINT) AS step_value,
                try_cast(tokens AS HUGEINT) AS tokens_value,
                try_cast(compute AS DOUBLE) AS compute_value
            FROM read_csv(
                [{files_sql}],
                header = true,
                all_varchar = true,
                filename = true
            )
        )
        SELECT
            count(*),
            count(*) FILTER (WHERE schedule.params IS NULL),
            count(*) FILTER (WHERE step_value IS NULL OR step_value < 0),
            count(tokens_value),
            count(*) FILTER (
                WHERE tokens_value IS NOT NULL
                  AND tokens_value <> {expected_tokens}
            ),
            count(compute_value),
            count(*) FILTER (
                WHERE compute_value IS NOT NULL AND {exact_mismatch}
            ),
            count(*) FILTER (
                WHERE compute_value IS NOT NULL AND {nominal_mismatch}
            )
        FROM raw
        LEFT JOIN _model_schedule AS schedule USING (params)
        """
    ).fetchone()
    assert row is not None
    return ScalingRawVerification(paths, *map(int, row))


def _verify_detail_tasks(
    connection: duckdb.DuckDBPyConnection,
    *,
    path: Path,
    source_sql: str | None = None,
) -> DetailTasksVerification:
    max_sequence_length = load_catalog().training.max_sequence_length
    source_sql = source_sql or f"SELECT * FROM read_parquet({sql_literal(path)})"
    row = connection.execute(
        f"""
        SELECT
            count(*),
            count(*) FILTER (WHERE schedule.params IS NULL),
            count(*) FILTER (WHERE tasks.step < 0),
            count(*) FILTER (
                WHERE try_cast(
                    json_extract_string(model_config, '$.max_length') AS BIGINT
                ) <> {max_sequence_length}
            ),
            count(*) FILTER (
                WHERE strpos(
                    json_extract_string(model_config, '$.model'),
                    '-' || tasks.params || '-'
                ) = 0
            ),
            count(*) FILTER (
                WHERE json_extract_string(model_config, '$.revision')
                    <> 'step' || tasks.step::VARCHAR || '-unsharded-hf'
            )
        FROM ({source_sql}) AS tasks
        LEFT JOIN _model_schedule AS schedule USING (params)
        """
    ).fetchone()
    assert row is not None
    return DetailTasksVerification(path, *map(int, row))


def verify_preprocessed_derivations(
    paths: DataArtifacts,
    selection: DatasetSelection,
) -> DerivationVerificationResult:
    processed: list[tuple[str, Path, str | None]] = []
    if selection.ppl:
        processed.append(("ppl", paths.get_path("ppl_processed"), None))
    if selection.olmes:
        processed.append(("olmes", paths.get_path("olmes_processed"), None))
    if selection.scaling_law:
        processed.extend(
            (
                (
                    "scaling-law evaluations",
                    paths.scaling_law_evaluations_path(),
                    None,
                ),
                (
                    "scaling-law checkpoint losses",
                    paths.scaling_law_checkpoint_losses_path(),
                    None,
                ),
            )
        )
    detail_path = paths.olmes_details_tasks_path() if selection.olmes_details else None
    detail_source_sql = (
        "SELECT * FROM _selected_detail_tasks" if detail_path is not None else None
    )
    if detail_path is not None:
        processed.append(("OLMES detail tasks", detail_path, detail_source_sql))
    raw_olmes_path = paths.get_path("dwn_raw") if selection.olmes else None
    raw_scaling_paths = paths.scaling_law_raw_paths() if selection.scaling_law else ()
    required_paths = (
        *(path for _, path, _ in processed),
        *((raw_olmes_path,) if raw_olmes_path is not None else ()),
        *raw_scaling_paths,
    )
    missing = tuple(path for path in required_paths if not path.is_file())
    if missing:
        raise FileNotFoundError(
            "missing derivation verification inputs: "
            + ", ".join(str(path) for path in missing)
        )

    if not processed:
        return DerivationVerificationResult(
            processed_outputs=(),
            raw_olmes=None,
            raw_scaling_law=None,
            detail_tasks=(),
        )

    connection = duckdb.connect()
    try:
        _register_model_schedule(connection)
        if detail_path is not None:
            selected_recipes = ", ".join(
                sql_literal(recipe) for recipe in selection.olmes_details
            )
            connection.execute(
                f"""
                CREATE TEMP TABLE _selected_detail_tasks AS
                SELECT *
                FROM read_parquet({sql_literal(detail_path)})
                WHERE recipe IN ({selected_recipes})
                """
            )
        checkpoint_union = " UNION ALL ".join(
            (
                f"SELECT params, step FROM ({source_sql})"
                if source_sql is not None
                else "SELECT params, step FROM read_parquet(" + sql_literal(path) + ")"
            )
            for _, path, source_sql in processed
        )
        create_model_enrichment_table(
            connection,
            checkpoint_select_sql=checkpoint_union,
        )
        return DerivationVerificationResult(
            processed_outputs=tuple(
                _verify_schedule_table(
                    connection,
                    name=name,
                    path=path,
                    expect_full_enrichment=True,
                    source_sql=source_sql,
                )
                for name, path, source_sql in processed
            ),
            raw_olmes=(
                _verify_schedule_table(
                    connection,
                    name="raw aggregate OLMES",
                    path=raw_olmes_path,
                    expect_full_enrichment=False,
                )
                if raw_olmes_path is not None
                else None
            ),
            raw_scaling_law=(
                _verify_scaling_raw(connection, paths=raw_scaling_paths)
                if raw_scaling_paths
                else None
            ),
            detail_tasks=(
                (
                    _verify_detail_tasks(
                        connection,
                        path=detail_path,
                        source_sql=detail_source_sql,
                    ),
                )
                if detail_path is not None
                else ()
            ),
        )
    finally:
        connection.close()


__all__ = [
    "DerivationVerificationResult",
    "DetailTasksVerification",
    "ScalingRawVerification",
    "ScheduleTableVerification",
    "verify_preprocessed_derivations",
]
