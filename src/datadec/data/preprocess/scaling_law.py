from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import duckdb
from duckdb import func

from datadec.config import (
    CHECKPOINT_ENRICHMENT_COLUMNS,
    OLMESContract,
    ScalingLawContract,
    ScalingLawTableContract,
    load_olmes_contract,
    load_scaling_law_contract,
)
from datadec.data.model_utils import create_model_schedules
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
from datadec.data.preprocess.model_enrichment import create_model_enrichment_table

RAW_COLUMNS: tuple[str, ...] = (
    "group",
    "model",
    "task",
    "chinchilla",
    "step",
    "tokens",
    "compute",
    "metrics",
    "eval/c4_en-validation/CrossEntropyLoss",
    "eval/dolma_common-crawl-validation/CrossEntropyLoss",
    "eval/pile-validation/CrossEntropyLoss",
    "eval/wikitext_103-validation/CrossEntropyLoss",
    "train/CrossEntropyLoss",
    "throughput/total_tokens",
    "seed",
)

RAW_CHECKPOINT_FIELDS: dict[str, str] = {
    "tokens": "tokens",
    "compute": "compute",
    "c4_en_validation_cross_entropy": ("eval/c4_en-validation/CrossEntropyLoss"),
    "dolma_common_crawl_validation_cross_entropy": (
        "eval/dolma_common-crawl-validation/CrossEntropyLoss"
    ),
    "pile_validation_cross_entropy": ("eval/pile-validation/CrossEntropyLoss"),
    "wikitext_103_validation_cross_entropy": (
        "eval/wikitext_103-validation/CrossEntropyLoss"
    ),
    "train_cross_entropy": "train/CrossEntropyLoss",
    "throughput_total_tokens": "throughput/total_tokens",
}

_IGNORED_METRIC_FIELDS: frozenset[str] = frozenset(
    {
        "predicted_index_raw",
        "predicted_index_per_token",
        "predicted_index_per_char",
        "predicted_index_per_byte",
        "predicted_index_uncond",
        "primary_metric",
    }
)


@dataclass(frozen=True, slots=True)
class ScalingLawPreprocessResult:
    input_paths: tuple[Path, ...]
    evaluations_output_path: Path
    checkpoint_losses_output_path: Path
    input_row_count: int
    clean_row_count: int
    excluded_row_count: int
    superseded_row_count: int
    evaluation_count: int
    checkpoint_count: int
    elapsed_seconds: float


def preprocess_scaling_law(
    paths: DataDecidePaths,
    *,
    verbose: bool = False,
) -> ScalingLawPreprocessResult:
    started_at = perf_counter()
    contract = load_scaling_law_contract()
    olmes_contract = load_olmes_contract()
    input_paths = paths.scaling_law_raw_paths()
    evaluations_output_path = paths.scaling_law_evaluations_path()
    checkpoint_losses_output_path = paths.scaling_law_checkpoint_losses_path()

    missing = tuple(path for path in input_paths if not path.is_file())
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "missing required scaling-law raw CSV files: " + missing_text
        )
    for path in input_paths:
        _validate_csv_header(path)

    temporary_paths = (
        evaluations_output_path.with_name(f".{evaluations_output_path.name}.tmp"),
        checkpoint_losses_output_path.with_name(
            f".{checkpoint_losses_output_path.name}.tmp"
        ),
    )
    exports_replaced = False
    connection = duckdb.connect()
    try:
        _register_metrics_parser(connection, olmes_contract=olmes_contract)
        _load_raw_sources(
            connection,
            input_paths=input_paths,
            source_precedence=contract.source_precedence,
        )
        input_row_count = _count(connection, "_scaling_raw")
        _validate_and_normalize_rows(
            connection,
            contract=contract,
            olmes_contract=olmes_contract,
        )
        clean_row_count = _count(connection, "_scaling_clean")
        excluded_row_count = input_row_count - clean_row_count

        _reject_same_priority_evaluation_duplicates(connection)
        _build_selected_checkpoints(connection, contract=contract)
        checkpoint_count = _count(connection, "_scaling_checkpoints")
        create_model_enrichment_table(
            connection,
            checkpoint_select_sql="SELECT params, step FROM _scaling_checkpoints",
        )
        _build_selected_evaluations(
            connection,
            contract=contract,
            olmes_contract=olmes_contract,
        )
        evaluation_count = _count(connection, "_scaling_evaluations")
        superseded_row_count = clean_row_count - evaluation_count

        evaluation_export = prepare_parquet_export(
            connection,
            select_sql=_table_export_sql(
                source_table="_scaling_evaluations",
                table=contract.tables.evaluations,
            ),
            output_path=evaluations_output_path,
        )
        checkpoint_export = prepare_parquet_export(
            connection,
            select_sql=_table_export_sql(
                source_table="_scaling_checkpoints",
                table=contract.tables.checkpoint_losses,
            ),
            output_path=checkpoint_losses_output_path,
        )
        _validate_pending_export(
            connection,
            export=evaluation_export,
            table=contract.tables.evaluations,
            expected_count=evaluation_count,
        )
        _validate_pending_export(
            connection,
            export=checkpoint_export,
            table=contract.tables.checkpoint_losses,
            expected_count=checkpoint_count,
        )
        replace_parquet_exports((evaluation_export, checkpoint_export))
        exports_replaced = True
    finally:
        connection.close()
        if not exports_replaced:
            for temporary_path in temporary_paths:
                remove_owned_file(temporary_path)

    result = ScalingLawPreprocessResult(
        input_paths=input_paths,
        evaluations_output_path=evaluations_output_path,
        checkpoint_losses_output_path=checkpoint_losses_output_path,
        input_row_count=input_row_count,
        clean_row_count=clean_row_count,
        excluded_row_count=excluded_row_count,
        superseded_row_count=superseded_row_count,
        evaluation_count=evaluation_count,
        checkpoint_count=checkpoint_count,
        elapsed_seconds=perf_counter() - started_at,
    )
    if verbose:
        for input_path in result.input_paths:
            print(f"scaling-law input: {input_path}")
        print(f"scaling-law evaluations output: {result.evaluations_output_path}")
        print(
            "scaling-law checkpoint losses output: "
            f"{result.checkpoint_losses_output_path}"
        )
        print(f"scaling-law input rows: {result.input_row_count}")
        print(f"scaling-law clean rows: {result.clean_row_count}")
        print(f"scaling-law excluded policy rows: {result.excluded_row_count}")
        print(f"scaling-law superseded rows: {result.superseded_row_count}")
        print(f"scaling-law evaluations: {result.evaluation_count}")
        print(f"scaling-law checkpoints: {result.checkpoint_count}")
        print(f"scaling-law elapsed seconds: {result.elapsed_seconds:.3f}")
    return result


def _validate_csv_header(path: Path) -> None:
    with path.open("r", encoding="utf-8", newline="") as file:
        header = next(csv.reader(file), None)
    if header != list(RAW_COLUMNS):
        raise ValueError(
            f"invalid scaling-law CSV header in {path}: "
            f"expected {list(RAW_COLUMNS)!r}, got {header!r}"
        )


def _register_metrics_parser(
    connection: duckdb.DuckDBPyConnection,
    *,
    olmes_contract: OLMESContract,
) -> None:
    output_fields = tuple(
        field for field in olmes_contract.metrics.aggregate if field != "primary_metric"
    )
    allowed_fields = frozenset(output_fields) | _IGNORED_METRIC_FIELDS

    def parse_metrics(
        raw: str | None,
        source_file: str,
        source_row: int,
    ) -> dict[str, float | None]:
        context = f"{source_file} row {source_row}"
        if raw is None or not raw.strip():
            raise ValueError(
                f"invalid scaling-law metrics at {context}: expected a Python dict"
            )
        try:
            expression = ast.parse(raw, mode="eval")
        except (SyntaxError, ValueError) as error:
            raise ValueError(
                f"invalid scaling-law metrics syntax at {context}: {error}"
            ) from error
        if not isinstance(expression.body, ast.Dict):
            raise ValueError(
                f"invalid scaling-law metrics at {context}: expected a Python dict"
            )

        result = dict.fromkeys(output_fields)
        seen: set[str] = set()
        for key_node, value_node in zip(
            expression.body.keys, expression.body.values, strict=True
        ):
            try:
                key = ast.literal_eval(key_node)
                value = ast.literal_eval(value_node)
            except (ValueError, TypeError, SyntaxError) as error:
                raise ValueError(
                    f"invalid scaling-law metrics value at {context}"
                ) from error
            if not isinstance(key, str):
                raise ValueError(
                    f"invalid scaling-law metrics key at {context}: {key!r}"
                )
            if key in seen:
                raise ValueError(
                    f"duplicate scaling-law metrics key at {context}: {key!r}"
                )
            seen.add(key)
            if key not in allowed_fields:
                raise ValueError(
                    f"unknown scaling-law metrics key at {context}: {key!r}"
                )
            numeric_value: float | None = None
            if (
                value is not None
                and not isinstance(value, bool)
                and isinstance(value, (int, float))
            ):
                try:
                    numeric_value = float(value)
                except OverflowError:
                    pass
            if value is not None and (
                numeric_value is None or not math.isfinite(numeric_value)
            ):
                raise ValueError(
                    f"invalid scaling-law metric {key!r} at {context}: "
                    f"{value!r}; expected a finite number or None"
                )
            if key in result:
                result[key] = numeric_value
        return result

    return_type = duckdb.struct_type(
        {field: duckdb.sqltype("DOUBLE") for field in output_fields}
    )
    connection.create_function(
        "_datadec_parse_scaling_metrics",
        parse_metrics,
        parameters=[
            duckdb.sqltype("VARCHAR"),
            duckdb.sqltype("VARCHAR"),
            duckdb.sqltype("BIGINT"),
        ],
        return_type=return_type,
        null_handling=func.FunctionNullHandling.SPECIAL,
    )


def _load_raw_sources(
    connection: duckdb.DuckDBPyConnection,
    *,
    input_paths: tuple[Path, ...],
    source_precedence: tuple[str, ...],
) -> None:
    column_schema = ", ".join(
        f"{sql_literal(column)}: 'VARCHAR'" for column in RAW_COLUMNS
    )
    selects: list[str] = []
    for priority, (path, filename) in enumerate(
        zip(input_paths, source_precedence, strict=True)
    ):
        selects.append(
            f"""
            SELECT
                {priority}::INTEGER AS source_priority,
                {sql_literal(filename)}::VARCHAR AS source_file,
                row_number() OVER ()::BIGINT AS source_row,
                *
            FROM read_csv(
                {sql_literal(path)},
                columns = {{{column_schema}}},
                header = true,
                auto_detect = false,
                all_varchar = true,
                strict_mode = true,
                null_padding = false,
                ignore_errors = false
            )
            """
        )
    connection.execute(
        "CREATE TEMP TABLE _scaling_raw AS " + " UNION ALL ".join(selects)
    )


def _validate_and_normalize_rows(
    connection: duckdb.DuckDBPyConnection,
    *,
    contract: ScalingLawContract,
    olmes_contract: OLMESContract,
) -> None:
    seed_expression = _optional_int64_sql("seed")
    invalid_seed = connection.execute(
        f"""
        SELECT source_file, source_row, seed
        FROM _scaling_raw
        WHERE seed IS NOT NULL
          AND trim(seed) <> ''
          AND {seed_expression} IS NULL
        ORDER BY source_priority, source_row
        LIMIT 1
        """
    ).fetchone()
    if invalid_seed is not None:
        source_file, source_row, value = invalid_seed
        raise ValueError(
            f"invalid scaling-law seed at {source_file} row {source_row}: "
            f"{value!r}; expected a finite integral int64 value"
        )

    allowed_seed_values = (
        *contract.seed_map,
        *contract.seed_policy.excluded_legacy_values,
    )
    allowed_seed_sql = ", ".join(str(value) for value in allowed_seed_values)
    unknown_seed = connection.execute(
        f"""
        SELECT source_file, source_row, seed
        FROM _scaling_raw
        WHERE {seed_expression} IS NOT NULL
          AND {seed_expression} NOT IN ({allowed_seed_sql})
        ORDER BY source_priority, source_row
        LIMIT 1
        """
    ).fetchone()
    if unknown_seed is not None:
        source_file, source_row, value = unknown_seed
        raise ValueError(
            f"unknown scaling-law seed at {source_file} row {source_row}: {value!r}"
        )

    excluded_seed_sql = ", ".join(
        str(value) for value in contract.seed_policy.excluded_legacy_values
    )
    excluded_group_sql = ", ".join(
        sql_literal(value) for value in contract.excluded_source_groups
    )
    connection.execute(
        f"""
        CREATE TEMP VIEW _scaling_clean_raw AS
        SELECT *
        FROM _scaling_raw
        WHERE {seed_expression} IS NOT NULL
          AND {seed_expression} NOT IN ({excluded_seed_sql})
          AND {quote_identifier("group")} NOT IN ({excluded_group_sql})
        """
    )

    source_group_map = {
        **contract.source_group_map,
        **{
            alias: contract.source_group_map[canonical]
            for alias, canonical in contract.source_group_aliases.items()
        },
    }
    _validate_known_string(
        connection,
        column="group",
        allowed=tuple(source_group_map),
    )
    _validate_known_string(connection, column="model", allowed=contract.models)
    _validate_nonblank_string(connection, column="task")
    _validate_nonblank_string(connection, column="chinchilla")
    _validate_required_int64(connection, column="step")
    _validate_optional_int64(connection, column="tokens")
    _validate_optional_float64(connection, column="compute")
    for raw_column in tuple(RAW_CHECKPOINT_FIELDS.values())[2:]:
        _validate_optional_float64(connection, column=raw_column)

    recipe_case = _mapping_case_sql(quote_identifier("group"), source_group_map)
    data_case = _mapping_case_sql(
        quote_identifier("group"),
        {
            source_group: olmes_contract.recipe_map[recipe]
            for source_group, recipe in source_group_map.items()
        },
    )
    seed_case = _mapping_case_sql(
        _optional_int64_sql("seed"),
        contract.seed_map,
        literal_keys=True,
    )
    normalized_fields = {
        "tokens": _optional_int64_sql("tokens"),
        **{
            output_field: _optional_float64_sql(raw_field)
            for output_field, raw_field in RAW_CHECKPOINT_FIELDS.items()
            if output_field != "tokens"
        },
    }
    normalized_sql = ",\n".join(
        f"{expression} AS {quote_identifier(field)}"
        for field, expression in normalized_fields.items()
    )
    connection.execute(
        f"""
        CREATE TEMP TABLE _scaling_clean AS
        SELECT
            source_priority,
            source_file,
            source_row,
            {recipe_case}::VARCHAR AS recipe,
            {data_case}::VARCHAR AS data,
            model::VARCHAR AS params,
            {_optional_int64_sql("seed")}::BIGINT AS seed_value,
            {seed_case}::VARCHAR AS seed,
            {_optional_int64_sql("step")}::BIGINT AS step,
            task::VARCHAR AS task,
            chinchilla::VARCHAR AS chinchilla,
            {normalized_sql},
            _datadec_parse_scaling_metrics(
                metrics, source_file, source_row
            ) AS parsed_metrics
        FROM _scaling_clean_raw
        """
    )


def _validate_known_string(
    connection: duckdb.DuckDBPyConnection,
    *,
    column: str,
    allowed: tuple[str, ...],
) -> None:
    quoted = quote_identifier(column)
    allowed_sql = ", ".join(sql_literal(value) for value in allowed)
    invalid = connection.execute(
        f"""
        SELECT source_file, source_row, {quoted}
        FROM _scaling_clean_raw
        WHERE {quoted} IS NULL OR {quoted} NOT IN ({allowed_sql})
        ORDER BY source_priority, source_row
        LIMIT 1
        """
    ).fetchone()
    if invalid is not None:
        source_file, source_row, value = invalid
        raise ValueError(
            f"unknown scaling-law {column} at {source_file} row {source_row}: {value!r}"
        )


def _validate_nonblank_string(
    connection: duckdb.DuckDBPyConnection,
    *,
    column: str,
) -> None:
    quoted = quote_identifier(column)
    invalid = connection.execute(
        f"""
        SELECT source_file, source_row, {quoted}
        FROM _scaling_clean_raw
        WHERE {quoted} IS NULL
           OR trim({quoted}) = ''
           OR {quoted} <> trim({quoted})
        ORDER BY source_priority, source_row
        LIMIT 1
        """
    ).fetchone()
    if invalid is not None:
        source_file, source_row, value = invalid
        raise ValueError(
            f"invalid scaling-law {column} at {source_file} row {source_row}: "
            f"{value!r}; expected a nonblank string without surrounding whitespace"
        )


def _validate_required_int64(
    connection: duckdb.DuckDBPyConnection,
    *,
    column: str,
) -> None:
    quoted = quote_identifier(column)
    converted = _optional_int64_sql(column)
    invalid = connection.execute(
        f"""
        SELECT source_file, source_row, {quoted}
        FROM _scaling_clean_raw
        WHERE {converted} IS NULL
        ORDER BY source_priority, source_row
        LIMIT 1
        """
    ).fetchone()
    if invalid is not None:
        source_file, source_row, value = invalid
        raise ValueError(
            f"invalid scaling-law {column} at {source_file} row {source_row}: "
            f"{value!r}; expected a finite integral int64 value"
        )


def _validate_optional_int64(
    connection: duckdb.DuckDBPyConnection,
    *,
    column: str,
) -> None:
    quoted = quote_identifier(column)
    converted = _optional_int64_sql(column)
    invalid = connection.execute(
        f"""
        SELECT source_file, source_row, {quoted}
        FROM _scaling_clean_raw
        WHERE {quoted} IS NOT NULL
          AND trim({quoted}) <> ''
          AND {converted} IS NULL
        ORDER BY source_priority, source_row
        LIMIT 1
        """
    ).fetchone()
    if invalid is not None:
        source_file, source_row, value = invalid
        raise ValueError(
            f"invalid scaling-law {column} at {source_file} row {source_row}: "
            f"{value!r}; expected a finite integral int64 value or blank"
        )


def _validate_optional_float64(
    connection: duckdb.DuckDBPyConnection,
    *,
    column: str,
) -> None:
    quoted = quote_identifier(column)
    converted = _optional_float64_sql(column)
    invalid = connection.execute(
        f"""
        SELECT source_file, source_row, {quoted}
        FROM _scaling_clean_raw
        WHERE {quoted} IS NOT NULL
          AND trim({quoted}) <> ''
          AND {converted} IS NULL
        ORDER BY source_priority, source_row
        LIMIT 1
        """
    ).fetchone()
    if invalid is not None:
        source_file, source_row, value = invalid
        raise ValueError(
            f"invalid scaling-law {column} at {source_file} row {source_row}: "
            f"{value!r}; expected a finite float64 value or blank"
        )


def _reject_same_priority_evaluation_duplicates(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    duplicate = connection.execute(
        """
        SELECT
            source_file,
            max(source_row) AS duplicate_row,
            recipe,
            params,
            seed_value,
            step,
            task
        FROM _scaling_clean
        GROUP BY
            source_priority, source_file, recipe, params, seed_value, step, task
        HAVING count(*) > 1
        ORDER BY source_priority, duplicate_row
        LIMIT 1
        """
    ).fetchone()
    if duplicate is not None:
        source_file, source_row, recipe, params, seed_value, step, task = duplicate
        raise ValueError(
            f"duplicate same-priority scaling-law evaluation at {source_file} "
            f"row {source_row}: recipe={recipe!r}, params={params!r}, "
            f"seed_value={seed_value}, step={step}, task={task!r}"
        )


def _build_selected_checkpoints(
    connection: duckdb.DuckDBPyConnection,
    *,
    contract: ScalingLawContract,
) -> None:
    schedules = {schedule.params: schedule for schedule in create_model_schedules()}
    model_schedule_rows = ", ".join(
        "("
        + ", ".join(
            (
                sql_literal(model),
                str(schedules[model].tokens_per_step),
                str(schedules[model].exact_parameter_count),
                str(schedules[model].flops_per_token_per_parameter),
            )
        )
        + ")"
        for model in contract.models
    )
    connection.execute(
        f"""
        CREATE TEMP TABLE _scaling_model_schedule AS
        SELECT *
        FROM (VALUES {model_schedule_rows}) AS schedule(
            params,
            tokens_per_step,
            exact_parameter_count,
            flops_per_token_per_parameter
        )
        """
    )
    schedule_mismatch = connection.execute(
        """
        SELECT
            source_file,
            source_row,
            c.params,
            step,
            tokens,
            step * tokens_per_step AS expected_tokens
        FROM _scaling_clean AS c
        INNER JOIN _scaling_model_schedule AS schedule USING (params)
        WHERE tokens IS NOT NULL AND tokens <> step * tokens_per_step
        ORDER BY source_priority, source_row
        LIMIT 1
        """
    ).fetchone()
    if schedule_mismatch is not None:
        (
            source_file,
            source_row,
            params,
            step,
            tokens,
            expected_tokens,
        ) = schedule_mismatch
        raise ValueError(
            f"scaling-law checkpoint schedule mismatch at {source_file} row "
            f"{source_row}: params={params!r}, step={step}, tokens={tokens!r}, "
            f"expected_tokens={expected_tokens}"
        )

    checkpoint_fields = tuple(RAW_CHECKPOINT_FIELDS)[2:]
    aggregates = [
        "count(DISTINCT chinchilla) AS chinchilla_distinct_count",
        "min(chinchilla) AS chinchilla",
    ]
    for field in checkpoint_fields:
        quoted = quote_identifier(field)
        aggregates.extend(
            (
                f"count(DISTINCT {quoted}) AS {quote_identifier(field + '_distinct_count')}",
                f"min({quoted}) AS {quoted}",
            )
        )
    aggregate_sql = ",\n".join(aggregates)
    has_data_sql = " OR ".join(
        f"{quote_identifier(field)} IS NOT NULL" for field in checkpoint_fields
    )
    connection.execute(
        f"""
        CREATE TEMP TABLE _scaling_checkpoint_sources AS
        SELECT
            source_priority,
            source_file,
            recipe,
            data,
            params,
            seed_value,
            seed,
            step,
            bool_or({has_data_sql}) AS has_checkpoint_data,
            {aggregate_sql}
        FROM _scaling_clean
        GROUP BY
            source_priority,
            source_file,
            recipe,
            data,
            params,
            seed_value,
            seed,
            step
        """
    )
    connection.execute(
        """
        CREATE TEMP VIEW _scaling_ranked_checkpoint_sources AS
        SELECT
            *,
            row_number() OVER (
                PARTITION BY recipe, params, seed_value, step
                ORDER BY has_checkpoint_data DESC, source_priority DESC
            ) AS source_rank
        FROM _scaling_checkpoint_sources
        """
    )

    conflict_checks = [
        ("chinchilla", "chinchilla_distinct_count"),
        *((field, field + "_distinct_count") for field in checkpoint_fields),
    ]
    conflict_condition = " OR ".join(
        f"{quote_identifier(count_field)} > 1" for _, count_field in conflict_checks
    )
    selected_conflict = connection.execute(
        f"""
        SELECT *
        FROM _scaling_ranked_checkpoint_sources
        WHERE source_rank = 1 AND ({conflict_condition})
        ORDER BY recipe, params, seed_value, step
        LIMIT 1
        """
    ).fetchone()
    if selected_conflict is not None:
        columns = [description[0] for description in connection.description]
        record = dict(zip(columns, selected_conflict, strict=True))
        conflicts = [
            field for field, count_field in conflict_checks if record[count_field] > 1
        ]
        raise ValueError(
            "conflicting checkpoint values in selected scaling-law source "
            f"{record['source_file']!r}: recipe={record['recipe']!r}, "
            f"params={record['params']!r}, seed_value={record['seed_value']}, "
            f"step={record['step']}, fields={conflicts!r}"
        )

    selected_fields = ",\n".join(quote_identifier(field) for field in checkpoint_fields)
    connection.execute(
        f"""
        CREATE TEMP TABLE _scaling_checkpoints AS
        SELECT
            source_file,
            recipe,
            data,
            params,
            seed_value,
            seed,
            step,
            chinchilla,
            step * tokens_per_step::BIGINT AS tokens,
            (
                step::DOUBLE * tokens_per_step * exact_parameter_count::DOUBLE
                * flops_per_token_per_parameter
            )::DOUBLE AS compute,
            {selected_fields}
        FROM _scaling_ranked_checkpoint_sources AS checkpoints
        INNER JOIN _scaling_model_schedule AS schedule USING (params)
        WHERE source_rank = 1
        """
    )


def _build_selected_evaluations(
    connection: duckdb.DuckDBPyConnection,
    *,
    contract: ScalingLawContract,
    olmes_contract: OLMESContract,
) -> None:
    connection.execute(
        """
        CREATE TEMP VIEW _scaling_ranked_evaluations AS
        SELECT
            *,
            row_number() OVER (
                PARTITION BY recipe, params, seed_value, step, task
                ORDER BY source_priority DESC
            ) AS source_rank
        FROM _scaling_clean
        """
    )
    source_metrics = tuple(
        field for field in olmes_contract.metrics.aggregate if field != "primary_metric"
    )
    metric_sql = ",\n".join(
        f"e.parsed_metrics.{quote_identifier(field)}::DOUBLE "
        f"AS {quote_identifier(field)}"
        for field in source_metrics
    )
    policy = olmes_contract.metrics.aggregate_primary_metric.model_dump()
    primary_cases = [
        "WHEN starts_with(e.task, 'mmlu_') "
        f"THEN e.parsed_metrics.{quote_identifier(policy['mmlu'])}"
    ]
    primary_cases.extend(
        f"WHEN e.task = {sql_literal(task)} "
        f"THEN e.parsed_metrics.{quote_identifier(metric)}"
        for task, metric in policy.items()
        if task != "mmlu"
    )
    primary_metric_sql = "CASE " + " ".join(primary_cases) + " ELSE NULL END"

    output_columns = tuple(
        column.name for column in contract.tables.evaluations.columns
    )
    expected_columns = (
        "source_file",
        "recipe",
        "data",
        "params",
        "seed_value",
        "seed",
        "step",
        "task",
        "chinchilla",
        "tokens",
        "compute",
        *CHECKPOINT_ENRICHMENT_COLUMNS[2:],
        *source_metrics,
        "primary_metric",
    )
    if output_columns != expected_columns:
        raise AssertionError(
            "scaling-law evaluation output columns drift from configured metrics"
        )
    connection.execute(
        f"""
        CREATE TEMP TABLE _scaling_evaluations AS
        SELECT
            e.source_file,
            e.recipe,
            e.data,
            e.params,
            e.seed_value,
            e.seed,
            e.step,
            e.task,
            e.chinchilla,
            c.tokens,
            c.compute,
            {metric_sql},
            {primary_metric_sql}::DOUBLE AS primary_metric
        FROM _scaling_ranked_evaluations AS e
        INNER JOIN _scaling_checkpoints AS c
            USING (recipe, params, seed_value, step)
        WHERE e.source_rank = 1
        """
    )


def _table_export_sql(
    *,
    source_table: str,
    table: ScalingLawTableContract,
) -> str:
    enrichment_columns = set(CHECKPOINT_ENRICHMENT_COLUMNS[2:])
    expressions = ", ".join(
        f"CAST({('enrichment' if column.name in enrichment_columns else 'source')}."
        f"{quote_identifier(column.name)} AS {duckdb_type(column.logical_type)}) "
        f"AS {quote_identifier(column.name)}"
        for column in table.columns
    )
    sort_key = ", ".join(
        f"source.{quote_identifier(column)}" for column in table.sort_key
    )
    return f"""
        SELECT {expressions}
        FROM {quote_identifier(source_table)} AS source
        INNER JOIN _model_enrichment AS enrichment USING (params, step)
        ORDER BY {sort_key}
    """


def _validate_pending_export(
    connection: duckdb.DuckDBPyConnection,
    *,
    export: PendingParquetExport,
    table: ScalingLawTableContract,
    expected_count: int,
) -> None:
    if export.row_count != expected_count:
        raise AssertionError(
            f"scaling-law export count mismatch for {export.output_path}: "
            f"expected {expected_count}, got {export.row_count}"
        )
    schema = connection.execute(
        "DESCRIBE SELECT * FROM read_parquet(?)", [str(export.temporary_path)]
    ).fetchall()
    actual = tuple((row[0], row[1]) for row in schema)
    expected = tuple(
        (column.name, duckdb_type(column.logical_type)) for column in table.columns
    )
    if actual != expected:
        raise AssertionError(
            f"scaling-law export schema mismatch for {export.output_path}: "
            f"expected {expected!r}, got {actual!r}"
        )

    required_columns = tuple(
        column.name for column in table.columns if not column.nullable
    )
    missing_required = " OR ".join(
        f"{quote_identifier(column)} IS NULL" for column in required_columns
    )
    null_count = connection.execute(
        f"SELECT count(*) FROM read_parquet(?) WHERE {missing_required}",
        [str(export.temporary_path)],
    ).fetchone()
    assert null_count is not None
    if null_count[0]:
        raise AssertionError(
            f"scaling-law export has {null_count[0]} rows with null required values: "
            f"{export.output_path}"
        )

    key_sql = ", ".join(quote_identifier(column) for column in table.primary_key)
    duplicate = connection.execute(
        f"""
        SELECT {key_sql}
        FROM read_parquet(?)
        GROUP BY {key_sql}
        HAVING count(*) > 1
        LIMIT 1
        """,
        [str(export.temporary_path)],
    ).fetchone()
    if duplicate is not None:
        raise AssertionError(
            f"scaling-law export has duplicate primary key {duplicate!r}: "
            f"{export.output_path}"
        )

    sort_sql = ", ".join(quote_identifier(column) for column in table.sort_key)
    out_of_order = connection.execute(
        f"""
        SELECT count(*)
        FROM (
            SELECT
                row_number() OVER () AS physical_ordinal,
                row_number() OVER (ORDER BY {sort_sql}) AS sorted_ordinal
            FROM read_parquet(?)
        )
        WHERE physical_ordinal <> sorted_ordinal
        """,
        [str(export.temporary_path)],
    ).fetchone()
    assert out_of_order is not None
    if out_of_order[0]:
        raise AssertionError(
            f"scaling-law export is not sorted by {table.sort_key!r}: "
            f"{export.output_path}"
        )


def _mapping_case_sql(
    expression: str,
    mapping: dict[Any, str],
    *,
    literal_keys: bool = False,
) -> str:
    branches = []
    for source, target in mapping.items():
        source_sql = str(source) if literal_keys else sql_literal(source)
        branches.append(f"WHEN {source_sql} THEN {sql_literal(target)}")
    return f"CASE {expression} {' '.join(branches)} END"


def _optional_int64_sql(column: str) -> str:
    quoted = quote_identifier(column)
    text = f"trim({quoted})"
    decimal = f"try_cast({text} AS DECIMAL(38, 18))"
    return (
        "CASE "
        f"WHEN {quoted} IS NULL OR {text} = '' THEN NULL "
        f"WHEN {decimal} IS NULL OR {decimal} <> trunc({decimal}) THEN NULL "
        f"ELSE try_cast({text} AS BIGINT) END"
    )


def _optional_float64_sql(column: str) -> str:
    quoted = quote_identifier(column)
    text = f"trim({quoted})"
    converted = f"try_cast({text} AS DOUBLE)"
    return (
        "CASE "
        f"WHEN {quoted} IS NULL OR {text} = '' THEN NULL "
        f"WHEN {converted} IS NULL OR NOT isfinite({converted}) THEN NULL "
        f"ELSE {converted} END"
    )


def _count(connection: duckdb.DuckDBPyConnection, table: str) -> int:
    row = connection.execute(f"SELECT count(*) FROM {table}").fetchone()
    assert row is not None
    return row[0]


__all__ = [
    "RAW_COLUMNS",
    "ScalingLawPreprocessResult",
    "preprocess_scaling_law",
]
