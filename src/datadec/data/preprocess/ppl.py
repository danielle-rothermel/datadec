from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from datadec.config import (
    CHECKPOINT_ENRICHMENT_TYPES,
    PPL_IDENTITY_COLUMNS,
    PPL_METRIC_COLUMNS,
    PPL_OUTPUT_COLUMNS,
)
from datadec.data.model_utils import checkpoint_enrichment
from datadec.data.paths import DataDecidePaths
from datadec.data.preprocess.duckdb import (
    duckdb_type,
    prepare_parquet_export,
    quote_identifier,
    replace_parquet_exports,
    sql_literal,
)
from datadec.data.preprocess.model_enrichment import (
    create_model_enrichment_table,
    enrichment_select_expressions,
)

if TYPE_CHECKING:
    from datadec.data.ingest.enums import DataRecipeName, ModelSizeName, Seed
    from datadec.data.ingest.metrics import PerplexityMetrics

PplRunKey: TypeAlias = tuple["ModelSizeName", "DataRecipeName", "Seed"]
PplRowsByKey: TypeAlias = dict[
    PplRunKey,
    dict[int, "PerplexityMetrics"],
]

_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


@dataclass(frozen=True, slots=True)
class PplPreprocessResult:
    input_path: Path
    output_path: Path
    checkpoint_count: int
    training_run_count: int


def group_perplexity_rows(ppl_df: pd.DataFrame) -> PplRowsByKey:
    from datadec.data.ingest.enums import DataRecipeName, ModelSizeName, Seed
    from datadec.data.ingest.metrics import PerplexityMetrics

    _assert_metric_field_parity(PerplexityMetrics)
    grouped: PplRowsByKey = defaultdict(dict)
    for row_index, record in enumerate(ppl_df.to_dict(orient="records")):
        run_key = (
            ModelSizeName(record["params"]),
            DataRecipeName(record["data"]),
            Seed(record["seed"]),
        )
        step = _normalize_step(record["step"], row_index=row_index)
        metrics = PerplexityMetrics.model_validate(record)
        if step in grouped[run_key]:
            params, data, seed = run_key
            raise ValueError(
                f"duplicate PPL checkpoint at row {row_index}: "
                f"params={params.value!r}, data={data.value!r}, "
                f"seed={seed.value!r}, step={step}"
            )
        grouped[run_key][step] = metrics
    return dict(grouped)


def flatten_perplexity_rows(grouped: PplRowsByKey) -> pd.DataFrame:
    from datadec.data.ingest.metrics import PerplexityMetrics

    _assert_metric_field_parity(PerplexityMetrics)
    rows: list[dict[str, object]] = []
    for run_key in sorted(grouped, key=lambda key: tuple(item.value for item in key)):
        params, data, seed = run_key
        for step in sorted(grouped[run_key]):
            metrics = grouped[run_key][step]
            rows.append(
                {
                    "params": params.value,
                    "data": data.value,
                    "seed": seed.value,
                    "step": step,
                    **checkpoint_enrichment(params.value, step),
                    **{field: getattr(metrics, field) for field in PPL_METRIC_COLUMNS},
                }
            )
    return _typed_output_dataframe(rows)


def preprocess_ppl(
    paths: DataDecidePaths,
    *,
    verbose: bool = False,
) -> PplPreprocessResult:
    input_path = paths.get_path("ppl_raw")
    output_path = paths.get_path("ppl_processed")

    from datadec.data.ingest.enums import DataRecipeName, ModelSizeName, Seed
    from datadec.data.ingest.metrics import RAW_PPL_TO_FIELD, PerplexityMetrics

    _assert_metric_field_parity(PerplexityMetrics)
    if tuple(RAW_PPL_TO_FIELD.values()) != PPL_METRIC_COLUMNS:
        raise AssertionError(
            "persisted PPL metric columns drift from raw metric mapping: "
            f"expected={PPL_METRIC_COLUMNS!r}, "
            f"actual={tuple(RAW_PPL_TO_FIELD.values())!r}"
        )

    parquet_file = pq.ParquetFile(input_path)
    connection = duckdb.connect()
    try:
        if parquet_file.metadata.num_rows == 0:
            select_sql = _empty_output_select_sql()
            training_run_count = 0
        else:
            raw_columns = set(parquet_file.schema_arrow.names)
            row_index_column = "_datadec_row_index"
            while row_index_column in raw_columns:
                row_index_column = f"_{row_index_column}"
            connection.execute(
                f"""
                CREATE TEMP VIEW _ppl_raw AS
                SELECT *, row_number() OVER () - 1
                    AS {quote_identifier(row_index_column)}
                FROM read_parquet({sql_literal(input_path)})
                """
            )
            _validate_raw_ppl(
                connection,
                raw_columns=raw_columns,
                row_index_column=row_index_column,
                model_sizes=tuple(item.value for item in ModelSizeName),
                data_recipes=tuple(item.value for item in DataRecipeName),
                seeds=tuple(item.value for item in Seed),
                raw_metric_columns=tuple(RAW_PPL_TO_FIELD),
            )
            create_model_enrichment_table(
                connection,
                checkpoint_select_sql="""
                    SELECT
                        CAST(params AS VARCHAR) AS params,
                        CAST(trim(CAST(step AS VARCHAR)) AS BIGINT) AS step
                    FROM _ppl_raw
                """,
            )
            select_sql = _output_select_sql(
                raw_columns=raw_columns,
                raw_metric_to_field=RAW_PPL_TO_FIELD,
            )
            training_run_count_row = connection.execute(
                """
                SELECT count(DISTINCT (params, data, seed))
                FROM _ppl_raw
                """
            ).fetchone()
            assert training_run_count_row is not None
            training_run_count = training_run_count_row[0]

        export = prepare_parquet_export(
            connection,
            select_sql=select_sql,
            output_path=output_path,
            key_value_metadata={"pandas": _pandas_parquet_metadata()},
        )
        replace_parquet_exports((export,))
    finally:
        connection.close()

    result = PplPreprocessResult(
        input_path=input_path,
        output_path=output_path,
        checkpoint_count=export.row_count,
        training_run_count=training_run_count,
    )
    if verbose:
        print(f"ppl input: {result.input_path}")
        print(f"ppl output: {result.output_path}")
        print(f"ppl checkpoints: {result.checkpoint_count}")
        print(f"ppl training runs: {result.training_run_count}")
    return result


def _pandas_parquet_metadata() -> str:
    columns = []
    for name in PPL_OUTPUT_COLUMNS:
        if name in PPL_IDENTITY_COLUMNS[:3]:
            pandas_type = "object"
            numpy_type = "string"
        elif name == "step" or dict(CHECKPOINT_ENRICHMENT_TYPES).get(name) == "int64":
            pandas_type = numpy_type = "int64"
        elif dict(CHECKPOINT_ENRICHMENT_TYPES).get(name) == "bool":
            pandas_type = numpy_type = "bool"
        elif dict(CHECKPOINT_ENRICHMENT_TYPES).get(name) == "string":
            pandas_type = "object"
            numpy_type = "string"
        else:
            pandas_type = numpy_type = "float64"
        columns.append(
            {
                "name": name,
                "field_name": name,
                "pandas_type": pandas_type,
                "numpy_type": numpy_type,
                "metadata": None,
            }
        )
    return json.dumps(
        {
            "index_columns": [],
            "column_indexes": [],
            "columns": columns,
            "attributes": {},
            "creator": {"library": "duckdb"},
            "pandas_version": pd.__version__,
        },
        separators=(",", ":"),
    )


def _validate_raw_ppl(
    connection: duckdb.DuckDBPyConnection,
    *,
    raw_columns: set[str],
    row_index_column: str,
    model_sizes: tuple[str, ...],
    data_recipes: tuple[str, ...],
    seeds: tuple[str, ...],
    raw_metric_columns: tuple[str, ...],
) -> None:
    row_index = quote_identifier(row_index_column)
    missing = [column for column in PPL_IDENTITY_COLUMNS if column not in raw_columns]
    if missing:
        raise ValueError(f"missing required PPL input columns: {missing!r}")

    row_count = connection.execute("SELECT count(*) FROM _ppl_raw").fetchone()
    assert row_count is not None
    if row_count[0] and not raw_columns.intersection(raw_metric_columns):
        raise ValueError(
            "missing PPL metric columns at row 0: expected at least one raw metric"
        )

    invalid_values: list[tuple[int, int, str, object]] = []
    for field_order, (column, allowed) in enumerate(
        (
            ("params", model_sizes),
            ("data", data_recipes),
            ("seed", seeds),
        )
    ):
        allowed_sql = ", ".join(sql_literal(value) for value in allowed)
        quoted = quote_identifier(column)
        invalid = connection.execute(
            f"""
            SELECT {row_index}, {quoted}
            FROM _ppl_raw
            WHERE {quoted} IS NULL
               OR CAST({quoted} AS VARCHAR) NOT IN ({allowed_sql})
            ORDER BY {row_index}
            LIMIT 1
            """
        ).fetchone()
        if invalid is not None:
            invalid_values.append((invalid[0], field_order, column, invalid[1]))

    step_text = "trim(CAST(step AS VARCHAR))"
    step_decimal = f"try_cast({step_text} AS DECIMAL(38, 18))"
    invalid_step = connection.execute(
        f"""
        SELECT {row_index}, step
        FROM _ppl_raw
        WHERE step IS NULL
           OR typeof(step) = 'BOOLEAN'
           OR {step_decimal} IS NULL
           OR {step_decimal} <> trunc({step_decimal})
           OR try_cast({step_text} AS BIGINT) IS NULL
        ORDER BY {row_index}
        LIMIT 1
        """
    ).fetchone()
    if invalid_step is not None:
        invalid_values.append((invalid_step[0], 3, "step", invalid_step[1]))

    if invalid_values:
        row_index, _, field, value = min(invalid_values)
        if field == "step":
            raise ValueError(
                f"invalid PPL step at row {row_index}: {value!r}; "
                "expected a finite integral int64 value"
            )
        raise ValueError(f"unknown PPL {field} at row {row_index}: {value!r}")

    normalized_step = f"CAST({step_text} AS BIGINT)"
    duplicate = connection.execute(
        f"""
        SELECT {row_index}, params, data, seed, normalized_step
        FROM (
            SELECT
                {row_index},
                CAST(params AS VARCHAR) AS params,
                CAST(data AS VARCHAR) AS data,
                CAST(seed AS VARCHAR) AS seed,
                {normalized_step} AS normalized_step,
                row_number() OVER (
                    PARTITION BY params, data, seed, {normalized_step}
                    ORDER BY {row_index}
                ) AS duplicate_ordinal
            FROM _ppl_raw
        )
        WHERE duplicate_ordinal > 1
        ORDER BY {row_index}
        LIMIT 1
        """
    ).fetchone()
    if duplicate is not None:
        row_index, params, data, seed, step = duplicate
        raise ValueError(
            f"duplicate PPL checkpoint at row {row_index}: "
            f"params={params!r}, data={data!r}, seed={seed!r}, step={step}"
        )


def _empty_output_select_sql() -> str:
    expressions = [
        f"CAST(NULL AS {duckdb_type('string')}) AS {quote_identifier(column)}"
        for column in PPL_IDENTITY_COLUMNS[:3]
    ]
    expressions.append(f"CAST(NULL AS {duckdb_type('int64')}) AS step")
    expressions.extend(
        f"CAST(NULL AS {duckdb_type(logical_type)}) AS {quote_identifier(column)}"
        for column, logical_type in CHECKPOINT_ENRICHMENT_TYPES
    )
    expressions.extend(
        f"CAST(NULL AS {duckdb_type('float64')}) AS {quote_identifier(column)}"
        for column in PPL_METRIC_COLUMNS
    )
    return f"SELECT {', '.join(expressions)} WHERE false"


def _output_select_sql(
    *,
    raw_columns: set[str],
    raw_metric_to_field: dict[str, str],
) -> str:
    step_text = "trim(CAST(step AS VARCHAR))"
    identity_expressions = [
        f"CAST({quote_identifier(column)} AS {duckdb_type('string')}) "
        f"AS {quote_identifier(column)}"
        for column in PPL_IDENTITY_COLUMNS[:3]
    ]
    identity_expressions.append(f"CAST({step_text} AS {duckdb_type('int64')}) AS step")
    metric_expressions = []
    raw_by_field = {field: raw for raw, field in raw_metric_to_field.items()}
    for field in PPL_METRIC_COLUMNS:
        raw_column = raw_by_field[field]
        if raw_column not in raw_columns:
            value = f"CAST(NULL AS {duckdb_type('float64')})"
        else:
            raw_identifier = quote_identifier(raw_column)
            converted = f"try_cast({raw_identifier} AS {duckdb_type('float64')})"
            value = (
                "CASE "
                f"WHEN typeof({raw_identifier}) = 'BOOLEAN' "
                f"OR isnan({converted}) "
                "THEN NULL "
                f"ELSE {converted} END"
            )
        metric_expressions.append(f"{value} AS {quote_identifier(field)}")
    identity_columns = ", ".join(
        f"normalized.{quote_identifier(column)}" for column in PPL_IDENTITY_COLUMNS
    )
    metric_columns = ", ".join(
        f"normalized.{quote_identifier(column)}" for column in PPL_METRIC_COLUMNS
    )
    sort_key = ", ".join(
        f"normalized.{quote_identifier(column)}" for column in PPL_IDENTITY_COLUMNS
    )
    return f"""
        WITH normalized AS (
            SELECT
                {", ".join(identity_expressions)},
                {", ".join(metric_expressions)}
            FROM _ppl_raw
        )
        SELECT
            {identity_columns},
            {enrichment_select_expressions(table_alias="enrichment")},
            {metric_columns}
        FROM normalized
        INNER JOIN _model_enrichment AS enrichment USING (params, step)
        ORDER BY {sort_key}
    """


def _assert_metric_field_parity(metrics_type: type[PerplexityMetrics]) -> None:
    actual = tuple(metrics_type.model_fields)
    if actual != PPL_METRIC_COLUMNS:
        raise AssertionError(
            "persisted PPL metric columns drift from PerplexityMetrics: "
            f"expected={PPL_METRIC_COLUMNS!r}, actual={actual!r}"
        )


def _normalize_step(value: Any, *, row_index: int) -> int:
    invalid = isinstance(value, (bool, np.bool_)) or value is None or value is pd.NA
    if not invalid:
        try:
            if not isinstance(
                value,
                (str, int, float, Decimal, np.integer, np.floating),
            ):
                raise InvalidOperation
            text = str(value).strip()
            if not text:
                raise InvalidOperation
            decimal_value = Decimal(text)
            invalid = (
                not decimal_value.is_finite()
                or decimal_value != decimal_value.to_integral_value()
            )
        except (InvalidOperation, ValueError):
            invalid = True
    if invalid:
        raise ValueError(
            f"invalid PPL step at row {row_index}: {value!r}; "
            "expected a finite integral int64 value"
        )

    step = int(decimal_value)
    if not _INT64_MIN <= step <= _INT64_MAX:
        raise ValueError(
            f"invalid PPL step at row {row_index}: {value!r}; "
            "expected a finite integral int64 value"
        )
    return step


def _typed_output_dataframe(rows: list[dict[str, object]]) -> pd.DataFrame:
    values_by_column = {
        column: [row[column] for row in rows] for column in PPL_OUTPUT_COLUMNS
    }
    columns: dict[str, pd.Series[Any]] = {
        "params": pd.Series(values_by_column["params"], dtype="string"),
        "data": pd.Series(values_by_column["data"], dtype="string"),
        "seed": pd.Series(values_by_column["seed"], dtype="string"),
        "step": pd.Series(values_by_column["step"], dtype="int64"),
    }
    columns.update(
        {
            field: pd.Series(
                values_by_column[field],
                dtype="string" if logical_type == "string" else logical_type,
            )
            for field, logical_type in CHECKPOINT_ENRICHMENT_TYPES
        }
    )
    columns.update(
        {
            field: pd.Series(values_by_column[field], dtype="float64")
            for field in PPL_METRIC_COLUMNS
        }
    )
    return pd.DataFrame(columns, columns=PPL_OUTPUT_COLUMNS)
