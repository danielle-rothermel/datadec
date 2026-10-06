from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import duckdb
import numpy as np
import orjson
import pandas as pd
import pyarrow.parquet as pq
from dr_ds import coerce_float

from datadec.config import (
    CHECKPOINT_ENRICHMENT_COLUMNS,
    OLMESContract,
    load_olmes_contract,
)
from datadec.data import constants as consts
from datadec.data.model_utils import checkpoint_enrichment
from datadec.data.artifacts import DataArtifacts
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
from datadec.data.preprocess.ppl import _normalize_step

if TYPE_CHECKING:
    from datadec.data.ingest.enums import DataRecipeName, ModelSizeName, Seed

_EXCLUDED_METRIC_KEYS: frozenset[str] = frozenset(
    name for name in consts.DROP_METRICS if name.startswith("predicted_index_")
)

_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_RAW_IDENTITY_COLUMNS: tuple[str, ...] = (
    "params",
    "data",
    "seed",
    "step",
    "task",
    "chinchilla",
    "tokens",
    "compute",
)

OlmesRunKey: TypeAlias = tuple["ModelSizeName", "DataRecipeName", "Seed"]
OlmesRowsByKey: TypeAlias = dict[
    OlmesRunKey,
    dict[int, dict[str, "OlmesAggregateRow"]],
]


@dataclass(frozen=True, slots=True)
class OlmesAggregateRow:
    params: ModelSizeName
    data: DataRecipeName
    seed: Seed
    step: int
    task: str
    chinchilla: str
    tokens: int
    compute: float
    metrics: dict[str, float | None]


@dataclass(frozen=True, slots=True)
class OlmesPreprocessResult:
    input_path: Path
    output_path: Path
    row_count: int
    training_run_count: int


def _output_columns(contract: OLMESContract) -> tuple[str, ...]:
    return tuple(column.name for column in contract.tables.aggregate.columns)


def _identity_columns(contract: OLMESContract) -> tuple[str, ...]:
    metric_count = len(contract.metrics.aggregate)
    all_columns = _output_columns(contract)
    return all_columns[: len(all_columns) - metric_count]


def _source_metric_columns(contract: OLMESContract) -> tuple[str, ...]:
    return tuple(
        name for name in contract.metrics.aggregate if name != "primary_metric"
    )


def _assert_output_schema_parity(contract: OLMESContract) -> None:
    table = contract.tables.aggregate
    expected = _output_columns(contract)
    actual = tuple(column.name for column in table.columns)
    if actual != expected:
        raise AssertionError(
            "persisted OLMES output columns drift from contract.tables.aggregate: "
            f"expected={expected!r}, actual={actual!r}"
        )

    identity = _identity_columns(contract)
    expected_identity = (
        "params",
        "data",
        "seed",
        "step",
        "task",
        "chinchilla",
        *CHECKPOINT_ENRICHMENT_COLUMNS,
    )
    if identity != expected_identity:
        raise AssertionError(
            "OLMES identity columns drift from contract: "
            f"expected={expected_identity!r}, actual={identity!r}"
        )

    expected_columns = (*expected_identity, *contract.metrics.aggregate)
    if expected != expected_columns:
        raise AssertionError(
            "OLMES aggregate columns drift from contract metrics: "
            f"expected={expected_columns!r}, actual={expected!r}"
        )

    source_metrics = _source_metric_columns(contract)
    expected_source_count = len(contract.metrics.aggregate) - 1
    if len(source_metrics) != expected_source_count:
        raise AssertionError(
            "OLMES source metric columns must exclude only primary_metric: "
            f"expected {expected_source_count}, got {len(source_metrics)}"
        )


def _pandas_dtype(
    logical_type: Literal["string", "int64", "float64", "bool"],
) -> str:
    if logical_type == "string":
        return "string"
    return logical_type


def group_olmes_rows(
    olmes_df: pd.DataFrame,
    *,
    contract: OLMESContract | None = None,
) -> OlmesRowsByKey:
    from datadec.data.ingest.enums import DataRecipeName, ModelSizeName, Seed

    contract = contract or load_olmes_contract()
    _assert_output_schema_parity(contract)
    source_metric_columns = _source_metric_columns(contract)

    grouped: OlmesRowsByKey = defaultdict(lambda: defaultdict(dict))
    seen_keys: set[tuple[str, str, str, int, str]] = set()

    for row_index, record in enumerate(olmes_df.to_dict(orient="records")):
        run_key = (
            ModelSizeName(record["params"]),
            DataRecipeName(record["data"]),
            Seed(record["seed"]),
        )
        step = _normalize_step(record["step"], row_index=row_index)
        task = _require_string(record.get("task"), row_index=row_index, field="task")
        chinchilla = _require_string(
            record.get("chinchilla"), row_index=row_index, field="chinchilla"
        )
        tokens = _normalize_int64(
            record.get("tokens"), row_index=row_index, field="tokens"
        )
        compute = _normalize_float64(
            record.get("compute"), row_index=row_index, field="compute"
        )

        primary_key = (
            run_key[0].value,
            run_key[1].value,
            run_key[2].value,
            step,
            task,
        )
        if primary_key in seen_keys:
            params, data, seed = run_key
            raise ValueError(
                f"duplicate OLMES row at row {row_index}: "
                f"params={params.value!r}, data={data.value!r}, "
                f"seed={seed.value!r}, step={step}, task={task!r}"
            )
        seen_keys.add(primary_key)

        metrics_payload = _parse_metrics_payload(record.get("metrics"))
        metrics = _extract_source_metrics(metrics_payload, source_metric_columns)

        row = OlmesAggregateRow(
            params=run_key[0],
            data=run_key[1],
            seed=run_key[2],
            step=step,
            task=task,
            chinchilla=chinchilla,
            tokens=tokens,
            compute=compute,
            metrics=metrics,
        )
        grouped[run_key][step][task] = row

    return dict(grouped)


def flatten_olmes_rows(
    grouped: OlmesRowsByKey,
    *,
    contract: OLMESContract | None = None,
) -> pd.DataFrame:
    contract = contract or load_olmes_contract()
    _assert_output_schema_parity(contract)
    output_columns = _output_columns(contract)
    source_metric_columns = _source_metric_columns(contract)
    policy = contract.metrics.aggregate_primary_metric.model_dump()

    rows: list[dict[str, object]] = []
    for run_key in sorted(grouped, key=lambda key: tuple(item.value for item in key)):
        params, data, seed = run_key
        for step in sorted(grouped[run_key]):
            for task in sorted(grouped[run_key][step]):
                row = grouped[run_key][step][task]
                primary_metric = _resolve_primary_metric(
                    task, row.metrics, policy=policy
                )
                enrichment = checkpoint_enrichment(params.value, step)
                expected_tokens = int(enrichment["tokens"])
                expected_compute = float(enrichment["compute"])
                if row.tokens != expected_tokens or not np.isclose(
                    row.compute,
                    expected_compute,
                    rtol=1e-12,
                    atol=1e-6,
                ):
                    raise ValueError(
                        "OLMES token/compute values contradict the canonical "
                        f"schedule for params={params.value!r}, step={step}"
                    )
                rows.append(
                    {
                        "params": params.value,
                        "data": data.value,
                        "seed": seed.value,
                        "step": step,
                        "task": task,
                        "chinchilla": row.chinchilla,
                        **enrichment,
                        **{
                            field: row.metrics.get(field)
                            for field in source_metric_columns
                        },
                        "primary_metric": primary_metric,
                    }
                )

    return _typed_output_dataframe(rows, contract=contract, columns=output_columns)


def preprocess_olmes(
    paths: DataArtifacts,
    *,
    input_path: Path | None = None,
    output_path: Path | None = None,
    verbose: bool = False,
) -> OlmesPreprocessResult:
    contract = load_olmes_contract()
    _assert_output_schema_parity(contract)
    resolved_input = input_path or paths.get_path("dwn_raw")
    resolved_output = output_path or paths.get_path("olmes_processed")

    from datadec.data.ingest.enums import DataRecipeName, ModelSizeName, Seed

    parquet_file = pq.ParquetFile(resolved_input)
    connection = duckdb.connect()
    try:
        if parquet_file.metadata.num_rows == 0:
            select_sql = _empty_output_select_sql(contract)
            training_run_count = 0
        else:
            raw_columns = set(parquet_file.schema_arrow.names)
            row_index_column = "_datadec_row_index"
            while row_index_column in raw_columns:
                row_index_column = f"_{row_index_column}"
            connection.execute(
                f"""
                CREATE TEMP VIEW _olmes_raw AS
                SELECT *, row_number() OVER () - 1
                    AS {quote_identifier(row_index_column)}
                FROM read_parquet({sql_literal(resolved_input)})
                """
            )
            _validate_raw_olmes(
                connection,
                contract=contract,
                raw_columns=raw_columns,
                row_index_column=row_index_column,
                model_sizes=tuple(item.value for item in ModelSizeName),
                data_recipes=tuple(item.value for item in DataRecipeName),
                seeds=tuple(item.value for item in Seed),
            )
            create_model_enrichment_table(
                connection,
                checkpoint_select_sql="""
                    SELECT
                        CAST(params AS VARCHAR) AS params,
                        CAST(trim(CAST(step AS VARCHAR)) AS BIGINT) AS step
                    FROM _olmes_raw
                """,
            )
            _validate_raw_schedule(connection, row_index_column=row_index_column)
            select_sql = _output_select_sql(contract)
            training_run_count_row = connection.execute(
                """
                SELECT count(DISTINCT (
                    CAST(params AS VARCHAR),
                    CAST(data AS VARCHAR),
                    CAST(seed AS VARCHAR)
                ))
                FROM _olmes_raw
                """
            ).fetchone()
            assert training_run_count_row is not None
            training_run_count = training_run_count_row[0]

        export = prepare_parquet_export(
            connection,
            select_sql=select_sql,
            output_path=resolved_output,
            key_value_metadata={"pandas": _pandas_parquet_metadata(contract)},
        )
        replace_parquet_exports((export,))
    finally:
        connection.close()

    result = OlmesPreprocessResult(
        input_path=resolved_input,
        output_path=resolved_output,
        row_count=export.row_count,
        training_run_count=training_run_count,
    )
    if verbose:
        print(f"olmes input: {result.input_path}")
        print(f"olmes output: {result.output_path}")
        print(f"olmes rows: {result.row_count}")
        print(f"olmes training runs: {result.training_run_count}")
    return result


def _pandas_parquet_metadata(contract: OLMESContract) -> str:
    columns = []
    for column in contract.tables.aggregate.columns:
        pandas_type = numpy_type = column.logical_type
        if column.logical_type == "string":
            pandas_type = "object"
            numpy_type = "string"
        columns.append(
            {
                "name": column.name,
                "field_name": column.name,
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


def _metrics_json_sql() -> str:
    metrics = quote_identifier("metrics")
    return (
        "CASE "
        f"WHEN typeof({metrics}) IN ('VARCHAR', 'BLOB') "
        f"THEN replace(CAST({metrics} AS VARCHAR), "
        f"{sql_literal(chr(39))}, {sql_literal(chr(34))}) "
        f"ELSE CAST(to_json({metrics}) AS VARCHAR) END"
    )


def _normalized_int64_sql(column: str) -> str:
    text = f"trim(CAST({quote_identifier(column)} AS VARCHAR))"
    return f"CAST({text} AS BIGINT)"


def _validate_raw_olmes(
    connection: duckdb.DuckDBPyConnection,
    *,
    contract: OLMESContract,
    raw_columns: set[str],
    row_index_column: str,
    model_sizes: tuple[str, ...],
    data_recipes: tuple[str, ...],
    seeds: tuple[str, ...],
) -> None:
    required_columns = (*_RAW_IDENTITY_COLUMNS, "metrics")
    missing = [column for column in required_columns if column not in raw_columns]
    if missing:
        raise ValueError(f"missing required OLMES input columns: {missing!r}")

    row_index = quote_identifier(row_index_column)
    conditions: list[tuple[str, str]] = []
    for column, allowed in (
        ("params", model_sizes),
        ("data", data_recipes),
        ("seed", seeds),
    ):
        identifier = quote_identifier(column)
        allowed_sql = ", ".join(sql_literal(value) for value in allowed)
        conditions.append(
            (
                column,
                f"{identifier} IS NULL "
                f"OR CAST({identifier} AS VARCHAR) NOT IN ({allowed_sql})",
            )
        )

    for column in ("task", "chinchilla"):
        identifier = quote_identifier(column)
        conditions.append(
            (
                column,
                f"{identifier} IS NULL OR isnan(try_cast({identifier} AS DOUBLE))",
            )
        )

    for column in ("step", "tokens"):
        identifier = quote_identifier(column)
        text = f"trim(CAST({identifier} AS VARCHAR))"
        decimal_value = f"try_cast({text} AS DECIMAL(38, 18))"
        conditions.append(
            (
                column,
                f"{identifier} IS NULL "
                f"OR typeof({identifier}) = 'BOOLEAN' "
                f"OR {decimal_value} IS NULL "
                f"OR {decimal_value} <> trunc({decimal_value}) "
                f"OR try_cast({text} AS BIGINT) IS NULL",
            )
        )

    compute = quote_identifier("compute")
    compute_value = f"try_cast(trim(CAST({compute} AS VARCHAR)) AS DOUBLE)"
    conditions.append(
        (
            "compute",
            f"{compute} IS NULL "
            f"OR typeof({compute}) = 'BOOLEAN' "
            f"OR {compute_value} IS NULL "
            f"OR NOT isfinite({compute_value})",
        )
    )

    metrics_json = quote_identifier("_datadec_metrics_json")
    conditions.extend(
        (
            (
                "metrics_json",
                f"{metrics_json} IS NULL OR NOT json_valid({metrics_json})",
            ),
            (
                "metrics_object",
                f"json_type(try_cast({metrics_json} AS JSON)) <> 'OBJECT'",
            ),
        )
    )
    invalid_kind = (
        "CASE "
        + " ".join(
            f"WHEN {condition} THEN {sql_literal(kind)}"
            for kind, condition in conditions
        )
        + " END"
    )
    invalid = connection.execute(
        f"""
        SELECT {row_index}, invalid_kind
        FROM (
            SELECT *, {invalid_kind} AS invalid_kind
            FROM (
                SELECT *, {_metrics_json_sql()} AS {metrics_json}
                FROM _olmes_raw
            )
        )
        WHERE invalid_kind IS NOT NULL
        ORDER BY {row_index}
        LIMIT 1
        """
    ).fetchone()
    if invalid is not None:
        invalid_row_index, kind = invalid
        source_column = "metrics" if kind.startswith("metrics_") else kind
        value_row = connection.execute(
            f"""
            SELECT {quote_identifier(source_column)}
            FROM _olmes_raw
            WHERE {row_index} = ?
            """,
            [invalid_row_index],
        ).fetchone()
        assert value_row is not None
        value = value_row[0]
        if kind == "metrics_json":
            raise ValueError(
                f"invalid OLMES metrics at row {invalid_row_index}: {value!r}; "
                "expected a valid JSON or Python-dict object"
            )
        if kind == "metrics_object":
            raise ValueError(
                f"invalid OLMES metrics at row {invalid_row_index}: {value!r}; "
                "expected an object payload"
            )
        if kind in {"params", "data", "seed"}:
            raise ValueError(
                f"unknown OLMES {kind} at row {invalid_row_index}: {value!r}"
            )
        if kind in {"task", "chinchilla"}:
            raise ValueError(
                f"invalid OLMES {kind} at row {invalid_row_index}: {value!r}; "
                "expected a non-null string"
            )
        if kind == "step":
            raise ValueError(
                f"invalid PPL step at row {invalid_row_index}: {value!r}; "
                "expected a finite integral int64 value"
            )
        if kind == "tokens":
            raise ValueError(
                f"invalid OLMES tokens at row {invalid_row_index}: {value!r}; "
                "expected a finite integral int64 value"
            )
        assert kind == "compute"
        raise ValueError(
            f"invalid OLMES compute at row {invalid_row_index}: {value!r}; "
            "expected a finite float64 value"
        )

    normalized_step = _normalized_int64_sql("step")
    duplicate = connection.execute(
        f"""
        SELECT {row_index}, params, data, seed, normalized_step, task
        FROM (
            SELECT
                {row_index},
                CAST(params AS VARCHAR) AS params,
                CAST(data AS VARCHAR) AS data,
                CAST(seed AS VARCHAR) AS seed,
                {normalized_step} AS normalized_step,
                CAST(task AS VARCHAR) AS task,
                row_number() OVER (
                    PARTITION BY params, data, seed, {normalized_step}, task
                    ORDER BY {row_index}
                ) AS duplicate_ordinal
            FROM _olmes_raw
        )
        WHERE duplicate_ordinal > 1
        ORDER BY {row_index}
        LIMIT 1
        """
    ).fetchone()
    if duplicate is not None:
        row_index_value, params, data, seed, step, task = duplicate
        raise ValueError(
            f"duplicate OLMES row at row {row_index_value}: "
            f"params={params!r}, data={data!r}, seed={seed!r}, "
            f"step={step}, task={task!r}"
        )


def _empty_output_select_sql(contract: OLMESContract) -> str:
    expressions = [
        f"CAST(NULL AS {duckdb_type(column.logical_type)}) "
        f"AS {quote_identifier(column.name)}"
        for column in contract.tables.aggregate.columns
    ]
    return f"SELECT {', '.join(expressions)} WHERE false"


def _validate_raw_schedule(
    connection: duckdb.DuckDBPyConnection,
    *,
    row_index_column: str,
) -> None:
    row_index = quote_identifier(row_index_column)
    mismatch = connection.execute(
        f"""
        SELECT
            raw.{row_index},
            CAST(raw.params AS VARCHAR),
            CAST(trim(CAST(raw.step AS VARCHAR)) AS BIGINT),
            CAST(trim(CAST(raw.tokens AS VARCHAR)) AS BIGINT),
            enrichment.tokens,
            CAST(trim(CAST(raw.compute AS VARCHAR)) AS DOUBLE),
            enrichment.compute
        FROM _olmes_raw AS raw
        INNER JOIN _model_enrichment AS enrichment
            ON CAST(raw.params AS VARCHAR) = enrichment.params
           AND CAST(trim(CAST(raw.step AS VARCHAR)) AS BIGINT) = enrichment.step
        WHERE CAST(trim(CAST(raw.tokens AS VARCHAR)) AS BIGINT)
                <> enrichment.tokens
           OR abs(
                CAST(trim(CAST(raw.compute AS VARCHAR)) AS DOUBLE)
                - enrichment.compute
              ) > greatest(1e-6, 1e-12 * abs(enrichment.compute))
        ORDER BY raw.{row_index}
        LIMIT 1
        """
    ).fetchone()
    if mismatch is not None:
        row, params, step, tokens, expected_tokens, compute, expected_compute = mismatch
        raise ValueError(
            f"OLMES schedule contradiction at row {row}: params={params!r}, "
            f"step={step}, tokens={tokens}, expected_tokens={expected_tokens}, "
            f"compute={compute}, expected_compute={expected_compute}"
        )


def _coerced_metric_sql(metrics_identifier: str, field: str) -> str:
    raw_value = f"{metrics_identifier}.{quote_identifier(field)}"
    converted = f"try_cast(trim({raw_value}) AS DOUBLE)"
    return f"CASE WHEN isnan({converted}) THEN NULL ELSE {converted} END"


def _output_select_sql(contract: OLMESContract) -> str:
    source_metric_columns = _source_metric_columns(contract)
    metrics_schema = json.dumps(
        {field: "VARCHAR" for field in source_metric_columns},
        separators=(",", ":"),
    )
    parsed_metrics = quote_identifier("_datadec_metrics")
    identity_expressions = [
        f"CAST(parsed.{quote_identifier(column)} AS {duckdb_type('string')}) "
        f"AS {quote_identifier(column)}"
        for column in ("params", "data", "seed")
    ]
    identity_expressions.extend(
        (
            "CAST(trim(CAST(parsed.step AS VARCHAR)) AS BIGINT) AS step",
            f"CAST(parsed.task AS {duckdb_type('string')}) AS task",
            f"CAST(parsed.chinchilla AS {duckdb_type('string')}) AS chinchilla",
            enrichment_select_expressions(table_alias="enrichment"),
        )
    )
    metric_expressions = [
        f"{_coerced_metric_sql(parsed_metrics, field)} AS {quote_identifier(field)}"
        for field in source_metric_columns
    ]
    policy = contract.metrics.aggregate_primary_metric.model_dump()
    primary_cases = [
        f"WHEN starts_with(task, 'mmlu_') THEN {quote_identifier(policy['mmlu'])}"
    ]
    primary_cases.extend(
        f"WHEN task = {sql_literal(task)} THEN {quote_identifier(metric)}"
        for task, metric in policy.items()
        if task != "mmlu"
    )
    primary_metric = "CASE " + " ".join(primary_cases) + " ELSE NULL END"
    non_primary_output_columns = ", ".join(
        quote_identifier(column) for column in _output_columns(contract)[:-1]
    )
    sort_key = ", ".join(
        quote_identifier(column) for column in contract.tables.aggregate.sort_key
    )
    return f"""
        WITH parsed AS (
            SELECT
                *,
                from_json({_metrics_json_sql()}, {sql_literal(metrics_schema)})
                    AS {parsed_metrics}
            FROM _olmes_raw
        ),
        coerced AS (
            SELECT
                {", ".join(identity_expressions)},
                {", ".join(metric_expressions)}
            FROM parsed
            INNER JOIN _model_enrichment AS enrichment
                ON CAST(parsed.params AS VARCHAR) = enrichment.params
               AND CAST(trim(CAST(parsed.step AS VARCHAR)) AS BIGINT)
                    = enrichment.step
        )
        SELECT {non_primary_output_columns},
               {primary_metric} AS primary_metric
        FROM coerced
        ORDER BY {sort_key}
    """


def _parse_metrics_payload(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, bytes):
        return orjson.loads(raw)
    text = str(raw).replace("'", '"')
    return orjson.loads(text)


def _extract_source_metrics(
    payload: dict[str, Any],
    source_metric_columns: tuple[str, ...],
) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {}
    for field in source_metric_columns:
        if field in payload and field not in _EXCLUDED_METRIC_KEYS:
            metrics[field] = coerce_float(payload[field])
        else:
            metrics[field] = None
    return metrics


def _resolve_primary_metric(
    task: str,
    metrics: dict[str, float | None],
    *,
    policy: dict[str, str],
) -> float | None:
    policy_key = "mmlu" if task.startswith("mmlu_") else task
    source_metric = policy.get(policy_key)
    if source_metric is None:
        return None
    return metrics.get(source_metric)


def _require_string(value: Any, *, row_index: int, field: str) -> str:
    invalid = value is None or value is pd.NA
    if not invalid and isinstance(value, (float, np.floating)):
        invalid = bool(np.isnan(value))
    if invalid:
        raise ValueError(
            f"invalid OLMES {field} at row {row_index}: {value!r}; "
            "expected a non-null string"
        )
    return str(value)


def _normalize_int64(value: Any, *, row_index: int, field: str) -> int:
    invalid = isinstance(value, (bool, np.bool_)) or value is None or value is pd.NA
    decimal_value: Decimal | None = None
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
    if invalid or decimal_value is None:
        raise ValueError(
            f"invalid OLMES {field} at row {row_index}: {value!r}; "
            "expected a finite integral int64 value"
        )

    normalized = int(decimal_value)
    if not _INT64_MIN <= normalized <= _INT64_MAX:
        raise ValueError(
            f"invalid OLMES {field} at row {row_index}: {value!r}; "
            "expected a finite integral int64 value"
        )
    return normalized


def _normalize_float64(value: Any, *, row_index: int, field: str) -> float:
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
            invalid = not decimal_value.is_finite()
        except (InvalidOperation, ValueError):
            invalid = True
    if invalid:
        raise ValueError(
            f"invalid OLMES {field} at row {row_index}: {value!r}; "
            "expected a finite float64 value"
        )
    return float(decimal_value)


def _typed_output_dataframe(
    rows: list[dict[str, object]],
    *,
    contract: OLMESContract,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    column_types = {
        column.name: column.logical_type for column in contract.tables.aggregate.columns
    }
    values_by_column = {column: [row.get(column) for row in rows] for column in columns}
    typed_columns: dict[str, pd.Series[Any]] = {}
    for column in columns:
        logical_type = column_types[column]
        dtype = _pandas_dtype(logical_type)
        typed_columns[column] = pd.Series(values_by_column[column], dtype=dtype)
    return pd.DataFrame(typed_columns, columns=list(columns))
