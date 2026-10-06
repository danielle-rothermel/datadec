from __future__ import annotations

import hashlib
import io
import re
import tarfile
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import duckdb
import numpy as np
import orjson
from dr_ds import coerce_float
from fsspec.implementations.memory import MemoryFileSystem

from datadec.config import OLMESContract, OLMESTableContract, load_olmes_contract
from datadec.config import load_source_manifest
from datadec.data.model_utils import checkpoint_enrichment
from datadec.data.artifacts import DataArtifacts
from datadec.data.preprocess.duckdb import (
    DuckDbLogicalType,
    PendingParquetExport,
    duckdb_type,
    prepare_parquet_export,
    quote_identifier,
    remove_owned_file,
    replace_parquet_exports,
    sql_literal,
)
from datadec.data.preprocess.ppl import _normalize_step

_CHECKPOINT_MEMBER_RE = re.compile(
    r"^(?P<recipe>[^/]+)/(?P<params>[^/]+)/seed-(?P<seed_value>\d+)/step-(?P<step>\d+)\.tar\.gz$"
)
_METRICS_SUFFIX = "-metrics.json"
_PREDICTIONS_SUFFIX = "-predictions.jsonl"
_STAGING_FILENAME = ".olmes-details.duckdb"
_MEMORY_ROOT = "/datadec-olmes-details"
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1

type _LogicalType = DuckDbLogicalType
_STRING_LOGICAL_TYPE: _LogicalType = "string"
_INT64_LOGICAL_TYPE: _LogicalType = "int64"


@dataclass(frozen=True, slots=True)
class OlmesDetailsPreprocessResult:
    recipe: str
    input_path: Path
    output_tasks_path: Path
    output_instances_path: Path
    output_choices_path: Path
    row_count: int
    instance_count: int
    choice_count: int
    checkpoint_count: int

    @property
    def output_path(self) -> Path:
        return self.output_tasks_path


@dataclass(frozen=True, slots=True)
class _CheckpointPayload:
    task_rows: tuple[dict[str, object], ...]
    predictions_by_task: dict[str, bytes]


def _output_columns(contract: OLMESContract, table_name: str) -> tuple[str, ...]:
    table = getattr(contract.tables, table_name)
    return tuple(column.name for column in table.columns)


def _metric_columns(contract: OLMESContract, table_name: str) -> tuple[str, ...]:
    return getattr(contract.metrics, table_name)


def _assert_table_schema_parity(
    contract: OLMESContract,
    *,
    table_name: str,
    label: str,
) -> None:
    table = getattr(contract.tables, table_name)
    expected = _output_columns(contract, table_name)
    actual = tuple(column.name for column in table.columns)
    if actual != expected:
        raise AssertionError(
            f"persisted OLMES {label} columns drift from contract: "
            f"expected={expected!r}, actual={actual!r}"
        )
    metric_columns = _metric_columns(contract, table_name)
    trailing = tuple(column.name for column in table.columns[-len(metric_columns) :])
    if trailing != metric_columns:
        raise AssertionError(
            f"OLMES {label} metric columns drift from contract: "
            f"expected={metric_columns!r}, actual={trailing!r}"
        )


def _assert_detailed_tasks_schema_parity(contract: OLMESContract) -> None:
    _assert_table_schema_parity(
        contract, table_name="detailed_tasks", label="detail task"
    )


def _assert_detailed_instances_schema_parity(contract: OLMESContract) -> None:
    _assert_table_schema_parity(
        contract, table_name="detailed_instances", label="detail instance"
    )


def _assert_detailed_choices_schema_parity(contract: OLMESContract) -> None:
    _assert_table_schema_parity(
        contract, table_name="detailed_choices", label="detail choice"
    )


def _assert_all_detailed_schema_parity(contract: OLMESContract) -> None:
    _assert_detailed_tasks_schema_parity(contract)
    _assert_detailed_instances_schema_parity(contract)
    _assert_detailed_choices_schema_parity(contract)


def _recipe_tar_path(paths: DataArtifacts, recipe: str) -> Path:
    manifest = load_source_manifest()
    filename = manifest.olmes_details.filename_template.format(recipe=recipe)
    return paths.data_dir / manifest.olmes_details.output_root / filename


def _canonical_json(value: object) -> str:
    return orjson.dumps(value, option=orjson.OPT_SORT_KEYS).decode()


def _require_hash(value: Any, *, context: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"invalid OLMES detail {field} in {context}: {value!r}; "
            "expected a non-empty string"
        )
    return value


def _normalize_int64_field(value: Any, *, context: str, field: str) -> int:
    invalid = isinstance(value, (bool, np.bool_)) or value is None
    decimal_value: Decimal | None = None
    if not invalid:
        try:
            if not isinstance(
                value, (str, int, float, Decimal, np.integer, np.floating)
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
            f"invalid OLMES detail {field} in {context}: {value!r}; "
            "expected a finite integral int64 value"
        )
    normalized = int(decimal_value)
    if not _INT64_MIN <= normalized <= _INT64_MAX:
        raise ValueError(
            f"invalid OLMES detail {field} in {context}: {value!r}; "
            "expected a finite integral int64 value"
        )
    return normalized


def _normalize_float64_field(value: Any, *, context: str, field: str) -> float:
    invalid = isinstance(value, (bool, np.bool_)) or value is None
    decimal_value: Decimal | None = None
    if not invalid:
        try:
            if not isinstance(
                value, (str, int, float, Decimal, np.integer, np.floating)
            ):
                raise InvalidOperation
            text = str(value).strip()
            if not text:
                raise InvalidOperation
            decimal_value = Decimal(text)
            invalid = not decimal_value.is_finite()
        except (InvalidOperation, ValueError):
            invalid = True
    if invalid or decimal_value is None:
        raise ValueError(
            f"invalid OLMES detail {field} in {context}: {value!r}; "
            "expected a finite float64 value"
        )
    return float(decimal_value)


def _build_task_row(
    metrics_payload: dict[str, Any],
    *,
    recipe: str,
    params: str,
    seed_value: int,
    step: int,
    task: str,
    contract: OLMESContract,
    context: str,
) -> dict[str, object]:
    if recipe not in contract.recipe_map:
        raise ValueError(f"unknown OLMES detail recipe in {context}: {recipe!r}")
    if seed_value not in contract.seed_map:
        raise ValueError(
            f"unknown OLMES detail seed_value in {context}: {seed_value!r}"
        )
    task_name = metrics_payload.get("task_name")
    task_config = metrics_payload.get("task_config")
    if not isinstance(task_config, dict):
        raise ValueError(
            f"invalid OLMES detail task_config in {context}: expected object"
        )
    config_task_name = task_config.get("task_name")
    if task_name != task or config_task_name != task:
        raise ValueError(
            f"task identity mismatch in {context}: path task={task!r}, "
            f"task_name={task_name!r}, task_config.task_name={config_task_name!r}"
        )
    model_config = metrics_payload.get("model_config")
    compute_config = metrics_payload.get("compute_config")
    if not isinstance(model_config, dict) or not isinstance(compute_config, dict):
        raise ValueError(
            f"invalid OLMES detail config objects in {context}: "
            "expected model_config and compute_config objects"
        )
    metrics_block = metrics_payload.get("metrics")
    if not isinstance(metrics_block, dict):
        raise ValueError(f"invalid OLMES detail metrics in {context}: expected object")
    primary_metric = task_config.get("primary_metric")
    if not isinstance(primary_metric, str) or not primary_metric.strip():
        raise ValueError(
            f"invalid OLMES detail primary_metric in {context}: {primary_metric!r}"
        )
    metric_values = {
        field: coerce_float(metrics_block[field]) if field in metrics_block else None
        for field in contract.metrics.detailed_tasks
    }
    enrichment = checkpoint_enrichment(params, step)
    if model_config.get("max_length") != enrichment["max_sequence_length"]:
        raise ValueError(
            f"model max_length contradicts canonical model details in {context}: "
            f"raw={model_config.get('max_length')!r}, "
            f"expected={enrichment['max_sequence_length']!r}"
        )
    return {
        "recipe": recipe,
        "data": contract.recipe_map[recipe],
        "params": params,
        "seed_value": seed_value,
        "seed": contract.seed_map[seed_value],
        "step": step,
        **enrichment,
        "task": task,
        "task_hash": _require_hash(
            metrics_payload.get("task_hash"), context=context, field="task_hash"
        ),
        "model_hash": _require_hash(
            metrics_payload.get("model_hash"), context=context, field="model_hash"
        ),
        "model_config": _canonical_json(model_config),
        "task_config": _canonical_json(task_config),
        "compute_config": _canonical_json(compute_config),
        "processing_time": _normalize_float64_field(
            metrics_payload.get("processing_time"),
            context=context,
            field="processing_time",
        ),
        "current_date": _require_hash(
            metrics_payload.get("current_date"), context=context, field="current_date"
        ),
        "num_instances": _normalize_int64_field(
            metrics_payload.get("num_instances"),
            context=context,
            field="num_instances",
        ),
        "task_idx": _normalize_int64_field(
            metrics_payload.get("task_idx"), context=context, field="task_idx"
        ),
        "primary_metric": primary_metric,
        **metric_values,
    }


def _index_checkpoint_members(
    inner: tarfile.TarFile,
    *,
    expected_step: int,
    context: str,
) -> tuple[dict[str, tarfile.TarInfo], dict[str, tarfile.TarInfo]]:
    metrics_by_task: dict[str, tarfile.TarInfo] = {}
    predictions_by_task: dict[str, tarfile.TarInfo] = {}
    step_prefix = f"step-{expected_step}/"
    for member in inner.getmembers():
        if not member.isfile():
            continue
        if not member.name.startswith(step_prefix):
            raise ValueError(
                f"unexpected member path in {context}: {member.name!r}; "
                f"expected prefix {step_prefix!r}"
            )
        filename = member.name[len(step_prefix) :]
        if filename.endswith(_METRICS_SUFFIX):
            task = filename[: -len(_METRICS_SUFFIX)]
            target = metrics_by_task
        elif filename.endswith(_PREDICTIONS_SUFFIX):
            task = filename[: -len(_PREDICTIONS_SUFFIX)]
            target = predictions_by_task
        else:
            continue
        if task in target:
            raise ValueError(
                f"duplicate OLMES detail task payload in {context}: {task!r}"
            )
        target[task] = member
    metrics_tasks = set(metrics_by_task)
    predictions_tasks = set(predictions_by_task)
    if metrics_tasks != predictions_tasks:
        details: list[str] = []
        missing_predictions = sorted(metrics_tasks - predictions_tasks)
        missing_metrics = sorted(predictions_tasks - metrics_tasks)
        if missing_predictions:
            details.append(f"missing predictions for {missing_predictions!r}")
        if missing_metrics:
            details.append(f"missing metrics for {missing_metrics!r}")
        raise ValueError(
            f"unpaired OLMES detail task files in {context}: {'; '.join(details)}"
        )
    return metrics_by_task, predictions_by_task


def _read_checkpoint_payload(
    inner_bytes: bytes,
    *,
    recipe: str,
    params: str,
    seed_value: int,
    step: int,
    contract: OLMESContract,
) -> _CheckpointPayload:
    context = (
        f"recipe={recipe!r}, params={params!r}, seed_value={seed_value}, step={step}"
    )
    task_rows: list[dict[str, object]] = []
    predictions: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(inner_bytes), mode="r:gz") as inner:
        metrics_by_task, predictions_by_task = _index_checkpoint_members(
            inner, expected_step=step, context=context
        )
        for task in sorted(metrics_by_task):
            task_context = f"{context}, task={task!r}"
            metrics_file = inner.extractfile(metrics_by_task[task])
            predictions_file = inner.extractfile(predictions_by_task[task])
            if metrics_file is None or predictions_file is None:
                raise ValueError(f"missing OLMES detail task payload in {task_context}")
            metrics_payload = orjson.loads(metrics_file.read())
            if not isinstance(metrics_payload, dict):
                raise ValueError(
                    f"invalid OLMES detail metrics JSON in {task_context}: "
                    "expected object"
                )
            task_rows.append(
                _build_task_row(
                    metrics_payload,
                    recipe=recipe,
                    params=params,
                    seed_value=seed_value,
                    step=step,
                    task=task,
                    contract=contract,
                    context=task_context,
                )
            )
            predictions[task] = predictions_file.read()
    return _CheckpointPayload(tuple(task_rows), predictions)


def _parse_checkpoint_member_path(
    member_name: str,
    *,
    expected_recipe: str,
) -> tuple[str, str, int, int] | None:
    match = _CHECKPOINT_MEMBER_RE.match(member_name)
    if match is None:
        return None
    recipe = match.group("recipe")
    if recipe.casefold() != expected_recipe.casefold():
        raise ValueError(
            f"unexpected recipe in checkpoint member {member_name!r}: "
            f"expected {expected_recipe!r}"
        )
    return (
        expected_recipe,
        match.group("params"),
        int(match.group("seed_value")),
        _normalize_step(match.group("step"), row_index=0),
    )


def _create_data_table(
    connection: duckdb.DuckDBPyConnection,
    *,
    name: str,
    contract: OLMESTableContract,
) -> None:
    definitions = []
    for column in contract.columns:
        nullable = "" if column.nullable else " NOT NULL"
        definitions.append(
            f"{quote_identifier(column.name)} "
            f"{duckdb_type(column.logical_type)}{nullable}"
        )
    connection.execute(
        f"CREATE TABLE IF NOT EXISTS {quote_identifier(name)} "
        f"({', '.join(definitions)})"
    )


def _contract_fingerprint(contract: OLMESContract) -> str:
    payload = orjson.dumps(
        contract.model_dump(mode="json"),
        option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SORT_KEYS,
    )
    return hashlib.sha256(payload).hexdigest()


def _staging_database_path(tasks_path: Path) -> Path:
    return tasks_path.parent / _STAGING_FILENAME


def _initialize_staging_database(
    connection: duckdb.DuckDBPyConnection,
    *,
    staging_path: Path,
    input_path: Path,
    recipe: str,
    contract: OLMESContract,
) -> set[tuple[str, str, int, int]]:
    source = input_path.resolve()
    source_stat = source.stat()
    expected_metadata = (
        recipe,
        str(source),
        source_stat.st_size,
        source_stat.st_mtime_ns,
        _contract_fingerprint(contract),
    )
    connection.execute("BEGIN TRANSACTION")
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS processing_metadata (
                recipe VARCHAR NOT NULL,
                source_path VARCHAR NOT NULL,
                source_size UBIGINT NOT NULL,
                source_mtime_ns UBIGINT NOT NULL,
                contract_sha256 VARCHAR NOT NULL
            )
            """
        )
        metadata = connection.execute(
            """
            SELECT recipe, source_path, source_size, source_mtime_ns, contract_sha256
            FROM processing_metadata
            """
        ).fetchall()
        if not metadata:
            connection.execute(
                "INSERT INTO processing_metadata VALUES (?, ?, ?, ?, ?)",
                expected_metadata,
            )
        elif metadata != [expected_metadata]:
            raise ValueError(
                f"OLMES detail staging database does not match the current input: "
                f"{staging_path}; remove it before processing a different source"
            )
        for name, table in (
            ("tasks", contract.tables.detailed_tasks),
            ("instances", contract.tables.detailed_instances),
            ("choices", contract.tables.detailed_choices),
        ):
            _create_data_table(connection, name=name, contract=table)
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS completed_checkpoints (
                recipe VARCHAR NOT NULL,
                params VARCHAR NOT NULL,
                seed_value BIGINT NOT NULL,
                step BIGINT NOT NULL,
                member_name VARCHAR NOT NULL,
                task_count BIGINT NOT NULL,
                instance_count BIGINT NOT NULL,
                choice_count BIGINT NOT NULL,
                PRIMARY KEY (recipe, params, seed_value, step)
            )
            """
        )
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    return {
        (row[0], row[1], row[2], row[3])
        for row in connection.execute(
            "SELECT recipe, params, seed_value, step FROM completed_checkpoints"
        ).fetchall()
    }


def _json_text(expression: str) -> str:
    return f"json_extract_string({expression}, '$')"


def _valid_json_value(
    expression: str,
    *,
    logical_type: _LogicalType,
    nullable: bool,
) -> str:
    text = _json_text(expression)
    if logical_type == "string":
        valid = f"json_type({expression}) = 'VARCHAR'"
    elif logical_type == "bool":
        valid = f"json_type({expression}) = 'BOOLEAN'"
    elif logical_type == "float64":
        valid = (
            f"json_type({expression}) IN ('BIGINT', 'UBIGINT', 'DOUBLE', 'VARCHAR') "
            f"AND try_cast({text} AS DOUBLE) IS NOT NULL "
            f"AND isfinite(try_cast({text} AS DOUBLE))"
        )
    else:
        decimal_value = f"try_cast({text} AS DECIMAL(38, 18))"
        double_value = f"try_cast({text} AS DOUBLE)"
        valid = (
            f"json_type({expression}) IN ('BIGINT', 'UBIGINT', 'DOUBLE', 'VARCHAR') "
            f"AND {decimal_value} IS NOT NULL "
            f"AND {decimal_value} = trunc({decimal_value}) "
            f"AND {double_value} IS NOT NULL "
            f"AND isfinite({double_value}) "
            f"AND {double_value} = trunc({double_value}) "
            f"AND try_cast({text} AS BIGINT) IS NOT NULL"
        )
    if nullable:
        return f"({expression} IS NULL OR ({valid}))"
    return f"({expression} IS NOT NULL AND ({valid}))"


def _cast_json_value(
    expression: str,
    logical_type: _LogicalType,
) -> str:
    return f"CAST({_json_text(expression)} AS {duckdb_type(logical_type)})"


def _prediction_json_columns(contract: OLMESContract) -> dict[str, str]:
    instance_metrics = ", ".join(
        f"{quote_identifier(name)} JSON" for name in contract.metrics.detailed_instances
    )
    choice_metrics = ", ".join(
        f"{quote_identifier(name)} JSON" for name in contract.metrics.detailed_choices
    )
    return {
        "doc_id": "JSON",
        "native_id": "JSON",
        "metrics": f"STRUCT({instance_metrics})",
        "model_output": f"STRUCT({choice_metrics})[]",
        "label": "JSON",
        "task_hash": "JSON",
        "model_hash": "JSON",
    }


def _validation_case(
    fields: list[tuple[str, str, _LogicalType, bool]],
) -> str:
    clauses = [
        f"WHEN NOT {_valid_json_value(expression, logical_type=logical_type, nullable=nullable)} "
        f"THEN {sql_literal(name)}"
        for name, expression, logical_type, nullable in fields
    ]
    return "CASE " + " ".join(clauses) + " END"


def _raise_invalid_prediction_value(
    connection: duckdb.DuckDBPyConnection,
    *,
    contract: OLMESContract,
    context: str,
) -> None:
    instance_meta = {
        column.name: column for column in contract.tables.detailed_instances.columns
    }
    instance_fields = cast(
        list[tuple[str, str, _LogicalType, bool]],
        [
            ("doc_id", "p.doc_id", _INT64_LOGICAL_TYPE, False),
            ("label", "p.label", _INT64_LOGICAL_TYPE, False),
            ("task_hash", "p.task_hash", _STRING_LOGICAL_TYPE, False),
            ("model_hash", "p.model_hash", _STRING_LOGICAL_TYPE, False),
        ],
    )
    instance_fields.extend(
        (
            name,
            f"p.metrics.{quote_identifier(name)}",
            instance_meta[name].logical_type,
            instance_meta[name].nullable,
        )
        for name in contract.metrics.detailed_instances
    )
    instance_case = _validation_case(instance_fields)
    invalid_instance = connection.execute(
        f"""
        SELECT p.task, p.filename, {instance_case} AS invalid_field
        FROM _checkpoint_predictions AS p
        WHERE p.metrics IS NULL
           OR p.model_output IS NULL
           OR {instance_case} IS NOT NULL
           OR NOT (
                p.native_id IS NULL
                OR json_type(p.native_id) IN ('VARCHAR', 'BIGINT', 'UBIGINT')
           )
        LIMIT 1
        """
    ).fetchone()
    if invalid_instance is not None:
        task, filename, field = invalid_instance
        field = field or "metrics, model_output, or native_id"
        raise ValueError(
            f"invalid OLMES detail prediction field {field!r} in {context}, "
            f"task={task!r}, file={filename!r}"
        )

    choice_meta = {
        column.name: column for column in contract.tables.detailed_choices.columns
    }
    choice_fields = [
        (
            name,
            f"u.choice.{quote_identifier(name)}",
            choice_meta[name].logical_type,
            choice_meta[name].nullable,
        )
        for name in contract.metrics.detailed_choices
    ]
    choice_case = _validation_case(choice_fields)
    invalid_choice = connection.execute(
        f"""
        SELECT p.task, p.filename, u.ordinality - 1 AS choice_index,
               {choice_case} AS invalid_field
        FROM _checkpoint_predictions AS p,
             UNNEST(p.model_output) WITH ORDINALITY AS u(choice, ordinality)
        WHERE {choice_case} IS NOT NULL
        LIMIT 1
        """
    ).fetchone()
    if invalid_choice is not None:
        task, filename, choice_index, field = invalid_choice
        raise ValueError(
            f"invalid OLMES detail choice field {field!r} in {context}, "
            f"task={task!r}, choice_index={choice_index}, file={filename!r}"
        )


def _validate_checkpoint_predictions(
    connection: duckdb.DuckDBPyConnection,
    *,
    recipe: str,
    params: str,
    seed_value: int,
    step: int,
    contract: OLMESContract,
) -> None:
    context = (
        f"recipe={recipe!r}, params={params!r}, seed_value={seed_value}, step={step}"
    )
    _raise_invalid_prediction_value(connection, contract=contract, context=context)
    hash_mismatch = connection.execute(
        """
        SELECT p.task, p.filename
        FROM _checkpoint_predictions AS p
        JOIN tasks AS t
          ON t.recipe = ? AND t.params = ? AND t.seed_value = ? AND t.step = ?
         AND t.task = p.task
        WHERE json_extract_string(p.task_hash, '$') IS DISTINCT FROM t.task_hash
           OR json_extract_string(p.model_hash, '$') IS DISTINCT FROM t.model_hash
        LIMIT 1
        """,
        [recipe, params, seed_value, step],
    ).fetchone()
    if hash_mismatch is not None:
        raise ValueError(
            f"hash mismatch in {context}, task={hash_mismatch[0]!r}, "
            f"file={hash_mismatch[1]!r}"
        )
    doc_id = _cast_json_value("p.doc_id", "int64")
    duplicate = connection.execute(
        f"""
        SELECT p.task, {doc_id} AS doc_id
        FROM _checkpoint_predictions AS p
        GROUP BY p.task, {doc_id}
        HAVING count(*) > 1
        LIMIT 1
        """
    ).fetchone()
    if duplicate is not None:
        raise ValueError(
            f"duplicate OLMES detail instance row in {context}, "
            f"task={duplicate[0]!r}: doc_id={duplicate[1]}"
        )
    count_mismatch = connection.execute(
        """
        SELECT t.task, t.num_instances, count(p.doc_id)
        FROM tasks AS t
        LEFT JOIN _checkpoint_predictions AS p ON p.task = t.task
        WHERE t.recipe = ? AND t.params = ? AND t.seed_value = ? AND t.step = ?
        GROUP BY t.task, t.num_instances
        HAVING t.num_instances <> count(p.doc_id)
        LIMIT 1
        """,
        [recipe, params, seed_value, step],
    ).fetchone()
    if count_mismatch is not None:
        raise ValueError(
            f"instance count mismatch in {context}, task={count_mismatch[0]!r}: "
            f"declared={count_mismatch[1]}, predictions={count_mismatch[2]}"
        )


def _insert_task_rows(
    connection: duckdb.DuckDBPyConnection,
    *,
    rows: tuple[dict[str, object], ...],
    contract: OLMESContract,
) -> None:
    columns = _output_columns(contract, "detailed_tasks")
    placeholders = ", ".join("?" for _ in columns)
    column_sql = ", ".join(quote_identifier(column) for column in columns)
    connection.executemany(
        f"INSERT INTO tasks ({column_sql}) VALUES ({placeholders})",
        [[row.get(column) for column in columns] for row in rows],
    )


def _instance_select_sql(contract: OLMESContract) -> str:
    metric_meta = {
        column.name: column for column in contract.tables.detailed_instances.columns
    }
    expressions = {
        "recipe": "t.recipe",
        "data": "t.data",
        "params": "t.params",
        "seed_value": "t.seed_value",
        "seed": "t.seed",
        "step": "t.step",
        "task": "t.task",
        "task_hash": "t.task_hash",
        "model_hash": "t.model_hash",
        "doc_id": _cast_json_value("p.doc_id", "int64"),
        "native_id": (
            "CASE WHEN p.native_id IS NULL THEN NULL "
            "ELSE json_extract_string(p.native_id, '$') END"
        ),
        "native_id_kind": (
            "CASE json_type(p.native_id) "
            "WHEN 'VARCHAR' THEN 'string' "
            "WHEN 'BIGINT' THEN 'integer' "
            "WHEN 'UBIGINT' THEN 'integer' "
            "ELSE 'null' END"
        ),
        "label": _cast_json_value("p.label", "int64"),
    }
    expressions.update(
        {
            name: _cast_json_value(
                f"p.metrics.{quote_identifier(name)}", metric_meta[name].logical_type
            )
            for name in contract.metrics.detailed_instances
        }
    )
    columns = _output_columns(contract, "detailed_instances")
    select_list = ",\n               ".join(
        f"{expressions[name]} AS {quote_identifier(name)}" for name in columns
    )
    return f"""
        SELECT {select_list}
        FROM _checkpoint_predictions AS p
        JOIN tasks AS t
          ON t.recipe = ? AND t.params = ? AND t.seed_value = ? AND t.step = ?
         AND t.task = p.task
    """


def _choice_select_sql(contract: OLMESContract) -> str:
    metric_meta = {
        column.name: column for column in contract.tables.detailed_choices.columns
    }
    expressions = {
        "recipe": "t.recipe",
        "data": "t.data",
        "params": "t.params",
        "seed_value": "t.seed_value",
        "seed": "t.seed",
        "step": "t.step",
        "task": "t.task",
        "doc_id": _cast_json_value("p.doc_id", "int64"),
        "choice_index": "u.ordinality - 1",
    }
    expressions.update(
        {
            name: _cast_json_value(
                f"u.choice.{quote_identifier(name)}", metric_meta[name].logical_type
            )
            for name in contract.metrics.detailed_choices
        }
    )
    columns = _output_columns(contract, "detailed_choices")
    select_list = ",\n               ".join(
        f"{expressions[name]} AS {quote_identifier(name)}" for name in columns
    )
    return f"""
        SELECT {select_list}
        FROM _checkpoint_predictions AS p
        JOIN tasks AS t
          ON t.recipe = ? AND t.params = ? AND t.seed_value = ? AND t.step = ?
         AND t.task = p.task
        CROSS JOIN UNNEST(p.model_output) WITH ORDINALITY AS u(choice, ordinality)
    """


def _drop_checkpoint_relations(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute("DROP VIEW IF EXISTS _checkpoint_json")
    connection.execute("DROP TABLE IF EXISTS _checkpoint_predictions")
    connection.execute("DROP TABLE IF EXISTS _checkpoint_files")


def _empty_checkpoint_predictions_sql(contract: OLMESContract) -> str:
    fields = [
        "CAST(NULL AS VARCHAR) AS task",
        "CAST(NULL AS JSON) AS doc_id",
        "CAST(NULL AS JSON) AS native_id",
    ]
    instance_metrics = ", ".join(
        f"{quote_identifier(name)} JSON" for name in contract.metrics.detailed_instances
    )
    choice_metrics = ", ".join(
        f"{quote_identifier(name)} JSON" for name in contract.metrics.detailed_choices
    )
    fields.extend(
        (
            f"CAST(NULL AS STRUCT({instance_metrics})) AS metrics",
            f"CAST(NULL AS STRUCT({choice_metrics})[]) AS model_output",
            "CAST(NULL AS JSON) AS label",
            "CAST(NULL AS JSON) AS task_hash",
            "CAST(NULL AS JSON) AS model_hash",
            "CAST(NULL AS VARCHAR) AS filename",
        )
    )
    return "SELECT " + ", ".join(fields) + " WHERE false"


def _ingest_checkpoint(
    connection: duckdb.DuckDBPyConnection,
    memory_filesystem: MemoryFileSystem,
    *,
    payload: _CheckpointPayload,
    recipe: str,
    params: str,
    seed_value: int,
    step: int,
    member_name: str,
    contract: OLMESContract,
) -> tuple[int, int, int]:
    checkpoint_token = uuid4().hex
    file_rows: list[tuple[str, str]] = []
    memory_paths: list[str] = []
    for index, (task, prediction_bytes) in enumerate(
        payload.predictions_by_task.items()
    ):
        memory_path = f"{_MEMORY_ROOT}/{checkpoint_token}/{index}.jsonl"
        memory_url = f"memory://{memory_path}"
        memory_filesystem.pipe_file(memory_path, prediction_bytes)
        memory_paths.append(memory_path)
        file_rows.append((memory_url, task))

    instance_count = 0
    choice_count = 0
    connection.execute("BEGIN TRANSACTION")
    try:
        _drop_checkpoint_relations(connection)
        _insert_task_rows(connection, rows=payload.task_rows, contract=contract)
        connection.execute(
            """
            CREATE TEMP TABLE _checkpoint_files (
                filename VARCHAR PRIMARY KEY,
                task VARCHAR NOT NULL
            )
            """
        )
        if file_rows:
            connection.executemany(
                "INSERT INTO _checkpoint_files VALUES (?, ?)", file_rows
            )
            relation = connection.read_json(
                cast(Any, [filename for filename, _ in file_rows]),
                columns=_prediction_json_columns(contract),
                format="newline_delimited",
                filename=True,
            )
            relation.create_view("_checkpoint_json", replace=True)
            connection.execute(
                """
                CREATE TEMP TABLE _checkpoint_predictions AS
                SELECT files.task, raw.*
                FROM _checkpoint_json AS raw
                JOIN _checkpoint_files AS files USING (filename)
                """
            )
        else:
            connection.execute(
                "CREATE TEMP TABLE _checkpoint_predictions AS "
                + _empty_checkpoint_predictions_sql(contract)
            )

        _validate_checkpoint_predictions(
            connection,
            recipe=recipe,
            params=params,
            seed_value=seed_value,
            step=step,
            contract=contract,
        )
        parameters = [recipe, params, seed_value, step]
        connection.execute(
            f"INSERT INTO instances {_instance_select_sql(contract)}", parameters
        )
        connection.execute(
            f"INSERT INTO choices {_choice_select_sql(contract)}", parameters
        )
        counts = connection.execute(
            """
            SELECT count(*), coalesce(sum(len(model_output)), 0)
            FROM _checkpoint_predictions
            """
        ).fetchone()
        assert counts is not None
        instance_count, choice_count = counts
        task_count = len(payload.task_rows)
        connection.execute(
            """
            INSERT INTO completed_checkpoints
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                recipe,
                params,
                seed_value,
                step,
                member_name,
                task_count,
                instance_count,
                choice_count,
            ],
        )
        connection.execute("COMMIT")
    except BaseException as exc:
        connection.execute("ROLLBACK")
        if isinstance(exc, duckdb.Error):
            context = (
                f"recipe={recipe!r}, params={params!r}, "
                f"seed_value={seed_value}, step={step}"
            )
            raise ValueError(
                f"invalid OLMES detail checkpoint data in {context}: {exc}"
            ) from exc
        raise
    finally:
        _drop_checkpoint_relations(connection)
        for memory_path in memory_paths:
            if memory_filesystem.isfile(memory_path):
                memory_filesystem.rm_file(memory_path)
    return task_count, int(instance_count), int(choice_count)


def _validate_staging_counts(connection: duckdb.DuckDBPyConnection) -> None:
    actual = connection.execute(
        """
        SELECT
            (SELECT count(*) FROM tasks),
            (SELECT count(*) FROM instances),
            (SELECT count(*) FROM choices)
        """
    ).fetchone()
    expected = connection.execute(
        """
        SELECT
            coalesce(sum(task_count), 0),
            coalesce(sum(instance_count), 0),
            coalesce(sum(choice_count), 0)
        FROM completed_checkpoints
        """
    ).fetchone()
    if actual != expected:
        raise ValueError(
            "OLMES detail staging counts do not match committed checkpoints: "
            f"expected={expected!r}, actual={actual!r}"
        )


def _export_parquet(
    connection: duckdb.DuckDBPyConnection,
    *,
    table_name: str,
    table: OLMESTableContract,
    output_path: Path,
) -> PendingParquetExport:
    columns = ", ".join(quote_identifier(column.name) for column in table.columns)
    sort_key = ", ".join(quote_identifier(column) for column in table.sort_key)
    return prepare_parquet_export(
        connection,
        select_sql=f"""
            SELECT {columns}
            FROM {quote_identifier(table_name)}
            ORDER BY {sort_key}
        """,
        output_path=output_path,
    )


def _finalize_outputs(
    connection: duckdb.DuckDBPyConnection,
    *,
    contract: OLMESContract,
    tasks_path: Path,
    instances_path: Path,
    choices_path: Path,
) -> tuple[int, int, int]:
    _validate_staging_counts(connection)
    exports = (
        _export_parquet(
            connection,
            table_name="tasks",
            table=contract.tables.detailed_tasks,
            output_path=tasks_path,
        ),
        _export_parquet(
            connection,
            table_name="instances",
            table=contract.tables.detailed_instances,
            output_path=instances_path,
        ),
        _export_parquet(
            connection,
            table_name="choices",
            table=contract.tables.detailed_choices,
            output_path=choices_path,
        ),
    )
    replace_parquet_exports(exports)
    return tuple(export.row_count for export in exports)


def _remove_completed_staging_database(staging_path: Path) -> None:
    for path in (
        staging_path,
        staging_path.with_name(f"{staging_path.name}.wal"),
    ):
        remove_owned_file(path)
    temporary_directory = Path(f"{staging_path}.tmp")
    if temporary_directory.exists():
        if temporary_directory.is_symlink() or not temporary_directory.is_dir():
            raise ValueError(
                f"unexpected DuckDB temporary path type: {temporary_directory}"
            )
        temporary_directory.rmdir()


def preprocess_olmes_details(
    paths: DataArtifacts,
    recipe: str,
    *,
    contract: OLMESContract | None = None,
    input_path: Path | None = None,
    output_tasks_path: Path | None = None,
    output_instances_path: Path | None = None,
    output_choices_path: Path | None = None,
    verbose: bool = False,
) -> OlmesDetailsPreprocessResult:
    contract = contract or load_olmes_contract()
    _assert_all_detailed_schema_parity(contract)
    resolved_input = input_path or _recipe_tar_path(paths, recipe)
    resolved_tasks = output_tasks_path or paths.olmes_details_tasks_path(recipe)
    resolved_instances = output_instances_path or paths.olmes_details_instances_path(
        recipe
    )
    resolved_choices = output_choices_path or paths.olmes_details_choices_path(recipe)
    if not resolved_input.is_file():
        raise FileNotFoundError(f"OLMES detail archive not found: {resolved_input}")

    resolved_tasks.parent.mkdir(parents=True, exist_ok=True)
    staging_path = _staging_database_path(resolved_tasks)
    connection = duckdb.connect(str(staging_path))
    memory_filesystem = MemoryFileSystem()
    connection.register_filesystem(memory_filesystem)
    completed_successfully = False
    started = time.monotonic()
    try:
        completed = _initialize_staging_database(
            connection,
            staging_path=staging_path,
            input_path=resolved_input,
            recipe=recipe,
            contract=contract,
        )
        initial_completed_count = len(completed)
        if verbose and initial_completed_count:
            print(
                f"olmes-details resuming after {initial_completed_count} "
                "committed checkpoints"
            )

        seen_checkpoints: set[tuple[str, str, int, int]] = set()
        processed_count = 0
        with tarfile.open(resolved_input, mode="r:gz") as outer:
            for member in outer:
                if not member.isfile():
                    continue
                parsed = _parse_checkpoint_member_path(
                    member.name, expected_recipe=recipe
                )
                if parsed is None:
                    continue
                checkpoint_key = parsed
                if checkpoint_key in seen_checkpoints:
                    _, params, seed_value, step = checkpoint_key
                    raise ValueError(
                        "duplicate OLMES detail task row in "
                        f"recipe={recipe!r}, params={params!r}, "
                        f"seed_value={seed_value}, step={step}"
                    )
                seen_checkpoints.add(checkpoint_key)
                if checkpoint_key in completed:
                    continue

                _, params, seed_value, step = checkpoint_key
                inner_file = outer.extractfile(member)
                if inner_file is None:
                    raise ValueError(
                        f"unable to read checkpoint member from {resolved_input}: "
                        f"{member.name!r}"
                    )
                payload = _read_checkpoint_payload(
                    inner_file.read(),
                    recipe=recipe,
                    params=params,
                    seed_value=seed_value,
                    step=step,
                    contract=contract,
                )
                _ingest_checkpoint(
                    connection,
                    memory_filesystem,
                    payload=payload,
                    recipe=recipe,
                    params=params,
                    seed_value=seed_value,
                    step=step,
                    member_name=member.name,
                    contract=contract,
                )
                completed.add(checkpoint_key)
                processed_count += 1
                if verbose and (processed_count == 1 or processed_count % 10 == 0):
                    elapsed = time.monotonic() - started
                    print(
                        f"olmes-details committed {processed_count} new checkpoints "
                        f"({len(completed)} total) in {elapsed:.1f}s"
                    )

        if not seen_checkpoints:
            raise ValueError(
                "OLMES detail archive contains no recognized checkpoints: "
                f"{resolved_input}"
            )

        stored_checkpoints = {
            (row[0], row[1], row[2], row[3])
            for row in connection.execute(
                "SELECT recipe, params, seed_value, step FROM completed_checkpoints"
            ).fetchall()
        }
        if stored_checkpoints != seen_checkpoints:
            missing = sorted(stored_checkpoints - seen_checkpoints)
            raise ValueError(
                "OLMES detail staging database contains checkpoints absent from "
                f"the source archive: {missing[:5]!r}"
            )
        task_count, instance_count, choice_count = _finalize_outputs(
            connection,
            contract=contract,
            tasks_path=resolved_tasks,
            instances_path=resolved_instances,
            choices_path=resolved_choices,
        )
        checkpoint_count = len(stored_checkpoints)
        completed_successfully = True
    finally:
        connection.close()

    if completed_successfully:
        _remove_completed_staging_database(staging_path)

    result = OlmesDetailsPreprocessResult(
        recipe=recipe,
        input_path=resolved_input,
        output_tasks_path=resolved_tasks,
        output_instances_path=resolved_instances,
        output_choices_path=resolved_choices,
        row_count=task_count,
        instance_count=instance_count,
        choice_count=choice_count,
        checkpoint_count=checkpoint_count,
    )
    if verbose:
        print(f"olmes-details recipe: {result.recipe}")
        print(f"olmes-details input: {result.input_path}")
        print(f"olmes-details tasks output: {result.output_tasks_path}")
        print(f"olmes-details instances output: {result.output_instances_path}")
        print(f"olmes-details choices output: {result.output_choices_path}")
        print(f"olmes-details task rows: {result.row_count}")
        print(f"olmes-details instance rows: {result.instance_count}")
        print(f"olmes-details choice rows: {result.choice_count}")
        print(f"olmes-details checkpoints: {result.checkpoint_count}")
        print(f"olmes-details elapsed seconds: {time.monotonic() - started:.3f}")
    return result
