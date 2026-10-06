from __future__ import annotations

import duckdb
import pyarrow as pa

from datadec.config import (
    CHECKPOINT_ENRICHMENT_COLUMNS,
    CHECKPOINT_ENRICHMENT_TYPES,
    MODEL_DETAIL_TYPES,
    CheckpointLogicalType,
)
from datadec.data.model_utils import (
    MODEL_DETAIL_COLUMNS,
    checkpoint_enrichment,
)
from datadec.data.preprocess.duckdb import quote_identifier

_ARROW_TYPES: dict[CheckpointLogicalType, pa.DataType] = {
    "string": pa.string(),
    "int64": pa.int64(),
    "float64": pa.float64(),
    "bool": pa.bool_(),
}


def create_model_enrichment_table(
    connection: duckdb.DuckDBPyConnection,
    *,
    checkpoint_select_sql: str,
) -> None:
    checkpoints = connection.execute(
        f"""
        SELECT DISTINCT
            CAST(params AS VARCHAR) AS params,
            CAST(step AS BIGINT) AS step
        FROM ({checkpoint_select_sql})
        ORDER BY params, step
        """
    ).fetchall()
    rows = [
        {
            "params": str(params),
            "step": int(step),
            **checkpoint_enrichment(str(params), int(step)),
        }
        for params, step in checkpoints
    ]
    schema = pa.schema(
        [
            pa.field("params", pa.string(), nullable=False),
            pa.field("step", pa.int64(), nullable=False),
        ]
        + [
            pa.field(name, _ARROW_TYPES[logical_type], nullable=False)
            for name, logical_type in CHECKPOINT_ENRICHMENT_TYPES
        ]
    )
    relation_name = "_model_enrichment_arrow"
    connection.register(relation_name, pa.Table.from_pylist(rows, schema=schema))
    try:
        connection.execute(
            f"CREATE TEMP TABLE _model_enrichment AS SELECT * FROM {relation_name}"
        )
        connection.execute(
            "CREATE UNIQUE INDEX _model_enrichment_key "
            "ON _model_enrichment (params, step)"
        )
    finally:
        connection.unregister(relation_name)


def enrichment_select_expressions(*, table_alias: str) -> str:
    return ", ".join(
        f"{table_alias}.{quote_identifier(column)} AS {quote_identifier(column)}"
        for column in CHECKPOINT_ENRICHMENT_COLUMNS
    )


assert tuple(name for name, _ in MODEL_DETAIL_TYPES) == MODEL_DETAIL_COLUMNS

__all__ = [
    "create_model_enrichment_table",
    "enrichment_select_expressions",
]
