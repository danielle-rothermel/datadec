"""Read local evaluations and construct complete subject/benchmark macro means."""

from pathlib import Path

import pandas as pd

from datadec.data.artifacts import DataArtifacts
from datadec.data.read import ProcessedTable, read_processed_table
from repro.config import EvaluationConfig

# Processed OLMES checkpoint identity, independent of experiment selections.
KEY = ["params", "step", "data", "seed"]


def _macro_average(
    rows: pd.DataFrame, task: str, count: int, metrics: tuple[str, ...]
) -> pd.DataFrame:
    groups = rows.groupby(KEY, observed=True, sort=False)
    metadata = ["compute", "total_steps"]
    if (groups["task"].nunique() != count).any():
        raise ValueError(
            f"{task} requires all {count} component tasks at every checkpoint"
        )
    if (groups[metadata].nunique(dropna=False) != 1).any().any():
        raise ValueError(f"inconsistent checkpoint metadata across {task} tasks")
    # A metric is unavailable for the macro task if any component lacks it.
    means = groups[list(metrics)].mean().where(groups[list(metrics)].count() == count)
    result = means.join(groups[metadata].first()).reset_index()
    result["task"] = task
    return result


def prepare_evaluations(raw: pd.DataFrame, config: EvaluationConfig) -> pd.DataFrame:
    """Macro-average configured subjects, then benchmarks with equal weights.

    Return a sorted checkpoint/task index for repeated generic helper calls.
    Missing task coverage fails explicitly; missing metric values stay missing.
    """
    required = [*KEY, "task", "compute", "total_steps", *config.metrics]
    rows = raw[required].copy()
    if rows[[*KEY, "task"]].isna().any().any():
        raise ValueError("checkpoint identity columns cannot be null")
    if rows.duplicated([*KEY, "task"]).any():
        raise ValueError("duplicate evaluation identity")
    subjects = rows.loc[rows["task"].str.startswith(config.aggregation.subject_prefix)]
    if subjects["task"].nunique() != config.aggregation.subject_count:
        raise ValueError(
            f"expected the {config.aggregation.subject_count} {config.aggregation.subject_task} subjects"
        )
    singles = rows.loc[
        rows["task"].isin(set(config.benchmarks) - {config.aggregation.subject_task})
    ]
    benchmarks = pd.concat(
        [
            singles,
            _macro_average(
                subjects,
                config.aggregation.subject_task,
                config.aggregation.subject_count,
                config.metrics,
            ),
        ]
    )
    macro = _macro_average(
        benchmarks,
        config.aggregation.macro_task,
        len(config.benchmarks),
        config.metrics,
    )
    return (
        pd.concat([benchmarks, macro], ignore_index=True)
        .set_index(["params", "step", "task"], drop=False)
        .sort_index()
    )


def load_evaluations(data_dir: Path, config: EvaluationConfig) -> pd.DataFrame:
    """Read the local processed OLMES table through the standard data API."""
    table = read_processed_table(
        DataArtifacts(data_dir),
        ProcessedTable.OLMES,
        columns=[*KEY, "task", "compute", "total_steps", *config.metrics],
    )
    return prepare_evaluations(table.to_pandas(), config)
