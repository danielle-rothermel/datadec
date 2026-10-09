"""Select exact checkpoint rows and validate recipe/seed score coverage."""

import numpy as np
import pandas as pd

from datadec.recipes import DataRecipeName


class UnavailableRankingError(ValueError):
    """The requested checkpoint, recipes, seeds, or finite scores are missing."""


def checkpoint_rows(
    evaluations: pd.DataFrame, size: str, step: int, task: str
) -> pd.DataFrame:
    if isinstance(evaluations.index, pd.MultiIndex) and evaluations.index.names == [
        "params",
        "step",
        "task",
    ]:
        try:
            return evaluations.loc[[(size, step, task)]]
        except KeyError as error:
            raise UnavailableRankingError(
                f"missing checkpoint: {size}, step {step}, {task}"
            ) from error
    rows = evaluations.loc[
        (evaluations["params"] == size)
        & (evaluations["step"] == step)
        & (evaluations["task"] == task)
    ]
    if rows.empty:
        raise UnavailableRankingError(
            f"missing checkpoint: {size}, step {step}, {task}"
        )
    return rows


def checkpoint_scores(
    rows: pd.DataFrame,
    metric: str,
    recipes: tuple[DataRecipeName, ...],
    seeds: tuple[str, ...],
) -> tuple[pd.DataFrame, float]:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and unique")
    selected = rows.loc[rows["seed"].isin(seeds)]
    if selected.duplicated(["data", "seed"]).any():
        raise ValueError("duplicate recipe/seed rows at checkpoint")
    scores = selected.pivot(index="data", columns="seed", values=metric)
    if set(scores.index) != set(recipes) or set(scores.columns) != set(seeds):
        raise UnavailableRankingError("checkpoint does not contain all recipes/seeds")
    scores = scores.reindex(index=list(recipes), columns=list(seeds))
    if not np.isfinite(scores.to_numpy(dtype=float, na_value=np.nan)).all():
        raise UnavailableRankingError(f"missing or non-finite {metric} scores")
    compute = selected["compute"].unique()
    if len(compute) != 1 or not np.isfinite(compute[0]) or compute[0] < 0:
        raise ValueError("checkpoint compute must be one finite nonnegative value")
    return scores, float(compute[0])
