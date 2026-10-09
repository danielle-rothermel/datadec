"""Recipe-order reversals across observed completed scales."""

import numpy as np
import pandas as pd

from datadec.config import load_olmes_contract


def recipe_crossovers(
    evaluations: pd.DataFrame, *, tasks: tuple[str, ...], metric: str
) -> pd.DataFrame:
    """Count strict recipe-order reversals across adjacent completed model scales.

    Seeds are averaged per recipe at each scale. Only observed checkpoints that
    reach their configured training budget enter this diagnostic. Exact ties do
    not count as strict crossovers. The diagnostic does not fit scaling laws.
    """
    rows = evaluations.reset_index(drop=True)
    rows = rows.loc[(rows["step"] >= rows["total_steps"]) & rows["task"].isin(tasks)]
    scores = (
        rows.groupby(["task", "params", "compute", "data"])[metric]
        .mean()
        .unstack("data")
    )
    expected_recipes = set(load_olmes_contract().recipe_map.values())
    if set(scores.columns) != expected_recipes:
        raise ValueError("crossover diagnostic requires all catalog recipes")
    records = []
    for task, group in scores.groupby(level="task"):
        group = group.sort_index(level="compute")
        if group.isna().any().any():
            raise ValueError("incomplete recipe coverage for crossover diagnostic")
        a, b = np.triu_indices(len(group.columns), k=1)
        values = group.to_numpy()
        signs = np.sign(values[:, a] - values[:, b])
        for index in range(1, len(group)):
            previous, current = group.index[index - 1], group.index[index]
            records.append(
                {
                    "task": task,
                    "smaller_size": previous[1],
                    "larger_size": current[1],
                    "smaller_compute": previous[2],
                    "larger_compute": current[2],
                    "pair_count": len(a),
                    "strict_crossovers": int(
                        np.sum(signs[index - 1] * signs[index] < 0)
                    ),
                }
            )
    return pd.DataFrame.from_records(records)
