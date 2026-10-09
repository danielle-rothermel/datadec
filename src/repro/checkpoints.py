"""Observed checkpoint schedules and compute-budget selection."""

import numpy as np
import pandas as pd


def observed_checkpoints(evaluations: pd.DataFrame) -> pd.DataFrame:
    """List actual checkpoint compute and whether the scheduled budget is reached."""
    rows = evaluations.reset_index(drop=True)
    metadata = ["compute", "total_steps"]
    if (
        (rows.groupby(["params", "step"])[metadata].nunique(dropna=False) != 1)
        .any()
        .any()
    ):
        raise ValueError("inconsistent checkpoint compute or schedule")
    result = rows[["params", "step", *metadata]].drop_duplicates().copy()
    result["schedule_complete"] = result["step"] >= result["total_steps"]
    return result.sort_values(["compute", "params", "step"]).reset_index(drop=True)


def select_compute_budget(
    checkpoints: pd.DataFrame,
    size: str,
    budget: float,
    *,
    relative_tolerance: float | None = None,
) -> int | None:
    """Largest observed positive-compute checkpoint <= budget, never interpolate.

    For approximate matched-compute comparisons, a supplied relative tolerance
    bounds the undershoot: (budget - actual) / budget. No match returns None.
    """
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError("compute budget must be finite and positive")
    if relative_tolerance is not None and not 0 <= relative_tolerance <= 1:
        raise ValueError("relative tolerance must be between 0 and 1")
    eligible = checkpoints.loc[
        (checkpoints["params"] == size)
        & (checkpoints["compute"] > 0)
        & (checkpoints["compute"] <= budget)
    ].sort_values("compute")
    if eligible.empty:
        return None
    chosen = eligible.iloc[-1]
    if (
        relative_tolerance is not None
        and (budget - chosen["compute"]) / budget > relative_tolerance
    ):
        return None
    return int(chosen["step"])
