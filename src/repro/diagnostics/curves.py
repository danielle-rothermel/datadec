"""Descriptive compute/accuracy trends."""

import numpy as np
import pandas as pd


def curve_summary(sweep: pd.DataFrame) -> pd.DataFrame:
    """Descriptive fits and reversals per task/metric/size; no significance claims."""
    records = []
    available = sweep.loc[sweep["available"]]
    for (task, metric, size), group in available.groupby(
        ["task", "metric", "predictor_size"]
    ):
        group = group.sort_values("compute")
        x = np.log10(group["compute_ratio"].to_numpy(dtype=float))
        y = group["decision_accuracy"].to_numpy(dtype=float)
        slope = r_squared = None
        if len(x) >= 2 and np.ptp(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            variance = float(np.sum((y - y.mean()) ** 2))
            if np.ptp(y) > 0:
                r_squared = (
                    1 - float(np.sum((y - (slope * x + intercept)) ** 2)) / variance
                )
        records.append(
            {
                "task": task,
                "metric": metric,
                "predictor_size": size,
                "checkpoints": len(group),
                "log10_compute_slope": slope,
                "r_squared": r_squared,
                "adjacent_decreases": int(np.sum(np.diff(y) < 0)),
                "first_accuracy": float(y[0]),
                "last_accuracy": float(y[-1]),
                "max_accuracy": float(y.max()),
            }
        )
    return pd.DataFrame.from_records(records)
