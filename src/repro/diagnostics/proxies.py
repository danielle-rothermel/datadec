"""Metric advantages at identical checkpoints."""

import pandas as pd

from repro.approaches import approach_rankings


def proxy_comparisons(sweep: pd.DataFrame, *, baseline_metric: str) -> pd.DataFrame:
    """Align proxies with curated target Accuracy at exactly the same checkpoint."""
    sweep = approach_rankings(sweep)
    available = sweep.loc[sweep["available"]]
    key = ["approach", "predictor_size", "predictor_step", "task"]
    baseline = available.loc[
        available["metric"] == baseline_metric, [*key, "decision_accuracy"]
    ].rename(columns={"decision_accuracy": "primary_accuracy"})
    result = available.reset_index().merge(
        baseline, on=key, how="left", validate="many_to_one"
    )
    result["advantage_over_primary"] = (
        result["decision_accuracy"] - result["primary_accuracy"]
    )
    return result[
        [
            "evidence_id",
            *key,
            "metric",
            "compute_ratio",
            "decision_accuracy",
            "primary_accuracy",
            "advantage_over_primary",
        ]
    ]
