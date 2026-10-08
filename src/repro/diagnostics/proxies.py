"""Metric advantages at identical checkpoints."""

import pandas as pd

from repro.config import EvaluationConfig


def proxy_comparisons(sweep: pd.DataFrame, config: EvaluationConfig) -> pd.DataFrame:
    """Align proxies with curated target Accuracy at exactly the same checkpoint."""
    available = sweep.loc[sweep["available"]]
    key = ["predictor_size", "predictor_step", "task"]
    baseline = available.loc[
        available["metric"] == config.target.metric, [*key, "decision_accuracy"]
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
