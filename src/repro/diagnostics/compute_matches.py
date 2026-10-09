"""Intermediate versus completed checkpoints at comparable compute."""

import pandas as pd

from repro.approaches import approach_rankings
from repro.checkpoints import select_compute_budget


def matched_compute_comparisons(
    sweep: pd.DataFrame, *, baseline_metric: str, relative_tolerance: float
) -> pd.DataFrame:
    """Compare observed scheduled endpoints to other sizes within a budget tolerance."""
    sweep = approach_rankings(sweep)
    checkpoints = sweep.rename(
        columns={"predictor_size": "params", "predictor_step": "step"}
    )[["params", "step", "compute"]].drop_duplicates()
    primary = sweep.loc[sweep["available"] & (sweep["metric"] == baseline_metric)]
    records = []
    for endpoint in primary.loc[primary["schedule_complete"]].itertuples():
        for size in checkpoints["params"].unique():
            if size == endpoint.predictor_size:
                continue
            step = select_compute_budget(
                checkpoints,
                size,
                endpoint.compute,
                relative_tolerance=relative_tolerance,
            )
            match = primary.loc[
                (primary["predictor_size"] == size)
                & (primary["predictor_step"] == step)
                & (primary["task"] == endpoint.task)
                & (primary["approach"] == endpoint.approach)
                & ~primary["schedule_complete"]
            ]
            if match.empty:
                continue
            candidate = match.iloc[0]
            records.append(
                {
                    "approach": endpoint.approach,
                    "final_evidence_id": endpoint.Index,
                    "intermediate_evidence_id": int(candidate.name),
                    "relative_compute_gap": 1 - candidate["compute"] / endpoint.compute,
                    "accuracy_difference": candidate["decision_accuracy"]
                    - endpoint.decision_accuracy,
                }
            )
    return pd.DataFrame(
        records,
        columns=[
            "approach",
            "final_evidence_id",
            "intermediate_evidence_id",
            "relative_compute_gap",
            "accuracy_difference",
        ],
    )
