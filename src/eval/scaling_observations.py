"""Align evaluation and loss observations without interpolating checkpoints."""

import numpy as np
import pandas as pd


def scaling_law_observations(
    evaluations: pd.DataFrame,
    checkpoint_losses: pd.DataFrame,
    *,
    metric: str,
    loss_metric: str,
) -> pd.DataFrame:
    keys = ["data", "params", "seed", "step"]
    if (
        evaluations.duplicated([*keys, "task"]).any()
        or checkpoint_losses.duplicated(keys).any()
    ):
        raise ValueError("duplicate scaling-law observation identity")
    if (
        evaluations[[*keys, "task"]].isna().any().any()
        or checkpoint_losses[keys].isna().any().any()
    ):
        raise ValueError("scaling-law identity cannot be null")
    values = evaluations.rename(
        columns={
            metric: "score",
            "compute": "evaluation_compute",
            "source_file": "evaluation_source",
        }
    )
    losses = checkpoint_losses.rename(
        columns={
            loss_metric: "loss",
            "compute": "loss_compute",
            "source_file": "loss_source",
        }
    )
    left = [*keys, "task", "score", "evaluation_compute"]
    right = [*keys, "loss", "loss_compute"]
    if "evaluation_source" in values:
        left.append("evaluation_source")
    if "loss_source" in losses:
        right.append("loss_source")
    joined = values[left].merge(
        losses[right], on=keys, how="left", validate="many_to_one", indicator=True
    )
    matched = joined["_merge"] == "both"
    if (
        joined.loc[matched, "evaluation_compute"] != joined.loc[matched, "loss_compute"]
    ).any():
        raise ValueError("scaling-law evaluation and loss compute disagree")
    joined["available"] = (
        matched & np.isfinite(joined["score"]) & np.isfinite(joined["loss"])
    )
    joined["metric"] = metric
    joined["loss_metric"] = loss_metric
    return joined.drop(columns="_merge")
