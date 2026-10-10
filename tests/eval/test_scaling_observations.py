import pandas as pd
import pytest

from eval.scaling_observations import scaling_law_observations


def observations():
    evaluations = pd.DataFrame(
        [
            dict(
                data="C4",
                params="4M",
                seed="a",
                step=1,
                task="task",
                compute=2.0,
                accuracy=0.2,
            ),
            dict(
                data="C4",
                params="4M",
                seed="b",
                step=1,
                task="task",
                compute=2.0,
                accuracy=0.4,
            ),
        ]
    )
    losses = pd.DataFrame(
        [dict(data="C4", params="4M", seed="a", step=1, compute=2.0, cross_entropy=3.0)]
    )
    return evaluations, losses


def test_alignment_retains_unmatched_seeds_without_borrowing_loss_values():
    evaluations, losses = observations()
    output = scaling_law_observations(
        evaluations, losses, metric="accuracy", loss_metric="cross_entropy"
    )
    assert output.available.tolist() == [True, False]
    assert output.loc[0, "loss"] == 3.0
    assert pd.isna(output.loc[1, "loss"])
    assert output["metric"].tolist() == ["accuracy"] * 2


def test_alignment_rejects_compute_conflicts_and_duplicate_loss_keys():
    evaluations, losses = observations()
    with pytest.raises(ValueError, match="compute disagree"):
        scaling_law_observations(
            evaluations,
            losses.assign(compute=3.0),
            metric="accuracy",
            loss_metric="cross_entropy",
        )
    with pytest.raises(ValueError, match="duplicate"):
        scaling_law_observations(
            evaluations,
            pd.concat([losses, losses]),
            metric="accuracy",
            loss_metric="cross_entropy",
        )
