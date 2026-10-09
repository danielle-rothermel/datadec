from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from eval.results import Checkpoint
from repro.observations import (
    RankingObservation,
    RankingStatistics,
    observations_frame,
    observations_from_frame,
)


def test_observation_boundary_pins_sweep_schema_and_unavailable_nulls(tmp_path):
    measured = RankingObservation(
        predictor=Checkpoint("4M", 5, 10.0),
        target=Checkpoint("1B", 20, 100.0),
        task="mmlu",
        metric="primary_metric",
        schedule_complete=False,
        statistics=RankingStatistics((0.5, 1.0), 2),
        reason="",
    )
    missing = RankingObservation.unavailable(
        measured.predictor,
        measured.target,
        measured.task,
        "correct_prob",
        schedule_complete=False,
        reason="missing checkpoint",
    )
    frame = observations_frame((measured, missing))
    assert frame.index.name == "evidence_id"
    assert frame.iloc[0].to_dict() == {
        "predictor_size": "4M",
        "predictor_step": 5,
        "target_size": "1B",
        "target_step": 20,
        "task": "mmlu",
        "metric": "primary_metric",
        "compute": 10.0,
        "target_compute": 100.0,
        "compute_ratio": 0.1,
        "schedule_complete": False,
        "available": True,
        "reason": "",
        "decision_accuracy": 0.75,
        "decision_accuracy_std": 0.25,
        "seed_accuracies": [0.5, 1.0],
        "recipe_count": 2.0,
        "pair_count": 1.0,
    }
    assert not frame.loc[1, "available"]
    assert frame.loc[1, "reason"] == "missing checkpoint"
    assert (
        frame.loc[
            1,
            [
                "decision_accuracy",
                "decision_accuracy_std",
                "seed_accuracies",
                "recipe_count",
                "pair_count",
            ],
        ]
        .isna()
        .all()
    )
    path = tmp_path / "rankings.parquet"
    frame.to_parquet(path)
    pd.testing.assert_frame_equal(pd.read_parquet(path), frame)
    with pytest.raises(FrozenInstanceError):
        measured.reason = "mutated"


def test_statistics_derive_summaries_and_pair_count():
    from dataclasses import fields

    statistics = RankingStatistics((0.5, 1.0), 3)
    assert {field.name for field in fields(statistics)} == {
        "seed_accuracies",
        "recipe_count",
    }
    assert statistics.decision_accuracy == 0.75
    assert statistics.decision_accuracy_std == 0.25
    assert statistics.pair_count == 3


def test_observation_reader_uses_canonical_values_not_materialized_summaries():
    observed = RankingObservation(
        Checkpoint("4M", 5, 10.0),
        Checkpoint("1B", 20, 100.0),
        "mmlu",
        "primary_metric",
        False,
        RankingStatistics((0.5, 1.0), 3),
        "",
    )
    frame = observations_frame((observed,))
    frame.loc[
        0, ["decision_accuracy", "decision_accuracy_std", "pair_count", "compute_ratio"]
    ] = -999
    assert observations_from_frame(frame)[0] == observed
    with pytest.raises(ValueError, match="unique"):
        observations_from_frame(pd.concat([frame, frame]))
