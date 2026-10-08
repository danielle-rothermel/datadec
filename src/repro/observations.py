"""Typed sweep observations and their tabular output boundary."""

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from eval.results import Checkpoint, RankingResult


@dataclass(frozen=True, slots=True)
class RankingStatistics:
    decision_accuracy: float
    decision_accuracy_std: float
    seed_accuracies: tuple[float, ...]
    recipe_count: int
    pair_count: int


@dataclass(frozen=True, slots=True)
class RankingObservation:
    predictor: Checkpoint
    target: Checkpoint
    task: str
    metric: str
    schedule_complete: bool
    statistics: RankingStatistics | None
    reason: str

    @classmethod
    def successful(
        cls, result: RankingResult, *, schedule_complete: bool
    ) -> "RankingObservation":
        return cls(
            predictor=result.predictor,
            target=result.predicted,
            task=result.task,
            metric=result.predictor_task_metric,
            schedule_complete=schedule_complete,
            statistics=RankingStatistics(
                decision_accuracy=result.decision_accuracy,
                decision_accuracy_std=result.decision_accuracy_std,
                seed_accuracies=tuple(
                    r.decision_accuracy for r in result.seed_rankings
                ),
                recipe_count=len(result.recipes),
                pair_count=len(result.seed_rankings[0].decisions),
            ),
            reason="",
        )

    @classmethod
    def unavailable(
        cls,
        predictor: Checkpoint,
        target: Checkpoint,
        task: str,
        metric: str,
        *,
        schedule_complete: bool,
        reason: str,
    ) -> "RankingObservation":
        return cls(predictor, target, task, metric, schedule_complete, None, reason)


def observations_frame(observations: Sequence[RankingObservation]) -> pd.DataFrame:
    """Own the persisted sweep column names and unavailable-value representation."""
    records = []
    for observation in observations:
        statistics = observation.statistics
        records.append(
            {
                "predictor_size": observation.predictor.size,
                "predictor_step": observation.predictor.step,
                "target_size": observation.target.size,
                "target_step": observation.target.step,
                "task": observation.task,
                "metric": observation.metric,
                "compute": observation.predictor.compute,
                "target_compute": observation.target.compute,
                "compute_ratio": observation.predictor.compute
                / observation.target.compute,
                "schedule_complete": observation.schedule_complete,
                "available": statistics is not None,
                "reason": observation.reason,
                "decision_accuracy": statistics.decision_accuracy
                if statistics
                else None,
                "decision_accuracy_std": statistics.decision_accuracy_std
                if statistics
                else None,
                "seed_accuracies": list(statistics.seed_accuracies)
                if statistics
                else None,
                "recipe_count": statistics.recipe_count if statistics else None,
                "pair_count": statistics.pair_count if statistics else None,
            }
        )
    frame = pd.DataFrame.from_records(records)
    frame.index.name = "evidence_id"
    return frame
