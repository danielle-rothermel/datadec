"""Typed sweep observations and their tabular output boundary."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pandas as pd
from pydantic import TypeAdapter

from eval.approaches import PredictionApproach
from eval.results import Checkpoint, RankingResult


@dataclass(frozen=True, slots=True)
class RankingStatistics:
    seed_accuracies: tuple[float, ...]
    recipe_count: int
    aggregate_decision_accuracy: float

    @property
    def decision_accuracy(self) -> float:
        return float(np.mean(self.seed_accuracies))

    @property
    def decision_accuracy_std(self) -> float:
        return float(np.std(self.seed_accuracies))

    def accuracy(self, approach: PredictionApproach) -> float:
        return {
            PredictionApproach.PER_SEED: self.decision_accuracy,
            PredictionApproach.AGGREGATE: self.aggregate_decision_accuracy,
        }[approach]

    @property
    def pair_count(self) -> int:
        return self.recipe_count * (self.recipe_count - 1) // 2


@dataclass(frozen=True, slots=True)
class RankingObservation:
    predictor: Checkpoint
    target: Checkpoint
    task: str
    metric: str
    schedule_complete: bool
    statistics: RankingStatistics | None
    reason: str

    @property
    def compute_ratio(self) -> float:
        return self.predictor.compute / self.target.compute

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
                seed_accuracies=result.seed_accuracies(),
                recipe_count=result.recipe_count,
                aggregate_decision_accuracy=result.aggregate_decision_accuracy(),
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
                "compute_ratio": observation.compute_ratio,
                "schedule_complete": observation.schedule_complete,
                "available": statistics is not None,
                "reason": observation.reason,
                "decision_accuracy": statistics.decision_accuracy
                if statistics
                else None,
                "decision_accuracy_std": statistics.decision_accuracy_std
                if statistics
                else None,
                "aggregate_decision_accuracy": statistics.aggregate_decision_accuracy
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


_OBSERVATION_ADAPTER = TypeAdapter(RankingObservation)


def observations_from_frame(frame: pd.DataFrame) -> Mapping[int, RankingObservation]:
    """Read canonical values from saved rows; derive summaries from seed accuracies."""
    if not frame.index.is_unique:
        raise ValueError("ranking evidence IDs must be unique")
    observations = {}
    for index, row in frame.iterrows():
        observations[int(index)] = _OBSERVATION_ADAPTER.validate_python(
            {
                "predictor": {
                    "size": row["predictor_size"],
                    "step": row["predictor_step"],
                    "compute": row["compute"],
                },
                "target": {
                    "size": row["target_size"],
                    "step": row["target_step"],
                    "compute": row["target_compute"],
                },
                "task": row["task"],
                "metric": row["metric"],
                "schedule_complete": row["schedule_complete"],
                "statistics": {
                    "seed_accuracies": tuple(row["seed_accuracies"]),
                    "recipe_count": row["recipe_count"],
                    "aggregate_decision_accuracy": row["aggregate_decision_accuracy"],
                }
                if row["available"]
                else None,
                "reason": row["reason"],
            }
        )
    return MappingProxyType(observations)
