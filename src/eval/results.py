"""Checkpoint scores and pairwise ranking results."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Checkpoint:
    size: str
    step: int
    compute: float


@dataclass(frozen=True, slots=True)
class PairwiseDecision:
    recipe_a: str
    recipe_b: str
    predictor_sign: int
    target_sign: int

    @property
    def correct(self) -> bool:
        return self.predictor_sign == self.target_sign


@dataclass(frozen=True, slots=True)
class SeedRanking:
    seed: str
    predictor_scores: tuple[float, ...]
    decisions: tuple[PairwiseDecision, ...]

    @property
    def decision_accuracy(self) -> float:
        return sum(pair.correct for pair in self.decisions) / len(self.decisions)


@dataclass(frozen=True, slots=True)
class RankingResult:
    predictor: Checkpoint
    predicted: Checkpoint
    task: str
    predictor_task_metric: str
    predicted_task_metric: str
    recipes: tuple[str, ...]
    target_seeds: tuple[str, ...]
    target_scores: tuple[float, ...]
    seed_rankings: tuple[SeedRanking, ...]

    @property
    def decision_accuracy(self) -> float:
        return float(np.mean([r.decision_accuracy for r in self.seed_rankings]))

    @property
    def decision_accuracy_std(self) -> float:
        """Population standard deviation across predictor seeds (ddof=0)."""
        return float(np.std([r.decision_accuracy for r in self.seed_rankings]))

    @property
    def compute_ratio(self) -> float:
        return self.predictor.compute / self.predicted.compute
