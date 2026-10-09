"""Checkpoint scores and pairwise ranking results."""

from dataclasses import dataclass

import numpy as np

from datadec.recipes import DataRecipeName
from eval.recipe_scores import MultiRecipeScores


@dataclass(frozen=True, slots=True)
class Checkpoint:
    size: str
    step: int
    compute: float


@dataclass(frozen=True, slots=True)
class PairwiseDecision:
    recipe_a: DataRecipeName
    recipe_b: DataRecipeName
    predictor_sign: int
    target_sign: int

    @property
    def correct(self) -> bool:
        return self.predictor_sign == self.target_sign


@dataclass(frozen=True, slots=True)
class SingleSeedRanking:
    seed: str
    predictor_per_recipe_scores: MultiRecipeScores
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
    recipes: tuple[DataRecipeName, ...]
    target_seeds: tuple[str, ...]
    target_per_recipe_scores: MultiRecipeScores
    seed_rankings: tuple[SingleSeedRanking, ...]

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
