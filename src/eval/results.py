"""Checkpoint scores and ranking calculations derived from immutable scores."""

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
class RecipeRanking:
    predictor_per_recipe_scores: MultiRecipeScores
    predictor_higher_is_better: bool

    def decision(self, recipe_a: DataRecipeName, recipe_b: DataRecipeName) -> int:
        """Return +1 if A ranks above B, -1 if below, or 0 for a tie."""
        scores = self.predictor_per_recipe_scores
        return int(np.sign(scores[recipe_a] - scores[recipe_b])) * (
            1 if self.predictor_higher_is_better else -1
        )

    def decision_accuracy(
        self,
        target_per_recipe_scores: MultiRecipeScores,
        *,
        target_higher_is_better: bool,
    ) -> float:
        """Compare all unordered recipe pairs; ties agree only with other ties."""
        recipes = tuple(self.predictor_per_recipe_scores)
        if len(recipes) < 2 or set(recipes) != set(target_per_recipe_scores):
            raise ValueError("accuracy requires matching sets of at least two recipes")
        predictor = np.array([self.predictor_per_recipe_scores[r] for r in recipes])
        target = np.array([target_per_recipe_scores[r] for r in recipes])
        left, right = np.triu_indices(len(recipes), k=1)
        predictor_signs = np.sign(predictor[left] - predictor[right]) * (
            1 if self.predictor_higher_is_better else -1
        )
        target_signs = np.sign(target[left] - target[right]) * (
            1 if target_higher_is_better else -1
        )
        return float(np.mean(predictor_signs == target_signs))


@dataclass(frozen=True, slots=True, kw_only=True)
class SingleSeedRanking(RecipeRanking):
    seed: str


@dataclass(frozen=True, slots=True)
class RankingResult:
    predictor: Checkpoint
    predicted: Checkpoint
    task: str
    predictor_task_metric: str
    predicted_task_metric: str
    target_seeds: tuple[str, ...]
    target_per_recipe_scores: MultiRecipeScores
    predicted_higher_is_better: bool
    seed_rankings: tuple[SingleSeedRanking, ...]

    @property
    def recipes(self) -> tuple[DataRecipeName, ...]:
        return tuple(self.target_per_recipe_scores)

    @property
    def recipe_count(self) -> int:
        return len(self.target_per_recipe_scores)

    @property
    def pair_count(self) -> int:
        return self.recipe_count * (self.recipe_count - 1) // 2

    @property
    def aggregate_ranking(self) -> RecipeRanking:
        """Rank recipes using scores averaged across predictor seeds."""
        recipes = self.recipes
        scores = np.array(
            [
                [seed.predictor_per_recipe_scores[recipe] for recipe in recipes]
                for seed in self.seed_rankings
            ]
        )
        return RecipeRanking(
            predictor_per_recipe_scores=MultiRecipeScores(
                dict(zip(recipes, map(float, scores.mean(axis=0)), strict=True))
            ),
            predictor_higher_is_better=self.seed_rankings[0].predictor_higher_is_better,
        )

    def aggregate_decision_accuracy(self) -> float:
        """Accuracy of the predictor seed-mean scores against target seed-mean scores."""
        return self.aggregate_ranking.decision_accuracy(
            self.target_per_recipe_scores,
            target_higher_is_better=self.predicted_higher_is_better,
        )

    def seed_accuracies(self) -> tuple[float, ...]:
        return tuple(
            ranking.decision_accuracy(
                self.target_per_recipe_scores,
                target_higher_is_better=self.predicted_higher_is_better,
            )
            for ranking in self.seed_rankings
        )

    def decision_accuracy(self) -> float:
        """Mean of individual predictor seeds' decision accuracies."""
        return float(np.mean(self.seed_accuracies()))

    def decision_accuracy_std(self) -> float:
        """Population standard deviation across predictor seeds (ddof=0)."""
        return float(np.std(self.seed_accuracies()))

    @property
    def compute_ratio(self) -> float:
        return self.predictor.compute / self.predicted.compute
