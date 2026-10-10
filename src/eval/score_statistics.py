"""Exact-checkpoint scores and statistics across seeds and recipes."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pandas as pd

from datadec.recipes import (
    DataRecipeName,
    RecipeNameResolver,
    load_recipe_name_resolver,
)
from eval.checkpoint_scores import checkpoint_rows, checkpoint_scores
from eval.recipe_scores import MultiRecipeScores
from eval.results import Checkpoint


@dataclass(frozen=True, slots=True)
class CheckpointRecipeScores:
    checkpoint: Checkpoint
    task: str
    metric: str
    per_seed_scores: Mapping[str, MultiRecipeScores]

    def __post_init__(self) -> None:
        scores = dict(self.per_seed_scores)
        if not scores:
            raise ValueError("scores require at least one seed")
        recipes = set(next(iter(scores.values())))
        if not recipes or any(set(values) != recipes for values in scores.values()):
            raise ValueError("all seeds must contain the same nonempty recipe set")
        if not all(
            np.isfinite(value)
            for values in scores.values()
            for value in values.values()
        ):
            raise ValueError("scores must be finite")
        object.__setattr__(self, "per_seed_scores", MappingProxyType(scores))

    @property
    def recipes(self) -> tuple[DataRecipeName, ...]:
        return tuple(next(iter(self.per_seed_scores.values())))

    def matrix(self) -> np.ndarray:
        """Rows are recipes; columns retain the selected seed order."""
        return np.array(
            [
                [scores[recipe] for scores in self.per_seed_scores.values()]
                for recipe in self.recipes
            ],
            dtype=float,
        )


def recipe_scores_at_checkpoint(
    evaluations: pd.DataFrame,
    size: str,
    step: int,
    task: str,
    metric: str,
    *,
    seeds: tuple[str, ...] | None = None,
    recipes: tuple[str | DataRecipeName, ...] | None = None,
    recipe_name_resolver: RecipeNameResolver | None = None,
) -> CheckpointRecipeScores:
    if step < 0:
        raise ValueError("checkpoint step must be nonnegative")
    resolver = (
        recipe_name_resolver
        if recipe_name_resolver is not None
        else load_recipe_name_resolver()
    )
    rows = checkpoint_rows(evaluations, size, step, task).copy()
    rows["data"] = rows["data"].map(resolver.resolve)
    selected_recipes = (
        tuple(resolver.resolve(recipe) for recipe in recipes)
        if recipes is not None
        else tuple(resolver.resolve(name) for name in sorted(set(rows["data"])))
    )
    if not selected_recipes or len(set(selected_recipes)) != len(selected_recipes):
        raise ValueError("nonempty unique recipes are required")
    selected_seeds = (
        seeds if seeds is not None else tuple(sorted(rows["seed"].unique()))
    )
    values, compute = checkpoint_scores(rows, metric, selected_recipes, selected_seeds)
    return CheckpointRecipeScores(
        Checkpoint(size, step, compute),
        task,
        metric,
        {
            seed: MultiRecipeScores(
                dict(zip(selected_recipes, map(float, values[seed]), strict=True))
            )
            for seed in selected_seeds
        },
    )


@dataclass(frozen=True, slots=True)
class RecipeScoreSummary:
    scores: CheckpointRecipeScores
    ddof: int

    def __post_init__(self) -> None:
        if self.ddof not in (0, 1) or len(self.scores.per_seed_scores) <= self.ddof:
            raise ValueError("ddof must be 0 or 1 with more seeds than ddof")

    @property
    def mean_per_recipe_scores(self) -> MultiRecipeScores:
        return MultiRecipeScores(
            dict(
                zip(
                    self.scores.recipes,
                    map(float, self.scores.matrix().mean(axis=1)),
                    strict=True,
                )
            )
        )

    @property
    def std_per_recipe_scores(self) -> MultiRecipeScores:
        return MultiRecipeScores(
            dict(
                zip(
                    self.scores.recipes,
                    map(float, self.scores.matrix().std(axis=1, ddof=self.ddof)),
                    strict=True,
                )
            )
        )


def summarize_recipe_scores(
    scores: CheckpointRecipeScores, *, ddof: int = 0
) -> RecipeScoreSummary:
    return RecipeScoreSummary(scores, ddof)


@dataclass(frozen=True, slots=True)
class ScoreNoiseAndSpread:
    summary: RecipeScoreSummary

    @property
    def noise(self) -> float:
        return float(np.mean(tuple(self.summary.std_per_recipe_scores.values())))

    @property
    def spread(self) -> float:
        return float(
            np.std(
                tuple(self.summary.mean_per_recipe_scores.values()),
                ddof=self.summary.ddof,
            )
        )


def score_noise_and_spread(
    scores: CheckpointRecipeScores, *, ddof: int = 0
) -> ScoreNoiseAndSpread:
    if len(scores.recipes) <= ddof:
        raise ValueError("spread requires more recipes than ddof")
    return ScoreNoiseAndSpread(summarize_recipe_scores(scores, ddof=ddof))
