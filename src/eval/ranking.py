"""Predict recipe order at one exact checkpoint from another checkpoint."""

import pandas as pd

from eval.checkpoint_scores import checkpoint_rows, checkpoint_scores
from datadec.recipes import (
    DataRecipeName,
    RecipeNameResolver,
    load_recipe_name_resolver,
)
from eval.recipe_scores import MultiRecipeScores
from eval.results import Checkpoint, RankingResult, SingleSeedRanking


def predict_recipe_ranking(
    evaluations: pd.DataFrame,
    predictor_size: str,
    predicted_size: str,
    task: str,
    predictor_task_metric: str,
    predicted_task_metric: str,
    *,
    predictor_step: int,
    predicted_step: int,
    predictor_seeds: tuple[str, ...] | None = None,
    predicted_seeds: tuple[str, ...] | None = None,
    recipes: tuple[str | DataRecipeName, ...] | None = None,
    recipe_name_resolver: RecipeNameResolver | None = None,
    predictor_higher_is_better: bool = True,
    predicted_higher_is_better: bool = True,
) -> RankingResult:
    """Rank from each predictor seed or their mean, against target seed-mean scores.

    Input columns: params, step, task, data (recipe), seed, compute, and the
    requested metrics. An optional sorted (params, step, task) MultiIndex speeds
    repeated calls. Steps are exact; no interpolation or nearest lookup occurs.
    Omitted recipe/seed sets use the observed predictor recipes and each side's
    observed seeds. Pass explicit sets when checking expected dataset coverage.
    Every selected seed must cover every recipe. Ties match only other ties.
    Recipe names resolve through the configured aliases to DataRecipeName keys.
    """
    if predictor_step < 0 or predicted_step < 0:
        raise ValueError("checkpoint steps must be nonnegative")
    resolver = (
        recipe_name_resolver
        if recipe_name_resolver is not None
        else load_recipe_name_resolver()
    )
    predictor = checkpoint_rows(
        evaluations, predictor_size, predictor_step, task
    ).copy()
    target = checkpoint_rows(evaluations, predicted_size, predicted_step, task).copy()
    predictor["data"] = predictor["data"].map(resolver.resolve)
    target["data"] = target["data"].map(resolver.resolve)
    canonical_recipes = (
        tuple(resolver.resolve(recipe) for recipe in recipes)
        if recipes is not None
        else tuple(resolver.resolve(name) for name in sorted(set(predictor["data"])))
    )
    if len(canonical_recipes) < 2 or len(set(canonical_recipes)) != len(
        canonical_recipes
    ):
        raise ValueError("at least two unique recipes are required")
    predictor_seeds = (
        predictor_seeds
        if predictor_seeds is not None
        else tuple(sorted(predictor["seed"].unique()))
    )
    predicted_seeds = (
        predicted_seeds
        if predicted_seeds is not None
        else tuple(sorted(target["seed"].unique()))
    )
    pred_scores, pred_compute = checkpoint_scores(
        predictor, predictor_task_metric, canonical_recipes, predictor_seeds
    )
    target_scores, target_compute = checkpoint_scores(
        target, predicted_task_metric, canonical_recipes, predicted_seeds
    )
    if target_compute <= 0:
        raise ValueError("target checkpoint compute must be positive")
    truth = target_scores.mean(axis=1).to_numpy()
    rankings = tuple(
        SingleSeedRanking(
            seed=seed,
            predictor_per_recipe_scores=MultiRecipeScores(
                dict(zip(canonical_recipes, map(float, pred_scores[seed]), strict=True))
            ),
            predictor_higher_is_better=predictor_higher_is_better,
        )
        for seed in predictor_seeds
    )
    return RankingResult(
        predictor=Checkpoint(predictor_size, predictor_step, pred_compute),
        predicted=Checkpoint(predicted_size, predicted_step, target_compute),
        task=task,
        predictor_task_metric=predictor_task_metric,
        predicted_task_metric=predicted_task_metric,
        target_seeds=predicted_seeds,
        target_per_recipe_scores=MultiRecipeScores(
            dict(zip(canonical_recipes, map(float, truth), strict=True))
        ),
        predicted_higher_is_better=predicted_higher_is_better,
        seed_rankings=rankings,
    )
