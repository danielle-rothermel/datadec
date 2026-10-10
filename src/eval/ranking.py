"""Predict recipe order at one exact checkpoint from another checkpoint."""

import pandas as pd

from datadec.recipes import (
    DataRecipeName,
    RecipeNameResolver,
)
from eval.results import RankingResult, RecipeRanking, SingleSeedRanking
from eval.score_statistics import recipe_scores_at_checkpoint, summarize_recipe_scores


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
    if recipes is not None and len(recipes) < 2:
        raise ValueError("at least two unique recipes are required")
    predictor = recipe_scores_at_checkpoint(
        evaluations,
        predictor_size,
        predictor_step,
        task,
        predictor_task_metric,
        seeds=predictor_seeds,
        recipes=recipes,
        recipe_name_resolver=recipe_name_resolver,
    )
    if len(predictor.recipes) < 2:
        raise ValueError("at least two unique recipes are required")
    target = recipe_scores_at_checkpoint(
        evaluations,
        predicted_size,
        predicted_step,
        task,
        predicted_task_metric,
        seeds=predicted_seeds,
        recipes=predictor.recipes,
        recipe_name_resolver=recipe_name_resolver,
    )
    if target.checkpoint.compute <= 0:
        raise ValueError("target checkpoint compute must be positive")
    return RankingResult(
        predictor=predictor.checkpoint,
        predicted=target.checkpoint,
        task=task,
        predictor_task_metric=predictor_task_metric,
        predicted_task_metric=predicted_task_metric,
        target_seeds=tuple(target.per_seed_scores),
        target_per_recipe_scores=summarize_recipe_scores(target).mean_per_recipe_scores,
        predicted_higher_is_better=predicted_higher_is_better,
        seed_rankings=tuple(
            SingleSeedRanking(seed, RecipeRanking(scores, predictor_higher_is_better))
            for seed, scores in predictor.per_seed_scores.items()
        ),
    )
