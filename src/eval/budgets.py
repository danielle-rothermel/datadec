"""Recipe ranking at an observed checkpoint within a per-run compute budget."""

from dataclasses import dataclass

import pandas as pd

from datadec.recipes import DataRecipeName, RecipeNameResolver
from eval.checkpoint_scores import UnavailableRankingError
from eval.checkpoints import select_compute_budget
from eval.ranking import predict_recipe_ranking
from eval.results import RankingResult


@dataclass(frozen=True, slots=True)
class BudgetRanking:
    budget: float
    ranking: RankingResult

    @property
    def relative_undershoot(self) -> float:
        return 1 - self.ranking.predictor.compute / self.budget

    @property
    def predictor_seed_count(self) -> int:
        return len(self.ranking.seed_rankings)

    @property
    def total_predictor_compute_per_recipe(self) -> float:
        return self.ranking.predictor.compute * self.predictor_seed_count


def ranking_at_compute_budget(
    evaluations: pd.DataFrame,
    predictor_size: str,
    predicted_size: str,
    task: str,
    predictor_task_metric: str,
    predicted_task_metric: str,
    *,
    compute_budget: float,
    predicted_step: int,
    relative_tolerance: float | None = None,
    predictor_seeds: tuple[str, ...] | None = None,
    predicted_seeds: tuple[str, ...] | None = None,
    recipes: tuple[str | DataRecipeName, ...] | None = None,
    recipe_name_resolver: RecipeNameResolver | None = None,
    predictor_higher_is_better: bool = True,
    predicted_higher_is_better: bool = True,
) -> BudgetRanking:
    """Select by scheduled compute, never by target accuracy or metric availability.

    Budget and selected checkpoint compute are per seed and per recipe. Total
    predictor compute per recipe includes every seed used by the prediction.
    """
    checkpoints = (
        evaluations.loc[
            evaluations["params"] == predictor_size, ["params", "step", "compute"]
        ]
        .reset_index(drop=True)
        .drop_duplicates()
    )
    if checkpoints.duplicated(["params", "step"]).any():
        raise ValueError("inconsistent checkpoint compute")
    step = select_compute_budget(
        checkpoints,
        predictor_size,
        compute_budget,
        relative_tolerance=relative_tolerance,
    )
    if step is None:
        raise UnavailableRankingError(
            "no observed checkpoint within compute budget/tolerance"
        )
    result = predict_recipe_ranking(
        evaluations,
        predictor_size,
        predicted_size,
        task,
        predictor_task_metric,
        predicted_task_metric,
        predictor_step=step,
        predicted_step=predicted_step,
        predictor_seeds=predictor_seeds,
        predicted_seeds=predicted_seeds,
        recipes=recipes,
        recipe_name_resolver=recipe_name_resolver,
        predictor_higher_is_better=predictor_higher_is_better,
        predicted_higher_is_better=predicted_higher_is_better,
    )
    return BudgetRanking(compute_budget, result)
