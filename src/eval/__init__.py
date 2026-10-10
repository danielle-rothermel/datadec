"""Reusable evaluation helpers, independent of paper reproduction policy."""

from eval.approaches import PredictionApproach
from eval.checkpoint_scores import UnavailableRankingError
from eval.ranking import predict_recipe_ranking
from eval.recipe_scores import MultiRecipeScores
from eval.results import RankingResult, RecipeRanking, SingleSeedRanking

from eval.budgets import BudgetRanking, ranking_at_compute_budget
from eval.score_statistics import (
    CheckpointRecipeScores,
    RecipeScoreSummary,
    ScoreNoiseAndSpread,
    recipe_scores_at_checkpoint,
    summarize_recipe_scores,
    score_noise_and_spread,
)
from eval.prediction_errors import PredictionError, prediction_errors
from eval.scaling_observations import scaling_law_observations

__all__ = [
    "MultiRecipeScores",
    "PredictionApproach",
    "RankingResult",
    "RecipeRanking",
    "SingleSeedRanking",
    "UnavailableRankingError",
    "predict_recipe_ranking",
    "BudgetRanking",
    "ranking_at_compute_budget",
    "CheckpointRecipeScores",
    "RecipeScoreSummary",
    "ScoreNoiseAndSpread",
    "recipe_scores_at_checkpoint",
    "summarize_recipe_scores",
    "score_noise_and_spread",
    "PredictionError",
    "prediction_errors",
    "scaling_law_observations",
]
