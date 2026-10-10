from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from datadec.recipes import DataRecipeName
from eval.budgets import ranking_at_compute_budget
from eval.checkpoint_scores import UnavailableRankingError
from eval.prediction_errors import prediction_errors
from eval.recipe_scores import MultiRecipeScores
from eval.score_statistics import (
    recipe_scores_at_checkpoint,
    score_noise_and_spread,
    summarize_recipe_scores,
)


@pytest.fixture
def scores():
    return pd.DataFrame(
        [
            dict(
                params=size,
                step=step,
                compute=compute,
                task="task",
                seed=seed,
                data=recipe,
                score=value,
            )
            for size, step, compute in [
                ("small", 1, 2.0),
                ("small", 2, 4.0),
                ("large", 3, 10.0),
            ]
            for seed, values in [("a", [0.0, 2.0]), ("b", [2.0, 4.0])]
            for recipe, value in zip(["C4", "Falcon"], values, strict=True)
        ]
    )


def test_noise_is_within_recipe_and_spread_is_between_recipe_means(scores):
    original = scores.copy(deep=True)
    selected = recipe_scores_at_checkpoint(scores, "small", 1, "task", "score")
    summary = summarize_recipe_scores(selected)
    assert summary.mean_per_recipe_scores == MultiRecipeScores(
        {DataRecipeName.C4: 1.0, DataRecipeName.FALCON: 3.0}
    )
    assert summary.std_per_recipe_scores == MultiRecipeScores(
        {DataRecipeName.C4: 1.0, DataRecipeName.FALCON: 1.0}
    )
    stats = score_noise_and_spread(selected)
    assert stats.noise == stats.spread == 1.0
    sample = score_noise_and_spread(selected, ddof=1)
    assert sample.noise == sample.spread == pytest.approx(2**0.5)
    pd.testing.assert_frame_equal(scores, original)
    with pytest.raises(TypeError):
        selected.per_seed_scores["c"] = selected.per_seed_scores[
            "a"
        ]  # ty: ignore[invalid-assignment]
    with pytest.raises(FrozenInstanceError):
        selected.task = "other"  # ty: ignore[invalid-assignment]


def test_score_summary_rejects_insufficient_seeds_for_sample_std(scores):
    selected = recipe_scores_at_checkpoint(
        scores, "small", 1, "task", "score", seeds=("a",)
    )
    with pytest.raises(ValueError, match="more seeds"):
        summarize_recipe_scores(selected, ddof=1)


def test_budget_selection_reports_actual_compute_and_total_seed_cost(scores):
    selected = ranking_at_compute_budget(
        scores,
        "small",
        "large",
        "task",
        "score",
        "score",
        compute_budget=3.0,
        predicted_step=3,
    )
    assert selected.ranking.predictor.step == 1
    assert selected.relative_undershoot == pytest.approx(1 / 3)
    assert selected.total_predictor_compute_per_recipe == 4.0
    assert selected.predictor_seed_count == 2
    assert (
        selected.ranking.decision_accuracy()
        == selected.ranking.aggregate_decision_accuracy()
        == 1.0
    )
    exact = ranking_at_compute_budget(
        scores,
        "small",
        "large",
        "task",
        "score",
        "score",
        compute_budget=4.0,
        predicted_step=3,
    )
    assert exact.ranking.predictor.step == 2


def test_budget_does_not_fall_back_when_chosen_checkpoint_lacks_scores(scores):
    scores.loc[(scores.params == "small") & (scores.step == 2), "score"] = float("nan")
    with pytest.raises(UnavailableRankingError, match="finite"):
        ranking_at_compute_budget(
            scores,
            "small",
            "large",
            "task",
            "score",
            "score",
            compute_budget=4.0,
            predicted_step=3,
        )
    with pytest.raises(UnavailableRankingError, match="tolerance"):
        ranking_at_compute_budget(
            scores,
            "small",
            "large",
            "task",
            "score",
            "score",
            compute_budget=3.0,
            predicted_step=3,
            relative_tolerance=0.1,
        )


def test_prediction_errors_use_observed_denominator_and_keep_zero_missing():
    predicted = MultiRecipeScores({DataRecipeName.C4: 3.0, DataRecipeName.FALCON: 2.0})
    actual = MultiRecipeScores({DataRecipeName.FALCON: 0.0, DataRecipeName.C4: 2.0})
    errors = prediction_errors(predicted, actual)
    assert errors[DataRecipeName.C4].absolute_error == 1.0
    assert errors[DataRecipeName.C4].relative_error == 0.5
    assert errors[DataRecipeName.FALCON].relative_error is None
    with pytest.raises(ValueError, match="matching"):
        prediction_errors(predicted, MultiRecipeScores({DataRecipeName.C4: 2.0}))
