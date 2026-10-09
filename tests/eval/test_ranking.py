import numpy as np
import pandas as pd
import pytest

from datadec.recipes import DataRecipeName, RecipeNameResolver
from eval import (
    MultiRecipeScores,
    RecipeRanking,
    SingleSeedRanking,
    UnavailableRankingError,
    predict_recipe_ranking,
)


@pytest.fixture
def evaluations():
    rows = []
    for size, step, compute, seeds in (
        ("small", 10, 2.0, {"p1": [3, 2, 1], "p2": [1, 20, 3]}),
        ("large", 20, 10.0, {"t1": [3, 2, 1], "t2": [9, 4, 0]}),
    ):
        for seed, scores in seeds.items():
            for recipe, score in zip(("C4", "Falcon", "Dolma1.7"), scores, strict=True):
                rows.append(
                    dict(
                        params=size,
                        step=step,
                        compute=compute,
                        task="task",
                        seed=seed,
                        data=recipe,
                        score=float(score),
                        loss=-score,
                    )
                )
    return pd.DataFrame(rows)


def rank(evaluations, **kwargs):
    options = dict(
        predictor_step=10,
        predicted_step=20,
        predictor_seeds=("p1", "p2"),
        predicted_seeds=("t1", "t2"),
        recipes=("C4", "Falcon", "Dolma1.7"),
    )
    options.update(kwargs)
    return predict_recipe_ranking(
        evaluations, "small", "large", "task", "score", "score", **options
    )


def test_seed_decisions_are_averaged_after_target_seed_mean(evaluations):
    result = rank(evaluations)
    assert result.target_per_recipe_scores == MultiRecipeScores(
        {
            DataRecipeName.C4: 6.0,
            DataRecipeName.FALCON: 3.0,
            DataRecipeName.DOLMA17: 0.5,
        }
    )
    assert isinstance(result.seed_rankings[0], SingleSeedRanking)
    assert all(isinstance(recipe, DataRecipeName) for recipe in result.recipes)
    assert result.seed_accuracies() == (1.0, 1 / 3)
    assert result.decision_accuracy() == pytest.approx(2 / 3)
    # Averaging predictor scores first gives the distinct aggregate accuracy 1/3.
    assert result.seed_rankings[
        1
    ].ranking.predictor_per_recipe_scores == MultiRecipeScores(
        {
            DataRecipeName.C4: 1.0,
            DataRecipeName.FALCON: 20.0,
            DataRecipeName.DOLMA17: 3.0,
        }
    )
    assert result.compute_ratio == 0.2
    assert result.decision_accuracy_std() == pytest.approx(1 / 3)
    assert result.recipe_count == 3
    assert result.pair_count == 3
    assert (
        result.seed_rankings[0].ranking.decision(
            DataRecipeName.C4, DataRecipeName.FALCON
        )
        == 1
    )
    assert (
        result.seed_rankings[1].ranking.decision(
            DataRecipeName.C4, DataRecipeName.FALCON
        )
        == -1
    )


def test_exact_ties_match_only_ties(evaluations):
    evaluations.loc[evaluations["data"] == "C4", "score"] = 2.0
    evaluations.loc[evaluations["data"] == "Falcon", "score"] = 2.0
    result = rank(evaluations)
    assert all(
        r.ranking.decision(DataRecipeName.C4, DataRecipeName.FALCON) == 0
        for r in result.seed_rankings
    )
    evaluations.loc[
        (evaluations["data"] == "C4") & (evaluations["seed"] == "p1"), "score"
    ] = 3.0
    assert (
        rank(evaluations)
        .seed_rankings[0]
        .ranking.decision(DataRecipeName.C4, DataRecipeName.FALCON)
        == 1
    )
    assert rank(evaluations).seed_accuracies()[0] < 1.0


def test_metric_direction_is_explicit(evaluations):
    normal = rank(evaluations)
    loss = predict_recipe_ranking(
        evaluations,
        "small",
        "large",
        "task",
        "loss",
        "score",
        predictor_step=10,
        predicted_step=20,
        predictor_higher_is_better=False,
    )
    assert loss.decision_accuracy() == normal.decision_accuracy()


def test_indexed_and_unindexed_calls_match(evaluations):
    indexed = evaluations.set_index(["params", "step", "task"], drop=False).sort_index()
    assert rank(indexed) == rank(evaluations)


def test_exact_steps_do_not_fall_back(evaluations):
    with pytest.raises(UnavailableRankingError, match="missing checkpoint"):
        rank(evaluations, predictor_step=11)


@pytest.mark.parametrize("missing", ["recipe", "seed", "one_score"])
def test_incomplete_coverage_fails(evaluations, missing):
    if missing == "recipe":
        evaluations = evaluations.loc[
            ~((evaluations["params"] == "large") & (evaluations["data"] == "C4"))
        ]
    elif missing == "seed":
        evaluations = evaluations.loc[evaluations["seed"] != "t2"]
    else:
        evaluations = evaluations.drop(index=0)
    with pytest.raises(UnavailableRankingError):
        rank(evaluations)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_scores_fail(evaluations, bad):
    evaluations.loc[0, "score"] = bad
    with pytest.raises(UnavailableRankingError, match="non-finite"):
        rank(evaluations)


def test_duplicates_are_errors_not_averaged(evaluations):
    with pytest.raises(ValueError, match="duplicate"):
        rank(pd.concat([evaluations, evaluations.iloc[[0]]]))


def test_inconsistent_compute_fails(evaluations):
    evaluations.loc[0, "compute"] = 3.0
    with pytest.raises(ValueError, match="compute"):
        rank(evaluations)


def test_indexed_single_recipe_reports_invalid_comparison(evaluations):
    single = (
        evaluations.loc[evaluations["data"] == "C4"]
        .set_index(["params", "step", "task"], drop=False)
        .sort_index()
    )
    with pytest.raises(ValueError, match="at least two"):
        rank(single, recipes=("C4",))


def test_nullable_missing_score_is_unavailable(evaluations):
    evaluations["score"] = evaluations["score"].astype("Float64")
    evaluations.loc[0, "score"] = pd.NA
    with pytest.raises(UnavailableRankingError, match="non-finite"):
        rank(evaluations)


def test_source_aliases_produce_identical_rankings_without_mutating_input(evaluations):
    aliased = evaluations.copy()
    aliases = {"C4": "c4", "Falcon": "falcon", "Dolma1.7": "dolma1.7"}
    aliased.loc[aliased["params"] == "small", "data"] = aliased.loc[
        aliased["params"] == "small", "data"
    ].map(aliases)
    original = aliased.copy(deep=True)
    result = rank(aliased, recipes=("c4", "falcon", "dolma1.7"))
    assert result == rank(evaluations)
    pd.testing.assert_frame_equal(aliased, original)


def test_custom_alias_resolver_is_used(evaluations):
    aliased = evaluations.replace({"data": {"C4": "custom-c4"}})
    resolver = RecipeNameResolver.from_mapping({"custom-c4": "C4"})
    assert rank(aliased, recipe_name_resolver=resolver) == rank(evaluations)


def test_unknown_recipe_fails(evaluations):
    evaluations.loc[0, "data"] = "not-a-recipe"
    with pytest.raises(ValueError, match="unknown recipe name"):
        rank(evaluations)


def test_aliases_cannot_duplicate_requested_recipes(evaluations):
    with pytest.raises(ValueError, match="unique recipes"):
        rank(evaluations, recipes=("C4", "c4"))


def test_aliases_cannot_duplicate_checkpoint_scores(evaluations):
    duplicate = evaluations.iloc[[0]].assign(data="c4")
    with pytest.raises(ValueError, match="duplicate recipe/seed"):
        rank(pd.concat([evaluations, duplicate]))


def test_score_lookup_is_independent_of_recipe_order(evaluations):
    result = rank(evaluations, recipes=("Dolma1.7", "Falcon", "C4"))
    assert (
        result.seed_rankings[1].ranking.predictor_per_recipe_scores[DataRecipeName.C4]
        == 1.0
    )
    assert result.target_per_recipe_scores[DataRecipeName.DOLMA17] == 0.5
    assert result.decision_accuracy() == rank(evaluations).decision_accuracy()


def test_target_metric_direction_is_retained(evaluations):
    result = predict_recipe_ranking(
        evaluations,
        "small",
        "large",
        "task",
        "loss",
        "loss",
        predictor_step=10,
        predicted_step=20,
        predictor_higher_is_better=False,
        predicted_higher_is_better=False,
    )
    assert result.predicted_higher_is_better is False
    assert all(
        r.ranking.predictor_higher_is_better is False for r in result.seed_rankings
    )
    assert result.seed_accuracies() == rank(evaluations).seed_accuracies()
    assert (
        result.seed_rankings[0].ranking.decision(
            DataRecipeName.C4, DataRecipeName.FALCON
        )
        == 1
    )


def test_result_stores_no_derived_recipe_or_decision_collections(evaluations):
    from dataclasses import fields

    result = rank(evaluations)
    assert "recipes" not in {field.name for field in fields(result)}
    seed = result.seed_rankings[0]
    assert {field.name for field in fields(seed)} == {"seed", "ranking"}
    assert isinstance(seed.ranking, RecipeRanking)
    assert not isinstance(seed, RecipeRanking)
    assert "decisions" not in {field.name for field in fields(seed.ranking)}
    assert result.recipes == tuple(result.target_per_recipe_scores)


def test_single_seed_accuracy_requires_matching_recipe_sets(evaluations):
    seed = rank(evaluations).seed_rankings[0]
    with pytest.raises(ValueError, match="matching sets"):
        seed.ranking.decision_accuracy(
            MultiRecipeScores({DataRecipeName.C4: 1.0}), target_higher_is_better=True
        )


def test_single_seed_accuracy_handles_reordered_targets_and_ties():
    predictor = MultiRecipeScores(
        {
            DataRecipeName.C4: 1.0,
            DataRecipeName.FALCON: 1.0,
            DataRecipeName.DOLMA17: 2.0,
        }
    )
    seed = SingleSeedRanking(
        seed="seed",
        ranking=RecipeRanking(
            predictor_per_recipe_scores=predictor,
            predictor_higher_is_better=True,
        ),
    )
    target = MultiRecipeScores(
        {
            DataRecipeName.DOLMA17: 20.0,
            DataRecipeName.FALCON: 10.0,
            DataRecipeName.C4: 10.0,
        }
    )
    assert seed.ranking.decision(DataRecipeName.C4, DataRecipeName.FALCON) == 0
    assert seed.ranking.decision_accuracy(target, target_higher_is_better=True) == 1.0
    untied = MultiRecipeScores(
        {
            DataRecipeName.C4: 9.0,
            DataRecipeName.FALCON: 10.0,
            DataRecipeName.DOLMA17: 20.0,
        }
    )
    assert seed.ranking.decision_accuracy(untied, target_higher_is_better=True) == 2 / 3


def test_aggregate_prediction_averages_scores_before_decisions(evaluations):
    result = rank(evaluations)
    aggregate = result.aggregate_ranking
    assert aggregate.predictor_per_recipe_scores == MultiRecipeScores(
        {
            DataRecipeName.C4: 2.0,
            DataRecipeName.FALCON: 11.0,
            DataRecipeName.DOLMA17: 2.0,
        }
    )
    assert result.seed_accuracies() == (1.0, 1 / 3)
    assert result.decision_accuracy() == 2 / 3
    assert aggregate.decision(DataRecipeName.C4, DataRecipeName.FALCON) == -1
    assert aggregate.decision(DataRecipeName.C4, DataRecipeName.DOLMA17) == 0
    assert result.aggregate_decision_accuracy() == 1 / 3


def test_aggregate_prediction_respects_both_metric_directions(evaluations):
    normal = rank(evaluations)
    loss = predict_recipe_ranking(
        evaluations,
        "small",
        "large",
        "task",
        "loss",
        "loss",
        predictor_step=10,
        predicted_step=20,
        predictor_higher_is_better=False,
        predicted_higher_is_better=False,
    )
    assert loss.aggregate_ranking.predictor_higher_is_better is False
    assert loss.aggregate_decision_accuracy() == normal.aggregate_decision_accuracy()
    assert (
        loss.aggregate_ranking.decision(DataRecipeName.C4, DataRecipeName.FALCON) == -1
    )


def test_single_seed_and_aggregate_predictions_coincide_for_one_seed(evaluations):
    result = rank(evaluations, predictor_seeds=("p1",))
    assert (
        result.aggregate_ranking.predictor_per_recipe_scores
        == result.seed_rankings[0].ranking.predictor_per_recipe_scores
    )
    assert result.aggregate_decision_accuracy() == result.decision_accuracy() == 1.0
