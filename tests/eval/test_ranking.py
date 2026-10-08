import numpy as np
import pandas as pd
import pytest

from eval.ranking import UnavailableRankingError, predict_recipe_ranking


@pytest.fixture
def evaluations():
    rows = []
    for size, step, compute, seeds in (
        ("small", 10, 2.0, {"p1": [3, 2, 1], "p2": [1, 20, 3]}),
        ("large", 20, 10.0, {"t1": [3, 2, 1], "t2": [9, 4, 0]}),
    ):
        for seed, scores in seeds.items():
            for recipe, score in zip(("a", "b", "c"), scores, strict=True):
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
        recipes=("a", "b", "c"),
    )
    options.update(kwargs)
    return predict_recipe_ranking(
        evaluations, "small", "large", "task", "score", "score", **options
    )


def test_seed_decisions_are_averaged_after_target_seed_mean(evaluations):
    result = rank(evaluations)
    assert result.target_scores == (6.0, 3.0, 0.5)
    assert [r.decision_accuracy for r in result.seed_rankings] == [1.0, 1 / 3]
    assert result.decision_accuracy == pytest.approx(2 / 3)
    # Averaging predictor scores first would incorrectly produce accuracy 1/3.
    assert result.seed_rankings[1].predictor_scores == (1.0, 20.0, 3.0)
    assert result.compute_ratio == 0.2
    assert result.decision_accuracy_std == pytest.approx(1 / 3)
    assert [(p.recipe_a, p.recipe_b) for p in result.seed_rankings[0].decisions] == [
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
    ]


def test_exact_ties_match_only_ties(evaluations):
    evaluations.loc[evaluations["data"] == "a", "score"] = 2.0
    evaluations.loc[evaluations["data"] == "b", "score"] = 2.0
    result = rank(evaluations)
    assert all(r.decisions[0].correct for r in result.seed_rankings)
    evaluations.loc[
        (evaluations["data"] == "a") & (evaluations["seed"] == "p1"), "score"
    ] = 3.0
    assert not rank(evaluations).seed_rankings[0].decisions[0].correct


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
    assert loss.decision_accuracy == normal.decision_accuracy


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
            ~((evaluations["params"] == "large") & (evaluations["data"] == "a"))
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
        evaluations.loc[evaluations["data"] == "a"]
        .set_index(["params", "step", "task"], drop=False)
        .sort_index()
    )
    with pytest.raises(ValueError, match="at least two"):
        rank(single, recipes=("a",))


def test_nullable_missing_score_is_unavailable(evaluations):
    evaluations["score"] = evaluations["score"].astype("Float64")
    evaluations.loc[0, "score"] = pd.NA
    with pytest.raises(UnavailableRankingError, match="non-finite"):
        rank(evaluations)
