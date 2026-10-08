from dataclasses import asdict

import pandas as pd
import pytest

from repro.evaluate import (
    ClaimStatus,
    SCOPES,
    curve_summary,
    evaluate_claims,
    proxy_comparisons,
    recipe_crossovers,
)
from repro.ranking import (
    BENCHMARKS,
    METRICS,
    matched_compute_comparisons,
    observed_checkpoints,
    prepare_evaluations,
    select_compute_budget,
)


@pytest.fixture
def raw():
    rows = []
    tasks = [task for task in BENCHMARKS if task != "mmlu"] + [
        f"mmlu_{i}" for i in range(57)
    ]
    for size, step, compute in (("small", 5, 1.0), ("large", 10, 2.0)):
        for task in tasks:
            for seed in ("s1", "s2", "s3"):
                for recipe in ("a", "b"):
                    value = (0.0 if task.startswith("mmlu_") else 1.0) + (
                        0.1 if recipe == "b" else 0.0
                    )
                    rows.append(
                        dict(
                            params=size,
                            step=step,
                            compute=compute,
                            total_steps=10,
                            task=task,
                            data=recipe,
                            seed=seed,
                            **dict.fromkeys(METRICS, value),
                        )
                    )
    return pd.DataFrame(rows)


def test_macro_average_weights_mmlu_once(raw):
    prepared = prepare_evaluations(raw)
    assert prepared.loc[("small", 5, "olmes")].query("data == 'a'")[
        "primary_metric"
    ].tolist() == pytest.approx([0.9] * 3)
    assert prepared.loc[("small", 5, "mmlu")][
        "primary_metric"
    ].tolist() == pytest.approx([0, 0.1] * 3)
    checkpoints = observed_checkpoints(prepared)
    assert checkpoints["schedule_complete"].tolist() == [False, True]


def test_missing_component_fails_instead_of_reweighting(raw):
    with pytest.raises(ValueError, match="requires all 57"):
        prepare_evaluations(raw.drop(index=raw.loc[raw["task"] == "mmlu_0"].index[0]))
    with pytest.raises(ValueError, match="requires all 10"):
        prepare_evaluations(raw.loc[raw["task"] != "boolq"])


def test_missing_metric_is_not_averaged_over_remaining_subjects(raw):
    raw.loc[raw["task"] == "mmlu_0", "correct_prob"] = float("nan")
    prepared = prepare_evaluations(raw)
    assert prepared.loc[("small", 5, "mmlu")]["correct_prob"].isna().all()
    assert prepared.loc[("small", 5, "olmes")]["correct_prob"].isna().all()


def test_duplicate_task_rows_are_rejected(raw):
    with pytest.raises(ValueError, match="duplicate"):
        prepare_evaluations(pd.concat([raw, raw.iloc[[0]]]))


def test_budget_lookup_reports_no_match_and_enforces_tolerance():
    checkpoints = pd.DataFrame(
        dict(params=["a"] * 4, step=[0, 10, 20, 30], compute=[0, 90, 110, 200])
    )
    assert select_compute_budget(checkpoints, "a", 100) == 10
    assert select_compute_budget(checkpoints, "a", 110) == 20
    assert select_compute_budget(checkpoints, "a", 50) is None
    assert select_compute_budget(checkpoints, "b", 100) is None
    assert select_compute_budget(checkpoints, "a", 100, relative_tolerance=0.05) is None
    assert select_compute_budget(checkpoints, "a", 100, relative_tolerance=0.1) == 10
    with pytest.raises(ValueError):
        select_compute_budget(checkpoints, "a", float("inf"))


@pytest.fixture
def sweep():
    rows = []
    for task in BENCHMARKS + ("olmes",):
        for step, ratio in ((1, 0.00001), (2, 0.0001), (3, 0.1)):
            for metric in METRICS:
                rows.append(
                    dict(
                        predictor_size="150M",
                        predictor_step=step,
                        task=task,
                        metric=metric,
                        available=True,
                        reason="",
                        compute_ratio=ratio,
                        compute=ratio * 100,
                        schedule_complete=False,
                        decision_accuracy=0.81
                        if metric == "correct_prob_per_char"
                        else 0.6,
                    )
                )
    result = pd.DataFrame(rows)
    result.index.name = "evidence_id"
    return result


def test_numerical_bounds_and_all_qualitative_claim_scopes(sweep):
    results = {item.claim_id: item for item in evaluate_claims(sweep)}
    assert len(results) == len(SCOPES) == 33
    for claim in ("DD-0014", "DD-0015", "DD-0016"):
        assert results[claim].status == ClaimStatus.SUPPORTED
        assert (
            sweep.loc[list(results[claim].evidence_ids), "compute_ratio"] <= 0.0001
        ).all()
    assert results["DD-0169"].status == ClaimStatus.REQUIRES_JUDGMENT
    assert results["DD-0175"].status == ClaimStatus.REQUIRES_JUDGMENT
    assert all(item.evidence_ids for item in results.values())
    assert asdict(results["DD-0207"])["judgment"].endswith("separate inspection.")


def test_arc_requires_both_tasks_and_strict_accuracy_threshold(sweep):
    sweep.loc[sweep["task"] == "arc_challenge", "decision_accuracy"] = 0.80
    assert {r.claim_id: r.status for r in evaluate_claims(sweep)}[
        "DD-0015"
    ] == ClaimStatus.NOT_SUPPORTED
    sweep.loc[sweep["task"] == "arc_challenge", "available"] = False
    assert {r.claim_id: r.status for r in evaluate_claims(sweep)}[
        "DD-0015"
    ] == ClaimStatus.INSUFFICIENT_DATA


def test_missing_grid_and_zero_compute_cannot_support_bound(sweep):
    sweep.loc[sweep["task"] == "hellaswag", "compute_ratio"] = 0.0
    result = {r.claim_id: r for r in evaluate_claims(sweep)}["DD-0016"]
    assert result.status == ClaimStatus.INSUFFICIENT_DATA
    assert not result.evidence_ids


def test_proxy_comparisons_align_exact_checkpoint(sweep):
    comparison = proxy_comparisons(sweep)
    proxy = comparison.loc[comparison["metric"] == "correct_prob_per_char"]
    assert proxy["advantage_over_primary"].tolist() == pytest.approx(
        [0.21] * len(proxy)
    )
    curves = curve_summary(sweep)
    assert len(curves) == len(BENCHMARKS + ("olmes",)) * len(METRICS)
    assert curves["adjacent_decreases"].eq(0).all()


def test_matched_compute_requires_completed_endpoint_and_tolerance(sweep):
    endpoint = sweep.iloc[[0]].copy()
    endpoint["predictor_size"] = "final"
    endpoint["schedule_complete"] = True
    endpoint["compute"] = 0.00101
    endpoint["decision_accuracy"] = 0.7
    sweep = pd.concat([sweep, endpoint], ignore_index=True)
    pairs = matched_compute_comparisons(sweep, relative_tolerance=0.02)
    assert len(pairs) == 1
    assert pairs.iloc[0]["accuracy_difference"] == pytest.approx(-0.1)
    assert matched_compute_comparisons(sweep, relative_tolerance=0.001).empty


def test_crossovers_use_completed_scales_and_strict_order(raw):
    raw.loc[raw["params"] == "small", "total_steps"] = 5
    raw.loc[(raw["params"] == "large") & (raw["data"] == "a"), "primary_metric"] = 2.0
    result = recipe_crossovers(prepare_evaluations(raw))
    assert len(result) == 10
    assert result["strict_crossovers"].eq(1).all()


def test_paper_sweep_preserves_missing_seed_evidence():
    from datadec.config import load_olmes_contract
    from repro.ranking import SMALL_SEEDS, TARGET_SEEDS, TASKS, sweep_rankings

    rows = []
    recipes = tuple(sorted(load_olmes_contract().recipe_map.values()))
    for size, step, compute, seeds in (
        ("4M", 5, 1.0, SMALL_SEEDS),
        ("1B", 10, 100.0, TARGET_SEEDS),
    ):
        for task in TASKS:
            for seed in seeds:
                for index, recipe in enumerate(recipes):
                    rows.append(
                        dict(
                            params=size,
                            step=step,
                            compute=compute,
                            total_steps=10,
                            task=task,
                            seed=seed,
                            data=recipe,
                            **dict.fromkeys(METRICS, float(index)),
                        )
                    )
    evaluations = pd.DataFrame(rows)
    # One predictor score is absent, while the target remains complete.
    evaluations = (
        evaluations.loc[
            ~(
                (evaluations["params"] == "4M")
                & (evaluations["task"] == "boolq")
                & (evaluations["seed"] == SMALL_SEEDS[-1])
                & (evaluations["data"] == recipes[0])
            )
        ]
        .set_index(["params", "step", "task"], drop=False)
        .sort_index()
    )
    result = sweep_rankings(evaluations, target_step=10)
    assert len(result) == 2 * len(TASKS) * len(METRICS)
    missing = result.query("predictor_size == '4M' and task == 'boolq'")
    assert not missing["available"].any()
    assert missing["reason"].str.contains("non-finite").all()
    available = result.loc[result["available"]]
    assert available["decision_accuracy"].eq(1).all()
    assert available["recipe_count"].eq(25).all()
    assert available["pair_count"].eq(300).all()
