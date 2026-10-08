import pandas as pd
import pytest

from repro.aggregation import prepare_evaluations
from repro.diagnostics.curves import curve_summary
from repro.diagnostics.proxies import proxy_comparisons
from repro.diagnostics.crossovers import recipe_crossovers
from repro.diagnostics.compute_matches import matched_compute_comparisons


def test_proxy_comparisons_align_exact_checkpoint(sweep, config):
    comparison = proxy_comparisons(sweep, config)
    proxy = comparison.loc[comparison["metric"] == "correct_prob_per_char"]
    assert proxy["advantage_over_primary"].tolist() == pytest.approx(
        [0.21] * len(proxy)
    )
    curves = curve_summary(sweep)
    assert len(curves) == len(config.benchmarks + ("olmes",)) * len(config.metrics)
    assert curves["adjacent_decreases"].eq(0).all()


def test_matched_compute_requires_completed_endpoint_and_tolerance(sweep, config):
    endpoint = sweep.iloc[[0]].copy()
    endpoint["predictor_size"] = "final"
    endpoint["schedule_complete"] = True
    endpoint["compute"] = 0.00101
    endpoint["decision_accuracy"] = 0.7
    sweep = pd.concat([sweep, endpoint], ignore_index=True)
    pairs = matched_compute_comparisons(
        sweep,
        config.model_copy(
            update={
                "run": config.run.model_copy(update={"matched_compute_tolerance": 0.02})
            }
        ),
    )
    assert len(pairs) == 1
    assert pairs.iloc[0]["accuracy_difference"] == pytest.approx(-0.1)
    assert matched_compute_comparisons(
        sweep,
        config.model_copy(
            update={
                "run": config.run.model_copy(
                    update={"matched_compute_tolerance": 0.001}
                )
            }
        ),
    ).empty


@pytest.fixture
def crossover_raw(raw, config):
    from datadec.config import load_olmes_contract

    recipes = tuple(sorted(load_olmes_contract().recipe_map.values()))
    first = raw.loc[raw["data"] == "a"].copy()
    first["data"] = recipes[0]
    other = raw.loc[raw["data"] == "b"]
    expanded = pd.concat(
        [first, *(other.assign(data=recipe) for recipe in recipes[1:])]
    )
    expanded.loc[expanded["params"] == "small", "total_steps"] = 5
    expanded.loc[
        (expanded["params"] == "large") & (expanded["data"] == recipes[0]),
        "primary_metric",
    ] = 2.0
    return expanded


def test_crossovers_use_completed_scales_and_strict_order(crossover_raw, config):
    result = recipe_crossovers(prepare_evaluations(crossover_raw, config), config)
    assert len(result) == 10
    assert result["strict_crossovers"].eq(24).all()
    assert result["pair_count"].eq(300).all()


def test_crossovers_reject_recipe_absent_from_all_completed_scales(
    crossover_raw, config
):
    absent_recipe = crossover_raw["data"].iloc[0]
    incomplete = crossover_raw.loc[crossover_raw["data"] != absent_recipe]
    with pytest.raises(ValueError, match="requires all catalog recipes"):
        recipe_crossovers(prepare_evaluations(incomplete, config), config)


def test_crossovers_reject_recipe_absent_from_one_completed_scale(
    crossover_raw, config
):
    absent_recipe = crossover_raw["data"].iloc[0]
    incomplete = crossover_raw.loc[
        ~(
            (crossover_raw["data"] == absent_recipe)
            & (crossover_raw["params"] == "small")
        )
    ]
    with pytest.raises(ValueError, match="incomplete recipe coverage"):
        recipe_crossovers(prepare_evaluations(incomplete, config), config)
