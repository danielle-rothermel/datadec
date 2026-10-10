import pandas as pd
import pytest

from repro.aggregation import prepare_evaluations
from eval.checkpoints import observed_checkpoints


def test_macro_average_weights_mmlu_once(raw, config):
    prepared = prepare_evaluations(raw, config)
    assert prepared.loc[("small", 5, "olmes")].query("data == 'a'")[
        "primary_metric"
    ].tolist() == pytest.approx([0.9] * 3)
    assert prepared.loc[("small", 5, "mmlu")][
        "primary_metric"
    ].tolist() == pytest.approx([0, 0.1] * 3)
    checkpoints = observed_checkpoints(prepared)
    assert checkpoints["schedule_complete"].tolist() == [False, True]


def test_missing_component_fails_instead_of_reweighting(raw, config):
    with pytest.raises(ValueError, match="requires all 57"):
        prepare_evaluations(
            raw.drop(index=raw.loc[raw["task"] == "mmlu_0"].index[0]), config
        )
    with pytest.raises(ValueError, match="requires all 10"):
        prepare_evaluations(raw.loc[raw["task"] != "boolq"], config)


def test_missing_metric_is_not_averaged_over_remaining_subjects(raw, config):
    raw.loc[raw["task"] == "mmlu_0", "correct_prob"] = float("nan")
    prepared = prepare_evaluations(raw, config)
    assert prepared.loc[("small", 5, "mmlu")]["correct_prob"].isna().all()
    assert prepared.loc[("small", 5, "olmes")]["correct_prob"].isna().all()


def test_duplicate_task_rows_are_rejected(raw, config):
    with pytest.raises(ValueError, match="duplicate"):
        prepare_evaluations(pd.concat([raw, raw.iloc[[0]]]), config)
