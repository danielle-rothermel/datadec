"""Pins the likelihood-metric definitions and their pairing with OLMES accuracy labels."""

import math

import pytest

from datadec.po.metrics import (
    LIKELIHOOD_FOR_ACCURACY,
    PRIMARY_METRIC,
    likelihood_metrics,
    primary_likelihood_metric,
    resolve_score_metric,
)

# two choices: gold (index 1) has the higher raw sum, but choice 0 is much longer so it wins per-char and
# per-token, and gold is common unconditionally so choice 0 also wins the unconditional-corrected rule
OUTPUT = [
    {"sum_logits": -4.0, "num_chars": 20, "num_tokens": 10, "sum_logits_uncond": -6.0},
    {"sum_logits": -3.0, "num_chars": 6, "num_tokens": 3, "sum_logits_uncond": -3.5},
]


def _share(a: float, b: float) -> float:
    return math.exp(a) / (math.exp(a) + math.exp(b))


def test_likelihood_metrics_match_hand_computation():
    m = likelihood_metrics(OUTPUT, gold=1)
    assert m["n_choices"] == 2
    assert m["correct_logprob"] == -3.0
    assert m["norm_correct_prob"] == pytest.approx(_share(-3.0, -4.0))
    assert m["norm_correct_prob_per_char"] == pytest.approx(_share(-3.0 / 6, -4.0 / 20))
    assert m["norm_correct_prob_per_token"] == pytest.approx(_share(-3.0 / 3, -4.0 / 10))
    assert m["norm_correct_prob_uncond"] == pytest.approx(_share(-3.0 + 3.5, -4.0 + 6.0))
    for k in ("norm_correct_prob", "norm_correct_prob_per_char", "norm_correct_prob_per_token", "norm_correct_prob_uncond"):
        assert m[k.replace("correct_prob", "margin")] == pytest.approx(2 * m[k] - 1)


def test_argmax_agrees_with_accuracy_rule():
    m = likelihood_metrics(OUTPUT, gold=1)
    assert m["norm_correct_prob"] > 0.5           # acc_raw winner is choice 1 (-3 > -4)
    assert m["norm_correct_prob_per_char"] < 0.5  # acc_per_char winner is choice 0 (-0.2 > -0.5)
    assert m["norm_correct_prob_per_token"] < 0.5  # acc_per_token winner is choice 0 (-0.4 > -1.0)
    assert m["norm_correct_prob_uncond"] < 0.5    # acc_uncond winner is choice 0 (2.0 > 0.5)


def test_uncond_variant_absent_for_mc_style_outputs():
    m = likelihood_metrics([{"sum_logits": -1.0, "num_chars": 1, "num_tokens": 1}, {"sum_logits": -2.0, "num_chars": 1, "num_tokens": 1}], gold=0)
    assert m["norm_correct_prob_uncond"] is None and m["norm_margin_uncond"] is None
    assert m["norm_correct_prob"] == pytest.approx(m["norm_correct_prob_per_char"]) == pytest.approx(m["norm_correct_prob_per_token"])


def test_missing_gold_yields_nulls():
    m = likelihood_metrics(OUTPUT, gold=None)
    assert m["n_choices"] == 2 and m["norm_correct_prob"] is None


def test_primary_pairing_is_pinned():
    assert PRIMARY_METRIC == {("arc_easy", "rc"): "acc_per_char", ("arc_easy", "mc"): "acc_raw",
                              ("arc_challenge", "rc"): "acc_uncond", ("arc_challenge", "mc"): "acc_raw"}
    assert primary_likelihood_metric("acc_per_char") == "norm_correct_prob_per_char"
    assert primary_likelihood_metric("acc_uncond") == "norm_correct_prob_uncond"
    assert primary_likelihood_metric("acc_raw") == "norm_correct_prob"
    assert set(LIKELIHOOD_FOR_ACCURACY) == {"acc_raw", "acc_per_char", "acc_per_token", "acc_uncond"}
    assert resolve_score_metric("primary", "acc_uncond") == "acc_uncond"
    assert resolve_score_metric("primary_likelihood", "acc_uncond") == "norm_correct_prob_uncond"
    assert resolve_score_metric("acc_raw", "acc_uncond") == "acc_raw"
