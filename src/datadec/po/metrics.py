"""Per-item metric definitions shared by the sweeps, the report, and the GEPA adapter.

Accuracy labels come from OLMES (acc_raw, acc_per_char, acc_per_token, acc_uncond); each ranks the
choices under one scoring rule. The likelihood labels here are the softmax-over-choices analogue of
each rule, following DataDecide's naming: norm_correct_prob is the gold choice's share of the summed
raw probabilities (their norm_correct_prob), norm_correct_prob_per_char / _per_token share out the
length-normalised probabilities (their norm_correct_prob_per_char / _per_token), and
norm_correct_prob_uncond shares out the conditional/unconditional ratios (no DataDecide counterpart).
Each likelihood metric's argmax is the corresponding accuracy's winner. norm_margin_* is the gold
share minus the best wrong share (DataDecide's `margin` is a raw-probability difference; not used).

The generation formulations (gen_rc, gen_mc) reuse the RC / MC prompt, decode greedily and score the
normalized prefix of the generation (fork metric PrefixMatch): text_match under RC prompts,
label_match under MC prompts. They have accuracy only; primary_likelihood is None for them.

The GEPA adapter in the OLMES fork (local/olmes_gepa_adapter.py) carries a copy of
likelihood_metrics and the two maps because it runs in a different environment; keep them in sync.
"""

from __future__ import annotations

import math

PRIMARY_METRIC = {("arc_easy", "rc"): "acc_per_char", ("arc_easy", "mc"): "acc_raw",
                  ("arc_challenge", "rc"): "acc_uncond", ("arc_challenge", "mc"): "acc_raw",
                  # generation formulations (fork's PrefixMatch): no likelihood twin
                  ("arc_easy", "gen_rc"): "text_match", ("arc_easy", "gen_mc"): "label_match",
                  ("arc_challenge", "gen_rc"): "text_match", ("arc_challenge", "gen_mc"): "label_match"}
GENERATION_FORMULATIONS = ("gen_rc", "gen_mc")
GENERATION_METRICS = ("label_match", "text_match", "any_match", "no_answer", "pred_label", "pred_text",
                      "num_tokens", "max_tokens_reached")

# accuracy label -> likelihood label whose argmax is the same winner
LIKELIHOOD_FOR_ACCURACY = {
    "acc_raw": "norm_correct_prob",
    "acc_per_char": "norm_correct_prob_per_char",
    "acc_per_token": "norm_correct_prob_per_token",
    "acc_uncond": "norm_correct_prob_uncond",
}
MARGIN_FOR_LIKELIHOOD = {
    "norm_correct_prob": "norm_margin",
    "norm_correct_prob_per_char": "norm_margin_per_char",
    "norm_correct_prob_per_token": "norm_margin_per_token",
    "norm_correct_prob_uncond": "norm_margin_uncond",
}
SCORE_METRIC_ALIASES = ("primary", "primary_likelihood")


def primary_likelihood_metric(primary_metric: str) -> str | None:
    """The likelihood twin of an accuracy label; None for generation metrics, which have none."""
    return LIKELIHOOD_FOR_ACCURACY.get(primary_metric)


def resolve_score_metric(name: str, primary_metric: str) -> str:
    """Map 'primary' / 'primary_likelihood' to concrete labels; pass concrete labels through."""
    if name == "primary":
        return primary_metric
    if name == "primary_likelihood":
        twin = primary_likelihood_metric(primary_metric)
        if twin is None:
            raise ValueError(f"{primary_metric} has no likelihood twin")
        return twin
    return name


def _share(scores: list[float], gold: int) -> tuple[float, float]:
    m = max(scores)
    w = [math.exp(s - m) for s in scores]
    z = sum(w)
    probs = [x / z for x in w]
    others = [p for i, p in enumerate(probs) if i != gold]
    return probs[gold], probs[gold] - (max(others) if others else 0.0)


def likelihood_metrics(model_output: list[dict], gold: int | None) -> dict:
    """Softmax-over-choices likelihood metrics from OLMES per-choice outputs (sum_logits, num_chars, ...)."""
    ll = [o["sum_logits"] for o in model_output]
    out: dict = {"n_choices": len(ll)}
    keys = ("norm_correct_prob", "norm_correct_prob_per_char", "norm_correct_prob_per_token", "norm_correct_prob_uncond",
            "norm_margin", "norm_margin_per_char", "norm_margin_per_token", "norm_margin_uncond", "correct_logprob")
    if not ll or gold is None:
        return out | {k: None for k in keys}
    out["correct_logprob"] = ll[gold]
    variants = {
        "": ll,
        "_per_char": [o["sum_logits"] / o["num_chars"] for o in model_output] if all(o.get("num_chars") for o in model_output) else None,
        "_per_token": [o["sum_logits"] / o["num_tokens"] for o in model_output] if all(o.get("num_tokens") for o in model_output) else None,
        "_uncond": [o["sum_logits"] - o["sum_logits_uncond"] for o in model_output] if all(o.get("sum_logits_uncond") is not None for o in model_output) else None,
    }
    for suffix, scores in variants.items():
        if scores is None:
            out[f"norm_correct_prob{suffix}"] = None
            out[f"norm_margin{suffix}"] = None
        else:
            out[f"norm_correct_prob{suffix}"], out[f"norm_margin{suffix}"] = _share(scores, gold)
    return out
