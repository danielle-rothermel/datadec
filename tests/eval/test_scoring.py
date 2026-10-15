from __future__ import annotations

import math

import pytest

from datadec.eval.scoring import (
    ChoiceScore,
    DecisionRule,
    predicted_index,
    rule_value,
)


def test_decision_rule_values_are_stable() -> None:
    assert [rule.value for rule in DecisionRule] == [
        "raw",
        "per_token",
        "per_char",
        "per_byte",
        "pmi",
    ]


def test_counts_include_the_leading_space_and_utf8_bytes() -> None:
    score = ChoiceScore.for_continuation(" café", log_likelihood=-5.0, token_count=2)

    assert (score.char_count, score.byte_count) == (5, 6)


def test_each_rule_normalizes_as_olmes() -> None:
    score = ChoiceScore(
        log_likelihood=-12.0,
        token_count=3,
        char_count=6,
        byte_count=8,
        unconditional_log_likelihood=-20.0,
    )

    assert rule_value(DecisionRule.RAW, score) == -12.0
    assert rule_value(DecisionRule.PER_TOKEN, score) == -4.0
    assert rule_value(DecisionRule.PER_CHAR, score) == -2.0
    assert rule_value(DecisionRule.PER_BYTE, score) == -1.5
    assert rule_value(DecisionRule.PMI, score) == 8.0


def test_rules_can_pick_different_choices() -> None:
    short = ChoiceScore(-4.0, 1, 2, 2, -3.0)
    long = ChoiceScore(-6.0, 3, 12, 12, -9.0)

    assert predicted_index(DecisionRule.RAW, [short, long]) == 0
    assert predicted_index(DecisionRule.PER_CHAR, [short, long]) == 1
    assert predicted_index(DecisionRule.PMI, [short, long]) == 1


def test_ties_resolve_to_the_lowest_index() -> None:
    score = ChoiceScore(-1.0, 1, 1, 1, -1.0)

    assert predicted_index(DecisionRule.RAW, [score, score]) == 0


def test_pmi_requires_unconditional_values() -> None:
    with pytest.raises(ValueError, match="unconditional"):
        rule_value(DecisionRule.PMI, ChoiceScore(-1.0, 1, 1, 1))
    assert math.isclose(
        rule_value(DecisionRule.PER_BYTE, ChoiceScore(-1.0, 1, 1, 4)), -0.25
    )
