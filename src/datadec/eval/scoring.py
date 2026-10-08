"""Likelihood decision rules over scored multiple-choice continuations.

Rules follow OLMES: raw sums the continuation log-likelihood; per-token,
per-char and per-byte divide it by the continuation's token, character and
UTF-8 byte counts (characters and bytes include the leading space); PMI
subtracts the log-likelihood of the same continuation after the
unconditional context. Ties resolve to the lowest index, like
``numpy.argmax``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import UNIQUE, StrEnum, auto, verify

__all__ = ["ChoiceScore", "DecisionRule", "predicted_index", "rule_value"]


@verify(UNIQUE)
class DecisionRule(StrEnum):
    RAW = auto()
    PER_TOKEN = auto()
    PER_CHAR = auto()
    PER_BYTE = auto()
    PMI = auto()


@dataclass(frozen=True, slots=True)
class ChoiceScore:
    log_likelihood: float
    token_count: int
    char_count: int
    byte_count: int
    unconditional_log_likelihood: float | None = None

    @classmethod
    def for_continuation(
        cls,
        continuation: str,
        *,
        log_likelihood: float,
        token_count: int,
        unconditional_log_likelihood: float | None = None,
    ) -> ChoiceScore:
        return cls(
            log_likelihood=log_likelihood,
            token_count=token_count,
            char_count=len(continuation),
            byte_count=len(continuation.encode("utf-8")),
            unconditional_log_likelihood=unconditional_log_likelihood,
        )


def rule_value(rule: DecisionRule, score: ChoiceScore) -> float:
    match rule:
        case DecisionRule.RAW:
            return score.log_likelihood
        case DecisionRule.PER_TOKEN:
            return score.log_likelihood / score.token_count
        case DecisionRule.PER_CHAR:
            return score.log_likelihood / score.char_count
        case DecisionRule.PER_BYTE:
            return score.log_likelihood / score.byte_count
        case DecisionRule.PMI:
            if score.unconditional_log_likelihood is None:
                raise ValueError("PMI requires an unconditional log-likelihood")
            return score.log_likelihood - score.unconditional_log_likelihood


def predicted_index(rule: DecisionRule, scores: Sequence[ChoiceScore]) -> int:
    if not scores:
        raise ValueError("predicted_index requires at least one score")
    values = [rule_value(rule, score) for score in scores]
    return values.index(max(values))
