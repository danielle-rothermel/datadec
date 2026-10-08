"""Minimal OLMES ranked-classification (cloze) requests for ARC-Challenge.

Format, matching OLMES ``arc_challenge:rc::olmes``:

- query: ``"Question: {question}\\nAnswer:"``
- continuations: ``" " + choice_text`` for every choice
- context: five fixed shots, each ``query + " " + gold_text``, joined by
  ``"\\n\\n"``, then ``"\\n\\n"``, then the item's query
- unconditional context: ``"Answer:"`` with the same continuations
- numeric answer keys ``"1".."5"`` map to ``"A".."E"``; the gold index is
  the key's position in ``"ABCDE"``
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, cast

from datadec.eval.olmes_shots import ARC_CHALLENGE_SHOTS

if TYPE_CHECKING:
    from dr_hf import SourceRow

__all__ = [
    "ANSWER_PREFIX",
    "QUESTION_PREFIX",
    "SHOT_SEPARATOR",
    "UNCONDITIONAL_CONTEXT",
    "ArcChallengeDoc",
    "RcRequests",
    "arc_challenge_doc",
    "arc_challenge_fewshot_prefix",
    "arc_challenge_query",
    "build_arc_challenge_rc_requests",
    "requests_for_doc",
]

QUESTION_PREFIX: Final = "Question: "
ANSWER_PREFIX: Final = "Answer:"
CONTINUATION_PREFIX: Final = " "
SHOT_SEPARATOR: Final = "\n\n"
UNCONDITIONAL_CONTEXT: Final = ANSWER_PREFIX
ANSWER_LETTERS: Final = "ABCDE"
NUMERIC_ANSWER_KEYS: Final = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}


@dataclass(frozen=True, slots=True)
class ArcChallengeDoc:
    native_id: str
    question: str
    choices: tuple[str, ...]
    gold_index: int


@dataclass(frozen=True, slots=True)
class RcRequests:
    """Scoring requests for one item: conditional and unconditional."""

    native_id: str
    context: str
    continuations: tuple[str, ...]
    unconditional_context: str
    gold_index: int


def arc_challenge_doc(native_id: str, fields: Mapping[str, object]) -> ArcChallengeDoc:
    """Parse ``allenai/ai2_arc`` row fields (question, choices, answerKey)."""
    question = fields["question"]
    choices = fields["choices"]
    answer_key = fields["answerKey"]
    if not isinstance(question, str) or not isinstance(answer_key, str):
        raise TypeError(f"{native_id}: question and answerKey must be str")
    if not isinstance(choices, Mapping):
        raise TypeError(f"{native_id}: choices must be a mapping")
    raw_texts = cast("Mapping[str, object]", choices).get("text")
    if not isinstance(raw_texts, Sequence) or isinstance(raw_texts, str):
        raise TypeError(f"{native_id}: choices.text must be a list of str")
    texts: list[str] = []
    for text in raw_texts:
        if not isinstance(text, str):
            raise TypeError(f"{native_id}: choices.text must be a list of str")
        texts.append(text)
    letter = NUMERIC_ANSWER_KEYS.get(answer_key, answer_key)
    if letter not in ANSWER_LETTERS:
        raise ValueError(f"{native_id}: unknown answerKey {answer_key!r}")
    gold_index = ANSWER_LETTERS.index(letter)
    if gold_index >= len(texts):
        raise ValueError(f"{native_id}: answerKey beyond the choices")
    return ArcChallengeDoc(
        native_id=native_id,
        question=question,
        choices=tuple(texts),
        gold_index=gold_index,
    )


def arc_challenge_query(question: str) -> str:
    return f"{QUESTION_PREFIX}{question}\n{ANSWER_PREFIX}"


def arc_challenge_fewshot_prefix() -> str:
    """The five fixed shots, each with its gold answer, plus the separator."""
    shots = []
    for shot in ARC_CHALLENGE_SHOTS:
        doc = arc_challenge_doc(str(shot["id"]), shot)
        shots.append(
            arc_challenge_query(doc.question)
            + CONTINUATION_PREFIX
            + doc.choices[doc.gold_index]
        )
    return SHOT_SEPARATOR.join(shots) + SHOT_SEPARATOR


def requests_for_doc(doc: ArcChallengeDoc) -> RcRequests:
    return RcRequests(
        native_id=doc.native_id,
        context=arc_challenge_fewshot_prefix() + arc_challenge_query(doc.question),
        continuations=tuple(CONTINUATION_PREFIX + text for text in doc.choices),
        unconditional_context=UNCONDITIONAL_CONTEXT,
        gold_index=doc.gold_index,
    )


def build_arc_challenge_rc_requests(row: SourceRow) -> RcRequests:
    """OLMES cloze requests for one ARC-Challenge row read through dr-hf."""
    return requests_for_doc(arc_challenge_doc(row.native_id, row.fields))
