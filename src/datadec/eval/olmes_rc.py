from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from datadec.config.eval import FewShotContract, load_eval_contract

if TYPE_CHECKING:
    from dr_hf import SourceRow

__all__ = [
    "ArcChallengeDoc",
    "RcRequests",
    "arc_challenge_doc",
    "arc_challenge_fewshot_prefix",
    "arc_challenge_query",
    "build_arc_challenge_rc_requests",
    "gold_index_for",
    "requests_for_doc",
    "shot_doc",
]


@dataclass(frozen=True, slots=True)
class ArcChallengeDoc:
    native_id: str
    question: str
    choices: tuple[str, ...]
    gold_index: int


@dataclass(frozen=True, slots=True)
class RcRequests:
    native_id: str
    context: str
    continuations: tuple[str, ...]
    unconditional_context: str
    gold_index: int


def arc_challenge_doc(native_id: str, fields: Mapping[str, object]) -> ArcChallengeDoc:
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
    return ArcChallengeDoc(
        native_id=native_id,
        question=question,
        choices=tuple(texts),
        gold_index=gold_index_for(native_id, answer_key, len(texts)),
    )


def gold_index_for(native_id: str, answer_key: str, choice_count: int) -> int:
    fmt = load_eval_contract().olmes_rc
    letter = fmt.numeric_answer_keys.get(answer_key, answer_key)
    if letter not in fmt.answer_letters:
        raise ValueError(f"{native_id}: unknown answerKey {answer_key!r}")
    gold_index = fmt.answer_letters.index(letter)
    if gold_index >= choice_count:
        raise ValueError(f"{native_id}: answerKey beyond the choices")
    return gold_index


def shot_doc(shot: FewShotContract) -> ArcChallengeDoc:
    return ArcChallengeDoc(
        native_id=shot.id,
        question=shot.question,
        choices=shot.choices,
        gold_index=gold_index_for(shot.id, shot.answer_key, len(shot.choices)),
    )


def arc_challenge_query(question: str) -> str:
    fmt = load_eval_contract().olmes_rc
    return f"{fmt.question_prefix}{question}\n{fmt.answer_prefix}"


def arc_challenge_fewshot_prefix() -> str:
    fmt = load_eval_contract().olmes_rc
    shots = []
    for shot in fmt.arc_challenge_shots:
        doc = shot_doc(shot)
        shots.append(
            arc_challenge_query(doc.question)
            + fmt.continuation_prefix
            + doc.choices[doc.gold_index]
        )
    return fmt.shot_separator.join(shots) + fmt.shot_separator


def requests_for_doc(doc: ArcChallengeDoc) -> RcRequests:
    fmt = load_eval_contract().olmes_rc
    return RcRequests(
        native_id=doc.native_id,
        context=arc_challenge_fewshot_prefix() + arc_challenge_query(doc.question),
        continuations=tuple(fmt.continuation_prefix + text for text in doc.choices),
        unconditional_context=fmt.unconditional_context,
        gold_index=doc.gold_index,
    )


def build_arc_challenge_rc_requests(row: SourceRow) -> RcRequests:
    return requests_for_doc(arc_challenge_doc(row.native_id, row.fields))
