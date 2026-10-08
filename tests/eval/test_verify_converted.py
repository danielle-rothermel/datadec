from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from datadec.data.artifacts import DataArtifacts
from datadec.eval.scoring import ChoiceScore, DecisionRule
from datadec.eval.verify_converted import (
    PublishedChoice,
    PublishedItem,
    PublishedKey,
    ScoredItem,
    compare,
    load_published,
)

KEY = PublishedKey(recipe="dclm-baseline", params="150M", seed="default", step=10)


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _identity(step: int = 10) -> dict[str, object]:
    return {
        "recipe": "dclm-baseline",
        "params": "150M",
        "seed": "default",
        "step": step,
        "task": "arc_challenge",
    }


def test_load_published_reads_one_checkpoint(tmp_path: Path) -> None:
    root = tmp_path / "processed" / "olmes-details"
    instances = [
        {
            **_identity(step),
            "doc_id": 0,
            "native_id": "q0",
            "label": 1,
            "predicted_index_raw": 0,
            "predicted_index_per_token": 1,
            "predicted_index_per_char": 1,
            "predicted_index_uncond": 1,
        }
        for step in (10, 20)
    ]
    choices = [
        {
            **_identity(step),
            "doc_id": 0,
            "choice_index": index,
            "sum_logits": -1.0 - index,
            "sum_logits_uncond": -3.0,
            "num_tokens": 2,
            "num_tokens_all": 9,
            "logits_per_byte": bits,
        }
        for step in (10, 20)
        for index, bits in ((1, 0.5), (0, 0.9))
    ]
    _write(root / "instances" / "dclm-baseline.parquet", instances)
    _write(root / "choices" / "dclm-baseline.parquet", choices)

    published = load_published(DataArtifacts(tmp_path), KEY)

    item = published["q0"]
    assert item.label == 1
    assert [choice.sum_logits for choice in item.choices] == [-1.0, -2.0]
    assert item.predictions[DecisionRule.PMI] == 1
    # Per-byte comes from the lowest published bits per byte.
    assert item.predictions[DecisionRule.PER_BYTE] == 1


def _published(native_id: str, label: int, index: int) -> PublishedItem:
    return PublishedItem(
        native_id=native_id,
        doc_id=0,
        label=label,
        predictions=dict.fromkeys(DecisionRule, index),
        choices=(
            PublishedChoice(-1.0, -2.0, 1, 11, 1.0),
            PublishedChoice(-2.0, -2.5, 1, 11, 2.0),
        ),
    )


def _scored(native_id: str, gold: int, shift: float = 0.0) -> ScoredItem:
    return ScoredItem(
        native_id=native_id,
        gold_index=gold,
        choices=(
            ChoiceScore(-1.0 + shift, 1, 2, 2, -2.0),
            ChoiceScore(-2.0, 1, 2, 2, -2.5),
        ),
        conditional_input_tokens=20,
        warnings=0,
    )


def test_compare_reports_agreement_differences_and_acceptance() -> None:
    published = {"a": _published("a", 0, 0), "b": _published("b", 1, 0)}
    ours = [_scored("a", 0), _scored("b", 1, shift=0.01)]

    comparison = compare(ours, published)

    raw = comparison.rule(DecisionRule.RAW)
    assert raw.agreement == 1.0
    assert raw.accuracy_ours == raw.accuracy_published == 0.5
    assert comparison.sum_logits_max_abs_diff == pytest.approx(0.01)
    assert comparison.sum_logits_mean_abs_diff == pytest.approx(0.0025)
    assert comparison.continuation_token_agreement == 1.0
    assert comparison.context_token_agreement == 1.0
    assert comparison.accepted


def test_compare_lists_mismatches_and_fails_acceptance() -> None:
    published = {"a": _published("a", 0, 1)}

    comparison = compare([_scored("a", 0)], published)

    pmi = comparison.rule(DecisionRule.PMI)
    assert pmi.agreement == 0.0
    assert pmi.first_mismatches == ("a: ours 0, published 1",)
    assert not comparison.accepted


def test_compare_requires_the_same_items() -> None:
    with pytest.raises(ValueError, match="item sets differ"):
        compare([_scored("a", 0)], {"b": _published("b", 0, 0)})
