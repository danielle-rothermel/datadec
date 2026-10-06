from __future__ import annotations

import pandas as pd
import pytest

from datadec.config import load_olmes_contract
from datadec.data.verify.olmes_details import (
    verify_cross_source_parity,
    verify_detail_counts,
    verify_reconstructed_task_metrics,
)

CONTRACT = load_olmes_contract()
RECIPE = "dolma1.7-no-math-no-code"
TASK = "arc_challenge"


def test_bits_per_byte_corr_is_not_reconstructible_from_instances() -> None:
    assert CONTRACT.metrics.not_reproducible_from_details == ("bits_per_byte_corr",)
    reproducible = [
        metric
        for metric in CONTRACT.metrics.detailed_tasks
        if metric in CONTRACT.metrics.detailed_instances
        and metric not in CONTRACT.metrics.not_reproducible_from_details
    ]
    assert "bits_per_byte_corr" not in reproducible


def _fixture_tasks() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "recipe": RECIPE,
                "params": "150M",
                "seed_value": 14,
                "seed": "small aux 2",
                "step": 1250,
                "task": TASK,
                "num_instances": 2,
                "primary_metric": "acc_uncond",
                "primary_score": 0.5,
                "acc_raw": 0.5,
                "acc_per_token": 0.5,
                "acc_per_char": 0.5,
                "acc_uncond": 0.5,
                "logits_per_byte_corr": None,
                "logits_per_char_corr": None,
                "no_answer": None,
            }
        ]
    )


def _fixture_instances() -> pd.DataFrame:
    rows = []
    for doc_id, acc in ((0, 1), (1, 0)):
        rows.append(
            {
                "recipe": RECIPE,
                "params": "150M",
                "seed_value": 14,
                "seed": "small aux 2",
                "step": 1250,
                "task": TASK,
                "doc_id": doc_id,
                "acc_raw": acc,
                "acc_per_token": acc,
                "acc_per_char": acc,
                "acc_uncond": acc,
                "predicted_index_raw": acc,
                "predicted_index_per_token": acc,
                "predicted_index_per_char": acc,
                "correct_choice": 2,
                "predicted_index_per_byte": None,
                "predicted_index_uncond": acc,
                "acc_per_byte": None,
                "no_answer": None,
                "sum_logits_corr": None,
                "logits_per_token_corr": None,
                "logits_per_char_corr": None,
                "logits_per_byte_corr": None,
            }
        )
    return pd.DataFrame(rows)


def _fixture_choices() -> pd.DataFrame:
    rows = []
    for doc_id in (0, 1):
        for choice_index in range(4):
            rows.append(
                {
                    "recipe": RECIPE,
                    "params": "150M",
                    "seed_value": 14,
                    "seed": "small aux 2",
                    "step": 1250,
                    "task": TASK,
                    "doc_id": doc_id,
                    "choice_index": choice_index,
                    "sum_logits": -1.0,
                    "num_tokens": 6,
                    "num_tokens_all": 201,
                    "is_greedy": False,
                    "sum_logits_uncond": -0.5,
                    "logits_per_token": -0.1,
                    "logits_per_char": -0.2,
                    "logits_per_byte": None,
                    "num_chars": 33,
                }
            )
    return pd.DataFrame(rows)


def test_verify_detail_counts_on_fixture_outputs() -> None:
    verify_detail_counts(
        tasks_df=_fixture_tasks(),
        instances_df=_fixture_instances(),
        choices_df=_fixture_choices(),
    )


def test_verify_reconstructed_task_metrics_on_fixture_outputs() -> None:
    rows = verify_reconstructed_task_metrics(
        tasks_df=_fixture_tasks(),
        instances_df=_fixture_instances(),
        contract=CONTRACT,
    )
    assert rows == 1


def test_verify_cross_source_parity_on_fixture_overlap() -> None:
    tasks = _fixture_tasks()
    aggregate = pd.DataFrame(
        [
            {
                "params": "150M",
                "data": CONTRACT.recipe_map[RECIPE],
                "seed": "small aux 2",
                "step": 1250,
                "task": TASK,
                "primary_metric": 0.5,
            }
        ]
    )
    overlap = {("150M", 14, 1250)}
    rows = verify_cross_source_parity(
        recipe=RECIPE,
        tasks_df=tasks,
        aggregate_df=aggregate,
        overlapping=overlap,
        contract=CONTRACT,
    )
    assert rows == 1


def test_verify_cross_source_parity_detects_mismatch() -> None:
    tasks = _fixture_tasks()
    aggregate = pd.DataFrame(
        [
            {
                "params": "150M",
                "data": CONTRACT.recipe_map[RECIPE],
                "seed": "small aux 2",
                "step": 1250,
                "task": TASK,
                "primary_metric": 0.1,
            }
        ]
    )
    overlap = {("150M", 14, 1250)}
    with pytest.raises(AssertionError, match="primary metric mismatch"):
        verify_cross_source_parity(
            recipe=RECIPE,
            tasks_df=tasks,
            aggregate_df=aggregate,
            overlapping=overlap,
            contract=CONTRACT,
        )
