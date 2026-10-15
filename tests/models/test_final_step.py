from __future__ import annotations

import json

import pytest

from datadec.models.final_step import (
    FinalStepEvidence,
    choose_final_step,
    parse_final_paths,
    training_data_name,
)

FINAL_PATHS = "\n".join(
    json.dumps(entry)
    for entry in (
        {
            "model_name": "DCLM-baseline-150M-6198",
            "checkpoints_location": "weka://x",
            "revisions": ["step38157-unsharded-hf"],
        },
        {
            "model_name": "c4-150M-6198",
            "checkpoints_location": "weka://y",
            "revisions": ["step38150-unsharded-hf"],
        },
    )
)


def _evidence(**changes: object) -> FinalStepEvidence:
    values: dict[str, object] = {
        "eval_results_last_step": 37500,
        "final_paths_step": 38157,
        "final_paths_commit": "0" * 40,
        "paper_training_steps": 38157,
        "paper_batch_size": 192,
        "paper_source": "paper",
        "source_branch_steps": (36250, 37500, 38157),
        **changes,
    }
    return FinalStepEvidence(**values)  # type: ignore[arg-type]


def test_training_data_name_inverts_the_source_group_map() -> None:
    assert training_data_name("dclm-baseline") == "DCLM-baseline"
    assert training_data_name("dolma1.7") == "dolma17"


def test_parse_final_paths_reads_the_unsharded_step() -> None:
    assert parse_final_paths(FINAL_PATHS, "DCLM-baseline-150M-6198") == 38157
    assert parse_final_paths(FINAL_PATHS, "falcon-150M-6198") is None


def test_parse_final_paths_refuses_unknown_revisions() -> None:
    text = json.dumps({"model_name": "m", "revisions": ["latest"]})

    with pytest.raises(ValueError, match="unexpected final revision"):
        parse_final_paths(text, "m")


def test_authors_list_wins_and_disagreements_are_recorded() -> None:
    decision = choose_final_step(_evidence())

    assert decision.step == 38157
    assert decision.basis == "authors' final default-seed checkpoint list"
    assert decision.disagreements == ("DataDecide-eval-results ends at step 37500",)
    assert decision.run_completed


def test_paper_steps_apply_without_the_authors_list() -> None:
    decision = choose_final_step(_evidence(final_paths_step=None))

    assert decision.step == 38157
    assert decision.basis == "paper model table training steps"


def test_early_final_step_is_an_incomplete_run() -> None:
    decision = choose_final_step(
        _evidence(final_paths_step=37500, eval_results_last_step=37500)
    )

    assert not decision.run_completed
    assert decision.disagreements == ("paper table lists 38157 training steps",)


def test_chosen_step_must_have_a_source_branch() -> None:
    with pytest.raises(ValueError, match="no source branch"):
        choose_final_step(_evidence(source_branch_steps=(37500,)))
