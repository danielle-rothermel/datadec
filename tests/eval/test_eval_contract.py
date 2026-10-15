from __future__ import annotations

from datadec.config.eval import load_eval_contract
from datadec.config.olmes import load_olmes_contract
from datadec.eval.scoring import DecisionRule


def test_arc_challenge_source_literals_are_pinned() -> None:
    source = load_eval_contract().arc_challenge
    assert (source.repo_id, source.config, source.native_id_field) == (
        "allenai/ai2_arc",
        "ARC-Challenge",
        "id",
    )
    assert (source.evaluation_split, source.task) == ("test", "arc_challenge")


def test_olmes_rc_format_literals_are_pinned() -> None:
    fmt = load_eval_contract().olmes_rc
    assert fmt.question_prefix == "Question: "
    assert fmt.answer_prefix == "Answer:"
    assert fmt.continuation_prefix == " "
    assert fmt.shot_separator == "\n\n"
    assert fmt.unconditional_context == "Answer:"
    assert fmt.answer_letters == "ABCDE"
    assert fmt.numeric_answer_keys == {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}


def test_arc_challenge_shots_carry_provenance() -> None:
    fmt = load_eval_contract().olmes_rc
    assert fmt.arc_challenge_shots_source.repo == "https://github.com/allenai/olmes"
    assert fmt.arc_challenge_shots_source.license == "Apache-2.0"
    assert [shot.id for shot in fmt.arc_challenge_shots] == [
        "Mercury_SC_415702",
        "MCAS_2009_5_6516",
        "Mercury_7233695",
        "Mercury_7041615",
        "MCAS_1998_4_3",
    ]


def test_verification_acceptance_literals_are_pinned() -> None:
    acceptance = load_eval_contract().verification.acceptance
    assert acceptance.min_predicted_index_agreement_pmi == 0.98
    assert acceptance.min_predicted_index_agreement_per_char == 0.98
    assert acceptance.max_mean_abs_sum_logits_diff_nats == 0.05


def test_published_prediction_columns_are_olmes_detail_columns() -> None:
    columns = load_eval_contract().verification.published_prediction_columns
    assert {DecisionRule(rule) for rule in columns} == {
        DecisionRule.RAW,
        DecisionRule.PER_TOKEN,
        DecisionRule.PER_CHAR,
        DecisionRule.PMI,
    }
    assert set(columns.values()) <= set(
        load_olmes_contract().metrics.detailed_instances
    )
