from pathlib import Path

import pytest
from pydantic import ValidationError

from repro.evidence import collect_claim_evidence
from repro.config import DEFAULT_CONFIG_PATH, EvaluationConfig, load_evaluation_config


ROOT = Path(__file__).resolve().parents[2]


def test_default_config_matches_explicit_toml_and_claim_inventory(config):
    from repro.claims import load_claims

    explicit = load_evaluation_config(ROOT / DEFAULT_CONFIG_PATH)
    assert explicit == config
    inventory = load_claims(ROOT / config.run.claim_inventory)
    assert set(config.claims) <= inventory.keys()
    assert config.predictors.seeds_for("1B") == config.target.seeds
    assert config.predictors.seeds_for("150M") == config.predictors.default_seeds


@pytest.mark.parametrize(
    "change, message",
    [
        ("unknown_task_group", "unknown selection groups"),
        ("unknown_metric", "invalid metrics"),
        ("authored_judgment", "Extra inputs"),
        ("invalid_range", "minimum compute ratio"),
        ("duplicate_seeds", "seed sets"),
        ("unknown_setting", "Extra inputs"),
        ("invalid_tolerance", "less than or equal to 1"),
    ],
)
def test_invalid_config_is_rejected(config, change, message):
    values = config.model_dump()
    if change == "unknown_task_group":
        values["claims"]["DD-0014"]["task_groups"] = ["typo"]
    elif change == "unknown_metric":
        values["claims"]["DD-0014"]["metrics"] = ["typo"]
    elif change == "authored_judgment":
        values["claims"]["DD-0014"]["judgment"] = "Authored interpretation"
    elif change == "invalid_range":
        values["claims"]["DD-0014"]["min_compute_ratio"] = 0.9
    elif change == "duplicate_seeds":
        values["target"]["seeds"] = ["default", "default"]
    elif change == "unknown_setting":
        values["run"]["typo"] = True
    else:
        values["run"]["matched_compute_tolerance"] = 1.1
    with pytest.raises(ValidationError, match=message):
        EvaluationConfig.model_validate(values)


def test_custom_toml_controls_selection_without_claim_id_special_case(tmp_path, sweep):
    source = (ROOT / DEFAULT_CONFIG_PATH).read_text()
    source = source.replace("[claims.DD-0014]", "[claims.custom-selection]")
    source = source.replace("max_compute_ratio = 0.0001", "max_compute_ratio = 0.00001")
    path = tmp_path / "evaluation.toml"
    path.write_text(source)
    config = load_evaluation_config(path)
    evidence = {
        result.claim_id: result for result in collect_claim_evidence(sweep, config)
    }
    assert "DD-0014" not in evidence
    selected = evidence["custom-selection"]
    assert selected.has_measurements
    assert (sweep.loc[list(selected.evidence_ids), "compute_ratio"] <= 0.00001).all()


def test_named_groups_resolve_in_order_without_repeated_metrics(config):
    values = config.model_dump()
    values["claims"]["DD-0055"]["metrics"] = ["primary_metric"]
    updated = EvaluationConfig.model_validate(values)
    assert updated.metrics_for(updated.claims["DD-0055"]) == (
        "primary_metric",
        "correct_prob_per_char",
        "total_prob_per_char",
        "norm_correct_prob_per_char",
        "margin_per_char",
    )


def test_narrow_experiment_can_retain_unused_group_definitions(config):
    values = config.model_dump()
    values["sweep_metric_group"] = "primary"
    values["claims"] = {"DD-0010": values["claims"]["DD-0010"]}
    updated = EvaluationConfig.model_validate(values)
    assert updated.metrics == ("primary_metric",)
    assert set(updated.claims) == {"DD-0010"}
