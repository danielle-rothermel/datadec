from dataclasses import asdict

from repro.evidence import collect_claim_evidence


def test_all_configured_selections_link_scoped_rows(sweep, config):
    results = {item.claim_id: item for item in collect_claim_evidence(sweep, config)}
    assert len(results) == len(config.claims) == 33
    for claim in ("DD-0014", "DD-0015", "DD-0016"):
        assert (
            sweep.loc[list(results[claim].evidence_ids), "compute_ratio"] <= 0.0001
        ).all()
    assert all(item.has_measurements for item in results.values())
    assert set(asdict(results["DD-0207"])) == {
        "claim_id",
        "evidence_ids",
        "unavailable_ids",
        "measurements",
        "related_tables",
    }


def test_arc_preserves_both_tasks_and_their_observed_numbers(sweep, config):
    sweep.loc[sweep["task"] == "arc_challenge", "decision_accuracy"] = 0.80
    result = {r.claim_id: r for r in collect_claim_evidence(sweep, config)}["DD-0015"]
    assert {m.task: m.best.decision_accuracy for m in result.measurements} == {
        "arc_easy": 0.81,
        "arc_challenge": 0.80,
    }
    sweep.loc[sweep["task"] == "arc_challenge", "available"] = False
    result = {r.claim_id: r for r in collect_claim_evidence(sweep, config)}["DD-0015"]
    assert result.has_measurements
    missing = next(m for m in result.measurements if m.task == "arc_challenge")
    assert missing.best is None
    assert missing.available_comparisons == 0
    assert missing.unavailable_comparisons > 0


def test_zero_compute_produces_no_linked_measurements(sweep, config):
    sweep.loc[sweep["task"] == "hellaswag", "compute_ratio"] = 0.0
    result = {r.claim_id: r for r in collect_claim_evidence(sweep, config)}["DD-0016"]
    assert not result.has_measurements
    assert not result.evidence_ids


def test_measurements_change_with_input_data_and_preserve_witness(sweep, config):
    before = {r.claim_id: r for r in collect_claim_evidence(sweep, config)}[
        "DD-0014"
    ].measurements[0]
    assert before.best.decision_accuracy == 0.81
    assert before.best.metric == "correct_prob_per_char"
    assert before.best.compute_ratio == 0.00001
    index = before.best.evidence_id
    sweep.loc[index, "decision_accuracy"] = 0.93
    sweep.at[index, "seed_accuracies"] = [0.92, 0.93, 0.94]
    updated = {r.claim_id: r for r in collect_claim_evidence(sweep, config)}["DD-0014"]
    assert updated.measurements[0].best.decision_accuracy == 0.93
    assert updated.measurements[0].best.seed_accuracies == (0.92, 0.93, 0.94)
    assert updated.measurements[0].best.evidence_id == index


def test_missing_task_is_recorded_with_no_fabricated_measurement(sweep, config):
    sweep = sweep.loc[sweep["task"] != "arc_challenge"]
    result = {r.claim_id: r for r in collect_claim_evidence(sweep, config)}["DD-0015"]
    missing = next(m for m in result.measurements if m.task == "arc_challenge")
    assert missing.best is None
    assert missing.available_comparisons == 0
    assert missing.unavailable_comparisons == 0
