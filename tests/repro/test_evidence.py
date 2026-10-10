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
        "tasks",
        "related_tables",
        "table_links",
    }


def test_arc_preserves_both_tasks_and_their_observed_numbers(sweep, config):
    sweep.loc[sweep["task"] == "arc_challenge", "decision_accuracy"] = 0.80
    result = {r.claim_id: r for r in collect_claim_evidence(sweep, config)}["DD-0015"]
    assert {
        m.task: sweep.loc[m.best_per_seed_evidence_id, "decision_accuracy"]
        for m in result.tasks
    } == {
        "arc_easy": 0.81,
        "arc_challenge": 0.80,
    }
    sweep.loc[sweep["task"] == "arc_challenge", "available"] = False
    result = {r.claim_id: r for r in collect_claim_evidence(sweep, config)}["DD-0015"]
    assert result.has_measurements
    missing = next(m for m in result.tasks if m.task == "arc_challenge")
    assert missing.best_per_seed_evidence_id is None
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
    ].tasks[0]
    assert sweep.loc[before.best_per_seed_evidence_id, "decision_accuracy"] == 0.81
    assert (
        sweep.loc[before.best_per_seed_evidence_id, "metric"] == "correct_prob_per_char"
    )
    assert sweep.loc[before.best_per_seed_evidence_id, "compute_ratio"] == 0.00001
    index = before.best_per_seed_evidence_id
    sweep.loc[index, "decision_accuracy"] = 0.93
    sweep.at[index, "seed_accuracies"] = [0.92, 0.93, 0.94]
    updated = {r.claim_id: r for r in collect_claim_evidence(sweep, config)}["DD-0014"]
    assert (
        sweep.loc[updated.tasks[0].best_per_seed_evidence_id, "decision_accuracy"]
        == 0.93
    )
    assert tuple(
        sweep.loc[updated.tasks[0].best_per_seed_evidence_id, "seed_accuracies"]
    ) == (
        0.92,
        0.93,
        0.94,
    )
    assert updated.tasks[0].best_per_seed_evidence_id == index


def test_missing_task_is_recorded_with_no_fabricated_measurement(sweep, config):
    sweep = sweep.loc[sweep["task"] != "arc_challenge"]
    result = {r.claim_id: r for r in collect_claim_evidence(sweep, config)}["DD-0015"]
    missing = next(m for m in result.tasks if m.task == "arc_challenge")
    assert missing.best_per_seed_evidence_id is None
    assert missing.available_comparisons == 0
    assert missing.unavailable_comparisons == 0


def test_each_approach_selects_its_own_maximum(sweep, config):
    selected = sweep.loc[
        (sweep["task"] == "mmlu")
        & (sweep["compute_ratio"] <= 0.0001)
        & sweep["metric"].isin(config.metrics_for(config.claims["DD-0014"]))
    ]
    first, second = selected.index[:2]
    sweep.loc[first, "decision_accuracy"] = 0.99
    sweep.loc[second, "aggregate_decision_accuracy"] = 1.0
    claim = next(
        item
        for item in collect_claim_evidence(sweep, config)
        if item.claim_id == "DD-0014"
    )
    assert claim.tasks[0].best_per_seed_evidence_id == first
    assert claim.tasks[0].best_aggregate_evidence_id == second
