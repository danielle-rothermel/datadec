from dataclasses import asdict

from repro.claim_evaluation import evaluate_claims
from repro.results import ClaimStatus


def test_numerical_bounds_and_all_qualitative_claim_scopes(sweep, config):
    results = {item.claim_id: item for item in evaluate_claims(sweep, config)}
    assert len(results) == len(config.claims) == 33
    for claim in ("DD-0014", "DD-0015", "DD-0016"):
        assert results[claim].status == ClaimStatus.SUPPORTED
        assert (
            sweep.loc[list(results[claim].evidence_ids), "compute_ratio"] <= 0.0001
        ).all()
    assert results["DD-0169"].status == ClaimStatus.REQUIRES_JUDGMENT
    assert results["DD-0175"].status == ClaimStatus.REQUIRES_JUDGMENT
    assert all(item.evidence_ids for item in results.values())
    assert asdict(results["DD-0207"])["judgment"].endswith("separate inspection.")


def test_arc_requires_both_tasks_and_strict_accuracy_threshold(sweep, config):
    sweep.loc[sweep["task"] == "arc_challenge", "decision_accuracy"] = 0.80
    assert {r.claim_id: r.status for r in evaluate_claims(sweep, config)}[
        "DD-0015"
    ] == ClaimStatus.NOT_SUPPORTED
    sweep.loc[sweep["task"] == "arc_challenge", "available"] = False
    assert {r.claim_id: r.status for r in evaluate_claims(sweep, config)}[
        "DD-0015"
    ] == ClaimStatus.INSUFFICIENT_DATA


def test_missing_grid_and_zero_compute_cannot_support_bound(sweep, config):
    sweep.loc[sweep["task"] == "hellaswag", "compute_ratio"] = 0.0
    result = {r.claim_id: r for r in evaluate_claims(sweep, config)}["DD-0016"]
    assert result.status == ClaimStatus.INSUFFICIENT_DATA
    assert not result.evidence_ids
