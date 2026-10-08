from dataclasses import replace
from pathlib import Path

from repro.claim_evaluation import evaluate_claims
from repro.datasets import (
    InputArtifact,
    RunMetadata,
    write_claim_results,
    write_run_metadata,
)
from repro.reporting import claim_report_data, load_claim_report, render_claim_markdown
from repro.results import ClaimStatus


ROOT = Path(__file__).resolve().parents[2]


def test_report_uses_saved_numbers_and_verdict_and_reads_paper(sweep, config, tmp_path):
    result = next(r for r in evaluate_claims(sweep, config) if r.claim_id == "DD-0014")
    measurement = result.measurements[0]
    measured = replace(
        measurement.best, decision_accuracy=0.73, seed_accuracies=(0.72, 0.73, 0.74)
    )
    result = replace(
        result,
        status=ClaimStatus.NOT_SUPPORTED,
        measurements=(replace(measurement, best=measured, passes_bound=False),),
    )
    values = config.model_dump()
    values["run"]["claim_inventory"] = tmp_path / "claims.toml"
    values["run"]["paper_dir"] = tmp_path
    config = type(config).model_validate(values)
    quote = "Original paper claim."
    (tmp_path / "paper.tex").write_text("prefix " + quote + " suffix\n")
    config.run.claim_inventory.write_text(f"""[claims.DD-0014]
statement = "Inventory label."
original_entry_ids = ["DD-0014"]
[[claims.DD-0014.locations]]
source_file = "paper.tex"
line = 1
start = 7
end = {7 + len(quote)}
section = "Abstract"
""")
    write_claim_results((result,), tmp_path / "claim_results.parquet")
    write_run_metadata(
        RunMetadata(
            input=InputArtifact(
                path=tmp_path / "absent-olmes.parquet", sha256="fixture"
            ),
            configuration=config,
        ),
        tmp_path / "run.json",
    )
    report = load_claim_report(tmp_path)
    data = claim_report_data(report)["claims"][0]
    assert data["status"] == "not_supported"
    assert data["measurements"][0]["best"]["decision_accuracy"] == 0.73
    assert data["quotes"][0]["text"] == quote
    assert data["quotes"][0]["location"]["start"] == 7
    assert "summary" not in data
    markdown = render_claim_markdown(report)
    assert quote in markdown
    assert "0.730000" in markdown
    assert "not_supported" in markdown

    changed_quote = "Modified paper claim."
    assert len(changed_quote) == len(quote)
    (tmp_path / "paper.tex").write_text("prefix " + changed_quote + " suffix\n")
    updated = load_claim_report(tmp_path)
    assert updated.claims[0].quotes[0].text == changed_quote
    assert updated.claims[0].evidence == result


def test_report_keeps_missing_measurements_and_pending_criteria_visible(
    sweep, config, tmp_path
):
    result = next(r for r in evaluate_claims(sweep, config) if r.claim_id == "DD-0010")
    result = replace(
        result,
        status=ClaimStatus.INSUFFICIENT_DATA,
        evidence_ids=(),
        measurements=(
            replace(result.measurements[0], best=None, available_comparisons=0),
        ),
    )
    values = config.model_dump()
    values["run"]["claim_inventory"] = ROOT / config.run.claim_inventory
    values["run"]["paper_dir"] = ROOT / config.run.paper_dir
    config = type(config).model_validate(values)
    write_claim_results((result,), tmp_path / "claim_results.parquet")
    write_run_metadata(
        RunMetadata(
            input=InputArtifact(path=tmp_path / "missing", sha256="fixture"),
            configuration=config,
        ),
        tmp_path / "run.json",
    )
    report = load_claim_report(tmp_path)
    measurement = report.claims[0].evidence.measurements[0]
    assert measurement.best is None
    assert measurement.accuracy_gt is None
    assert measurement.passes_bound is None
    assert "insufficient_data" in render_claim_markdown(report)
    assert result.judgment in render_claim_markdown(report)
