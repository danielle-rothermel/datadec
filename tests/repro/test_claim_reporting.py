from dataclasses import replace
from io import StringIO
import os
from pathlib import Path

import pytest
from rich.console import Console

from repro.claim_evaluation import evaluate_claims
from repro.datasets import (
    InputArtifact,
    RunMetadata,
    latest_run,
    read_claim_results,
    write_claim_results,
    write_run_metadata,
)
from repro.reporting import load_claim_report, print_claim_report
from repro.results import ClaimStatus


ROOT = Path(__file__).resolve().parents[2]


def render(report, *, color=False, width=120):
    output = StringIO()
    print_claim_report(
        report,
        Console(
            file=output,
            width=width,
            force_terminal=color,
            no_color=False,
            color_system="standard" if color else None,
        ),
    )
    return output.getvalue()


@pytest.fixture
def saved_run(sweep, config, tmp_path):
    values = config.model_dump()
    values["run"]["claim_inventory"] = ROOT / config.run.claim_inventory
    values["run"]["paper_dir"] = ROOT / config.run.paper_dir
    config = type(config).model_validate(values)
    write_run_metadata(
        RunMetadata(
            input=InputArtifact(
                path=tmp_path / "absent-olmes.parquet", sha256="fixture"
            ),
            configuration=config,
        ),
        tmp_path / "run.json",
    )
    write_claim_results(
        evaluate_claims(sweep, config), tmp_path / "claim_results.parquet"
    )
    return tmp_path, config


def test_report_uses_saved_numbers_and_verdict_and_reads_paper(saved_run):
    run_dir, config = saved_run
    result = next(
        r
        for r in read_claim_results(run_dir / "claim_results.parquet")
        if r.claim_id == "DD-0014"
    )
    measurement = result.measurements[0]
    result = replace(
        result,
        status=ClaimStatus.NOT_SUPPORTED,
        measurements=(
            replace(
                measurement,
                best=replace(
                    measurement.best,
                    decision_accuracy=0.73,
                    seed_accuracies=(0.72, 0.73, 0.74),
                ),
                passes_bound=False,
            ),
        ),
    )
    inventory = run_dir / "claims.toml"
    quote = "Original paper claim."
    (run_dir / "paper.tex").write_text("prefix " + quote + " suffix\n")
    inventory.write_text(f"""[claims.DD-0014]
statement = "Inventory label."
original_entry_ids = ["DD-0014"]
[[claims.DD-0014.locations]]
source_file = "paper.tex"
line = 1
start = 7
end = {7 + len(quote)}
section = "Abstract"
""")
    values = config.model_dump()
    values["run"]["claim_inventory"] = inventory
    config = type(config).model_validate(values)
    write_run_metadata(
        RunMetadata(
            input=InputArtifact(path=run_dir / "absent", sha256="fixture"),
            configuration=config,
        ),
        run_dir / "run.json",
    )
    write_claim_results((result,), run_dir / "claim_results.parquet")
    report = load_claim_report(inventory, run_dir, run_dir)
    assert report.claims[0].evidence == result
    assert report.claims[0].quotes[0].text == quote
    assert report.claims[0].quotes[0].location.start == 7
    output = render(report)
    assert quote in output
    assert "0.730000" in output
    assert "not_supported" in output
    assert "mmlu > 0.8: false" in output
    assert "\x1b[" in render(report, color=True)
    assert "\x1b[" not in output
    assert "…" not in render(report, width=80)

    changed_quote = "Modified paper claim."
    assert len(changed_quote) == len(quote)
    (run_dir / "paper.tex").write_text("prefix " + changed_quote + " suffix\n")
    updated = load_claim_report(inventory, run_dir, run_dir)
    assert updated.claims[0].quotes[0].text == changed_quote
    assert updated.claims[0].evidence == result


def test_report_keeps_all_claims_and_pending_measurements_visible(saved_run):
    run_dir, config = saved_run
    result = next(
        r
        for r in read_claim_results(run_dir / "claim_results.parquet")
        if r.claim_id == "DD-0010"
    )
    result = replace(
        result,
        status=ClaimStatus.INSUFFICIENT_DATA,
        evidence_ids=(),
        measurements=(
            replace(result.measurements[0], best=None, available_comparisons=0),
        ),
    )
    write_claim_results((result,), run_dir / "claim_results.parquet")
    report = load_claim_report(
        config.run.claim_inventory, config.run.paper_dir, run_dir
    )
    assert len(report.claims) == 64
    assert report.claims[0].evidence == result
    assert report.claims[1].evidence is None
    output = render(report)
    assert "insufficient_data" in output
    assert "No measurement" in output
    assert result.judgment in output
    assert output.count("No relevant evidence has been extracted.") == 63


def test_quotes_remain_available_without_saved_results(config):
    report = load_claim_report(
        ROOT / config.run.claim_inventory, ROOT / config.run.paper_dir
    )
    assert report.metadata is None
    assert len(report.claims) == 64
    assert all(claim.evidence is None for claim in report.claims)
    output = render(report)
    assert "No saved analysis report found." in output
    assert output.count("No relevant evidence has been extracted.") == 64
    assert r"\rankingMethod{}" in output


def test_latest_run_uses_manifest_time_and_ignores_incomplete_or_other_inventory(
    saved_run,
):
    run_dir, config = saved_run
    reports = run_dir / "reports"
    for name, timestamp in (
        ("old", 100),
        ("new", 200),
        ("incomplete", 300),
        ("other", 400),
    ):
        output = reports / name
        output.mkdir(parents=True)
        metadata = RunMetadata(
            input=InputArtifact(path=run_dir / "absent", sha256="fixture"),
            configuration=config,
        )
        if name == "other":
            values = config.model_dump()
            values["run"]["claim_inventory"] = run_dir / "other.toml"
            metadata = metadata.model_copy(
                update={"configuration": type(config).model_validate(values)}
            )
        write_run_metadata(metadata, output / "run.json")
        if name != "incomplete":
            write_claim_results((), output / "claim_results.parquet")
        os.utime(output / "run.json", ns=(timestamp, timestamp))
    assert latest_run(reports, config.run.claim_inventory) == reports / "new"
    assert latest_run(run_dir / "absent", config.run.claim_inventory) is None


def test_explicit_run_does_not_accept_other_claim_inventory(saved_run):
    run_dir, config = saved_run
    other = run_dir / "other.toml"
    other.write_text(config.run.claim_inventory.read_text())
    with pytest.raises(ValueError, match="different claim inventory"):
        load_claim_report(other, config.run.paper_dir, run_dir)
