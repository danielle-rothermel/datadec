"""Present saved measurements beside claims extracted from paper text."""

from dataclasses import dataclass
from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from repro.claims import QuoteLocation, load_claims, read_quotes
from repro.datasets import RunMetadata, read_claim_evidence, read_run_metadata
from repro.results import ClaimEvidence


@dataclass(frozen=True, slots=True)
class PaperPassage:
    location: QuoteLocation
    text: str


@dataclass(frozen=True, slots=True)
class ReportedClaim:
    claim_id: str
    quotes: tuple[PaperPassage, ...]
    evidence: ClaimEvidence | None


@dataclass(frozen=True, slots=True)
class ClaimReport:
    run_dir: Path | None
    metadata: RunMetadata | None
    claims: tuple[ReportedClaim, ...]


def load_claim_report(
    claim_inventory: Path, paper_dir: Path, run_dir: Path | None = None
) -> ClaimReport:
    """Extract source selections and join saved numerical evidence by ID."""
    inventory = load_claims(claim_inventory)
    quotes = read_quotes(inventory, paper_dir)
    metadata = read_run_metadata(run_dir / "run.json") if run_dir is not None else None
    results = (
        {r.claim_id: r for r in read_claim_evidence(run_dir / "claim_evidence.parquet")}
        if run_dir is not None
        else {}
    )
    if (
        metadata is not None
        and metadata.configuration.run.claim_inventory.resolve()
        != claim_inventory.resolve()
    ):
        raise ValueError("saved run uses a different claim inventory")
    return ClaimReport(
        run_dir=run_dir,
        metadata=metadata,
        claims=tuple(
            ReportedClaim(
                claim_id=claim_id,
                quotes=tuple(
                    PaperPassage(location=location, text=text)
                    for location, text in zip(
                        claim.locations, quotes[claim_id], strict=True
                    )
                ),
                evidence=results.get(claim_id),
            )
            for claim_id, claim in inventory.items()
        ),
    )


def evidence_table(result: ClaimEvidence) -> Table:
    """Format stored observations as a table of numerical measurements."""
    table = Table(box=box.SIMPLE_HEAD, padding=(0, 1))
    for column in (
        "Task",
        "Max accuracy (std)",
        "Predictor @ step",
        "Metric",
        "Compute / target",
        "Available / unavailable",
    ):
        table.add_column(column, overflow="fold")
    for measurement in result.measurements:
        best = measurement.best
        coverage = f"{measurement.available_comparisons} / {measurement.unavailable_comparisons}"
        if best is None:
            cells = [measurement.task, "—", "—", "—", "—", coverage]
        else:
            cells = [
                measurement.task,
                f"{best.decision_accuracy:.6f} ({best.decision_accuracy_std:.6f})",
                f"{best.predictor_size} @ {best.predictor_step}",
                best.metric,
                f"{best.compute_ratio:.8g}",
                coverage,
            ]
        table.add_row(*(Text(cell, overflow="fold") for cell in cells))
    return table


def print_claim_report(report: ClaimReport, console: Console) -> None:
    console.print("Paper selections and linked measurements", style="bold")
    with_measurements = sum(
        claim.evidence is not None and claim.evidence.has_measurements
        for claim in report.claims
    )
    console.print(
        f"Selections: {len(report.claims)} | With measurements: {with_measurements} | "
        f"Without measurements: {len(report.claims) - with_measurements}"
    )
    if report.run_dir is None:
        console.print("Saved report: none", style="dim")
    else:
        console.print(Text(f"Saved report: {report.run_dir.resolve()}", style="dim"))
    console.print()
    for claim in report.claims:
        console.print(Text(claim.claim_id, style="bold cyan"))
        for passage in claim.quotes:
            location = passage.location
            console.print(Text(f"{location.source_file}:{location.line}", style="cyan"))
            console.print(Text(passage.text))
            console.print()
        result = claim.evidence
        if result is None:
            console.print("Linked ranking rows: 0", style="dim")
        else:
            console.print(
                f"Linked ranking rows: {len(result.evidence_ids)} | "
                f"Unavailable rows: {len(result.unavailable_ids)}"
            )
            console.print(evidence_table(result))
            console.print(
                Text(
                    "Data tables: "
                    + ", ".join(("rankings.parquet", *result.related_tables)),
                    style="dim",
                )
            )
        console.print()
