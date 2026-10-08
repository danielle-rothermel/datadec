"""Present saved measurements beside claims extracted from paper text."""

from dataclasses import dataclass
from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from repro.claims import QuoteLocation, load_claims, read_quotes
from repro.datasets import RunMetadata, read_claim_results, read_run_metadata
from repro.results import ClaimEvidence, ClaimStatus


@dataclass(frozen=True, slots=True)
class PaperPassage:
    location: QuoteLocation
    text: str


@dataclass(frozen=True, slots=True)
class ReportedClaim:
    claim_id: str
    statement: str
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
    """Read every source claim and join any saved evidence by claim ID."""
    inventory = load_claims(claim_inventory)
    quotes = read_quotes(inventory, paper_dir)
    metadata = read_run_metadata(run_dir / "run.json") if run_dir is not None else None
    results = (
        {r.claim_id: r for r in read_claim_results(run_dir / "claim_results.parquet")}
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
                statement=claim.statement,
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


_STATUS_STYLES = {
    ClaimStatus.SUPPORTED: "green",
    ClaimStatus.NOT_SUPPORTED: "red",
    ClaimStatus.REQUIRES_JUDGMENT: "yellow",
    ClaimStatus.INSUFFICIENT_DATA: "yellow",
}


def evidence_table(result: ClaimEvidence) -> Table:
    """Format stored observations without recalculating measurements or verdicts."""
    table = Table(box=box.SIMPLE_HEAD, padding=(0, 1))
    for column in (
        "Task",
        "Accuracy (std)",
        "Predictor @ step",
        "Metric",
        "Compute / target",
        "OK / missing",
    ):
        table.add_column(column, overflow="fold")
    for measurement in result.measurements:
        best = measurement.best
        coverage = f"{measurement.available_comparisons} / {measurement.unavailable_comparisons}"
        if best is None:
            cells = [measurement.task, "No measurement", "—", "—", "—", coverage]
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
    console.print("DataDecide claims and evidence", style="bold")
    if report.run_dir is None:
        console.print("No saved analysis report found.", style="yellow")
    else:
        console.print(Text(f"Saved report: {report.run_dir.resolve()}", style="dim"))
    console.print(
        "Evidence shows per-task observed maxima, seed standard deviations, compute / target, and available / unavailable comparisons.",
        style="dim",
    )
    console.print(
        "Full seed scores, FLOPs, and evidence IDs remain in claim_results.parquet.",
        style="dim",
    )
    console.print()
    for claim in report.claims:
        heading = Text(claim.claim_id, style="bold cyan")
        if claim.evidence is not None:
            status = claim.evidence.status
            heading.append(f" · {status}", style=_STATUS_STYLES[status])
        console.print(heading)
        console.print(Text(claim.statement, style="bold"))
        console.print()
        for passage in claim.quotes:
            location = passage.location
            console.print(
                Text(
                    f"{location.section} ({location.source_file}:{location.line})",
                    style="cyan",
                )
            )
            console.print(Text(passage.text, style="italic"))
            console.print()
        if claim.evidence is None:
            console.print("No relevant evidence has been extracted.", style="yellow")
        else:
            result = claim.evidence
            console.print(evidence_table(result))
            bounds = [
                f"{m.task} > {m.accuracy_gt:g}: "
                + ("pending" if m.passes_bound is None else str(m.passes_bound).lower())
                for m in result.measurements
                if m.accuracy_gt is not None
            ]
            if bounds:
                console.print(Text("Bound outcomes: " + "; ".join(bounds)))
            console.print(Text(f"Criterion / remaining judgment: {result.judgment}"))
            console.print(
                Text(
                    "Evidence datasets: "
                    + ", ".join(("rankings.parquet", *result.supporting_tables)),
                    style="dim",
                )
            )
        console.print()
