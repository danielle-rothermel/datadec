"""Present saved measurements beside claims extracted from paper text."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from eval.approaches import PredictionApproach
from repro.claims import QuoteLocation, load_claims, read_quotes
from repro.datasets import (
    RunMetadata,
    read_claim_evidence,
    read_rankings,
    read_run_metadata,
)
from repro.observations import RankingObservation
from repro.numerical_tables import load_linked_tables, print_linked_table
from repro.results import ClaimEvidence
from repro.table_evidence import EvidenceTable


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
    rankings: Mapping[int, RankingObservation]
    tables: Mapping[EvidenceTable, pd.DataFrame]


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
    rankings = (
        read_rankings(run_dir / "rankings.parquet") if run_dir is not None else {}
    )
    validate_evidence_references(tuple(results.values()), rankings)
    tables = (
        load_linked_tables(run_dir, tuple(results.values()), metadata.tables)
        if run_dir is not None and metadata is not None
        else {}
    )
    return ClaimReport(
        rankings=rankings,
        run_dir=run_dir,
        metadata=metadata,
        tables=tables,
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


def validate_evidence_references(
    evidence: tuple[ClaimEvidence, ...], rankings: Mapping[int, RankingObservation]
) -> None:
    """Require task/availability links to point to matching saved ranking rows."""
    for claim in evidence:
        for task in claim.tasks:
            for available, indices in (
                (True, task.evidence_ids),
                (False, task.unavailable_ids),
            ):
                for index in indices:
                    observation = rankings.get(index)
                    if observation is None:
                        raise ValueError(f"missing ranking evidence ID: {index}")
                    if (
                        observation.task != task.task
                        or (observation.statistics is not None) != available
                    ):
                        raise ValueError(
                            f"ranking evidence does not match task/availability: {index}"
                        )


def evidence_table(
    result: ClaimEvidence, rankings: Mapping[int, RankingObservation]
) -> Table:
    """Format stored observations as a table of numerical measurements."""
    table = Table(box=box.SIMPLE_HEAD, padding=(0, 1))
    for column in (
        "Task",
        "Approach",
        "Max accuracy (seed std)",
        "Predictor @ step",
        "Metric",
        "Compute / target",
        "Available / unavailable",
    ):
        table.add_column(column, overflow="fold")
    for measurement in result.tasks:
        for approach in PredictionApproach:
            index = measurement.best_evidence_id(approach)
            best = rankings[index] if index is not None else None
            coverage = f"{measurement.available_comparisons} / {measurement.unavailable_comparisons}"
            label = (
                "Per-seed mean"
                if approach is PredictionApproach.PER_SEED
                else "Mean scores"
            )
            if best is None:
                cells = [measurement.task, label, "—", "—", "—", "—", coverage]
            else:
                statistics = best.statistics
                if statistics is None:
                    raise ValueError(
                        "best evidence must reference an available ranking"
                    )
                std = (
                    f"{statistics.decision_accuracy_std:.6f}"
                    if approach is PredictionApproach.PER_SEED
                    else "—"
                )
                cells = [
                    measurement.task,
                    label,
                    f"{statistics.accuracy(approach):.6f} ({std})",
                    f"{best.predictor.size} @ {best.predictor.step}",
                    best.metric,
                    f"{best.compute_ratio:.8g}",
                    coverage,
                ]
            table.add_row(*(Text(cell, overflow="fold") for cell in cells))
    return table


def print_claim_report(
    report: ClaimReport,
    console: Console,
    *,
    max_rows: int | None = 12,
    claim_ids: tuple[str, ...] = (),
) -> None:
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
    unknown = set(claim_ids) - {claim.claim_id for claim in report.claims}
    if unknown:
        raise ValueError(f"unknown claim IDs: {sorted(unknown)}")
    for claim in report.claims:
        if claim_ids and claim.claim_id not in claim_ids:
            continue
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
            if result.tasks:
                console.print(evidence_table(result, report.rankings))
            for link in result.table_links:
                print_linked_table(
                    link, report.tables[link.table], console, max_rows=max_rows
                )
            names = dict.fromkeys(
                (
                    *(("rankings.parquet",) if result.tasks else ()),
                    *(link.table.value for link in result.table_links),
                )
            )
            if names:
                console.print(Text("Data tables: " + ", ".join(names), style="dim"))
        console.print()
