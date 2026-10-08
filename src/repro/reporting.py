"""Render persisted claim results alongside passages read from the paper."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from repro.claims import QuoteLocation, load_claims, read_quotes
from repro.datasets import RunMetadata, read_claim_results, read_run_metadata
from repro.results import ClaimEvidence, TaskMeasurement


@dataclass(frozen=True, slots=True)
class PaperPassage:
    location: QuoteLocation
    text: str


@dataclass(frozen=True, slots=True)
class ReportedClaim:
    evidence: ClaimEvidence
    statement: str
    quotes: tuple[PaperPassage, ...]


@dataclass(frozen=True, slots=True)
class ClaimReport:
    metadata: RunMetadata
    claims: tuple[ReportedClaim, ...]


def load_claim_report(run_dir: Path) -> ClaimReport:
    """Load measurements/verdicts from disk and extract each claim's source quotes."""
    metadata = read_run_metadata(run_dir / "run.json")
    results = read_claim_results(run_dir / "claim_results.parquet")
    inventory = load_claims(metadata.configuration.run.claim_inventory)
    claims = {result.claim_id: inventory[result.claim_id] for result in results}
    quotes = read_quotes(claims, metadata.configuration.run.paper_dir)
    return ClaimReport(
        metadata=metadata,
        claims=tuple(
            ReportedClaim(
                evidence=result,
                statement=claims[result.claim_id].statement,
                quotes=tuple(
                    PaperPassage(location=location, text=text)
                    for location, text in zip(
                        claims[result.claim_id].locations,
                        quotes[result.claim_id],
                        strict=True,
                    )
                ),
            )
            for result in results
        ),
    )


def claim_report_data(report: ClaimReport) -> dict[str, object]:
    return {
        **report.metadata.model_dump(mode="json"),
        "claims": [
            {
                **asdict(claim.evidence),
                "statement": claim.statement,
                "quotes": [asdict(passage) for passage in claim.quotes],
            }
            for claim in report.claims
        ],
    }


def _measurement_row(measurement: TaskMeasurement) -> str:
    best = measurement.best
    values = [
        measurement.task,
        str(measurement.available_comparisons),
        str(measurement.unavailable_comparisons),
    ]
    if best is None:
        values.extend(["—"] * 8)
    else:
        values.extend(
            [
                f"{best.decision_accuracy:.6f}",
                f"{best.decision_accuracy_std:.6f}",
                ", ".join(f"{value:.6f}" for value in best.seed_accuracies),
                best.predictor_size,
                str(best.predictor_step),
                best.metric,
                f"{best.compute:.8g}",
                f"{best.compute_ratio:.8g}",
            ]
        )
    values.append(
        "—" if measurement.accuracy_gt is None else f"> {measurement.accuracy_gt:g}"
    )
    values.append(
        "—"
        if measurement.passes_bound is None
        else str(measurement.passes_bound).lower()
    )
    return "| " + " | ".join(values) + " |"


def render_claim_markdown(report: ClaimReport) -> str:
    lines = ["# Single-scale claim results", ""]
    for claim in report.claims:
        result = claim.evidence
        lines.extend(
            [f"## {result.claim_id}: {result.status}", "", claim.statement, ""]
        )
        for passage in claim.quotes:
            location = passage.location
            lines.extend(
                [
                    f"{location.section} ({location.source_file}:{location.line})",
                    "",
                    *(f"> {line}" for line in passage.text.splitlines()),
                    "",
                ]
            )
        lines.extend(
            [
                "| Task | Available | Unavailable | Best accuracy | Seed std | Seed accuracies | Predictor | Step | Metric | FLOPs | Compute / target | Bound | Passes bound |",
                "| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | --- | ---: | ---: | --- | --- |",
                *(_measurement_row(measurement) for measurement in result.measurements),
                "",
                f"Analysis criterion / remaining judgment: {result.judgment}",
                "",
                "Supporting datasets: "
                + ", ".join(
                    f"[{name}]({name})"
                    for name in ("rankings.parquet", *result.supporting_tables)
                ),
                "",
            ]
        )
    return "\n".join(lines)


def write_claim_report(report: ClaimReport, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "claims.json").write_text(
        json.dumps(claim_report_data(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "claims.md").write_text(
        render_claim_markdown(report), encoding="utf-8"
    )
