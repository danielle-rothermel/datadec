"""Run configured single-scale experiments on local processed OLMES."""

import argparse
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict
from hashlib import file_digest
import json
import logging
from pathlib import Path

import pandas as pd

from datadec.data.artifacts import DataArtifacts
from repro.aggregation import load_evaluations
from repro.claim_evaluation import evaluate_claims
from repro.claims import Claim, load_claims
from repro.config import DEFAULT_CONFIG_PATH, EvaluationConfig, load_evaluation_config
from repro.checkpoints import observed_checkpoints
from repro.diagnostics.compute_matches import matched_compute_comparisons
from repro.diagnostics.crossovers import recipe_crossovers
from repro.diagnostics.curves import curve_summary
from repro.diagnostics.proxies import proxy_comparisons
from repro.results import ClaimEvidence
from repro.sweep import sweep_rankings


ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger(__name__)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--data-dir", type=Path, help="override the configured data directory"
    )
    parser.add_argument(
        "--output-dir", type=Path, help="override the configured output directory"
    )
    parser.add_argument(
        "--matched-compute-tolerance",
        type=float,
        help="override the configured maximum relative compute undershoot",
    )
    args = parser.parse_args(argv)
    if (
        args.matched_compute_tolerance is not None
        and not 0 <= args.matched_compute_tolerance <= 1
    ):
        parser.error("--matched-compute-tolerance must be between 0 and 1")
    return args


def resolve_config(args: argparse.Namespace) -> EvaluationConfig:
    """Resolve TOML paths from the checkout and apply explicit CLI overrides."""
    config = load_evaluation_config(args.config)
    values = config.model_dump()
    run = config.run.model_dump()
    run["data_dir"] = (
        args.data_dir if args.data_dir is not None else ROOT / config.run.data_dir
    )
    run["output_dir"] = (
        args.output_dir if args.output_dir is not None else ROOT / config.run.output_dir
    )
    run["claim_inventory"] = ROOT / config.run.claim_inventory
    if args.matched_compute_tolerance is not None:
        run["matched_compute_tolerance"] = args.matched_compute_tolerance
    values["run"] = run
    return EvaluationConfig.model_validate(values)


def load_run_claims(config: EvaluationConfig) -> dict[str, Claim]:
    """Check configured claim references before starting the expensive sweep."""
    claims = load_claims(config.run.claim_inventory)
    unknown = (
        set(config.claims) | set(config.skipped_math_code_claims)
    ) - claims.keys()
    if unknown:
        raise ValueError(f"unknown claim IDs in evaluation config: {sorted(unknown)}")
    return claims


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def hash_file(path: Path) -> str:
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def write_analysis_tables(
    output: Path,
    evaluations: pd.DataFrame,
    sweep: pd.DataFrame,
    config: EvaluationConfig,
) -> None:
    sweep.to_parquet(output / "rankings.parquet")
    curve_summary(sweep).to_csv(output / "curves.csv", index=False)
    proxy_comparisons(sweep, config).to_parquet(
        output / "proxy_comparisons.parquet", index=False
    )
    recipe_crossovers(evaluations, config).to_csv(
        output / "recipe_crossovers.csv", index=False
    )
    matched_compute_comparisons(sweep, config).to_csv(
        output / "matched_compute.csv", index=False
    )
    observed_checkpoints(evaluations).to_csv(output / "checkpoints.csv", index=False)


def build_claim_report(
    evidence: tuple[ClaimEvidence, ...],
    claims: Mapping[str, Claim],
    config: EvaluationConfig,
    source: Path,
    source_sha256: str,
) -> dict[str, object]:
    return {
        "input": {"path": str(source.resolve()), "sha256": source_sha256},
        "configuration": config.model_dump(mode="json"),
        "claims": [
            {
                **asdict(item),
                "statement": claims[item.claim_id].statement,
                "locations": [
                    asdict(location) for location in claims[item.claim_id].locations
                ],
            }
            for item in evidence
        ],
    }


def render_claim_markdown(
    evidence: tuple[ClaimEvidence, ...], claims: Mapping[str, Claim]
) -> str:
    lines = [
        "# Single-scale claim evidence",
        "",
        "Numerical verdicts concern the observed grid. Qualitative claims require judgment.",
        "",
    ]
    for item in evidence:
        lines.extend(
            [
                f"## {item.claim_id}: {item.status}",
                "",
                claims[item.claim_id].statement,
                "",
                item.summary,
                "",
                f"Remaining judgment / criterion: {item.judgment}",
                "",
            ]
        )
    return "\n".join(lines)


def write_run_outputs(
    evaluations: pd.DataFrame,
    sweep: pd.DataFrame,
    evidence: tuple[ClaimEvidence, ...],
    claims: Mapping[str, Claim],
    config: EvaluationConfig,
    source: Path,
    source_sha256: str,
) -> None:
    output = config.run.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_analysis_tables(output, evaluations, sweep, config)
    report = build_claim_report(evidence, claims, config, source, source_sha256)
    (output / "claims.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    (output / "claims.md").write_text(render_claim_markdown(evidence, claims))


def log_summary(evidence: tuple[ClaimEvidence, ...], output: Path) -> None:
    for status, count in sorted(Counter(item.status for item in evidence).items()):
        LOGGER.info("%s: %s", status, count)
    LOGGER.info("Wrote claim evidence to %s", output.resolve())


def main() -> None:
    config = resolve_config(parse_arguments())
    configure_logging()
    claims = load_run_claims(config)
    source = DataArtifacts(config.run.data_dir).get_path("olmes_processed")
    LOGGER.info("Reading %s", source)
    source_sha256 = hash_file(source)
    evaluations = load_evaluations(config.run.data_dir, config)
    sweep = sweep_rankings(evaluations, config, progress=LOGGER.info)
    evidence = evaluate_claims(sweep, config)
    write_run_outputs(
        evaluations, sweep, evidence, claims, config, source, source_sha256
    )
    log_summary(evidence, config.run.output_dir)


if __name__ == "__main__":
    main()
