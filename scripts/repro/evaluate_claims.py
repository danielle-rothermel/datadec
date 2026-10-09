"""Run configured single-scale experiments on local processed OLMES."""

import argparse
from hashlib import file_digest
import logging
import shlex
from pathlib import Path

import pandas as pd

from datadec.data.artifacts import DataArtifacts
from repro.aggregation import load_evaluations
from repro.evidence import collect_claim_evidence
from repro.claims import load_claims
from repro.config import DEFAULT_CONFIG_PATH, EvaluationConfig, load_evaluation_config
from repro.checkpoints import observed_checkpoints
from repro.diagnostics.compute_matches import matched_compute_comparisons
from repro.diagnostics.crossovers import recipe_crossovers
from repro.diagnostics.curves import curve_summary
from repro.diagnostics.proxies import proxy_comparisons
from repro.datasets import (
    InputArtifact,
    RunMetadata,
    write_claim_evidence,
    write_run_metadata,
)
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
    run["paper_dir"] = ROOT / config.run.paper_dir
    if args.matched_compute_tolerance is not None:
        run["matched_compute_tolerance"] = args.matched_compute_tolerance
    values["run"] = run
    return EvaluationConfig.model_validate(values)


def validate_run_claims(config: EvaluationConfig) -> None:
    """Check configured claim references before starting the expensive sweep."""
    claims = load_claims(config.run.claim_inventory)
    unknown = set(config.claims) - claims.keys()
    if unknown:
        raise ValueError(f"unknown claim IDs in evaluation config: {sorted(unknown)}")


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
    proxy_comparisons(sweep, baseline_metric=config.target.metric).to_parquet(
        output / "proxy_comparisons.parquet", index=False
    )
    recipe_crossovers(
        evaluations, tasks=config.benchmarks, metric=config.target.metric
    ).to_csv(output / "recipe_crossovers.csv", index=False)
    matched_compute_comparisons(
        sweep,
        baseline_metric=config.target.metric,
        relative_tolerance=config.run.matched_compute_tolerance,
    ).to_csv(output / "matched_compute.csv", index=False)
    observed_checkpoints(evaluations).to_csv(output / "checkpoints.csv", index=False)


def write_run_outputs(
    evaluations: pd.DataFrame,
    sweep: pd.DataFrame,
    evidence: tuple[ClaimEvidence, ...],
    config: EvaluationConfig,
    source: Path,
    source_sha256: str,
) -> None:
    output = config.run.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_analysis_tables(output, evaluations, sweep, config)
    write_claim_evidence(evidence, output / "claim_evidence.parquet")
    metadata = RunMetadata(
        input=InputArtifact(path=source.resolve(), sha256=source_sha256),
        configuration=config,
    )
    write_run_metadata(metadata, output / "run.json")


def log_saved_report(output: Path) -> None:
    LOGGER.info("Saved analysis report to %s", output.resolve())
    LOGGER.info(
        "View claims and evidence: uv run python scripts/repro/claims.py --run-dir %s",
        shlex.quote(str(output.resolve())),
    )


def main() -> None:
    config = resolve_config(parse_arguments())
    configure_logging()
    validate_run_claims(config)
    source = DataArtifacts(config.run.data_dir).get_path("olmes_processed")
    LOGGER.info("Reading %s", source)
    source_sha256 = hash_file(source)
    evaluations = load_evaluations(config.run.data_dir, config)
    sweep = sweep_rankings(evaluations, config, progress=LOGGER.info)
    evidence = collect_claim_evidence(sweep, config)
    write_run_outputs(evaluations, sweep, evidence, config, source, source_sha256)
    log_saved_report(config.run.output_dir)


if __name__ == "__main__":
    main()
