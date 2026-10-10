"""Build numerical datasets from configured local sources."""

from pathlib import Path

import pandas as pd

from datadec.data.artifacts import DataArtifacts
from repro.budget_measurements import budget_measurements
from repro.config import EvaluationConfig
from repro.diagnostics.compute_matches import matched_compute_comparisons
from repro.diagnostics.crossovers import recipe_crossovers
from repro.diagnostics.curves import curve_summary
from repro.diagnostics.proxies import proxy_comparisons
from repro.scaling_measurements import scaling_measurements
from repro.score_measurements import score_measurements
from repro.table_evidence import EvidenceTable


def additional_sources(config: EvaluationConfig) -> tuple[Path, ...]:
    if not config.measurements.scaling_metrics:
        return ()
    artifacts = DataArtifacts(config.run.data_dir)
    return (
        artifacts.scaling_law_evaluations_path(),
        artifacts.scaling_law_checkpoint_losses_path(),
        artifacts.published_result_table_path("cheap_decisions"),
    )


def build_analysis_tables(
    evaluations: pd.DataFrame, sweep: pd.DataFrame, config: EvaluationConfig
) -> dict[EvidenceTable, pd.DataFrame]:
    tables = {
        EvidenceTable.CURVES: curve_summary(sweep),
        EvidenceTable.PROXIES: proxy_comparisons(
            sweep, baseline_metric=config.target.metric
        ),
        EvidenceTable.CROSSOVERS: recipe_crossovers(
            evaluations, tasks=config.benchmarks, metric=config.target.metric
        ),
        EvidenceTable.COMPUTE_MATCHES: matched_compute_comparisons(
            sweep,
            baseline_metric=config.target.metric,
            relative_tolerance=config.run.matched_compute_tolerance,
        ),
    }
    if config.measurements.score_metrics:
        tables.update(score_measurements(evaluations, config, sweep))
    if config.measurements.budget_ratios:
        tables[EvidenceTable.BUDGET_ACCURACY] = budget_measurements(evaluations, config)
    if config.measurements.scaling_metrics:
        tables.update(scaling_measurements(config))
    for kind, frame in tables.items():
        tables[kind] = frame.reset_index(drop=True).rename_axis("row_id")
    return tables


def write_measurement_tables(
    tables: dict[EvidenceTable, pd.DataFrame], output: Path
) -> None:
    for kind, frame in tables.items():
        if kind.value.endswith(".parquet"):
            frame.to_parquet(output / kind.value, index=True)
        else:
            frame.to_csv(output / kind.value, index=False)
