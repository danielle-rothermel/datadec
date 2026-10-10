"""Load linked saved rows and render numerical evidence with Rich."""

from collections.abc import Mapping
from pathlib import Path

import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from repro.results import ClaimEvidence
from repro.table_evidence import EvidenceTable, TableEvidence


_DISPLAY_COLUMNS = {
    EvidenceTable.BUDGET_ACCURACY: (
        "size",
        "task",
        "metric",
        "requested_ratio",
        "step",
        "actual_ratio",
        "undershoot",
        "seed_count",
        "total_predictor_compute_per_recipe",
        "decision_accuracy",
        "aggregate_decision_accuracy",
    ),
    EvidenceTable.RECIPE_SCORES: (
        "size",
        "step",
        "schedule_complete",
        "task",
        "metric",
        "recipe",
        "seed",
        "score",
    ),
    EvidenceTable.SCORE_SUMMARIES: (
        "size",
        "step",
        "schedule_complete",
        "task",
        "metric",
        "recipe",
        "seed_count",
        "ddof",
        "mean_score",
        "score_std",
    ),
    EvidenceTable.NOISE_SPREAD: (
        "size",
        "step",
        "schedule_complete",
        "task",
        "metric",
        "seed_count",
        "recipe_count",
        "ddof",
        "noise",
        "spread",
        "decision_accuracy",
        "aggregate_decision_accuracy",
    ),
    EvidenceTable.SCALING_OBSERVATIONS: (
        "data",
        "params",
        "seed",
        "step",
        "task",
        "metric",
        "loss_metric",
        "evaluation_compute",
        "score",
        "loss",
        "available",
        "evaluation_source",
        "loss_source",
    ),
    EvidenceTable.SCALING_ERRORS: (
        "task",
        "metric",
        "setup",
        "recipe",
        "predicted",
        "observed",
        "absolute_error",
        "relative_error",
        "source_file",
        "source_unit",
    ),
    EvidenceTable.SCALING_SUMMARIES: (
        "task",
        "metric",
        "setup",
        "recipe_count",
        "mean_absolute_error",
        "mean_relative_error",
        "relative_error_count",
        "decision_accuracy",
    ),
    EvidenceTable.CURVES: (
        "approach",
        "task",
        "metric",
        "predictor_size",
        "checkpoints",
        "log10_compute_slope",
        "r_squared",
        "adjacent_decreases",
        "first_accuracy",
        "last_accuracy",
    ),
    EvidenceTable.PROXIES: (
        "approach",
        "task",
        "predictor_size",
        "predictor_step",
        "metric",
        "compute_ratio",
        "decision_accuracy",
        "primary_accuracy",
        "advantage_over_primary",
    ),
    EvidenceTable.COMPUTE_MATCHES: (
        "approach",
        "final_evidence_id",
        "intermediate_evidence_id",
        "relative_compute_gap",
        "accuracy_difference",
    ),
    EvidenceTable.CROSSOVERS: (
        "task",
        "smaller_size",
        "larger_size",
        "smaller_compute",
        "larger_compute",
        "pair_count",
        "strict_crossovers",
    ),
}


def load_linked_tables(
    run_dir: Path,
    evidence: tuple[ClaimEvidence, ...],
    available_tables: tuple[EvidenceTable, ...],
) -> Mapping[EvidenceTable, pd.DataFrame]:
    requested: dict[EvidenceTable, set[int]] = {}
    for claim in evidence:
        for link in claim.table_links:
            if link.table not in available_tables:
                raise ValueError(f"linked table is absent from manifest: {link.table}")
            requested.setdefault(link.table, set()).update(link.row_ids)
    tables = {}
    for kind, ids in requested.items():
        path = run_dir / kind.value
        if kind.value.endswith(".parquet"):
            frame = (
                pd.read_parquet(path, filters=[("row_id", "in", sorted(ids))])
                if ids
                else pd.read_parquet(path).iloc[:0]
            )
        else:
            frame = pd.read_csv(path).rename_axis("row_id")
            frame = frame.loc[frame.index.isin(ids)]
        if not frame.index.is_unique or set(ids) - set(frame.index):
            raise ValueError(f"missing or duplicate evidence row IDs in {kind}")
        tables[kind] = frame
    return tables


def _cell(value: object) -> Text:
    if pd.isna(value):
        return Text("—")
    if isinstance(value, float):
        return Text(f"{value:.6g}", overflow="fold")
    return Text(str(value), overflow="fold")


def print_linked_table(
    link: TableEvidence, frame: pd.DataFrame, console: Console, *, max_rows: int | None
) -> None:
    selected = frame.loc[list(link.row_ids)]
    displayed = selected if max_rows is None else selected.head(max_rows)
    console.print(
        Text(
            f"{link.table.value} | Rows shown: {len(displayed)} / {len(selected)}",
            style="bold green",
        )
    )
    if selected.empty:
        return
    columns = _DISPLAY_COLUMNS[link.table]
    # Repeat IDs across narrow column groups so terminal output stays readable.
    for offset in range(0, len(columns), 5):
        group = columns[offset : offset + 5]
        table = Table(box=box.SIMPLE_HEAD, padding=(0, 1))
        table.add_column("row_id", overflow="fold")
        for name in group:
            table.add_column(name.replace("_", " "), overflow="fold")
        for index, row in displayed.iterrows():
            table.add_row(_cell(index), *(_cell(row[column]) for column in group))
        console.print(table)
    if len(displayed) < len(selected):
        console.print("Use --max-rows 0 to display all linked rows.", style="dim")
