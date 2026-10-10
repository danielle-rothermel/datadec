"""Select related data rows and compute descriptive measurements."""

from dataclasses import replace

import pandas as pd

from eval.approaches import PredictionApproach
from repro.approaches import accuracy_column
from repro.config import EvidenceSelection, EvaluationConfig, TableSelection
from repro.results import ClaimEvidence, TaskEvidence
from repro.table_evidence import EvidenceTable, TableEvidence


def _select_evidence(
    sweep: pd.DataFrame, scope: EvidenceSelection, config: EvaluationConfig
) -> pd.DataFrame:
    selected = sweep.loc[
        sweep["task"].isin(config.tasks_for(scope))
        & sweep["metric"].isin(config.metrics_for(scope))
        & (sweep["compute_ratio"] > 0)
        & (sweep["compute_ratio"] >= scope.min_compute_ratio)
        & (sweep["compute_ratio"] <= scope.max_compute_ratio)
    ]
    if scope.predictor_size is not None:
        selected = selected.loc[selected["predictor_size"] == scope.predictor_size]
    return selected


def _link_task(selected: pd.DataFrame, task: str) -> TaskEvidence:
    rows = selected.loc[selected["task"] == task]
    available = rows.loc[rows["available"]]
    unavailable = rows.loc[~rows["available"]]
    return TaskEvidence(
        task=task,
        evidence_ids=tuple(int(index) for index in available.index),
        unavailable_ids=tuple(int(index) for index in unavailable.index),
        best_per_seed_evidence_id=int(
            available[accuracy_column(PredictionApproach.PER_SEED)].idxmax()
        )
        if not available.empty
        else None,
        best_aggregate_evidence_id=int(
            available[accuracy_column(PredictionApproach.AGGREGATE)].idxmax()
        )
        if not available.empty
        else None,
    )


def collect_claim_evidence(
    sweep: pd.DataFrame,
    config: EvaluationConfig,
    tables: dict[EvidenceTable, pd.DataFrame] | None = None,
) -> tuple[ClaimEvidence, ...]:
    """Collect configured row links and per-task maxima without interpretation."""
    results = []
    for claim_id, scope in config.claims.items():
        selected = _select_evidence(sweep, scope, config)
        tasks = tuple(_link_task(selected, task) for task in config.tasks_for(scope))
        results.append(
            ClaimEvidence(
                claim_id=claim_id,
                tasks=tasks,
                related_tables=scope.related_tables or config.default_related_tables,
            )
        )
    if tables is None:
        return tuple(results)
    by_id = {result.claim_id: result for result in results}

    for claim_id in dict.fromkeys((*config.claims, *config.measurement_claims)):
        result = by_id.get(claim_id, ClaimEvidence(claim_id, (), ()))
        links = []
        if claim_id in config.claims and EvidenceTable.BUDGET_ACCURACY in tables:
            scope = config.claims[claim_id]
            selected = select_table_rows(
                tables[EvidenceTable.BUDGET_ACCURACY],
                TableSelection(
                    table=EvidenceTable.BUDGET_ACCURACY,
                    tasks=config.tasks_for(scope),
                    metrics=config.metrics_for(scope),
                    sizes=(scope.predictor_size,) if scope.predictor_size else (),
                ),
            )
            selected = selected.loc[
                (selected["actual_ratio"] >= scope.min_compute_ratio)
                & (selected["actual_ratio"] <= scope.max_compute_ratio)
            ]
            links.append(
                TableEvidence(
                    EvidenceTable.BUDGET_ACCURACY, tuple(map(int, selected.index))
                )
            )
        for selection in config.measurement_claims.get(claim_id, ()):
            if selection.table not in tables:
                raise ValueError(
                    f"configured evidence table was not produced: {selection.table}"
                )
            selected = select_table_rows(tables[selection.table], selection)
            links.append(
                TableEvidence(selection.table, tuple(map(int, selected.index)))
            )
        for name in result.related_tables:
            kind = EvidenceTable(name)
            if kind not in tables:
                continue
            frame = tables[kind]
            if kind is EvidenceTable.COMPUTE_MATCHES:
                frame = frame.loc[frame["final_evidence_id"].isin(result.evidence_ids)]
            elif kind is EvidenceTable.PROXIES:
                frame = frame.loc[frame["evidence_id"].isin(result.evidence_ids)]
            else:
                scope = config.claims[claim_id]
                frame = select_table_rows(
                    frame,
                    TableSelection(
                        table=kind,
                        tasks=config.tasks_for(scope),
                        metrics=config.metrics_for(scope) if "metric" in frame else (),
                        sizes=(scope.predictor_size,) if scope.predictor_size else (),
                    ),
                )
            links.append(TableEvidence(kind, tuple(map(int, frame.index))))
        by_id[claim_id] = replace(result, table_links=tuple(links))
    return tuple(by_id.values())


def select_table_rows(frame: pd.DataFrame, selection: TableSelection) -> pd.DataFrame:
    selected = frame
    if frame.empty:
        return selected
    if "available" in selected:
        selected = selected.loc[selected["available"]]
    for column, values in (
        ("task", selection.tasks),
        ("metric", selection.metrics),
        ("size" if "size" in frame else "predictor_size", selection.sizes),
        ("step" if "step" in frame else "predictor_step", selection.steps),
        ("setup", selection.setups),
    ):
        if values:
            selected = selected.loc[selected[column].isin(values)]
    return selected
