"""Select related data rows and compute descriptive measurements."""

import pandas as pd

from repro.config import EvidenceSelection, EvaluationConfig
from repro.results import ClaimEvidence, TaskEvidence


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
        best_evidence_id=int(available["decision_accuracy"].idxmax())
        if not available.empty
        else None,
    )


def collect_claim_evidence(
    sweep: pd.DataFrame, config: EvaluationConfig
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
    return tuple(results)
