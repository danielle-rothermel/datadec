"""Select related data rows and compute descriptive measurements."""

import pandas as pd

from repro.config import EvidenceSelection, EvaluationConfig
from repro.results import (
    ClaimEvidence,
    PredictionMeasurement,
    TaskMeasurement,
)


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


def _measure_task(selected: pd.DataFrame, task: str) -> TaskMeasurement:
    rows = selected.loc[selected["task"] == task]
    available = rows.loc[rows["available"]]
    unavailable = rows.loc[~rows["available"]]
    best = None
    if not available.empty:
        row = available.loc[available["decision_accuracy"].idxmax()]
        best = PredictionMeasurement(
            evidence_id=int(row.name),
            predictor_size=str(row["predictor_size"]),
            predictor_step=int(row["predictor_step"]),
            metric=str(row["metric"]),
            compute=float(row["compute"]),
            compute_ratio=float(row["compute_ratio"]),
            decision_accuracy=float(row["decision_accuracy"]),
            decision_accuracy_std=float(row["decision_accuracy_std"]),
            seed_accuracies=tuple(float(value) for value in row["seed_accuracies"]),
        )
    return TaskMeasurement(
        task=task,
        available_comparisons=len(available),
        unavailable_comparisons=len(unavailable),
        best=best,
    )


def collect_claim_evidence(
    sweep: pd.DataFrame, config: EvaluationConfig
) -> tuple[ClaimEvidence, ...]:
    """Collect configured row links and per-task maxima without interpretation."""
    results = []
    for claim_id, scope in config.claims.items():
        selected = _select_evidence(sweep, scope, config)
        measurements = tuple(
            _measure_task(selected, task) for task in config.tasks_for(scope)
        )
        results.append(
            ClaimEvidence(
                claim_id=claim_id,
                evidence_ids=tuple(
                    int(index) for index in selected.loc[selected["available"]].index
                ),
                unavailable_ids=tuple(
                    int(index) for index in selected.loc[~selected["available"]].index
                ),
                measurements=measurements,
                related_tables=scope.related_tables or config.default_related_tables,
            )
        )
    return tuple(results)
