"""Compute structured claim measurements and configured numerical verdicts."""

import pandas as pd

from repro.config import ClaimAnalysis, EvaluationConfig
from repro.results import (
    ClaimEvidence,
    ClaimStatus,
    PredictionMeasurement,
    TaskMeasurement,
)


def _select_evidence(
    sweep: pd.DataFrame, scope: ClaimAnalysis, config: EvaluationConfig
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


def _measure_task(
    selected: pd.DataFrame, task: str, accuracy_gt: float | None
) -> TaskMeasurement:
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
    passes_bound = None
    if accuracy_gt is not None and best is not None:
        if best.decision_accuracy > accuracy_gt:
            passes_bound = True
        elif unavailable.empty:
            passes_bound = False
    return TaskMeasurement(
        task=task,
        available_comparisons=len(available),
        unavailable_comparisons=len(unavailable),
        best=best,
        accuracy_gt=accuracy_gt,
        passes_bound=passes_bound,
    )


def _claim_status(
    measurements: tuple[TaskMeasurement, ...], has_criterion: bool
) -> ClaimStatus:
    if not has_criterion:
        return (
            ClaimStatus.REQUIRES_JUDGMENT
            if any(m.best is not None for m in measurements)
            else ClaimStatus.INSUFFICIENT_DATA
        )
    if all(m.passes_bound is True for m in measurements):
        return ClaimStatus.SUPPORTED
    if all(m.best is not None and m.unavailable_comparisons == 0 for m in measurements):
        return ClaimStatus.NOT_SUPPORTED
    return ClaimStatus.INSUFFICIENT_DATA


def evaluate_claims(
    sweep: pd.DataFrame, config: EvaluationConfig
) -> tuple[ClaimEvidence, ...]:
    """Calculate per-task observations and verdicts without rendering prose."""
    results = []
    for claim_id, scope in config.claims.items():
        selected = _select_evidence(sweep, scope, config)
        accuracy_gt = (
            config.criteria[scope.criterion].accuracy_gt
            if scope.criterion is not None
            else None
        )
        measurements = tuple(
            _measure_task(selected, task, accuracy_gt)
            for task in config.tasks_for(scope)
        )
        results.append(
            ClaimEvidence(
                claim_id=claim_id,
                status=_claim_status(measurements, scope.criterion is not None),
                evidence_ids=tuple(
                    int(index) for index in selected.loc[selected["available"]].index
                ),
                unavailable_ids=tuple(
                    int(index) for index in selected.loc[~selected["available"]].index
                ),
                measurements=measurements,
                judgment=scope.judgment,
                supporting_tables=scope.supporting_tables
                or config.default_supporting_tables,
            )
        )
    return tuple(results)
