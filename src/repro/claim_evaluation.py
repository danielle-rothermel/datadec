"""Apply configured claim selectors and numerical bounds to a ranking sweep."""

import pandas as pd

from repro.config import AccuracyBound, ClaimAnalysis, EvaluationConfig
from repro.results import ClaimEvidence, ClaimStatus


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


def _summarize_comparisons(available: pd.DataFrame, unavailable: pd.DataFrame) -> str:
    summary = f"{len(available)} available comparisons; {len(unavailable)} unavailable."
    if not available.empty:
        best = available.loc[available["decision_accuracy"].idxmax()]
        summary += (
            f" Best accuracy {best['decision_accuracy']:.6f} at {best['predictor_size']}"
            f" step {best['predictor_step']}, {best['task']}, {best['metric']},"
            f" compute ratio {best['compute_ratio']:.8g}."
        )
    return summary


def _evaluate_accuracy_bound(
    available: pd.DataFrame,
    unavailable: pd.DataFrame,
    tasks: tuple[str, ...],
    bound: AccuracyBound,
) -> tuple[ClaimStatus, str]:
    # One observed proxy/checkpoint above the bound per named task suffices.
    maxima = available.groupby("task")["decision_accuracy"].max()
    if all(task in maxima and maxima[task] > bound.accuracy_gt for task in tasks):
        status = ClaimStatus.SUPPORTED
    elif unavailable.empty and all(task in maxima for task in tasks):
        status = ClaimStatus.NOT_SUPPORTED
    else:
        status = ClaimStatus.INSUFFICIENT_DATA
    return (
        status,
        f" Per-task maxima: {maxima.to_dict()}. Verdict concerns the observed grid only.",
    )


def evaluate_claims(
    sweep: pd.DataFrame, config: EvaluationConfig
) -> tuple[ClaimEvidence, ...]:
    """Link configured claims to observations and explicit limits of inference."""
    results = []
    for claim_id, scope in config.claims.items():
        selected = _select_evidence(sweep, scope, config)
        available = selected.loc[selected["available"]]
        unavailable = selected.loc[~selected["available"]]
        status = (
            ClaimStatus.REQUIRES_JUDGMENT
            if not available.empty
            else ClaimStatus.INSUFFICIENT_DATA
        )
        summary = _summarize_comparisons(available, unavailable)
        if scope.criterion is not None:
            status, bound_summary = _evaluate_accuracy_bound(
                available,
                unavailable,
                config.tasks_for(scope),
                config.criteria[scope.criterion],
            )
            summary += bound_summary
        results.append(
            ClaimEvidence(
                claim_id=claim_id,
                status=status,
                evidence_ids=tuple(int(index) for index in available.index),
                unavailable_ids=tuple(int(index) for index in unavailable.index),
                summary=summary,
                judgment=scope.judgment,
                supporting_tables=scope.supporting_tables
                or config.default_supporting_tables,
            )
        )
    return tuple(results)
