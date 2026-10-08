"""Claim evidence wrappers for the DataDecide single-scale experiments.

Only explicit numerical bounds receive automatic verdicts. Qualitative claims
retain their observations and a description of the judgment still required.
"""

from dataclasses import dataclass
from enum import UNIQUE, StrEnum, auto, verify

import numpy as np
import pandas as pd

from datadec.config import load_olmes_contract
from repro.ranking import BENCHMARKS, METRICS, PROXY_FAMILIES


@verify(UNIQUE)
class ClaimStatus(StrEnum):
    SUPPORTED = auto()
    NOT_SUPPORTED = auto()
    REQUIRES_JUDGMENT = auto()
    INSUFFICIENT_DATA = auto()


@dataclass(frozen=True, slots=True)
class EvidenceScope:
    tasks: tuple[str, ...]
    metrics: tuple[str, ...]
    judgment: str
    min_compute_ratio: float = 0.0
    max_compute_ratio: float = 1.0
    predictor_size: str | None = None


@dataclass(frozen=True, slots=True)
class ClaimEvidence:
    claim_id: str
    status: ClaimStatus
    evidence_ids: tuple[int, ...]
    unavailable_ids: tuple[int, ...]
    summary: str
    judgment: str
    supporting_tables: tuple[str, ...]


PRIMARY = ("primary_metric",)
CHAR_PROXIES = tuple(family + "_per_char" for family in PROXY_FAMILIES)
LIKELIHOOD_PROXIES = tuple(
    metric for metric in METRICS if metric.startswith(PROXY_FAMILIES)
)
OTHER_TASKS = tuple(
    task
    for task in BENCHMARKS
    if task not in ("arc_easy", "arc_challenge", "mmlu", "hellaswag")
)

# These are analysis scopes, not copies of claim text or paper quotes. Statements
# and source locations continue to have a single owner in the TOML inventory.
SCOPES = {
    "DD-0010": EvidenceScope(
        ("olmes",), PRIMARY, "Judge whether the measured baseline is strong."
    ),
    "DD-0011": EvidenceScope(
        ("olmes",),
        PRIMARY,
        "Compare observed 150M checkpoints with approximately 0.80; a fully trained endpoint may be absent.",
        predictor_size="150M",
    ),
    "DD-0014": EvidenceScope(
        ("mmlu",),
        LIKELIHOOD_PROXIES,
        "Strictly >0.80 within 0.0001 of target compute.",
        max_compute_ratio=0.0001,
    ),
    "DD-0015": EvidenceScope(
        ("arc_easy", "arc_challenge"),
        LIKELIHOOD_PROXIES,
        "Require the >0.80 budget bound separately for both ARC tasks.",
        max_compute_ratio=0.0001,
    ),
    "DD-0016": EvidenceScope(
        ("hellaswag",),
        LIKELIHOOD_PROXIES,
        "Strictly >0.80 within 0.0001 of target compute.",
        max_compute_ratio=0.0001,
    ),
    "DD-0051": EvidenceScope(
        BENCHMARKS,
        PRIMARY,
        "Compare task budget-to-accuracy curves and choose what counts as good prediction.",
    ),
    "DD-0052": EvidenceScope(
        ("arc_easy", "arc_challenge", "mmlu", "hellaswag"),
        PRIMARY,
        "Judge how much less compute constitutes the claimed difference.",
    ),
    "DD-0053": EvidenceScope(
        ("socialiqa",), PRIMARY, "Judge difficulty across observed sizes and budgets."
    ),
    "DD-0055": EvidenceScope(
        BENCHMARKS,
        ("primary_metric", *CHAR_PROXIES),
        "Choose the small-scale regime and equivalence tolerance.",
    ),
    "DD-0148": EvidenceScope(
        BENCHMARKS,
        PRIMARY,
        "Compare tasks at identical checkpoint compute; judge significance.",
    ),
    "DD-0149": EvidenceScope(
        ("arc_easy",), PRIMARY, "Choose small scale and predictable thresholds."
    ),
    "DD-0150": EvidenceScope(
        ("arc_easy", "hellaswag"),
        PRIMARY,
        "Choose a common accuracy threshold and judge the compute gap.",
    ),
    "DD-0164": EvidenceScope(
        ("olmes",),
        PRIMARY,
        "Judge the overall compute trend, accounting for local reversals.",
    ),
    "DD-0165": EvidenceScope(
        ("olmes",),
        PRIMARY,
        "Inspect matched-compute comparisons using the reported tolerance; choose accuracy equivalence tolerance.",
    ),
    "DD-0168": EvidenceScope(
        OTHER_TASKS, PRIMARY, "Judge reliability relative to ARC, MMLU, and HellaSwag."
    ),
    "DD-0169": EvidenceScope(
        ("olmes",),
        PRIMARY,
        "Judge roughly log-linear fit and deviations, not just positive slope.",
    ),
    "DD-0175": EvidenceScope(
        ("arc_easy",),
        PRIMARY,
        "At <=0.00001 target compute, choose what predictable and consistently mean.",
        max_compute_ratio=0.00001,
    ),
    "DD-0176": EvidenceScope(
        ("boolq",),
        PRIMARY,
        "Choose a nontrivial threshold and compare intermediate 1B with other scales.",
    ),
    "DD-0177": EvidenceScope(
        ("hellaswag",),
        PRIMARY,
        "Choose a change point and judge flat-then-log-linear behavior.",
    ),
    "DD-0178": EvidenceScope(
        ("socialiqa",),
        PRIMARY,
        "Choose a change point and judge flat-then-log-linear behavior.",
    ),
    "DD-0179": EvidenceScope(
        ("winogrande",),
        PRIMARY,
        "Choose a change point and judge flat-then-log-linear behavior.",
    ),
    "DD-0194": EvidenceScope(
        BENCHMARKS,
        PRIMARY,
        "Inspect recipe-order crossovers between observed completed scales; judge frequently. Missing endpoints limit coverage.",
    ),
    "DD-0196": EvidenceScope(
        BENCHMARKS,
        ("primary_metric", "correct_prob_per_char", "total_prob_per_char"),
        "Choose small-scale and equivalence thresholds.",
    ),
    "DD-0197": EvidenceScope(
        BENCHMARKS, METRICS, "Compare raw/token/character curves and judge similar."
    ),
    "DD-0198": EvidenceScope(
        BENCHMARKS,
        METRICS,
        "Define optimality across checkpoints and count tasks by normalization.",
    ),
    "DD-0199": EvidenceScope(
        BENCHMARKS, METRICS, "Choose small scales, most, and equivalence tolerance."
    ),
    "DD-0202": EvidenceScope(
        BENCHMARKS,
        ("primary_metric", *CHAR_PROXIES),
        "Classify the observed task curves into the proposed two types.",
    ),
    "DD-0203": EvidenceScope(
        BENCHMARKS,
        ("primary_metric", *CHAR_PROXIES),
        "Identify tasks with near-identical improving curves and choose similarity tolerance.",
    ),
    "DD-0204": EvidenceScope(
        BENCHMARKS,
        ("primary_metric", *CHAR_PROXIES),
        "Identify flat Correct/Total Prob curves approached by other proxies.",
    ),
    "DD-0205": EvidenceScope(
        BENCHMARKS,
        ("primary_metric", *CHAR_PROXIES),
        "Judge the tendency to overtake within the last order of magnitude.",
        min_compute_ratio=0.1,
    ),
    "DD-0206": EvidenceScope(
        BENCHMARKS,
        ("correct_prob_per_char", "total_prob_per_char"),
        "Judge near-target decreases relative to seed variation.",
        min_compute_ratio=0.1,
    ),
    "DD-0207": EvidenceScope(
        BENCHMARKS,
        ("primary_metric", "norm_correct_prob_per_char", "margin_per_char"),
        "Judge tracking; the incorrect-answer penalty is a metric-definition claim requiring separate inspection.",
    ),
    "DD-0208": EvidenceScope(
        BENCHMARKS,
        ("primary_metric", *CHAR_PROXIES),
        "Choose small-scale and benefit thresholds, then count benefiting tasks (claimed count: five).",
    ),
}


def curve_summary(sweep: pd.DataFrame) -> pd.DataFrame:
    """Descriptive fits and reversals per task/metric/size; no significance claims."""
    records = []
    available = sweep.loc[sweep["available"]]
    for (task, metric, size), group in available.groupby(
        ["task", "metric", "predictor_size"]
    ):
        group = group.sort_values("compute")
        x = np.log10(group["compute_ratio"].to_numpy(dtype=float))
        y = group["decision_accuracy"].to_numpy(dtype=float)
        slope = r_squared = None
        if len(x) >= 2 and np.ptp(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            variance = float(np.sum((y - y.mean()) ** 2))
            if np.ptp(y) > 0:
                r_squared = (
                    1 - float(np.sum((y - (slope * x + intercept)) ** 2)) / variance
                )
        records.append(
            {
                "task": task,
                "metric": metric,
                "predictor_size": size,
                "checkpoints": len(group),
                "log10_compute_slope": slope,
                "r_squared": r_squared,
                "adjacent_decreases": int(np.sum(np.diff(y) < 0)),
                "first_accuracy": float(y[0]),
                "last_accuracy": float(y[-1]),
                "max_accuracy": float(y.max()),
            }
        )
    return pd.DataFrame.from_records(records)


def proxy_comparisons(sweep: pd.DataFrame) -> pd.DataFrame:
    """Align proxies with curated target Accuracy at exactly the same checkpoint."""
    available = sweep.loc[sweep["available"]]
    key = ["predictor_size", "predictor_step", "task"]
    baseline = available.loc[
        available["metric"] == "primary_metric", [*key, "decision_accuracy"]
    ].rename(columns={"decision_accuracy": "primary_accuracy"})
    result = available.reset_index().merge(
        baseline, on=key, how="left", validate="many_to_one"
    )
    result["advantage_over_primary"] = (
        result["decision_accuracy"] - result["primary_accuracy"]
    )
    return result[
        [
            "evidence_id",
            *key,
            "metric",
            "compute_ratio",
            "decision_accuracy",
            "primary_accuracy",
            "advantage_over_primary",
        ]
    ]


def recipe_crossovers(evaluations: pd.DataFrame) -> pd.DataFrame:
    """Count strict recipe-order reversals across adjacent completed model scales.

    Seeds are averaged per recipe at each scale. Only observed checkpoints that
    reach their configured training budget enter this diagnostic. Exact ties do
    not count as strict crossovers. The diagnostic does not fit scaling laws.
    """
    rows = evaluations.reset_index(drop=True)
    rows = rows.loc[
        (rows["step"] >= rows["total_steps"]) & rows["task"].isin(BENCHMARKS)
    ]
    scores = (
        rows.groupby(["task", "params", "compute", "data"])["primary_metric"]
        .mean()
        .unstack("data")
    )
    expected_recipes = set(load_olmes_contract().recipe_map.values())
    if set(scores.columns) != expected_recipes:
        raise ValueError("crossover diagnostic requires all catalog recipes")
    records = []
    for task, group in scores.groupby(level="task"):
        group = group.sort_index(level="compute")
        if group.isna().any().any():
            raise ValueError("incomplete recipe coverage for crossover diagnostic")
        a, b = np.triu_indices(len(group.columns), k=1)
        values = group.to_numpy()
        signs = np.sign(values[:, a] - values[:, b])
        for index in range(1, len(group)):
            previous, current = group.index[index - 1], group.index[index]
            records.append(
                {
                    "task": task,
                    "smaller_size": previous[1],
                    "larger_size": current[1],
                    "smaller_compute": previous[2],
                    "larger_compute": current[2],
                    "pair_count": len(a),
                    "strict_crossovers": int(
                        np.sum(signs[index - 1] * signs[index] < 0)
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def evaluate_claims(sweep: pd.DataFrame) -> tuple[ClaimEvidence, ...]:
    """Link all candidate claims to observations and explicit limits of inference."""
    results = []
    for claim_id, scope in SCOPES.items():
        selected = sweep.loc[
            sweep["task"].isin(scope.tasks)
            & sweep["metric"].isin(scope.metrics)
            & (sweep["compute_ratio"] > 0)
            & (sweep["compute_ratio"] >= scope.min_compute_ratio)
            & (sweep["compute_ratio"] <= scope.max_compute_ratio)
        ]
        if scope.predictor_size is not None:
            selected = selected.loc[selected["predictor_size"] == scope.predictor_size]
        available = selected.loc[selected["available"]]
        unavailable = selected.loc[~selected["available"]]
        status = (
            ClaimStatus.REQUIRES_JUDGMENT
            if not available.empty
            else ClaimStatus.INSUFFICIENT_DATA
        )
        summary = (
            f"{len(available)} available comparisons; {len(unavailable)} unavailable."
        )
        if not available.empty:
            best = available.loc[available["decision_accuracy"].idxmax()]
            summary += (
                f" Best accuracy {best['decision_accuracy']:.6f} at {best['predictor_size']}"
                f" step {best['predictor_step']}, {best['task']}, {best['metric']},"
                f" compute ratio {best['compute_ratio']:.8g}."
            )
        if claim_id in ("DD-0014", "DD-0015", "DD-0016"):
            # These claims state an existence bound, not a universal guarantee:
            # one observed proxy/checkpoint above 0.80 per named task suffices.
            maxima = available.groupby("task")["decision_accuracy"].max()
            passed = all(task in maxima and maxima[task] > 0.80 for task in scope.tasks)
            if passed:
                status = ClaimStatus.SUPPORTED
            elif unavailable.empty and all(task in maxima for task in scope.tasks):
                status = ClaimStatus.NOT_SUPPORTED
            else:
                status = ClaimStatus.INSUFFICIENT_DATA
            summary += f" Per-task maxima: {maxima.to_dict()}. Verdict concerns the observed grid only."
        tables = ("curves.csv", "proxy_comparisons.parquet")
        if claim_id == "DD-0165":
            tables = ("matched_compute.csv",)
        elif claim_id == "DD-0194":
            tables = ("recipe_crossovers.csv",)
        results.append(
            ClaimEvidence(
                claim_id=claim_id,
                status=status,
                evidence_ids=tuple(int(index) for index in available.index),
                unavailable_ids=tuple(int(index) for index in unavailable.index),
                summary=summary,
                judgment=scope.judgment,
                supporting_tables=tables,
            )
        )
    return tuple(results)
