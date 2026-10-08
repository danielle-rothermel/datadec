"""DataDecide paper policy around the generic recipe ranking evaluator."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from datadec.config import load_olmes_contract
from datadec.data.artifacts import DataArtifacts
from datadec.data.read import ProcessedTable, read_processed_table
from eval.ranking import RankingResult, UnavailableRankingError, predict_recipe_ranking


BENCHMARKS = (
    "arc_challenge",
    "arc_easy",
    "boolq",
    "csqa",
    "hellaswag",
    "mmlu",
    "openbookqa",
    "piqa",
    "socialiqa",
    "winogrande",
)
TASKS = (*BENCHMARKS, "olmes")
PROXY_FAMILIES = ("correct_prob", "total_prob", "norm_correct_prob", "margin")
NORMALIZATIONS = ("", "_per_token", "_per_char")
METRICS = (
    "primary_metric",
    "acc_raw",
    "acc_per_token",
    "acc_per_char",
    *(family + norm for family in PROXY_FAMILIES for norm in NORMALIZATIONS),
)
SMALL_SEEDS = ("default", "small aux 2", "small aux 3")
TARGET_SEEDS = ("default", "large aux 2", "large aux 3")
KEY = ["params", "step", "data", "seed"]


def _macro_average(rows: pd.DataFrame, task: str, count: int) -> pd.DataFrame:
    groups = rows.groupby(KEY, observed=True, sort=False)
    metadata = ["compute", "total_steps"]
    if (groups["task"].nunique() != count).any():
        raise ValueError(
            f"{task} requires all {count} component tasks at every checkpoint"
        )
    if (groups[metadata].nunique(dropna=False) != 1).any().any():
        raise ValueError(f"inconsistent checkpoint metadata across {task} tasks")
    # A metric is unavailable for the macro task if any component lacks it.
    means = groups[list(METRICS)].mean().where(groups[list(METRICS)].count() == count)
    result = means.join(groups[metadata].first()).reset_index()
    result["task"] = task
    return result


def prepare_evaluations(raw: pd.DataFrame) -> pd.DataFrame:
    """Macro-average 57 MMLU subjects, then ten OLMES benchmarks equally.

    Return a sorted checkpoint/task index for repeated generic helper calls.
    Missing task coverage fails explicitly; missing metric values stay missing.
    """
    required = [*KEY, "task", "compute", "total_steps", *METRICS]
    rows = raw[required].copy()
    if rows[[*KEY, "task"]].isna().any().any():
        raise ValueError("checkpoint identity columns cannot be null")
    if rows.duplicated([*KEY, "task"]).any():
        raise ValueError("duplicate evaluation identity")
    subjects = rows.loc[rows["task"].str.startswith("mmlu_")]
    if subjects["task"].nunique() != 57:
        raise ValueError("expected the 57 MMLU subjects")
    singles = rows.loc[rows["task"].isin(set(BENCHMARKS) - {"mmlu"})]
    benchmarks = pd.concat([singles, _macro_average(subjects, "mmlu", 57)])
    macro = _macro_average(benchmarks, "olmes", len(BENCHMARKS))
    return (
        pd.concat([benchmarks, macro], ignore_index=True)
        .set_index(["params", "step", "task"], drop=False)
        .sort_index()
    )


def load_evaluations(data_dir: Path) -> pd.DataFrame:
    """Read the local processed OLMES table through the standard data API."""
    table = read_processed_table(
        DataArtifacts(data_dir),
        ProcessedTable.OLMES,
        columns=[*KEY, "task", "compute", "total_steps", *METRICS],
    )
    return prepare_evaluations(table.to_pandas())


def paper_ranking(
    evaluations: pd.DataFrame,
    predictor_size: str,
    predictor_step: int,
    task: str,
    metric: str = "primary_metric",
    *,
    target_size: str = "1B",
    target_step: int = 69369,
) -> RankingResult:
    """Use all catalog recipes, three predictor seeds, and three gold seeds."""
    return predict_recipe_ranking(
        evaluations,
        predictor_size,
        target_size,
        task,
        metric,
        "primary_metric",
        predictor_step=predictor_step,
        predicted_step=target_step,
        predictor_seeds=TARGET_SEEDS if predictor_size == "1B" else SMALL_SEEDS,
        predicted_seeds=TARGET_SEEDS if target_size == "1B" else SMALL_SEEDS,
        recipes=tuple(sorted(load_olmes_contract().recipe_map.values())),
    )


def observed_checkpoints(evaluations: pd.DataFrame) -> pd.DataFrame:
    """List actual checkpoint compute and whether the scheduled budget is reached."""
    rows = evaluations.reset_index(drop=True)
    metadata = ["compute", "total_steps"]
    if (
        (rows.groupby(["params", "step"])[metadata].nunique(dropna=False) != 1)
        .any()
        .any()
    ):
        raise ValueError("inconsistent checkpoint compute or schedule")
    result = rows[["params", "step", *metadata]].drop_duplicates().copy()
    result["schedule_complete"] = result["step"] >= result["total_steps"]
    return result.sort_values(["compute", "params", "step"]).reset_index(drop=True)


def select_compute_budget(
    checkpoints: pd.DataFrame,
    size: str,
    budget: float,
    *,
    relative_tolerance: float | None = None,
) -> int | None:
    """Largest observed positive-compute checkpoint <= budget, never interpolate.

    For approximate matched-compute comparisons, a supplied relative tolerance
    bounds the undershoot: (budget - actual) / budget. No match returns None.
    """
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError("compute budget must be finite and positive")
    if relative_tolerance is not None and not 0 <= relative_tolerance <= 1:
        raise ValueError("relative tolerance must be between 0 and 1")
    eligible = checkpoints.loc[
        (checkpoints["params"] == size)
        & (checkpoints["compute"] > 0)
        & (checkpoints["compute"] <= budget)
    ].sort_values("compute")
    if eligible.empty:
        return None
    chosen = eligible.iloc[-1]
    if (
        relative_tolerance is not None
        and (budget - chosen["compute"]) / budget > relative_tolerance
    ):
        return None
    return int(chosen["step"])


def sweep_rankings(
    evaluations: pd.DataFrame,
    *,
    target_size: str = "1B",
    target_step: int = 69369,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Evaluate every positive checkpoint at or below target compute.

    A row is one checkpoint/task/proxy comparison. Coverage failures are kept
    as unavailable rows with their reason; malformed or duplicate data aborts.
    """
    checkpoints = observed_checkpoints(evaluations)
    target = checkpoints.loc[
        (checkpoints["params"] == target_size) & (checkpoints["step"] == target_step)
    ]
    if len(target) != 1 or target.iloc[0]["compute"] <= 0:
        raise UnavailableRankingError(
            "target checkpoint requires positive observed compute"
        )
    target_compute = float(target.iloc[0]["compute"])
    checkpoints = checkpoints.loc[
        (checkpoints["compute"] > 0) & (checkpoints["compute"] <= target_compute)
    ]
    records = []
    for size, group in checkpoints.groupby("params", sort=False):
        if progress:
            progress(f"Evaluating {size}: {len(group)} checkpoints")
        for checkpoint in group.itertuples(index=False):
            for task in TASKS:
                for metric in METRICS:
                    record = {
                        "predictor_size": size,
                        "predictor_step": int(checkpoint.step),
                        "target_size": target_size,
                        "target_step": target_step,
                        "task": task,
                        "metric": metric,
                        "compute": float(checkpoint.compute),
                        "target_compute": target_compute,
                        "compute_ratio": float(checkpoint.compute) / target_compute,
                        "schedule_complete": bool(checkpoint.schedule_complete),
                        "available": False,
                        "reason": "",
                        "decision_accuracy": None,
                        "decision_accuracy_std": None,
                        "seed_accuracies": None,
                        "recipe_count": None,
                        "pair_count": None,
                    }
                    try:
                        result = paper_ranking(
                            evaluations,
                            size,
                            int(checkpoint.step),
                            task,
                            metric,
                            target_size=target_size,
                            target_step=target_step,
                        )
                    except UnavailableRankingError as error:
                        record["reason"] = str(error)
                    else:
                        record.update(
                            available=True,
                            decision_accuracy=result.decision_accuracy,
                            decision_accuracy_std=result.decision_accuracy_std,
                            seed_accuracies=[
                                r.decision_accuracy for r in result.seed_rankings
                            ],
                            recipe_count=len(result.recipes),
                            pair_count=len(result.seed_rankings[0].decisions),
                        )
                    records.append(record)
    result = pd.DataFrame.from_records(records)
    result.index.name = "evidence_id"
    return result


def matched_compute_comparisons(
    sweep: pd.DataFrame, *, relative_tolerance: float
) -> pd.DataFrame:
    """Compare observed scheduled endpoints to other sizes within a budget tolerance."""
    checkpoints = sweep.rename(
        columns={"predictor_size": "params", "predictor_step": "step"}
    )[["params", "step", "compute"]].drop_duplicates()
    primary = sweep.loc[sweep["available"] & (sweep["metric"] == "primary_metric")]
    records = []
    for endpoint in primary.loc[primary["schedule_complete"]].itertuples():
        for size in checkpoints["params"].unique():
            if size == endpoint.predictor_size:
                continue
            step = select_compute_budget(
                checkpoints,
                size,
                endpoint.compute,
                relative_tolerance=relative_tolerance,
            )
            match = primary.loc[
                (primary["predictor_size"] == size)
                & (primary["predictor_step"] == step)
                & (primary["task"] == endpoint.task)
                & ~primary["schedule_complete"]
            ]
            if match.empty:
                continue
            candidate = match.iloc[0]
            records.append(
                {
                    "final_evidence_id": endpoint.Index,
                    "intermediate_evidence_id": int(candidate.name),
                    "relative_compute_gap": 1 - candidate["compute"] / endpoint.compute,
                    "accuracy_difference": candidate["decision_accuracy"]
                    - endpoint.decision_accuracy,
                }
            )
    return pd.DataFrame(
        records,
        columns=[
            "final_evidence_id",
            "intermediate_evidence_id",
            "relative_compute_gap",
            "accuracy_difference",
        ],
    )
