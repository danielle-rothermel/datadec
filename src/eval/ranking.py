"""Predict recipe order at one exact checkpoint from another checkpoint."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


class UnavailableRankingError(ValueError):
    """The requested checkpoint, recipes, seeds, or finite scores are missing."""


@dataclass(frozen=True, slots=True)
class Checkpoint:
    size: str
    step: int
    compute: float


@dataclass(frozen=True, slots=True)
class PairwiseDecision:
    recipe_a: str
    recipe_b: str
    predictor_sign: int
    target_sign: int

    @property
    def correct(self) -> bool:
        return self.predictor_sign == self.target_sign


@dataclass(frozen=True, slots=True)
class SeedRanking:
    seed: str
    predictor_scores: tuple[float, ...]
    decisions: tuple[PairwiseDecision, ...]

    @property
    def decision_accuracy(self) -> float:
        return sum(pair.correct for pair in self.decisions) / len(self.decisions)


@dataclass(frozen=True, slots=True)
class RankingResult:
    predictor: Checkpoint
    predicted: Checkpoint
    task: str
    predictor_task_metric: str
    predicted_task_metric: str
    recipes: tuple[str, ...]
    target_seeds: tuple[str, ...]
    target_scores: tuple[float, ...]
    seed_rankings: tuple[SeedRanking, ...]

    @property
    def decision_accuracy(self) -> float:
        return float(np.mean([r.decision_accuracy for r in self.seed_rankings]))

    @property
    def decision_accuracy_std(self) -> float:
        """Population standard deviation across predictor seeds (ddof=0)."""
        return float(np.std([r.decision_accuracy for r in self.seed_rankings]))

    @property
    def compute_ratio(self) -> float:
        return self.predictor.compute / self.predicted.compute


def _checkpoint_rows(
    evaluations: pd.DataFrame, size: str, step: int, task: str
) -> pd.DataFrame:
    if isinstance(evaluations.index, pd.MultiIndex) and evaluations.index.names == [
        "params",
        "step",
        "task",
    ]:
        try:
            return evaluations.loc[[(size, step, task)]]
        except KeyError as error:
            raise UnavailableRankingError(
                f"missing checkpoint: {size}, step {step}, {task}"
            ) from error
    rows = evaluations.loc[
        (evaluations["params"] == size)
        & (evaluations["step"] == step)
        & (evaluations["task"] == task)
    ]
    if rows.empty:
        raise UnavailableRankingError(
            f"missing checkpoint: {size}, step {step}, {task}"
        )
    return rows


def _scores(
    rows: pd.DataFrame,
    metric: str,
    recipes: tuple[str, ...],
    seeds: tuple[str, ...],
) -> tuple[pd.DataFrame, float]:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and unique")
    selected = rows.loc[rows["seed"].isin(seeds)]
    if selected.duplicated(["data", "seed"]).any():
        raise ValueError("duplicate recipe/seed rows at checkpoint")
    scores = selected.pivot(index="data", columns="seed", values=metric)
    if set(scores.index) != set(recipes) or set(scores.columns) != set(seeds):
        raise UnavailableRankingError("checkpoint does not contain all recipes/seeds")
    scores = scores.reindex(index=list(recipes), columns=list(seeds))
    if not np.isfinite(scores.to_numpy(dtype=float, na_value=np.nan)).all():
        raise UnavailableRankingError(f"missing or non-finite {metric} scores")
    compute = selected["compute"].unique()
    if len(compute) != 1 or not np.isfinite(compute[0]) or compute[0] < 0:
        raise ValueError("checkpoint compute must be one finite nonnegative value")
    return scores, float(compute[0])


def predict_recipe_ranking(
    evaluations: pd.DataFrame,
    predictor_size: str,
    predicted_size: str,
    task: str,
    predictor_task_metric: str,
    predicted_task_metric: str,
    *,
    predictor_step: int,
    predicted_step: int,
    predictor_seeds: tuple[str, ...] | None = None,
    predicted_seeds: tuple[str, ...] | None = None,
    recipes: tuple[str, ...] | None = None,
    predictor_higher_is_better: bool = True,
    predicted_higher_is_better: bool = True,
) -> RankingResult:
    """Compare each predictor seed with target scores averaged across seeds.

    Input columns: params, step, task, data (recipe), seed, compute, and the
    requested metrics. An optional sorted (params, step, task) MultiIndex speeds
    repeated calls. Steps are exact; no interpolation or nearest lookup occurs.
    Omitted recipe/seed sets use the observed predictor recipes and each side's
    observed seeds. Pass explicit sets when checking expected dataset coverage.
    Every selected seed must cover every recipe. Ties match only other ties.
    """
    if predictor_step < 0 or predicted_step < 0:
        raise ValueError("checkpoint steps must be nonnegative")
    predictor = _checkpoint_rows(evaluations, predictor_size, predictor_step, task)
    target = _checkpoint_rows(evaluations, predicted_size, predicted_step, task)
    recipes = (
        recipes if recipes is not None else tuple(sorted(predictor["data"].unique()))
    )
    if len(recipes) < 2 or len(set(recipes)) != len(recipes):
        raise ValueError("at least two unique recipes are required")
    predictor_seeds = (
        predictor_seeds
        if predictor_seeds is not None
        else tuple(sorted(predictor["seed"].unique()))
    )
    predicted_seeds = (
        predicted_seeds
        if predicted_seeds is not None
        else tuple(sorted(target["seed"].unique()))
    )
    pred_scores, pred_compute = _scores(
        predictor, predictor_task_metric, recipes, predictor_seeds
    )
    target_scores, target_compute = _scores(
        target, predicted_task_metric, recipes, predicted_seeds
    )
    if target_compute <= 0:
        raise ValueError("target checkpoint compute must be positive")
    truth = target_scores.mean(axis=1).to_numpy()
    left, right = np.triu_indices(len(recipes), k=1)
    target_signs = np.sign(truth[left] - truth[right]) * (
        1 if predicted_higher_is_better else -1
    )
    rankings = []
    for seed in predictor_seeds:
        scores = pred_scores[seed].to_numpy()
        signs = np.sign(scores[left] - scores[right]) * (
            1 if predictor_higher_is_better else -1
        )
        rankings.append(
            SeedRanking(
                seed=seed,
                predictor_scores=tuple(float(score) for score in scores),
                decisions=tuple(
                    PairwiseDecision(recipes[a], recipes[b], int(p), int(t))
                    for a, b, p, t in zip(left, right, signs, target_signs, strict=True)
                ),
            )
        )
    return RankingResult(
        predictor=Checkpoint(predictor_size, predictor_step, pred_compute),
        predicted=Checkpoint(predicted_size, predicted_step, target_compute),
        task=task,
        predictor_task_metric=predictor_task_metric,
        predicted_task_metric=predicted_task_metric,
        recipes=recipes,
        target_seeds=predicted_seeds,
        target_scores=tuple(float(score) for score in truth),
        seed_rankings=tuple(rankings),
    )
