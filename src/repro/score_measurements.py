"""Persist absolute scores, recipe summaries, and noise/spread observations."""

import pandas as pd

from datadec.config import load_olmes_contract
from eval.checkpoint_scores import UnavailableRankingError
from eval.checkpoints import observed_checkpoints
from eval.score_statistics import recipe_scores_at_checkpoint, score_noise_and_spread
from repro.config import EvaluationConfig
from repro.table_evidence import EvidenceTable


def score_measurements(
    evaluations: pd.DataFrame,
    config: EvaluationConfig,
    sweep: pd.DataFrame | None = None,
) -> dict[EvidenceTable, pd.DataFrame]:
    raw_records, recipe_records, noise_records = [], [], []
    recipes = tuple(sorted(load_olmes_contract().recipe_map.values()))
    checkpoints = observed_checkpoints(evaluations)
    explicit = {
        (point.size, point.step) for point in config.measurements.score_checkpoints
    }
    observed = set(zip(checkpoints["params"], checkpoints["step"], strict=True))
    if explicit - observed:
        raise UnavailableRankingError(
            f"missing selected score checkpoints: {sorted(explicit - observed)}"
        )
    selected = checkpoints.schedule_complete | pd.Series(
        [
            (size, step) in explicit
            for size, step in zip(
                checkpoints["params"], checkpoints["step"], strict=True
            )
        ],
        index=checkpoints.index,
    )
    for checkpoint in checkpoints.loc[selected].itertuples(index=False):
        for task in config.tasks:
            for metric in config.measurements.score_metrics:
                identity = dict(
                    size=checkpoint.params,
                    step=int(checkpoint.step),
                    compute=checkpoint.compute,
                    task=task,
                    metric=metric,
                    schedule_complete=bool(checkpoint.schedule_complete),
                )
                seeds = config.predictors.seeds_for(checkpoint.params)
                try:
                    scores = recipe_scores_at_checkpoint(
                        evaluations,
                        checkpoint.params,
                        int(checkpoint.step),
                        task,
                        metric,
                        seeds=seeds,
                        recipes=recipes,
                    )
                except UnavailableRankingError as error:
                    noise_records.append(
                        dict(
                            **identity,
                            available=False,
                            reason=str(error),
                            seed_count=len(seeds),
                            recipe_count=len(recipes),
                            ddof=config.measurements.noise_ddof,
                            noise=None,
                            spread=None,
                        )
                    )
                    continue
                statistics = score_noise_and_spread(
                    scores, ddof=config.measurements.noise_ddof
                )
                means, stds = (
                    statistics.summary.mean_per_recipe_scores,
                    statistics.summary.std_per_recipe_scores,
                )
                for seed, values in scores.per_seed_scores.items():
                    raw_records.extend(
                        dict(**identity, seed=seed, recipe=recipe.value, score=value)
                        for recipe, value in values.items()
                    )
                recipe_records.extend(
                    dict(
                        **identity,
                        recipe=recipe.value,
                        seed_count=len(seeds),
                        ddof=config.measurements.noise_ddof,
                        mean_score=means[recipe],
                        score_std=stds[recipe],
                    )
                    for recipe in scores.recipes
                )
                noise_records.append(
                    dict(
                        **identity,
                        available=True,
                        reason="",
                        seed_count=len(seeds),
                        recipe_count=len(scores.recipes),
                        ddof=config.measurements.noise_ddof,
                        noise=statistics.noise,
                        spread=statistics.spread,
                    )
                )
    noise_frame = pd.DataFrame(noise_records)
    if sweep is not None and not noise_frame.empty:
        accuracies = sweep.loc[
            sweep["available"],
            [
                "predictor_size",
                "predictor_step",
                "task",
                "metric",
                "decision_accuracy",
                "aggregate_decision_accuracy",
            ],
        ].rename(columns={"predictor_size": "size", "predictor_step": "step"})
        noise_frame = noise_frame.merge(
            accuracies,
            on=["size", "step", "task", "metric"],
            how="left",
            validate="one_to_one",
        )
    else:
        noise_frame["decision_accuracy"] = None
        noise_frame["aggregate_decision_accuracy"] = None
    return {
        EvidenceTable.RECIPE_SCORES: pd.DataFrame(raw_records),
        EvidenceTable.SCORE_SUMMARIES: pd.DataFrame(recipe_records),
        EvidenceTable.NOISE_SPREAD: noise_frame,
    }
