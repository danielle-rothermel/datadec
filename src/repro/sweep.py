"""Paper seed/target policy and checkpoint/task/metric sweep execution."""

from collections.abc import Callable

import pandas as pd

from datadec.config import load_olmes_contract
from eval import RankingResult, UnavailableRankingError, predict_recipe_ranking
from eval.results import Checkpoint
from eval.checkpoints import observed_checkpoints
from repro.config import EvaluationConfig
from repro.observations import RankingObservation, observations_frame


def paper_ranking(
    evaluations: pd.DataFrame,
    predictor_size: str,
    predictor_step: int,
    task: str,
    config: EvaluationConfig,
    metric: str | None = None,
) -> RankingResult:
    """Apply configured predictor seeds and gold checkpoint to all catalog recipes."""
    return predict_recipe_ranking(
        evaluations,
        predictor_size,
        config.target.size,
        task,
        metric if metric is not None else config.target.metric,
        config.target.metric,
        predictor_step=predictor_step,
        predicted_step=config.target.step,
        predictor_seeds=config.predictors.seeds_for(predictor_size),
        predicted_seeds=config.target.seeds,
        recipes=tuple(sorted(load_olmes_contract().recipe_map.values())),
    )


def sweep_rankings(
    evaluations: pd.DataFrame,
    config: EvaluationConfig,
    *,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Evaluate every positive checkpoint at or below target compute.

    A row is one checkpoint/task/proxy comparison. Coverage failures are kept
    as unavailable rows with their reason; malformed or duplicate data aborts.
    """
    target_size = config.target.size
    target_step = config.target.step
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
    observations: list[RankingObservation] = []
    target_checkpoint = Checkpoint(target_size, target_step, target_compute)
    for size, group in checkpoints.groupby("params", sort=False):
        if progress:
            progress(f"Evaluating {size}: {len(group)} checkpoints")
        for checkpoint in group.itertuples(index=False):
            for task in config.tasks:
                for metric in config.metrics:
                    try:
                        result = paper_ranking(
                            evaluations,
                            size,
                            int(checkpoint.step),
                            task,
                            config,
                            metric,
                        )
                    except UnavailableRankingError as error:
                        observation = RankingObservation.unavailable(
                            Checkpoint(
                                str(size),
                                int(checkpoint.step),
                                float(checkpoint.compute),
                            ),
                            target_checkpoint,
                            task,
                            metric,
                            schedule_complete=bool(checkpoint.schedule_complete),
                            reason=str(error),
                        )
                    else:
                        observation = RankingObservation.successful(
                            result, schedule_complete=bool(checkpoint.schedule_complete)
                        )
                    observations.append(observation)
    return observations_frame(observations)
