"""Evaluate configured per-run budgets without selecting on observed accuracy."""

import pandas as pd

from datadec.config import load_olmes_contract
from eval.budgets import ranking_at_compute_budget
from eval.checkpoint_scores import UnavailableRankingError
from repro.config import EvaluationConfig


def budget_measurements(
    evaluations: pd.DataFrame, config: EvaluationConfig
) -> pd.DataFrame:
    target = evaluations.loc[
        (evaluations["params"] == config.target.size)
        & (evaluations["step"] == config.target.step)
    ]
    target_compute = float(target["compute"].iloc[0])
    records = []
    recipes = tuple(sorted(load_olmes_contract().recipe_map.values()))
    for size in evaluations["params"].unique():
        for ratio in config.measurements.budget_ratios:
            for task in config.tasks:
                for metric in config.measurements.budget_metrics:
                    identity = dict(
                        size=size,
                        task=task,
                        metric=metric,
                        requested_ratio=ratio,
                        requested_compute=ratio * target_compute,
                    )
                    try:
                        selected = ranking_at_compute_budget(
                            evaluations,
                            size,
                            config.target.size,
                            task,
                            metric,
                            config.target.metric,
                            compute_budget=ratio * target_compute,
                            predicted_step=config.target.step,
                            predictor_seeds=config.predictors.seeds_for(size),
                            predicted_seeds=config.target.seeds,
                            recipes=recipes,
                        )
                    except UnavailableRankingError as error:
                        records.append(
                            dict(
                                **identity,
                                available=False,
                                reason=str(error),
                                step=None,
                                actual_ratio=None,
                                undershoot=None,
                                seed_count=None,
                                total_predictor_compute_per_recipe=None,
                                decision_accuracy=None,
                                aggregate_decision_accuracy=None,
                            )
                        )
                        continue
                    result = selected.ranking
                    records.append(
                        dict(
                            **identity,
                            available=True,
                            reason="",
                            step=result.predictor.step,
                            actual_ratio=result.compute_ratio,
                            undershoot=selected.relative_undershoot,
                            seed_count=selected.predictor_seed_count,
                            total_predictor_compute_per_recipe=selected.total_predictor_compute_per_recipe,
                            decision_accuracy=result.decision_accuracy(),
                            aggregate_decision_accuracy=result.aggregate_decision_accuracy(),
                        )
                    )
    return pd.DataFrame(records)
