"""Process released predictions and aligned loss observations; no fitting."""

import pandas as pd

from datadec.config import load_scaling_law_contract
from datadec.data.artifacts import DataArtifacts
from datadec.data.read import ProcessedTable, read_processed_table
from datadec.recipes import RecipeNameResolver, load_recipe_name_resolver
from eval.prediction_errors import prediction_errors
from eval.recipe_scores import MultiRecipeScores
from eval.results import RecipeRanking
from eval.scaling_observations import scaling_law_observations
from repro.config import EvaluationConfig
from repro.table_evidence import EvidenceTable


def released_prediction_errors(
    predictions: pd.DataFrame, config: EvaluationConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = predictions.loc[
        predictions["metric"].isin(config.measurements.scaling_metrics)
        & predictions["setup"].isin(config.measurements.scaling_setups)
    ].copy()
    if selected.duplicated(["task", "mix", "metric", "setup"]).any():
        raise ValueError("duplicate released prediction identity")
    official = load_recipe_name_resolver()
    contract = load_scaling_law_contract()
    aliases = dict(official.aliases)
    aliases.update(
        {
            source: official.resolve(recipe)
            for source, recipe in contract.source_group_map.items()
        }
    )
    aliases.update(
        {
            source: aliases[canonical]
            for source, canonical in contract.source_group_aliases.items()
        }
    )
    resolver = RecipeNameResolver.from_mapping(aliases)
    records, summaries = [], []
    for (task, metric, setup), group in selected.groupby(
        ["task", "metric", "setup"], sort=True
    ):
        predicted = MultiRecipeScores.from_named_scores(
            dict(zip(group["mix"], group["stacked_pred"], strict=True)),
            resolver=resolver,
        )
        observed = MultiRecipeScores.from_named_scores(
            dict(zip(group["mix"], group["stacked_y"], strict=True)), resolver=resolver
        )
        errors = prediction_errors(predicted, observed)
        for row in group.itertuples(index=False):
            recipe = resolver.resolve(row.mix)
            error = errors[recipe]
            records.append(
                dict(
                    task=task,
                    metric=metric,
                    setup=setup,
                    recipe=recipe.value,
                    predicted=error.predicted,
                    observed=error.observed,
                    absolute_error=error.absolute_error,
                    relative_error=error.relative_error,
                    source_file=row.source_file,
                    source_unit=row.source_unit,
                )
            )
        relative = [
            error.relative_error
            for error in errors.values()
            if error.relative_error is not None
        ]
        summaries.append(
            dict(
                task=task,
                metric=metric,
                setup=setup,
                recipe_count=len(errors),
                mean_absolute_error=sum(
                    error.absolute_error for error in errors.values()
                )
                / len(errors),
                mean_relative_error=sum(relative) / len(relative) if relative else None,
                relative_error_count=len(relative),
                decision_accuracy=RecipeRanking(predicted, True).decision_accuracy(
                    observed, target_higher_is_better=True
                ),
            )
        )
    return pd.DataFrame(records), pd.DataFrame(summaries)


def scaling_measurements(config: EvaluationConfig) -> dict[EvidenceTable, pd.DataFrame]:
    artifacts = DataArtifacts(config.run.data_dir)
    evaluations = read_processed_table(
        artifacts,
        ProcessedTable.SCALING_LAW_EVALUATIONS,
        columns=[
            "source_file",
            "data",
            "params",
            "seed",
            "step",
            "task",
            "compute",
            *config.measurements.scaling_metrics,
        ],
    ).to_pandas()
    losses = read_processed_table(
        artifacts,
        ProcessedTable.SCALING_LAW_CHECKPOINT_LOSSES,
        columns=[
            "source_file",
            "data",
            "params",
            "seed",
            "step",
            "compute",
            config.measurements.loss_metric,
        ],
    ).to_pandas()
    observations = pd.concat(
        [
            scaling_law_observations(
                evaluations,
                losses,
                metric=metric,
                loss_metric=config.measurements.loss_metric,
            )
            for metric in config.measurements.scaling_metrics
        ],
        ignore_index=True,
    )
    predictions = read_processed_table(
        artifacts, ProcessedTable.PUBLISHED_RESULTS_CHEAP_DECISIONS
    ).to_pandas()
    errors, summaries = released_prediction_errors(predictions, config)
    return {
        EvidenceTable.SCALING_OBSERVATIONS: observations,
        EvidenceTable.SCALING_ERRORS: errors,
        EvidenceTable.SCALING_SUMMARIES: summaries,
    }
