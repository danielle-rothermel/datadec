"""Prediction errors in source score units, aligned by recipe identity."""

from dataclasses import dataclass

import numpy as np

from datadec.recipes import DataRecipeName
from eval.recipe_scores import MultiRecipeScores


@dataclass(frozen=True, slots=True)
class PredictionError:
    predicted: float
    observed: float

    @property
    def absolute_error(self) -> float:
        return abs(self.predicted - self.observed)

    @property
    def relative_error(self) -> float | None:
        return self.absolute_error / abs(self.observed) if self.observed != 0 else None


def prediction_errors(
    predicted_scores: MultiRecipeScores, observed_scores: MultiRecipeScores
) -> dict[DataRecipeName, PredictionError]:
    if not predicted_scores or set(predicted_scores) != set(observed_scores):
        raise ValueError("prediction errors require matching nonempty recipe sets")
    if not all(
        np.isfinite(value)
        for scores in (predicted_scores, observed_scores)
        for value in scores.values()
    ):
        raise ValueError("prediction error scores must be finite")
    return {
        recipe: PredictionError(predicted_scores[recipe], observed_scores[recipe])
        for recipe in predicted_scores
    }
