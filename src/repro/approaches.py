"""Map canonical ranking columns into labelled diagnostic comparisons."""

import pandas as pd

from eval.approaches import PredictionApproach


_ACCURACY_COLUMNS = {
    PredictionApproach.PER_SEED: "decision_accuracy",
    PredictionApproach.AGGREGATE: "aggregate_decision_accuracy",
}


def accuracy_column(approach: PredictionApproach) -> str:
    return _ACCURACY_COLUMNS[approach]


def approach_rankings(sweep: pd.DataFrame) -> pd.DataFrame:
    """Keep both approaches separate while retaining canonical evidence IDs."""
    frames = []
    for approach, column in _ACCURACY_COLUMNS.items():
        frame = sweep.drop(
            columns=[
                "aggregate_decision_accuracy",
                "seed_accuracies",
                "decision_accuracy_std",
            ]
        ).copy()
        frame["approach"] = approach.value
        frame["decision_accuracy"] = sweep[column]
        frames.append(frame)
    return pd.concat(frames)
