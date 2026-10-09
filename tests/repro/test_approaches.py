import pandas as pd

from eval.approaches import PredictionApproach
from repro.approaches import accuracy_column, approach_rankings


def test_persisted_approach_names_and_accuracy_columns():
    assert {
        approach.value: accuracy_column(approach) for approach in PredictionApproach
    } == {
        "per_seed": "decision_accuracy",
        "aggregate": "aggregate_decision_accuracy",
    }


def test_diagnostic_approaches_preserve_row_ids_without_mutating_input():
    sweep = pd.DataFrame(
        {
            "decision_accuracy": [0.5, 0.75],
            "aggregate_decision_accuracy": [0.6, 0.8],
            "seed_accuracies": [(0.4, 0.6), (0.7, 0.8)],
            "decision_accuracy_std": [0.1, 0.05],
        },
        index=[7, 12],
    )
    original = sweep.copy(deep=True)
    result = approach_rankings(sweep)
    assert result.index.tolist() == [7, 12, 7, 12]
    assert result["approach"].tolist() == [
        "per_seed",
        "per_seed",
        "aggregate",
        "aggregate",
    ]
    assert result["decision_accuracy"].tolist() == [0.5, 0.75, 0.6, 0.8]
    assert "decision_accuracy_std" not in result
    assert "seed_accuracies" not in result
    pd.testing.assert_frame_equal(sweep, original)
