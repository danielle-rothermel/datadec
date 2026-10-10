import pandas as pd
import pytest

from eval.checkpoints import select_compute_budget


def test_budget_lookup_reports_no_match_and_enforces_tolerance():
    checkpoints = pd.DataFrame(
        dict(params=["a"] * 4, step=[0, 10, 20, 30], compute=[0, 90, 110, 200])
    )
    assert select_compute_budget(checkpoints, "a", 100) == 10
    assert select_compute_budget(checkpoints, "a", 110) == 20
    assert select_compute_budget(checkpoints, "a", 50) is None
    assert select_compute_budget(checkpoints, "b", 100) is None
    assert select_compute_budget(checkpoints, "a", 100, relative_tolerance=0.05) is None
    assert select_compute_budget(checkpoints, "a", 100, relative_tolerance=0.1) == 10
    with pytest.raises(ValueError):
        select_compute_budget(checkpoints, "a", float("inf"))
