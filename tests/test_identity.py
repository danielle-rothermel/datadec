from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from datadec.data.preprocess.identity import normalize_step


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0),
        (1250.0, 1250),
        (" 42 ", 42),
        (Decimal(7), 7),
        (np.int64(9), 9),
    ],
)
def test_normalize_step_accepts_integral_int64_values(
    value: object,
    expected: int,
) -> None:
    assert normalize_step(value, row_index=3) == expected


@pytest.mark.parametrize(
    "value",
    [None, pd.NA, True, "", "x", 1.5, np.inf, 2**63, -(2**63) - 1],
)
def test_normalize_step_rejects_invalid_values_with_row_context(value: object) -> None:
    with pytest.raises(ValueError, match="invalid PPL step at row 5"):
        normalize_step(value, row_index=5)
