from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

import numpy as np
import pandas as pd

_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


def normalize_step(value: Any, *, row_index: int) -> int:
    """Normalize a checkpoint step to an integral int64 value."""
    invalid = isinstance(value, (bool, np.bool_)) or value is None or value is pd.NA
    if not invalid:
        try:
            if not isinstance(
                value,
                (str, int, float, Decimal, np.integer, np.floating),
            ):
                raise InvalidOperation
            text = str(value).strip()
            if not text:
                raise InvalidOperation
            decimal_value = Decimal(text)
            invalid = (
                not decimal_value.is_finite()
                or decimal_value != decimal_value.to_integral_value()
            )
        except (InvalidOperation, ValueError):
            invalid = True
    if invalid:
        raise ValueError(
            f"invalid PPL step at row {row_index}: {value!r}; "
            "expected a finite integral int64 value"
        )

    step = int(decimal_value)
    if not _INT64_MIN <= step <= _INT64_MAX:
        raise ValueError(
            f"invalid PPL step at row {row_index}: {value!r}; "
            "expected a finite integral int64 value"
        )
    return step


__all__ = ["normalize_step"]
