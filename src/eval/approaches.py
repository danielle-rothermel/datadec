"""The two predictor-seed aggregation approaches."""

from enum import UNIQUE, StrEnum, auto, verify


@verify(UNIQUE)
class PredictionApproach(StrEnum):
    PER_SEED = auto()
    AGGREGATE = auto()
