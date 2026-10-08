"""Computed claim measurements, verdicts, and evidence references."""

from dataclasses import dataclass
from enum import UNIQUE, StrEnum, auto, verify


@verify(UNIQUE)
class ClaimStatus(StrEnum):
    SUPPORTED = auto()
    NOT_SUPPORTED = auto()
    REQUIRES_JUDGMENT = auto()
    INSUFFICIENT_DATA = auto()


@dataclass(frozen=True, slots=True)
class PredictionMeasurement:
    evidence_id: int
    predictor_size: str
    predictor_step: int
    metric: str
    compute: float
    compute_ratio: float
    decision_accuracy: float
    decision_accuracy_std: float
    seed_accuracies: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class TaskMeasurement:
    task: str
    available_comparisons: int
    unavailable_comparisons: int
    best: PredictionMeasurement | None
    accuracy_gt: float | None
    passes_bound: bool | None


@dataclass(frozen=True, slots=True)
class ClaimEvidence:
    claim_id: str
    status: ClaimStatus
    evidence_ids: tuple[int, ...]
    unavailable_ids: tuple[int, ...]
    measurements: tuple[TaskMeasurement, ...]
    judgment: str
    supporting_tables: tuple[str, ...]
