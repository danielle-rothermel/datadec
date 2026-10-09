"""Computed measurements and references to linked data rows."""

from dataclasses import dataclass


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


@dataclass(frozen=True, slots=True)
class ClaimEvidence:
    claim_id: str
    evidence_ids: tuple[int, ...]
    unavailable_ids: tuple[int, ...]
    measurements: tuple[TaskMeasurement, ...]
    related_tables: tuple[str, ...]

    @property
    def has_measurements(self) -> bool:
        return any(measurement.best is not None for measurement in self.measurements)
