"""Claim-to-ranking references; numerical measurements live in the ranking dataset."""

from dataclasses import dataclass

from eval.approaches import PredictionApproach


@dataclass(frozen=True, slots=True)
class TaskEvidence:
    task: str
    evidence_ids: tuple[int, ...]
    unavailable_ids: tuple[int, ...]
    best_per_seed_evidence_id: int | None
    best_aggregate_evidence_id: int | None

    def __post_init__(self) -> None:
        for best in (self.best_per_seed_evidence_id, self.best_aggregate_evidence_id):
            if self.evidence_ids:
                if best not in self.evidence_ids:
                    raise ValueError(
                        "best evidence ID must reference an available linked row"
                    )
            elif best is not None:
                raise ValueError("best evidence ID requires available linked rows")

    def best_evidence_id(self, approach: PredictionApproach) -> int | None:
        return {
            PredictionApproach.PER_SEED: self.best_per_seed_evidence_id,
            PredictionApproach.AGGREGATE: self.best_aggregate_evidence_id,
        }[approach]

    @property
    def available_comparisons(self) -> int:
        return len(self.evidence_ids)

    @property
    def unavailable_comparisons(self) -> int:
        return len(self.unavailable_ids)


@dataclass(frozen=True, slots=True)
class ClaimEvidence:
    claim_id: str
    tasks: tuple[TaskEvidence, ...]
    related_tables: tuple[str, ...]

    @property
    def evidence_ids(self) -> tuple[int, ...]:
        return tuple(index for task in self.tasks for index in task.evidence_ids)

    @property
    def unavailable_ids(self) -> tuple[int, ...]:
        return tuple(index for task in self.tasks for index in task.unavailable_ids)

    @property
    def has_measurements(self) -> bool:
        return any(task.evidence_ids for task in self.tasks)
