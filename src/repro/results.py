"""Claim-to-ranking references; numerical measurements live in the ranking dataset."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TaskEvidence:
    task: str
    evidence_ids: tuple[int, ...]
    unavailable_ids: tuple[int, ...]
    best_evidence_id: int | None

    def __post_init__(self) -> None:
        if self.evidence_ids:
            if self.best_evidence_id not in self.evidence_ids:
                raise ValueError(
                    "best evidence ID must reference an available linked row"
                )
        elif self.best_evidence_id is not None:
            raise ValueError("best evidence ID requires available linked rows")

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
        return any(task.best_evidence_id is not None for task in self.tasks)
