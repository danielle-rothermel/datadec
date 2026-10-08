"""Claim verdicts and references to reproducible evidence."""

from dataclasses import dataclass
from enum import UNIQUE, StrEnum, auto, verify


@verify(UNIQUE)
class ClaimStatus(StrEnum):
    SUPPORTED = auto()
    NOT_SUPPORTED = auto()
    REQUIRES_JUDGMENT = auto()
    INSUFFICIENT_DATA = auto()


@dataclass(frozen=True, slots=True)
class ClaimEvidence:
    claim_id: str
    status: ClaimStatus
    evidence_ids: tuple[int, ...]
    unavailable_ids: tuple[int, ...]
    summary: str
    judgment: str
    supporting_tables: tuple[str, ...]
