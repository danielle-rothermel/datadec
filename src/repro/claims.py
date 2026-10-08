"""Locate and read claim quotes in paper source files."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class QuoteLocation:
    """A source line (1-based) and Python character slice (0-based, end exclusive)."""

    source_file: str
    line: int
    start: int
    end: int
    section: str


@dataclass(frozen=True, slots=True)
class Claim:
    """One assertion with its occurrences in the paper."""

    statement: str
    locations: tuple[QuoteLocation, ...]


def read_quotes(
    claims: Mapping[str, Claim], paper_dir: Path
) -> dict[str, tuple[str, ...]]:
    """Read all quotes per claim, preserving claim and source-location order."""
    sources = {
        source_file: (paper_dir / source_file).read_text(encoding="utf-8").splitlines()
        for source_file in {
            location.source_file
            for claim in claims.values()
            for location in claim.locations
        }
    }
    return {
        claim_id: tuple(
            sources[location.source_file][location.line - 1][
                location.start : location.end
            ]
            for location in claim.locations
        )
        for claim_id, claim in claims.items()
    }
