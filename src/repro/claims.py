"""Locate and read claim quotes in downloaded paper source files."""

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


def read_quotes(claims: Mapping[str, QuoteLocation], paper_dir: Path) -> dict[str, str]:
    """Read quotes from the supplied paper directory, preserving claim order."""
    sources = {
        source_file: (paper_dir / source_file).read_text(encoding="utf-8").splitlines()
        for source_file in {location.source_file for location in claims.values()}
    }
    return {
        claim_id: sources[location.source_file][location.line - 1][
            location.start : location.end
        ]
        for claim_id, location in claims.items()
    }
