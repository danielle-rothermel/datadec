"""Load claim inventories and read their quotes from paper source files."""

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


@dataclass(frozen=True, slots=True)
class QuoteLocation:
    """A source line (1-based) and Python character slice (0-based, end exclusive)."""

    source_file: str
    line: int
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class Claim:
    """Source selections grouped under a stable ID."""

    locations: tuple[QuoteLocation, ...]
    original_entry_ids: tuple[str, ...]


class _QuoteLocationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_file: str
    line: int = Field(ge=1)
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_character_range(self) -> Self:
        if self.end <= self.start:
            raise ValueError("quote end must be greater than start")
        return self


class _ClaimConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    locations: tuple[_QuoteLocationConfig, ...] = Field(min_length=1)
    original_entry_ids: tuple[str, ...] = Field(min_length=1)


class _ClaimInventoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claims: dict[str, _ClaimConfig]


def load_claims(path: Path) -> dict[str, Claim]:
    """Parse and validate a TOML inventory, preserving claim and location order."""
    with path.open("rb") as file:
        inventory = _ClaimInventoryConfig.model_validate(tomllib.load(file))
    return {
        claim_id: Claim(
            locations=tuple(
                QuoteLocation(
                    source_file=location.source_file,
                    line=location.line,
                    start=location.start,
                    end=location.end,
                )
                for location in claim.locations
            ),
            original_entry_ids=claim.original_entry_ids,
        )
        for claim_id, claim in inventory.claims.items()
    }


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
