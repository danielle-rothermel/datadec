"""Canonical recipe identities and configured source-name aliases."""

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import Self

from pydantic import TypeAdapter

from datadec.config import load_olmes_contract
from datadec.data.ingest.enums import DataRecipeName


@dataclass(frozen=True, slots=True)
class RecipeNameResolver:
    """Resolve exact configured spellings; official names always resolve to themselves."""

    aliases: Mapping[str, DataRecipeName]

    def __post_init__(self) -> None:
        aliases = dict(self.aliases)
        if any(not isinstance(recipe, DataRecipeName) for recipe in aliases.values()):
            raise TypeError("alias targets must be DataRecipeName members")
        for recipe in DataRecipeName:
            if recipe.value in aliases and aliases[recipe.value] != recipe:
                raise ValueError(
                    f"alias conflicts with official recipe: {recipe.value}"
                )
            aliases[recipe.value] = recipe
        object.__setattr__(self, "aliases", MappingProxyType(aliases))

    @classmethod
    def from_mapping(cls, aliases: Mapping[str, str | DataRecipeName]) -> Self:
        """Parse an alias-to-official-name dictionary, allowing many aliases per recipe."""
        parsed = TypeAdapter(dict[str, DataRecipeName]).validate_python(aliases)
        return cls(parsed)

    @classmethod
    def from_toml(cls, path: Path) -> Self:
        """Load the [recipe_map] table from a TOML file."""
        with path.open("rb") as file:
            raw = tomllib.load(file)
        return cls.from_mapping(raw["recipe_map"])

    def resolve(self, name: str | DataRecipeName) -> DataRecipeName:
        try:
            return self.aliases[name]
        except KeyError as error:
            raise ValueError(f"unknown recipe name: {name!r}") from error


@cache
def load_recipe_name_resolver() -> RecipeNameResolver:
    """Load source spellings from the bundled OLMES recipe map once."""
    return RecipeNameResolver.from_mapping(load_olmes_contract().recipe_map)
