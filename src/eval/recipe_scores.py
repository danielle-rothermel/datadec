"""Immutable scores keyed by canonical recipe identity."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Self

from datadec.recipes import (
    DataRecipeName,
    RecipeNameResolver,
    load_recipe_name_resolver,
)


@dataclass(frozen=True, slots=True)
class MultiRecipeScores(Mapping[DataRecipeName, float]):
    scores: Mapping[DataRecipeName, float]

    def __post_init__(self) -> None:
        if any(not isinstance(recipe, DataRecipeName) for recipe in self.scores):
            raise TypeError("score keys must be DataRecipeName members")
        object.__setattr__(self, "scores", MappingProxyType(dict(self.scores)))

    @classmethod
    def from_named_scores(
        cls,
        scores: Mapping[str, float],
        *,
        resolver: RecipeNameResolver | None = None,
    ) -> Self:
        """Resolve input names, rejecting aliases that would overwrite a score."""
        resolver = resolver if resolver is not None else load_recipe_name_resolver()
        canonical: dict[DataRecipeName, float] = {}
        for name, score in scores.items():
            recipe = resolver.resolve(name)
            if recipe in canonical:
                raise ValueError(f"duplicate score for recipe: {recipe.value}")
            canonical[recipe] = score
        return cls(canonical)

    def __getitem__(self, recipe: DataRecipeName) -> float:
        return self.scores[recipe]

    def __iter__(self) -> Iterator[DataRecipeName]:
        return iter(self.scores)

    def __len__(self) -> int:
        return len(self.scores)
