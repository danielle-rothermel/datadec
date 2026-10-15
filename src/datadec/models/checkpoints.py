from __future__ import annotations

from dataclasses import dataclass

from datadec.config import load_catalog, load_olmes_contract
from datadec.config.checkpoints import SeedLiteral, load_checkpoint_contract


@dataclass(frozen=True, slots=True)
class SeedIdentity:
    ordinal: int
    name: str
    value: int

    @property
    def branch_slug(self) -> str:
        return self.name.replace(" ", "-")


@dataclass(frozen=True, slots=True)
class DataDecideCheckpoint:
    recipe: str
    size: str
    seed_ordinal: int
    step: int

    def __post_init__(self) -> None:
        recipes = load_olmes_contract().recipe_map
        if self.recipe not in recipes:
            raise ValueError(
                f"Unknown DataDecide recipe {self.recipe!r}. "
                f"Available: {', '.join(sorted(recipes))}"
            )
        sizes = [model.name for model in load_catalog().models]
        if self.size not in sizes:
            raise ValueError(
                f"Unknown DataDecide size {self.size!r}. Available: {', '.join(sizes)}"
            )
        ordinals = load_checkpoint_contract().seeds.ordinals
        if not 0 <= self.seed_ordinal < len(ordinals):
            raise ValueError(
                f"seed_ordinal must be in 0..{len(ordinals) - 1}, "
                f"got {self.seed_ordinal}"
            )
        if self.step < 0:
            raise ValueError(f"step must be non-negative, got {self.step}")

    @property
    def recipe_display_name(self) -> str:
        return load_olmes_contract().recipe_map[self.recipe]

    @property
    def seed(self) -> SeedIdentity:
        seeds = load_checkpoint_contract().seeds
        entry = seeds.ordinals[self.seed_ordinal]
        literal: SeedLiteral = (
            entry.large if self.size in seeds.large_sizes else entry.small
        )
        return SeedIdentity(
            ordinal=self.seed_ordinal, name=literal.name, value=literal.value
        )

    @property
    def source_repo_id(self) -> str:
        template = load_checkpoint_contract().source.repo_id_template
        return template.format(recipe=self.recipe, size=self.size)

    @property
    def source_branch(self) -> str:
        template = load_checkpoint_contract().source.branch_template
        return template.format(step=self.step, seed_slug=self.seed.branch_slug)

    @property
    def target_repo_id(self) -> str:
        template = load_checkpoint_contract().target.repo_id_template
        return template.format(recipe=self.recipe, size=self.size)

    @property
    def target_branch(self) -> str:
        template = load_checkpoint_contract().target.branch_template
        return template.format(step=self.step, ordinal=self.seed_ordinal)

    @property
    def final_tag(self) -> str:
        template = load_checkpoint_contract().target.final_tag_template
        return template.format(ordinal=self.seed_ordinal)
