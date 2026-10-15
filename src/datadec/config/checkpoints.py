from __future__ import annotations

from functools import cache
from typing import Literal, Self

from pydantic import Field, model_validator

from datadec.config._load import ConfigModel, load_toml
from datadec.config.olmes import load_olmes_contract


def _validate_recipe_keys(mapping: dict[str, str], field: str) -> None:
    recipes = set(load_olmes_contract().recipe_map)
    unknown = sorted(set(mapping) - recipes)
    if unknown:
        raise ValueError(f"{field} names unknown recipes: {unknown}")


class CheckpointSourceContract(ConfigModel):
    repo_id_template: str
    branch_template: str
    files: tuple[str, ...] = Field(min_length=1)
    tokenizer_files: tuple[str, ...] = Field(min_length=1)
    repo_slugs: dict[str, str]

    @model_validator(mode="after")
    def validate_tokenizer_files(self) -> Self:
        if not set(self.tokenizer_files) <= set(self.files):
            raise ValueError("tokenizer_files must be a subset of files")
        return self

    @model_validator(mode="after")
    def validate_repo_slugs(self) -> Self:
        _validate_recipe_keys(self.repo_slugs, "repo_slugs")
        missing = sorted(set(load_olmes_contract().recipe_map) - set(self.repo_slugs))
        if missing:
            raise ValueError(f"repo_slugs lacks recipes: {missing}")
        slugs = list(self.repo_slugs.values())
        if any(not slug for slug in slugs) or len(slugs) != len(set(slugs)):
            raise ValueError("repo_slugs values must be non-empty and unique")
        return self


class CheckpointTargetContract(ConfigModel):
    repo_id_template: str
    branch_template: str
    final_tag_template: str
    private: bool


class SeedLiteral(ConfigModel):
    name: str = Field(min_length=1)
    value: int


class SeedOrdinal(ConfigModel):
    ordinal: int = Field(ge=0)
    small: SeedLiteral
    large: SeedLiteral


class SeedContract(ConfigModel):
    large_sizes: tuple[str, ...]
    ordinals: tuple[SeedOrdinal, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_ordinals(self) -> Self:
        ordinals = [seed.ordinal for seed in self.ordinals]
        if ordinals != list(range(len(ordinals))):
            raise ValueError("seed ordinals must be 0..n-1 in order")
        return self


class PaperSizeRow(ConfigModel):
    batch_size: int = Field(gt=0)
    training_steps: int = Field(gt=0)


class PaperTable(ConfigModel):
    source: str
    sizes: dict[str, PaperSizeRow]


class FinalStepContract(ConfigModel):
    final_paths_url: str
    final_paths_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    final_paths_training_data_names: dict[str, str]
    paper: PaperTable

    @model_validator(mode="after")
    def validate_training_data_names(self) -> Self:
        _validate_recipe_keys(
            self.final_paths_training_data_names, "final_paths_training_data_names"
        )
        return self


class SupportedSourceLayout(ConfigModel):
    required_absent: tuple[str, ...]
    required_values: dict[str, str | int | bool]

    @model_validator(mode="after")
    def validate_disjoint(self) -> Self:
        overlap = set(self.required_absent) & set(self.required_values)
        if overlap:
            raise ValueError(f"layout fields both absent and valued: {sorted(overlap)}")
        return self


class TensorMapRule(ConfigModel):
    source: str = Field(min_length=1)
    targets: tuple[str, ...] = Field(min_length=1)
    transform: Literal["copy", "split_rows_equal"]

    @model_validator(mode="after")
    def validate_targets(self) -> Self:
        if self.transform == "copy" and len(self.targets) != 1:
            raise ValueError("copy rules map to exactly one target")
        if self.transform == "split_rows_equal" and len(self.targets) < 2:
            raise ValueError("split_rows_equal rules need at least two targets")
        per_layer = "{layer}" in self.source
        if any(("{layer}" in target) != per_layer for target in self.targets):
            raise ValueError("source and targets must agree on the {layer} placeholder")
        return self


class ConversionContract(ConfigModel):
    source_model_type: str = Field(min_length=1)
    source_tensor_prefix: str
    target_model_type: str = Field(min_length=1)
    target_architecture: str = Field(min_length=1)
    weights_filename: str = Field(min_length=1)
    conversion_record_filename: str = Field(min_length=1)
    supported_source_layout: SupportedSourceLayout
    tensor_map: tuple[TensorMapRule, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_names(self) -> Self:
        sources = [rule.source for rule in self.tensor_map]
        targets = [target for rule in self.tensor_map for target in rule.targets]
        if len(sources) != len(set(sources)) or len(targets) != len(set(targets)):
            raise ValueError("tensor map sources and targets must be unique")
        return self


class ReferenceRequirement(ConfigModel):
    requirement: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class StrictCheckContract(ConfigModel):
    python: str = Field(min_length=1)
    prompts: tuple[str, ...] = Field(min_length=1)
    reference_environment: tuple[ReferenceRequirement, ...] = Field(min_length=1)


class CheckpointContract(ConfigModel):
    source: CheckpointSourceContract
    target: CheckpointTargetContract
    seeds: SeedContract
    final_step: FinalStepContract
    strict_check: StrictCheckContract
    conversion: ConversionContract


@cache
def load_checkpoint_contract() -> CheckpointContract:
    return CheckpointContract.model_validate(load_toml("checkpoints.toml"))
