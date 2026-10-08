from __future__ import annotations

from functools import cache
from typing import Self

from pydantic import Field, model_validator

from datadec.config._load import ConfigModel, load_toml


class CheckpointSourceContract(ConfigModel):
    repo_id_template: str
    branch_template: str
    files: tuple[str, ...] = Field(min_length=1)
    tokenizer_files: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_tokenizer_files(self) -> Self:
        if not set(self.tokenizer_files) <= set(self.files):
            raise ValueError("tokenizer_files must be a subset of files")
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
    paper: PaperTable


class CheckpointContract(ConfigModel):
    source: CheckpointSourceContract
    target: CheckpointTargetContract
    seeds: SeedContract
    final_step: FinalStepContract


@cache
def load_checkpoint_contract() -> CheckpointContract:
    return CheckpointContract.model_validate(load_toml("checkpoints.toml"))
