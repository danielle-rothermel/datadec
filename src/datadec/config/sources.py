from __future__ import annotations

from datetime import date
from functools import cache
from typing import Literal, Self

from pydantic import Field, model_validator

from datadec.config._load import ConfigModel, load_toml


class DatasetSource(ConfigModel):
    id: str
    provider: Literal["datasets"]
    repo_id: str
    revision: str
    split: str
    output: str


class DetailFile(ConfigModel):
    recipe: str
    expected_size: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DetailSource(ConfigModel):
    id: str
    provider: Literal["huggingface_hub"]
    repo_type: Literal["dataset"]
    repo_id: str
    revision: str
    filename_template: str
    output_root: str
    files: tuple[DetailFile, ...]

    @property
    def recipes(self) -> tuple[str, ...]:
        return tuple(file.recipe for file in self.files)

    def file_for_recipe(self, recipe: str) -> DetailFile:
        for file in self.files:
            if file.recipe == recipe:
                return file
        raise ValueError(f"unknown OLMES detail recipe: {recipe}")

    @model_validator(mode="after")
    def validate_unique_recipes(self) -> Self:
        if len(self.recipes) != len(set(self.recipes)):
            raise ValueError("OLMES detail recipes must be unique")
        return self


class ArchiveSource(ConfigModel):
    id: str
    provider: Literal["google_drive_folder"]
    url: str
    downloaded_on: date


class SourceManifest(ConfigModel):
    ppl: DatasetSource
    olmes: DatasetSource
    olmes_details: DetailSource
    archives: tuple[ArchiveSource, ...]


@cache
def load_source_manifest() -> SourceManifest:
    return SourceManifest.model_validate(load_toml("sources.toml"))
