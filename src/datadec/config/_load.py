from __future__ import annotations

import tomllib
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path

from pydantic import BaseModel, ConfigDict

_CONFIG_PACKAGE = "datadec"
_SOURCE_CONFIGS_DIR = Path(__file__).parents[3] / "configs"


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def config_file(filename: str) -> Traversable:
    packaged = files(_CONFIG_PACKAGE).joinpath("configs", filename)
    if packaged.is_file():
        return packaged

    source = _SOURCE_CONFIGS_DIR / filename
    if source.is_file():
        return source
    raise FileNotFoundError(f"DataDecide config file not found: {filename}")


def load_toml(filename: str) -> dict[str, object]:
    with config_file(filename).open("rb") as file:
        return tomllib.load(file)
