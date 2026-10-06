from __future__ import annotations

from functools import cache
from pathlib import PurePosixPath
from typing import Literal, Self

from pydantic import model_validator

from datadec.config._load import ConfigModel, load_toml
from datadec.config.catalog import DataDecideCatalog, load_catalog
from datadec.config.olmes import OLMESContract, load_olmes_contract
from datadec.config.published_results import (
    PublishedResultsManifest,
    load_published_results_manifest,
)


class ScalingLawSeedPolicy(ConfigModel):
    excluded_legacy_values: tuple[int, ...]
    missing: Literal["exclude_legacy_input"]
    unknown_non_null: Literal["error"]

    @model_validator(mode="after")
    def validate_unique_excluded_values(self) -> Self:
        if len(self.excluded_legacy_values) != len(set(self.excluded_legacy_values)):
            raise ValueError("scaling-law excluded legacy seeds must be unique")
        if self.excluded_legacy_values != (6198,):
            raise ValueError(
                "scaling-law excluded legacy seeds must contain only seed 6198"
            )
        return self


class ScalingLawColumnContract(ConfigModel):
    name: str
    logical_type: Literal["string", "int64", "float64", "bool"]
    nullable: bool


class ScalingLawTableContract(ConfigModel):
    path: str
    primary_key: tuple[str, ...]
    sort_key: tuple[str, ...]
    columns: tuple[ScalingLawColumnContract, ...]

    @model_validator(mode="after")
    def validate_table(self) -> Self:
        path = PurePosixPath(self.path)
        if (
            not path.parts
            or path.is_absolute()
            or path.as_posix() != self.path
            or ".." in path.parts
        ):
            raise ValueError(
                "scaling-law table paths must be normalized relative paths"
            )

        names = [column.name for column in self.columns]
        if len(names) != len(set(names)):
            raise ValueError("scaling-law table column names must be unique")

        for key_name, key in (
            ("primary key", self.primary_key),
            ("sort key", self.sort_key),
        ):
            if not key:
                raise ValueError(f"scaling-law table {key_name} must not be empty")
            if len(key) != len(set(key)):
                raise ValueError(f"scaling-law table {key_name} columns must be unique")
            missing = set(key).difference(names)
            if missing:
                missing_names = ", ".join(sorted(missing))
                raise ValueError(
                    f"scaling-law table {key_name} columns are missing: {missing_names}"
                )
        return self


class ScalingLawTables(ConfigModel):
    evaluations: ScalingLawTableContract
    checkpoint_losses: ScalingLawTableContract


class ScalingLawContract(ConfigModel):
    raw_directory: str
    source_precedence: tuple[str, ...]
    models: tuple[str, ...]
    excluded_source_groups: tuple[str, ...]
    source_group_aliases: dict[str, str]
    source_group_map: dict[str, str]
    seed_map: dict[int, str]
    seed_policy: ScalingLawSeedPolicy
    tables: ScalingLawTables

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        raw_directory = PurePosixPath(self.raw_directory)
        if (
            not raw_directory.parts
            or raw_directory.is_absolute()
            or raw_directory.as_posix() != self.raw_directory
            or ".." in raw_directory.parts
        ):
            raise ValueError(
                "scaling-law raw directory must be a normalized relative path"
            )

        if not self.source_precedence:
            raise ValueError("scaling-law source precedence must not be empty")
        if len(self.source_precedence) != len(set(self.source_precedence)):
            raise ValueError("scaling-law source precedence must be unique")
        for filename in self.source_precedence:
            path = PurePosixPath(filename)
            if len(path.parts) != 1 or path.name != filename:
                raise ValueError("scaling-law sources must be bare filenames")

        for name, mapping in (
            ("source group alias", self.source_group_aliases),
            ("source group", self.source_group_map),
            ("seed", self.seed_map),
        ):
            if not mapping:
                raise ValueError(f"scaling-law {name} mapping must not be empty")
            if len(mapping.values()) != len(set(mapping.values())):
                raise ValueError(f"scaling-law {name} mappings must be unique")

        if len(self.excluded_source_groups) != len(set(self.excluded_source_groups)):
            raise ValueError("scaling-law excluded source groups must be unique")
        configured_source_groups = (
            set(self.source_group_map)
            | set(self.source_group_aliases)
            | set(self.excluded_source_groups)
        )
        if len(configured_source_groups) != (
            len(self.source_group_map)
            + len(self.source_group_aliases)
            + len(self.excluded_source_groups)
        ):
            raise ValueError(
                "scaling-law canonical, aliased, and excluded source groups "
                "must be disjoint"
            )
        unknown_alias_targets = set(self.source_group_aliases.values()).difference(
            self.source_group_map
        )
        if unknown_alias_targets:
            raise ValueError(
                "scaling-law source group aliases must reference canonical groups"
            )

        if set(self.seed_map).intersection(self.seed_policy.excluded_legacy_values):
            raise ValueError("clean and excluded scaling-law seeds must be disjoint")

        if len(self.models) != len(set(self.models)):
            raise ValueError("scaling-law models must be unique")

        expected_keys = {
            "evaluations": ("recipe", "params", "seed_value", "step", "task"),
            "checkpoint_losses": ("recipe", "params", "seed_value", "step"),
        }
        expected_prefixes = {
            "evaluations": (
                ("source_file", "string", False),
                ("recipe", "string", False),
                ("data", "string", False),
                ("params", "string", False),
                ("seed_value", "int64", False),
                ("seed", "string", False),
                ("step", "int64", False),
                ("task", "string", False),
                ("chinchilla", "string", False),
                ("tokens", "int64", False),
                ("compute", "float64", False),
            ),
            "checkpoint_losses": (
                ("source_file", "string", False),
                ("recipe", "string", False),
                ("data", "string", False),
                ("params", "string", False),
                ("seed_value", "int64", False),
                ("seed", "string", False),
                ("step", "int64", False),
                ("chinchilla", "string", False),
                ("tokens", "int64", False),
                ("compute", "float64", False),
            ),
        }
        checkpoint_metrics = (
            "c4_en_validation_cross_entropy",
            "dolma_common_crawl_validation_cross_entropy",
            "pile_validation_cross_entropy",
            "wikitext_103_validation_cross_entropy",
            "train_cross_entropy",
            "throughput_total_tokens",
        )
        for name, table in (
            ("evaluations", self.tables.evaluations),
            ("checkpoint_losses", self.tables.checkpoint_losses),
        ):
            if (
                table.primary_key != expected_keys[name]
                or table.sort_key != expected_keys[name]
            ):
                raise ValueError(
                    f"scaling-law {name} primary and sort keys must match identity"
                )
            actual_prefix = tuple(
                (column.name, column.logical_type, column.nullable)
                for column in table.columns[: len(expected_prefixes[name])]
            )
            if actual_prefix != expected_prefixes[name]:
                raise ValueError(
                    f"scaling-law {name} identity and provenance columns are invalid"
                )

        loss_columns = self.tables.checkpoint_losses.columns[-len(checkpoint_metrics) :]
        if tuple(column.name for column in loss_columns) != checkpoint_metrics or any(
            column.logical_type != "float64" or not column.nullable
            for column in loss_columns
        ):
            raise ValueError(
                "scaling-law checkpoint loss metrics must be nullable float64 "
                "columns in canonical order"
            )

        if self.tables.evaluations.path == self.tables.checkpoint_losses.path:
            raise ValueError("scaling-law output table paths must be unique")
        return self

    def validate_references(
        self,
        *,
        catalog: DataDecideCatalog,
        olmes_contract: OLMESContract,
        published_results_manifest: PublishedResultsManifest,
    ) -> Self:
        manifest_sources = tuple(
            PurePosixPath(file.path).name
            for file in published_results_manifest.files
            if file.category == "scaling_law"
        )
        if self.source_precedence != manifest_sources:
            raise ValueError(
                "scaling-law source precedence must exactly cover the published "
                "results manifest in precedence order"
            )

        catalog_models = tuple(model.name for model in catalog.models)
        if self.models != catalog_models:
            raise ValueError("scaling-law models must exactly match catalog models")

        canonical_recipes = set(self.source_group_map.values())
        if canonical_recipes != set(olmes_contract.recipe_map):
            raise ValueError(
                "scaling-law source groups must map bijectively to OLMES recipes"
            )
        catalog_recipes = {
            recipe
            for family in catalog.data_recipe_families.values()
            for recipe in family
        }
        display_labels = {
            olmes_contract.recipe_map[recipe] for recipe in canonical_recipes
        }
        if display_labels != catalog_recipes:
            raise ValueError(
                "scaling-law recipe display labels must derive from OLMES recipes"
            )

        if self.seed_map != olmes_contract.seed_map:
            raise ValueError("scaling-law seed map must exactly match OLMES seeds")

        evaluation_metrics = self.tables.evaluations.columns[
            -len(olmes_contract.metrics.aggregate) :
        ]
        aggregate_columns = {
            column.name: column for column in olmes_contract.tables.aggregate.columns
        }
        if tuple(column.name for column in evaluation_metrics) != (
            olmes_contract.metrics.aggregate
        ):
            raise ValueError(
                "scaling-law evaluation metrics must exactly match OLMES aggregate "
                "metrics"
            )
        if any(
            column.logical_type != aggregate_columns[column.name].logical_type
            or column.nullable != aggregate_columns[column.name].nullable
            for column in evaluation_metrics
        ):
            raise ValueError(
                "scaling-law evaluation metric types must match OLMES aggregate columns"
            )
        return self


@cache
def load_scaling_law_contract() -> ScalingLawContract:
    contract = ScalingLawContract.model_validate(load_toml("scaling_law.toml"))
    return contract.validate_references(
        catalog=load_catalog(),
        olmes_contract=load_olmes_contract(),
        published_results_manifest=load_published_results_manifest(),
    )
