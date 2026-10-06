from __future__ import annotations

from functools import cache
from typing import Literal, Self

from pydantic import model_validator

from datadec.config._load import ConfigModel, load_toml
from datadec.config.catalog import DataDecideCatalog, load_catalog
from datadec.config.sources import SourceManifest, load_source_manifest


class OLMESColumnContract(ConfigModel):
    name: str
    logical_type: Literal["string", "int64", "float64", "bool"]
    nullable: bool


class OLMESTableContract(ConfigModel):
    path: str | None = None
    path_template: str | None = None
    primary_key: tuple[str, ...]
    sort_key: tuple[str, ...]
    columns: tuple[OLMESColumnContract, ...]

    @model_validator(mode="after")
    def validate_table(self) -> Self:
        if (self.path is None) == (self.path_template is None):
            raise ValueError("OLMES tables must define exactly one path")

        names = [column.name for column in self.columns]
        if len(names) != len(set(names)):
            raise ValueError("OLMES table column names must be unique")

        for key_name, key in (
            ("primary key", self.primary_key),
            ("sort key", self.sort_key),
        ):
            if not key:
                raise ValueError(f"OLMES table {key_name} must not be empty")
            if len(key) != len(set(key)):
                raise ValueError(f"OLMES table {key_name} columns must be unique")
            missing = set(key).difference(names)
            if missing:
                missing_names = ", ".join(sorted(missing))
                raise ValueError(
                    f"OLMES table {key_name} columns are missing: {missing_names}"
                )
        return self


class OLMESAggregatePrimaryMetricPolicy(ConfigModel):
    mmlu: str
    arc_challenge: str
    arc_easy: str
    boolq: str
    csqa: str
    hellaswag: str
    openbookqa: str
    piqa: str
    socialiqa: str
    winogrande: str


class OLMESMetricContract(ConfigModel):
    aggregate: tuple[str, ...]
    detailed_tasks: tuple[str, ...]
    detailed_instances: tuple[str, ...]
    detailed_choices: tuple[str, ...]
    not_reproducible_from_details: tuple[str, ...]
    aggregate_primary_metric: OLMESAggregatePrimaryMetricPolicy
    detailed_primary_metric_source: Literal["task_config.primary_metric"]
    detailed_primary_metric_column: Literal["primary_score"]

    @model_validator(mode="after")
    def validate_unique_metric_classifications(self) -> Self:
        for name, metrics in (
            ("aggregate", self.aggregate),
            ("detailed tasks", self.detailed_tasks),
            ("detailed instances", self.detailed_instances),
            ("detailed choices", self.detailed_choices),
            ("not reproducible", self.not_reproducible_from_details),
        ):
            if len(metrics) != len(set(metrics)):
                raise ValueError(f"OLMES {name} metrics must be unique")
        return self


class OLMESTables(ConfigModel):
    aggregate: OLMESTableContract
    detailed_tasks: OLMESTableContract
    detailed_instances: OLMESTableContract
    detailed_choices: OLMESTableContract


class OLMESIdentityContract(ConfigModel):
    native_id_kinds: tuple[Literal["integer", "string", "null"], ...]

    @model_validator(mode="after")
    def validate_native_id_kinds(self) -> Self:
        if (
            set(self.native_id_kinds) != {"integer", "string", "null"}
            or len(self.native_id_kinds) != 3
        ):
            raise ValueError("OLMES native ID kinds must cover integer, string, null")
        return self


class OLMESContract(ConfigModel):
    recipe_map: dict[str, str]
    seed_map: dict[int, str]
    identity: OLMESIdentityContract
    metrics: OLMESMetricContract
    tables: OLMESTables

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        for name, mapping in (
            ("recipe", self.recipe_map),
            ("seed", self.seed_map),
        ):
            if len(mapping.values()) != len(set(mapping.values())):
                raise ValueError(f"OLMES {name} mappings must be unique")

        for table in (
            self.tables.detailed_tasks,
            self.tables.detailed_instances,
            self.tables.detailed_choices,
        ):
            columns = {column.name: column for column in table.columns}
            seed_value = columns.get("seed_value")
            seed = columns.get("seed")
            if (
                seed_value is None
                or seed_value.logical_type != "int64"
                or seed_value.nullable
                or seed is None
                or seed.logical_type != "string"
                or seed.nullable
                or "seed_value" not in table.primary_key
                or "seed_value" not in table.sort_key
                or "seed" in table.primary_key
                or "seed" in table.sort_key
            ):
                raise ValueError(
                    "OLMES detailed tables use seed_value keys and seed attributes"
                )

        metric_tables = (
            (self.metrics.aggregate, self.tables.aggregate),
            (self.metrics.detailed_tasks, self.tables.detailed_tasks),
            (self.metrics.detailed_instances, self.tables.detailed_instances),
            (self.metrics.detailed_choices, self.tables.detailed_choices),
        )
        for metric_names, table in metric_tables:
            column_names = tuple(column.name for column in table.columns)
            if column_names[-len(metric_names) :] != metric_names:
                raise ValueError(
                    "OLMES metric classifications must match table column order"
                )

        aggregate_columns = {
            column.name: column for column in self.tables.aggregate.columns
        }
        for name in self.metrics.aggregate:
            column = aggregate_columns[name]
            if column.logical_type != "float64" or not column.nullable:
                raise ValueError(
                    "OLMES aggregate metrics must be nullable float64 columns"
                )

        policy_metrics = self.metrics.aggregate_primary_metric.model_dump().values()
        if not set(policy_metrics).issubset(self.metrics.aggregate):
            raise ValueError(
                "OLMES aggregate primary metric policy references unknown metrics"
            )
        if self.metrics.detailed_primary_metric_column not in (
            self.metrics.detailed_tasks
        ):
            raise ValueError(
                "OLMES detailed primary metric must be a detailed task metric"
            )
        if not set(self.metrics.not_reproducible_from_details).issubset(
            self.metrics.aggregate
        ):
            raise ValueError("OLMES non-reproducible metrics must be aggregate metrics")
        return self

    def validate_references(
        self,
        *,
        catalog: DataDecideCatalog,
        source_manifest: SourceManifest,
    ) -> Self:
        source_recipes = set(source_manifest.olmes_details.recipes)
        if set(self.recipe_map) != source_recipes:
            raise ValueError(
                "OLMES recipe mapping must exactly cover source detail recipes"
            )

        catalog_recipes = {
            recipe
            for family in catalog.data_recipe_families.values()
            for recipe in family
        }
        if set(self.recipe_map.values()) != catalog_recipes:
            raise ValueError(
                "OLMES recipe mapping must be a bijection with catalog recipes"
            )
        if set(self.seed_map.values()) != set(catalog.seed_map):
            raise ValueError("OLMES seed mapping must exactly cover catalog seeds")
        return self


@cache
def load_olmes_contract() -> OLMESContract:
    contract = OLMESContract.model_validate(load_toml("olmes.toml"))
    return contract.validate_references(
        catalog=load_catalog(), source_manifest=load_source_manifest()
    )
