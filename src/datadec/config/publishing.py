from __future__ import annotations

from functools import cache
from pathlib import PurePosixPath
from string import Formatter
from typing import Literal, Self

from pydantic import model_validator

from datadec.config._load import ConfigModel, load_toml
from datadec.config.olmes import OLMESContract, load_olmes_contract
from datadec.config.scaling_law import ScalingLawContract, load_scaling_law_contract
from datadec.config.sources import SourceManifest, load_source_manifest


def _validate_remote_path(path: str, *, description: str) -> None:
    parsed = PurePosixPath(path)
    if (
        not path
        or not parsed.parts
        or parsed.is_absolute()
        or parsed.as_posix() != path
        or ".." in parsed.parts
        or "\\" in path
    ):
        raise ValueError(f"{description} must be a normalized relative POSIX path")


def _template_fields(template: str, *, description: str) -> tuple[str, ...]:
    try:
        parsed = tuple(Formatter().parse(template))
    except ValueError as error:
        raise ValueError(f"{description} must be a valid format template") from error

    if any(format_spec or conversion for _, _, format_spec, conversion in parsed):
        raise ValueError(f"{description} must not use conversions or format specs")
    return tuple(field for _, field, _, _ in parsed if field is not None)


def _validate_template(
    template: str,
    *,
    description: str,
    expected_fields: tuple[str, ...],
) -> None:
    fields = _template_fields(template, description=description)
    if fields != expected_fields:
        expected = ", ".join(expected_fields)
        raise ValueError(f"{description} must contain exactly {{{expected}}}")


def _remote_path_for_local(local_path: str) -> str:
    path = PurePosixPath(local_path)
    if not path.parts or path.parts[0] != "processed":
        raise ValueError("published local table paths must be under processed/")
    return PurePosixPath(*path.parts[1:]).as_posix()


class PublishingTarget(ConfigModel):
    repo_id: Literal["drotherm/dd_parsed"]
    revision: Literal["main"]


class SingleFilePublishingContract(ConfigModel):
    remote_path: str
    commit_message: str

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        _validate_remote_path(self.remote_path, description="publication remote path")
        if not self.commit_message.strip():
            raise ValueError("publication commit message must not be empty")
        return self


class ScalingLawPublishingContract(ConfigModel):
    evaluations_remote_path: str
    checkpoint_losses_remote_path: str
    commit_message: str

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        paths = (self.evaluations_remote_path, self.checkpoint_losses_remote_path)
        for path in paths:
            _validate_remote_path(
                path, description="scaling-law publication remote path"
            )
        if len(paths) != len(set(paths)):
            raise ValueError("scaling-law publication remote paths must be unique")
        if not self.commit_message.strip():
            raise ValueError("scaling-law publication commit message must not be empty")
        return self


class OLMESDetailsPublishingContract(ConfigModel):
    tasks_remote_path_template: str
    instances_remote_path_template: str
    choices_remote_path_template: str
    commit_message_template: str

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        templates = self.remote_path_templates()
        if len(templates) != len(set(templates)):
            raise ValueError("OLMES detail remote path templates must be unique")
        for template in templates:
            _validate_template(
                template,
                description="OLMES detail remote path template",
                expected_fields=("recipe",),
            )
            _validate_remote_path(
                template.format(recipe="representative-recipe"),
                description="OLMES detail remote path template",
            )
        _validate_template(
            self.commit_message_template,
            description="OLMES detail commit message template",
            expected_fields=("recipe",),
        )
        return self

    def remote_path_templates(self) -> tuple[str, str, str]:
        return (
            self.tasks_remote_path_template,
            self.instances_remote_path_template,
            self.choices_remote_path_template,
        )


class PublishedResultsPublishingContract(ConfigModel):
    remote_root: str
    commit_message_template: str

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        _validate_remote_path(
            self.remote_root, description="published-results remote root"
        )
        _validate_template(
            self.commit_message_template,
            description="published-results commit message template",
            expected_fields=("unit",),
        )
        return self


class PublishingContract(ConfigModel):
    target: PublishingTarget
    ppl: SingleFilePublishingContract
    olmes: SingleFilePublishingContract
    scaling_law: ScalingLawPublishingContract
    olmes_details: OLMESDetailsPublishingContract
    published_results: PublishedResultsPublishingContract

    def validate_references(
        self,
        *,
        olmes_contract: OLMESContract,
        scaling_law_contract: ScalingLawContract,
        source_manifest: SourceManifest,
    ) -> Self:
        expected_paths = {
            "olmes": _remote_path_for_local(olmes_contract.tables.aggregate.path or ""),
            "scaling-law evaluations": _remote_path_for_local(
                scaling_law_contract.tables.evaluations.path
            ),
            "scaling-law checkpoint losses": _remote_path_for_local(
                scaling_law_contract.tables.checkpoint_losses.path
            ),
        }
        configured_paths = {
            "olmes": self.olmes.remote_path,
            "scaling-law evaluations": self.scaling_law.evaluations_remote_path,
            "scaling-law checkpoint losses": (
                self.scaling_law.checkpoint_losses_remote_path
            ),
        }
        for name, expected in expected_paths.items():
            if configured_paths[name] != expected:
                raise ValueError(
                    f"{name} remote path must correspond to its local table path"
                )

        detail_tables = (
            olmes_contract.tables.detailed_tasks,
            olmes_contract.tables.detailed_instances,
            olmes_contract.tables.detailed_choices,
        )
        expected_templates = tuple(
            _remote_path_for_local(table.path_template or "") for table in detail_tables
        )
        if self.olmes_details.remote_path_templates() != expected_templates:
            raise ValueError(
                "OLMES detail remote path templates must correspond to local tables"
            )

        recipes = source_manifest.olmes_details.recipes
        if set(recipes) != set(olmes_contract.recipe_map):
            raise ValueError(
                "OLMES detail publication recipes must match configured recipes"
            )
        expanded_detail_paths = tuple(
            template.format(recipe=recipe)
            for recipe in recipes
            for template in self.olmes_details.remote_path_templates()
        )
        remote_paths = (
            self.ppl.remote_path,
            self.olmes.remote_path,
            self.scaling_law.evaluations_remote_path,
            self.scaling_law.checkpoint_losses_remote_path,
            *expanded_detail_paths,
            self.published_results.remote_root,
        )
        if len(remote_paths) != len(set(remote_paths)):
            raise ValueError("publication remote paths must be unique")
        return self


@cache
def load_publishing_contract() -> PublishingContract:
    contract = PublishingContract.model_validate(load_toml("publishing.toml"))
    return contract.validate_references(
        olmes_contract=load_olmes_contract(),
        scaling_law_contract=load_scaling_law_contract(),
        source_manifest=load_source_manifest(),
    )
