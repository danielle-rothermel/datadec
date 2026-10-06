from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from datadec.config import (
    CHECKPOINT_ENRICHMENT_TYPES,
    PPL_IDENTITY_COLUMNS,
    PPL_METRIC_COLUMNS,
    PUBLISHED_RESULT_SCHEMAS,
    OLMESTableContract,
    PublishedResultSchema,
    PublishedResultsManifest,
    PublishingContract,
    ScalingLawTableContract,
    load_olmes_contract,
    load_published_results_manifest,
    load_publishing_contract,
    load_scaling_law_contract,
)
from datadec.data.artifacts import DataArtifacts
from datadec.data.selection import DatasetSelection, selected_published_result_sources

type ParquetLogicalType = Literal["string", "int64", "float64", "bool"]


@dataclass(frozen=True, slots=True)
class PublicationColumn:
    name: str
    logical_type: ParquetLogicalType
    nullable: bool | None = None


@dataclass(frozen=True, slots=True)
class PublicationFile:
    local_path: Path
    remote_path: str
    expected_schema: tuple[PublicationColumn, ...] | None = None


@dataclass(frozen=True, slots=True)
class PublicationUnit:
    name: str
    files: tuple[PublicationFile, ...]
    commit_message: str


def _publication_schema(
    table: OLMESTableContract | ScalingLawTableContract,
) -> tuple[PublicationColumn, ...]:
    return tuple(
        PublicationColumn(column.name, column.logical_type, column.nullable)
        for column in table.columns
    )


def _ppl_publication_schema() -> tuple[PublicationColumn, ...]:
    return (
        *(PublicationColumn(name, "string") for name in PPL_IDENTITY_COLUMNS[:3]),
        PublicationColumn("step", "int64"),
        *(
            PublicationColumn(name, logical_type)
            for name, logical_type in CHECKPOINT_ENRICHMENT_TYPES
        ),
        *(PublicationColumn(name, "float64") for name in PPL_METRIC_COLUMNS),
    )


def ppl_publication_unit(
    artifacts: DataArtifacts,
    *,
    contract: PublishingContract | None = None,
    output_path: Path | None = None,
) -> PublicationUnit:
    publishing = contract or load_publishing_contract()
    return PublicationUnit(
        name="ppl",
        files=(
            PublicationFile(
                local_path=output_path or artifacts.get_path("ppl_processed"),
                remote_path=publishing.ppl.remote_path,
                expected_schema=_ppl_publication_schema(),
            ),
        ),
        commit_message=publishing.ppl.commit_message,
    )


def olmes_publication_unit(
    artifacts: DataArtifacts,
    *,
    contract: PublishingContract | None = None,
    output_path: Path | None = None,
) -> PublicationUnit:
    publishing = contract or load_publishing_contract()
    table = load_olmes_contract().tables.aggregate
    return PublicationUnit(
        name="olmes",
        files=(
            PublicationFile(
                local_path=output_path or artifacts.get_path("olmes_processed"),
                remote_path=publishing.olmes.remote_path,
                expected_schema=_publication_schema(table),
            ),
        ),
        commit_message=publishing.olmes.commit_message,
    )


def scaling_law_publication_unit(
    artifacts: DataArtifacts,
    *,
    contract: PublishingContract | None = None,
    evaluations_output_path: Path | None = None,
    checkpoint_losses_output_path: Path | None = None,
) -> PublicationUnit:
    publishing = contract or load_publishing_contract()
    scaling_law = load_scaling_law_contract()
    return PublicationUnit(
        name="scaling-law",
        files=(
            PublicationFile(
                local_path=(
                    evaluations_output_path or artifacts.scaling_law_evaluations_path()
                ),
                remote_path=publishing.scaling_law.evaluations_remote_path,
                expected_schema=_publication_schema(scaling_law.tables.evaluations),
            ),
            PublicationFile(
                local_path=(
                    checkpoint_losses_output_path
                    or artifacts.scaling_law_checkpoint_losses_path()
                ),
                remote_path=publishing.scaling_law.checkpoint_losses_remote_path,
                expected_schema=_publication_schema(
                    scaling_law.tables.checkpoint_losses
                ),
            ),
        ),
        commit_message=publishing.scaling_law.commit_message,
    )


def olmes_details_tasks_publication_unit(
    artifacts: DataArtifacts,
    *,
    contract: PublishingContract | None = None,
    output_path: Path | None = None,
) -> PublicationUnit:
    publishing = contract or load_publishing_contract()
    return PublicationUnit(
        name="olmes-detail-tasks",
        files=(
            PublicationFile(
                output_path or artifacts.olmes_details_tasks_path(),
                publishing.olmes_details.tasks_remote_path,
                _publication_schema(load_olmes_contract().tables.detailed_tasks),
            ),
        ),
        commit_message="Publish consolidated OLMES task summaries",
    )


def olmes_details_publication_unit(
    artifacts: DataArtifacts,
    recipe: str,
    *,
    contract: PublishingContract | None = None,
    output_tasks_path: Path | None = None,
    output_instances_path: Path | None = None,
    output_choices_path: Path | None = None,
    include_tasks: bool = True,
) -> PublicationUnit:
    publishing = contract or load_publishing_contract()
    olmes = load_olmes_contract()
    detail = publishing.olmes_details
    tasks = olmes_details_tasks_publication_unit(
        artifacts, contract=publishing, output_path=output_tasks_path
    )
    return PublicationUnit(
        name=f"olmes-details:{recipe}",
        files=(
            *(tasks.files if include_tasks else ()),
            PublicationFile(
                output_instances_path or artifacts.olmes_details_instances_path(recipe),
                detail.instances_remote_path_template.format(recipe=recipe),
                _publication_schema(olmes.tables.detailed_instances),
            ),
            PublicationFile(
                output_choices_path or artifacts.olmes_details_choices_path(recipe),
                detail.choices_remote_path_template.format(recipe=recipe),
                _publication_schema(olmes.tables.detailed_choices),
            ),
        ),
        commit_message=detail.commit_message_template.format(recipe=recipe),
    )


def published_results_publication_units(
    artifacts: DataArtifacts,
    *,
    units: Sequence[str] = (),
    contract: PublishingContract | None = None,
    manifest: PublishedResultsManifest | None = None,
) -> tuple[PublicationUnit, ...]:
    publishing = contract or load_publishing_contract()
    published_results = manifest or load_published_results_manifest()
    sources = selected_published_result_sources(units, published_results)
    schemas = tuple(
        dict.fromkeys(source.schema for source in sources if source.schema is not None)
    )
    return tuple(
        PublicationUnit(
            name=f"published-results:{schema}",
            files=(
                _published_result_publication_file(
                    artifacts,
                    schema,
                    remote_root=publishing.published_results.remote_root,
                ),
            ),
            commit_message=publishing.published_results.commit_message_template.format(
                unit=schema
            ),
        )
        for schema in schemas
    )


def _published_result_publication_file(
    artifacts: DataArtifacts,
    schema_name: PublishedResultSchema,
    *,
    remote_root: str,
) -> PublicationFile:
    schema = PUBLISHED_RESULT_SCHEMAS[schema_name]
    return PublicationFile(
        local_path=artifacts.published_result_table_path(schema_name),
        remote_path=(PurePosixPath(remote_root) / f"{schema_name}.parquet").as_posix(),
        expected_schema=(
            PublicationColumn("source_file", "string", False),
            PublicationColumn("source_unit", "string", False),
            *(
                PublicationColumn(column.name, column.logical_type, column.nullable)
                for column in schema.columns
            ),
        ),
    )


def publication_units(
    artifacts: DataArtifacts,
    selection: DatasetSelection,
    *,
    contract: PublishingContract | None = None,
    manifest: PublishedResultsManifest | None = None,
) -> tuple[PublicationUnit, ...]:
    publishing = contract or load_publishing_contract()
    units: list[PublicationUnit] = []
    if selection.ppl:
        units.append(ppl_publication_unit(artifacts, contract=publishing))
    if selection.olmes:
        units.append(olmes_publication_unit(artifacts, contract=publishing))
    if selection.olmes_details:
        units.append(
            olmes_details_tasks_publication_unit(artifacts, contract=publishing)
        )
    units.extend(
        olmes_details_publication_unit(
            artifacts, recipe, contract=publishing, include_tasks=False
        )
        for recipe in selection.olmes_details
    )
    if selection.scaling_law:
        units.append(scaling_law_publication_unit(artifacts, contract=publishing))
    if selection.published_results:
        units.extend(
            published_results_publication_units(
                artifacts,
                units=selection.published_results,
                contract=publishing,
                manifest=manifest,
            )
        )
    return tuple(units)


__all__ = [
    "ParquetLogicalType",
    "PublicationColumn",
    "PublicationFile",
    "PublicationUnit",
    "olmes_details_publication_unit",
    "olmes_details_tasks_publication_unit",
    "olmes_publication_unit",
    "ppl_publication_unit",
    "publication_units",
    "published_results_publication_units",
    "scaling_law_publication_unit",
]
