from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from datadec.config import (
    CHECKPOINT_ENRICHMENT_TYPES,
    OLMESTableContract,
    PPL_IDENTITY_COLUMNS,
    PPL_METRIC_COLUMNS,
    PUBLISHED_RESULT_SCHEMAS,
    PublishedResultFile,
    PublishedResultsManifest,
    PublishingContract,
    ScalingLawTableContract,
    load_olmes_contract,
    load_published_results_manifest,
    load_publishing_contract,
    load_scaling_law_contract,
)
from datadec.data.artifacts import DataArtifacts
from datadec.data.selection import DatasetSelection, resolve_published_result_units

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


def olmes_details_publication_unit(
    artifacts: DataArtifacts,
    recipe: str,
    *,
    contract: PublishingContract | None = None,
    output_tasks_path: Path | None = None,
    output_instances_path: Path | None = None,
    output_choices_path: Path | None = None,
) -> PublicationUnit:
    publishing = contract or load_publishing_contract()
    olmes = load_olmes_contract()
    detail_contract = publishing.olmes_details
    return PublicationUnit(
        name=f"olmes-details:{recipe}",
        files=(
            PublicationFile(
                local_path=(
                    output_tasks_path or artifacts.olmes_details_tasks_path(recipe)
                ),
                remote_path=detail_contract.tasks_remote_path_template.format(
                    recipe=recipe
                ),
                expected_schema=_publication_schema(olmes.tables.detailed_tasks),
            ),
            PublicationFile(
                local_path=(
                    output_instances_path
                    or artifacts.olmes_details_instances_path(recipe)
                ),
                remote_path=detail_contract.instances_remote_path_template.format(
                    recipe=recipe
                ),
                expected_schema=_publication_schema(olmes.tables.detailed_instances),
            ),
            PublicationFile(
                local_path=(
                    output_choices_path or artifacts.olmes_details_choices_path(recipe)
                ),
                remote_path=detail_contract.choices_remote_path_template.format(
                    recipe=recipe
                ),
                expected_schema=_publication_schema(olmes.tables.detailed_choices),
            ),
        ),
        commit_message=detail_contract.commit_message_template.format(recipe=recipe),
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
    selected_units = resolve_published_result_units(units, published_results)
    result: list[PublicationUnit] = []
    for unit in selected_units:
        sources = tuple(
            source
            for source in published_results.files
            if source.category == "published_results"
            and source.publication_unit == unit
        )
        result.append(
            PublicationUnit(
                name=f"published-results:{unit}",
                files=tuple(
                    _published_result_publication_file(
                        artifacts,
                        source,
                        remote_root=publishing.published_results.remote_root,
                    )
                    for source in sources
                ),
                commit_message=(
                    publishing.published_results.commit_message_template.format(
                        unit=unit
                    )
                ),
            )
        )
    return tuple(result)


def _published_result_publication_file(
    artifacts: DataArtifacts,
    source: PublishedResultFile,
    *,
    remote_root: str,
) -> PublicationFile:
    schema_name = source.schema
    if schema_name is None:
        raise ValueError(f"published result has no schema: {source.path}")
    schema = PUBLISHED_RESULT_SCHEMAS[schema_name]
    return PublicationFile(
        local_path=artifacts.published_result_output_path(source),
        remote_path=(
            PurePosixPath(remote_root) / source.parquet_relative_path()
        ).as_posix(),
        expected_schema=tuple(
            PublicationColumn(column.name, column.logical_type, column.nullable)
            for column in schema.columns
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
    units.extend(
        olmes_details_publication_unit(artifacts, recipe, contract=publishing)
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
    "olmes_publication_unit",
    "ppl_publication_unit",
    "publication_units",
    "published_results_publication_units",
    "scaling_law_publication_unit",
]
