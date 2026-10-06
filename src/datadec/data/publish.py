from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq
from dr_hf import (
    DatasetFileCommitEntry,
    HFLocation,
    commit_dataset_files_to_hf,
)
from huggingface_hub import HfApi

from datadec.config import (
    CHECKPOINT_ENRICHMENT_TYPES,
    OLMESTableContract,
    PPL_IDENTITY_COLUMNS,
    PPL_METRIC_COLUMNS,
    PUBLISHED_RESULT_SCHEMAS,
    PublishedResultFile,
    PublishedResultsManifest,
    PublishingContract,
    PublishingTarget,
    ScalingLawTableContract,
    load_olmes_contract,
    load_published_results_manifest,
    load_publishing_contract,
    load_scaling_law_contract,
    load_source_manifest,
)
from datadec.data.selection import resolve_olmes_detail_recipes, resolve_published_result_units
from datadec.data.artifacts import DataArtifacts

type ParquetLogicalType = Literal["string", "int64", "float64", "bool"]

_ARROW_TYPES: dict[ParquetLogicalType, pa.DataType] = {
    "string": pa.string(),
    "int64": pa.int64(),
    "float64": pa.float64(),
    "bool": pa.bool_(),
}
_HASH_CHUNK_SIZE = 1024 * 1024


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
    cleanup_paths: tuple[Path, ...] = ()


@dataclass(frozen=True, slots=True)
class PublicationResult:
    unit_name: str
    created: bool
    commit_oid: str
    remote_paths: tuple[str, ...]
    deleted_sources: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class _ValidatedLocalFile:
    publication_file: PublicationFile
    size: int


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
    paths: DataArtifacts,
    *,
    contract: PublishingContract | None = None,
    output_path: Path | None = None,
) -> PublicationUnit:
    publishing = contract or load_publishing_contract()
    return PublicationUnit(
        name="ppl",
        files=(
            PublicationFile(
                local_path=output_path or paths.get_path("ppl_processed"),
                remote_path=publishing.ppl.remote_path,
                expected_schema=_ppl_publication_schema(),
            ),
        ),
        commit_message=publishing.ppl.commit_message,
    )


def olmes_publication_unit(
    paths: DataArtifacts,
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
                local_path=output_path or paths.get_path("olmes_processed"),
                remote_path=publishing.olmes.remote_path,
                expected_schema=_publication_schema(table),
            ),
        ),
        commit_message=publishing.olmes.commit_message,
    )


def scaling_law_publication_unit(
    paths: DataArtifacts,
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
                    evaluations_output_path or paths.scaling_law_evaluations_path()
                ),
                remote_path=publishing.scaling_law.evaluations_remote_path,
                expected_schema=_publication_schema(scaling_law.tables.evaluations),
            ),
            PublicationFile(
                local_path=(
                    checkpoint_losses_output_path
                    or paths.scaling_law_checkpoint_losses_path()
                ),
                remote_path=publishing.scaling_law.checkpoint_losses_remote_path,
                expected_schema=_publication_schema(
                    scaling_law.tables.checkpoint_losses
                ),
            ),
        ),
        commit_message=publishing.scaling_law.commit_message,
        cleanup_paths=paths.scaling_law_raw_paths(),
    )


def olmes_details_publication_unit(
    paths: DataArtifacts,
    recipe: str,
    *,
    contract: PublishingContract | None = None,
    output_tasks_path: Path | None = None,
    output_instances_path: Path | None = None,
    output_choices_path: Path | None = None,
    cleanup_source: bool = True,
) -> PublicationUnit:
    publishing = contract or load_publishing_contract()
    olmes = load_olmes_contract()
    detail_contract = publishing.olmes_details
    manifest = load_source_manifest().olmes_details
    archive = (
        paths.data_dir
        / manifest.output_root
        / manifest.filename_template.format(recipe=recipe)
    )
    return PublicationUnit(
        name=f"olmes-details:{recipe}",
        files=(
            PublicationFile(
                local_path=(
                    output_tasks_path or paths.olmes_details_tasks_path(recipe)
                ),
                remote_path=detail_contract.tasks_remote_path_template.format(
                    recipe=recipe
                ),
                expected_schema=_publication_schema(olmes.tables.detailed_tasks),
            ),
            PublicationFile(
                local_path=(
                    output_instances_path or paths.olmes_details_instances_path(recipe)
                ),
                remote_path=detail_contract.instances_remote_path_template.format(
                    recipe=recipe
                ),
                expected_schema=_publication_schema(olmes.tables.detailed_instances),
            ),
            PublicationFile(
                local_path=(
                    output_choices_path or paths.olmes_details_choices_path(recipe)
                ),
                remote_path=detail_contract.choices_remote_path_template.format(
                    recipe=recipe
                ),
                expected_schema=_publication_schema(olmes.tables.detailed_choices),
            ),
        ),
        commit_message=detail_contract.commit_message_template.format(recipe=recipe),
        cleanup_paths=(archive,) if cleanup_source else (),
    )


def published_results_publication_units(
    paths: DataArtifacts,
    *,
    units: Sequence[str] = (),
    contract: PublishingContract | None = None,
    manifest: PublishedResultsManifest | None = None,
) -> tuple[PublicationUnit, ...]:
    publishing = contract or load_publishing_contract()
    published_results = manifest or load_published_results_manifest()
    selected_units = resolve_published_result_units(units, published_results)
    publication_units: list[PublicationUnit] = []
    for unit in selected_units:
        sources = tuple(
            source
            for source in published_results.files
            if source.category == "published_results"
            and source.publication_unit == unit
        )
        publication_units.append(
            PublicationUnit(
                name=f"published-results:{unit}",
                files=tuple(
                    _published_result_publication_file(
                        paths,
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
                cleanup_paths=tuple(
                    paths.published_result_source_path(source) for source in sources
                ),
            )
        )
    return tuple(publication_units)


def _published_result_publication_file(
    paths: DataArtifacts,
    source: PublishedResultFile,
    *,
    remote_root: str,
) -> PublicationFile:
    schema_name = source.schema
    if schema_name is None:
        raise ValueError(f"published result has no schema: {source.path}")
    schema = PUBLISHED_RESULT_SCHEMAS[schema_name]
    return PublicationFile(
        local_path=paths.published_result_output_path(source),
        remote_path=(
            PurePosixPath(remote_root) / source.parquet_relative_path()
        ).as_posix(),
        expected_schema=tuple(
            PublicationColumn(column.name, column.logical_type, column.nullable)
            for column in schema.columns
        ),
    )


def _validate_local_file(file: PublicationFile) -> _ValidatedLocalFile:
    path = file.local_path
    if not path.is_file():
        raise ValueError(f"publication input is not a regular file: {path}")
    size = path.stat().st_size
    if size == 0:
        raise ValueError(f"publication input is empty: {path}")

    try:
        parquet = pq.ParquetFile(path)
    except (OSError, pa.ArrowInvalid) as error:
        raise ValueError(
            f"publication input is not readable Parquet: {path}"
        ) from error
    if parquet.metadata.num_rows == 0:
        raise ValueError(f"publication input has no rows: {path}")

    expected_schema = file.expected_schema
    if expected_schema is not None:
        actual = tuple(
            PublicationColumn(field.name, _logical_type(field.type, path=path))
            for field in parquet.schema_arrow
        )
        expected_types = tuple(
            PublicationColumn(column.name, column.logical_type)
            for column in expected_schema
        )
        if actual != expected_types:
            raise ValueError(
                f"publication input schema mismatch for {path}: "
                f"expected {expected_types!r}, got {actual!r}"
            )
        required_columns = tuple(
            column.name for column in expected_schema if column.nullable is False
        )
        for batch in parquet.iter_batches(columns=list(required_columns)):
            for name, column in zip(required_columns, batch.columns, strict=True):
                if column.null_count:
                    raise ValueError(
                        f"publication input has nulls in required column {name!r}: "
                        f"{path}"
                    )
    return _ValidatedLocalFile(publication_file=file, size=size)


def _logical_type(data_type: pa.DataType, *, path: Path) -> ParquetLogicalType:
    for logical_type, expected in _ARROW_TYPES.items():
        if data_type == expected:
            return logical_type
    raise ValueError(
        f"unsupported Parquet type {data_type} in publication input: {path}"
    )


def _local_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _lfs_sha256(remote_file: object) -> str | None:
    lfs = getattr(remote_file, "lfs", None)
    if lfs is None:
        return None
    if isinstance(lfs, Mapping):
        value = lfs.get("sha256") or lfs.get("oid")
    else:
        value = getattr(lfs, "sha256", None)
    return value if isinstance(value, str) and value else None


def _verify_remote_files(
    api: HfApi,
    *,
    target: PublishingTarget,
    commit_oid: str,
    local_files: tuple[_ValidatedLocalFile, ...],
) -> None:
    remote_paths = [file.publication_file.remote_path for file in local_files]
    remote_items = api.get_paths_info(
        target.repo_id,
        remote_paths,
        repo_type="dataset",
        revision=commit_oid,
    )
    remote_by_path = {getattr(item, "path", None): item for item in remote_items}
    missing = [path for path in remote_paths if path not in remote_by_path]
    if missing:
        raise RuntimeError(
            "published files are missing at immutable commit "
            f"{commit_oid}: {', '.join(missing)}"
        )

    for local in local_files:
        path = local.publication_file.remote_path
        remote = remote_by_path[path]
        remote_size = getattr(remote, "size", None)
        if remote_size != local.size:
            raise RuntimeError(
                f"published file size mismatch at {commit_oid} for {path}: "
                f"expected {local.size}, got {remote_size!r}"
            )
        remote_sha256 = _lfs_sha256(remote)
        if remote_sha256 is None:
            raise RuntimeError(
                f"published file LFS SHA-256 missing at {commit_oid} for {path}"
            )
        local_sha256 = _local_sha256(local.publication_file.local_path)
        if local_sha256 != remote_sha256:
            raise RuntimeError(
                f"published file LFS SHA-256 mismatch at {commit_oid} for "
                f"{path}: expected {local_sha256}, got {remote_sha256}"
            )


def publish_unit(
    unit: PublicationUnit,
    *,
    target: PublishingTarget | None = None,
    keep_sources: bool = False,
    hf_token: str | None = None,
    api: HfApi | None = None,
) -> PublicationResult:
    if not unit.files:
        raise ValueError(f"publication unit has no files: {unit.name}")
    remote_paths = tuple(file.remote_path for file in unit.files)
    if len(remote_paths) != len(set(remote_paths)):
        raise ValueError(f"publication unit has duplicate remote paths: {unit.name}")

    local_files = tuple(_validate_local_file(file) for file in unit.files)
    resolved_target = target or load_publishing_contract().target
    resolved_api = api or HfApi(token=hf_token)
    repo_info = resolved_api.repo_info(
        resolved_target.repo_id,
        repo_type="dataset",
        revision=resolved_target.revision,
    )
    expected_parent = getattr(repo_info, "sha", None)
    if not isinstance(expected_parent, str) or not expected_parent:
        raise RuntimeError(
            f"could not resolve {resolved_target.repo_id}@{resolved_target.revision}"
        )

    org, repo_name = resolved_target.repo_id.split("/", maxsplit=1)
    commit_result = commit_dataset_files_to_hf(
        [
            DatasetFileCommitEntry(
                local_path=local.publication_file.local_path,
                repo_path=local.publication_file.remote_path,
            )
            for local in local_files
        ],
        HFLocation(org=org, repo_name=repo_name),
        revision=resolved_target.revision,
        expected_parent=expected_parent,
        commit_message=unit.commit_message,
        create_pr=False,
        hf_token=hf_token,
    )
    _verify_remote_files(
        resolved_api,
        target=resolved_target,
        commit_oid=commit_result.commit_oid,
        local_files=local_files,
    )

    deleted_sources: tuple[Path, ...] = ()
    if not keep_sources:
        deleted_sources = tuple(path for path in unit.cleanup_paths if path.is_file())
        for path in deleted_sources:
            path.unlink()

    return PublicationResult(
        unit_name=unit.name,
        created=commit_result.created,
        commit_oid=commit_result.commit_oid,
        remote_paths=remote_paths,
        deleted_sources=deleted_sources,
    )


def publish_existing_outputs(
    paths: DataArtifacts,
    *,
    ppl: bool = False,
    olmes: bool = False,
    olmes_details: Sequence[str] = (),
    scaling_law: bool = False,
    published_results: bool = False,
    keep_sources: bool = False,
    hf_token: str | None = None,
) -> list[PublicationResult]:
    if (
        not ppl
        and not olmes
        and not olmes_details
        and not scaling_law
        and not published_results
    ):
        raise ValueError("select at least one output to publish")

    publishing = load_publishing_contract()
    recipes = resolve_olmes_detail_recipes(
        olmes_details, load_source_manifest().olmes_details
    )
    units: list[PublicationUnit] = []
    if ppl:
        units.append(ppl_publication_unit(paths, contract=publishing))
    if olmes:
        units.append(olmes_publication_unit(paths, contract=publishing))
    if scaling_law:
        units.append(scaling_law_publication_unit(paths, contract=publishing))
    if published_results:
        units.extend(published_results_publication_units(paths, contract=publishing))
    units.extend(
        olmes_details_publication_unit(paths, recipe, contract=publishing)
        for recipe in recipes
    )
    return [
        publish_unit(
            unit,
            target=publishing.target,
            keep_sources=keep_sources,
            hf_token=hf_token,
        )
        for unit in units
    ]


__all__ = [
    "PublicationColumn",
    "PublicationFile",
    "PublicationResult",
    "PublicationUnit",
    "olmes_details_publication_unit",
    "olmes_publication_unit",
    "ppl_publication_unit",
    "published_results_publication_units",
    "publish_existing_outputs",
    "publish_unit",
    "scaling_law_publication_unit",
]
