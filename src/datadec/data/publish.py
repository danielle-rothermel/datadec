from __future__ import annotations

import hashlib
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from dr_hf import (
    DatasetFileCommitEntry,
    HFLocation,
    commit_dataset_files_to_hf,
)
from huggingface_hub import HfApi

from datadec.config import (
    PublishingTarget,
    load_olmes_contract,
    load_publishing_contract,
)
from datadec.data.artifacts import DataArtifacts
from datadec.data.preprocess.duckdb import (
    prepare_parquet_export,
    quote_identifier,
    remove_owned_file,
    replace_parquet_exports,
    sql_literal,
)
from datadec.data.publication import (
    ParquetLogicalType,
    PublicationColumn,
    PublicationFile,
    PublicationUnit,
    publication_units,
)
from datadec.data.selection import DatasetSelection

_ARROW_TYPES: dict[ParquetLogicalType, pa.DataType] = {
    "string": pa.string(),
    "int64": pa.int64(),
    "float64": pa.float64(),
    "bool": pa.bool_(),
}
_HASH_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class PublicationResult:
    unit_name: str
    created: bool
    commit_oid: str
    remote_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ValidatedLocalFile:
    publication_file: PublicationFile
    size: int


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


def validate_publication_unit(unit: PublicationUnit) -> None:
    """Validate a complete local publication unit without accessing the network."""
    _validated_local_files(unit)


def _validated_local_files(
    unit: PublicationUnit,
) -> tuple[_ValidatedLocalFile, ...]:
    if not unit.files:
        raise ValueError(f"publication unit has no files: {unit.name}")
    remote_paths = tuple(file.remote_path for file in unit.files)
    if len(remote_paths) != len(set(remote_paths)):
        raise ValueError(f"publication unit has duplicate remote paths: {unit.name}")
    return tuple(_validate_local_file(file) for file in unit.files)


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


def _resolve_revision(api: HfApi, target: PublishingTarget) -> str:
    repo_info = api.repo_info(
        target.repo_id,
        repo_type="dataset",
        revision=target.revision,
    )
    commit_oid = getattr(repo_info, "sha", None)
    if not isinstance(commit_oid, str) or not commit_oid:
        raise RuntimeError(f"could not resolve {target.repo_id}@{target.revision}")
    return commit_oid


def verify_published_unit(
    unit: PublicationUnit,
    *,
    target: PublishingTarget | None = None,
    hf_token: str | None = None,
    api: HfApi | None = None,
) -> PublicationResult:
    """Verify local files against one immutable commit without uploading."""
    local_files = _validated_local_files(unit)
    resolved_target = target or load_publishing_contract().target
    resolved_api = api or HfApi(token=hf_token)
    commit_oid = _resolve_revision(resolved_api, resolved_target)
    _verify_remote_files(
        resolved_api,
        target=resolved_target,
        commit_oid=commit_oid,
        local_files=local_files,
    )
    return PublicationResult(
        unit_name=unit.name,
        created=False,
        commit_oid=commit_oid,
        remote_paths=tuple(file.remote_path for file in unit.files),
    )


def publish_unit(
    unit: PublicationUnit,
    *,
    target: PublishingTarget | None = None,
    hf_token: str | None = None,
    api: HfApi | None = None,
) -> PublicationResult:
    local_files = _validated_local_files(unit)
    resolved_target = target or load_publishing_contract().target
    resolved_api = api or HfApi(token=hf_token)
    expected_parent = _resolve_revision(resolved_api, resolved_target)
    if _preserve_remote_task_recipes(
        unit, api=resolved_api, target=resolved_target, commit_oid=expected_parent
    ):
        local_files = _validated_local_files(unit)

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
    return PublicationResult(
        unit_name=unit.name,
        created=commit_result.created,
        commit_oid=commit_result.commit_oid,
        remote_paths=tuple(file.remote_path for file in unit.files),
    )


def _preserve_remote_task_recipes(
    unit: PublicationUnit,
    *,
    api: HfApi,
    target: PublishingTarget,
    commit_oid: str,
) -> bool:
    """Keep published recipes absent from a locally created shared task table.

    A standalone no-upload run can create only one recipe locally. Publishing
    that table must retain other remotely published recipes, just as local
    preprocessing retains recipes already present on disk.
    """
    remote_path = load_publishing_contract().olmes_details.tasks_remote_path
    task_files = tuple(file for file in unit.files if file.remote_path == remote_path)
    if not task_files:
        return False
    file = task_files[0]
    remote = api.get_paths_info(
        target.repo_id, [remote_path], repo_type="dataset", revision=commit_oid
    )
    if not remote:
        return False
    if _local_sha256(file.local_path) == _lfs_sha256(remote[0]):
        return False

    with tempfile.TemporaryDirectory(prefix="datadec-published-tasks-") as directory:
        cached_path = Path(
            api.hf_hub_download(
                target.repo_id,
                remote_path,
                repo_type="dataset",
                revision=commit_oid,
                cache_dir=directory,
            )
        )
        remote_file = _validate_local_file(replace(file, local_path=cached_path))
        _verify_remote_files(
            api, target=target, commit_oid=commit_oid, local_files=(remote_file,)
        )
        connection = duckdb.connect()
        export = None
        try:
            retained = (
                f"SELECT * FROM read_parquet({sql_literal(cached_path)}) WHERE recipe NOT IN "
                f"(SELECT DISTINCT recipe FROM read_parquet({sql_literal(file.local_path)}))"
            )
            count = connection.execute(f"SELECT count(*) FROM ({retained})").fetchone()
            assert count is not None
            if count[0] == 0:
                return False
            order = ", ".join(
                quote_identifier(key)
                for key in load_olmes_contract().tables.detailed_tasks.sort_key
            )
            export = prepare_parquet_export(
                connection,
                select_sql=f"SELECT * FROM (SELECT * FROM read_parquet({sql_literal(file.local_path)}) UNION ALL {retained}) ORDER BY {order}",
                output_path=file.local_path,
            )
            replace_parquet_exports((export,))
        finally:
            connection.close()
            if export is not None:
                remove_owned_file(export.temporary_path)
    return True


def publish_existing_outputs(
    artifacts: DataArtifacts,
    selection: DatasetSelection,
    *,
    hf_token: str | None = None,
) -> tuple[PublicationResult, ...]:
    publishing = load_publishing_contract()
    return tuple(
        publish_unit(
            unit,
            target=publishing.target,
            hf_token=hf_token,
        )
        for unit in publication_units(artifacts, selection, contract=publishing)
    )


__all__ = [
    "PublicationResult",
    "publish_existing_outputs",
    "publish_unit",
    "validate_publication_unit",
    "verify_published_unit",
]
