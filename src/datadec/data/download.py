from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Protocol, cast
from urllib.request import Request, urlopen

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download

from datadec.config import (
    DatasetSource,
    DetailSource,
    PublishedResultFile,
    PublishedResultsManifest,
    SourceManifest,
    load_published_results_manifest,
    load_publishing_contract,
    load_source_manifest,
)
from datadec.data.artifacts import DataArtifacts
from datadec.data.publication import (
    PublicationFile,
    PublicationUnit,
    olmes_details_tasks_publication_unit,
    publication_units,
)
from datadec.data.publish import validate_publication_unit
from datadec.data.selection import DatasetSelection, selected_published_result_sources

_GOOGLE_DRIVE_DOWNLOAD_URL = (
    "https://drive.usercontent.google.com/download?id={file_id}"
    "&export=download&confirm=t"
)
_DOWNLOAD_CHUNK_SIZE = 1024 * 1024
_DOWNLOAD_TIMEOUT_SECONDS = 60
_CONTENT_RANGE_PATTERN = re.compile(r"bytes (\d+)-(\d+)/(\d+)")


@dataclass(frozen=True, slots=True)
class DownloadResult:
    source: str
    destination: Path
    status: Literal["downloaded", "reused"]


class _ParquetDataset(Protocol):
    def to_parquet(self, path: Path) -> object: ...


def _is_valid_dataset_parquet(path: Path) -> bool:
    try:
        metadata = pq.ParquetFile(path).metadata
    except (OSError, pa.ArrowInvalid):
        return False
    return metadata.num_rows > 0 and metadata.num_columns > 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(_DOWNLOAD_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_file_identity(
    path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    description: str,
) -> None:
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"{description} has unexpected size: {path} has {actual_size} bytes, "
            f"expected {expected_size}"
        )
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{description} has unexpected SHA-256: {path} has {actual_sha256}, "
            f"expected {expected_sha256}"
        )


def _matches_file_identity(
    path: Path, *, expected_size: int, expected_sha256: str
) -> bool:
    try:
        return (
            path.is_file()
            and path.stat().st_size == expected_size
            and (_sha256(path) == expected_sha256)
        )
    except OSError:
        return False


def _download_dataset_source(
    paths: DataArtifacts,
    source: DatasetSource,
    *,
    force: bool,
) -> DownloadResult:
    destination = paths.dataset_source_path(source)
    if destination.exists() and not force and _is_valid_dataset_parquet(destination):
        return DownloadResult(source.id, destination, "reused")

    destination.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = paths.data_dir / "cache" / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)
    load_kwargs: dict[str, object] = {
        "split": source.split,
        "revision": source.revision,
        "cache_dir": cache_dir,
    }
    if force:
        load_kwargs["download_mode"] = "force_redownload"
    dataset = cast(_ParquetDataset, load_dataset(source.repo_id, **load_kwargs))
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        dataset.to_parquet(temporary_path)
        if not _is_valid_dataset_parquet(temporary_path):
            raise ValueError(
                f"downloaded dataset is not a non-empty Parquet: {source.id}"
            )
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return DownloadResult(source.id, destination, "downloaded")


def _download_detail_source(
    paths: DataArtifacts,
    source: DetailSource,
    recipe: str,
    *,
    force: bool,
) -> DownloadResult:
    artifact = source.file_for_recipe(recipe)
    filename = source.filename_template.format(recipe=recipe)
    output_root = paths.data_dir / source.output_root
    destination = paths.olmes_detail_source_path(recipe, source)
    result_source = f"{source.id}:{recipe}"
    if destination.exists() and not force:
        _validate_file_identity(
            destination,
            expected_size=artifact.expected_size,
            expected_sha256=artifact.sha256,
            description=f"existing OLMES detail archive for {recipe}",
        )
        return DownloadResult(result_source, destination, "reused")

    output_root.mkdir(parents=True, exist_ok=True)
    cache_dir = paths.data_dir / "cache" / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id=source.repo_id,
        repo_type=source.repo_type,
        filename=filename,
        revision=source.revision,
        cache_dir=cache_dir,
        local_dir=output_root,
        force_download=force,
    )
    _validate_file_identity(
        destination,
        expected_size=artifact.expected_size,
        expected_sha256=artifact.sha256,
        description=f"downloaded OLMES detail archive for {recipe}",
    )
    return DownloadResult(result_source, destination, "downloaded")


def _response_status(response: object) -> int:
    status = getattr(response, "status", None)
    if status is not None:
        return int(status)
    getcode = getattr(response, "getcode")
    return int(getcode())


def _download_published_result_file(
    paths: DataArtifacts,
    source: PublishedResultFile,
    *,
    force: bool,
) -> DownloadResult:
    destination = paths.published_result_download_path(source)
    result_source = f"{source.category.replace('_', '-')}:{source.path}"
    if destination.exists() and not force:
        _validate_file_identity(
            destination,
            expected_size=source.expected_size,
            expected_sha256=source.sha256,
            description=f"existing file for {source.path}",
        )
        return DownloadResult(result_source, destination, "reused")

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    partial_size = 0 if force or not partial.exists() else partial.stat().st_size
    if partial_size > source.expected_size:
        raise ValueError(
            f"partial file has unexpected size for {source.path}: {partial} has "
            f"{partial_size} bytes, expected at most {source.expected_size}"
        )
    if partial_size == source.expected_size and not force:
        _validate_file_identity(
            partial,
            expected_size=source.expected_size,
            expected_sha256=source.sha256,
            description=f"completed partial file for {source.path}",
        )
        partial.replace(destination)
        return DownloadResult(result_source, destination, "downloaded")

    request = Request(
        _GOOGLE_DRIVE_DOWNLOAD_URL.format(file_id=source.id),
        headers={"Range": f"bytes={partial_size}-"} if partial_size else {},
    )
    try:
        with urlopen(  # noqa: S310 - pinned HTTPS endpoint
            request, timeout=_DOWNLOAD_TIMEOUT_SECONDS
        ) as response:
            status = _response_status(response)
            mode = "wb"
            if partial_size:
                if status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    match = _CONTENT_RANGE_PATTERN.fullmatch(content_range)
                    if (
                        match is None
                        or int(match.group(1)) != partial_size
                        or int(match.group(2)) != source.expected_size - 1
                        or int(match.group(3)) != source.expected_size
                    ):
                        raise ValueError(
                            f"invalid Content-Range for resume: {content_range!r}"
                        )
                    mode = "ab"
                elif status != 200:
                    raise ValueError(f"unexpected HTTP status {status} while resuming")
            elif status != 200:
                raise ValueError(f"unexpected HTTP status {status}")

            with partial.open(mode) as file:
                while chunk := response.read(_DOWNLOAD_CHUNK_SIZE):
                    file.write(chunk)

        _validate_file_identity(
            partial,
            expected_size=source.expected_size,
            expected_sha256=source.sha256,
            description=f"downloaded file for {source.path}",
        )
        partial.replace(destination)
    except Exception as exc:
        raise RuntimeError(
            f"failed to download {source.path} to {destination}"
        ) from exc

    return DownloadResult(result_source, destination, "downloaded")


def download_sources(
    paths: DataArtifacts,
    selection: DatasetSelection,
    *,
    force: bool = False,
    verbose: bool = False,
    manifest: SourceManifest | None = None,
    published_results_manifest: PublishedResultsManifest | None = None,
) -> tuple[DownloadResult, ...]:
    manifest = manifest or load_source_manifest()
    results: list[DownloadResult] = []

    def record(result: DownloadResult) -> None:
        results.append(result)
        if verbose:
            print(f"{result.source}: {result.status} -> {result.destination}")

    if selection.ppl:
        record(_download_dataset_source(paths, manifest.ppl, force=force))
    if selection.olmes:
        record(_download_dataset_source(paths, manifest.olmes, force=force))
    for recipe in selection.olmes_details:
        record(
            _download_detail_source(paths, manifest.olmes_details, recipe, force=force)
        )

    if (
        selection.scaling_law
        or selection.published_results
        or selection.published_figures
    ):
        drive_manifest = published_results_manifest or load_published_results_manifest()
        categories: list[str] = []
        if selection.scaling_law:
            categories.append("scaling_law")
        if selection.published_results:
            categories.append("published_results")
        if selection.published_figures:
            categories.append("published_figures")
        selected_sources = (
            {
                source.path
                for source in selected_published_result_sources(
                    selection.published_results, drive_manifest
                )
            }
            if selection.published_results
            else set()
        )
        for category in categories:
            for source in drive_manifest.files:
                if source.category == category and (
                    category != "published_results" or source.path in selected_sources
                ):
                    record(_download_published_result_file(paths, source, force=force))

    return tuple(results)


def _remote_lfs_sha256(remote_file: object) -> str | None:
    lfs = getattr(remote_file, "lfs", None)
    if lfs is None:
        return None
    if isinstance(lfs, Mapping):
        value = lfs.get("sha256") or lfs.get("oid")
    else:
        value = getattr(lfs, "sha256", None)
    return value if isinstance(value, str) and value else None


def _validate_processed_file(
    unit: PublicationUnit,
    publication_file: PublicationFile,
    path: Path,
) -> None:
    validate_publication_unit(
        PublicationUnit(
            name=unit.name,
            files=(replace(publication_file, local_path=path),),
            commit_message=unit.commit_message,
        )
    )


def download_processed_outputs(
    artifacts: DataArtifacts,
    selection: DatasetSelection,
    *,
    force: bool = False,
    hf_token: str | None = None,
    api: HfApi | None = None,
    verbose: bool = False,
) -> tuple[DownloadResult, ...]:
    """Download selected published outputs from one immutable repository commit."""
    if selection.published_figures:
        raise ValueError(
            "published figures are raw-only; use download_sources for that selection"
        )
    units = publication_units(artifacts, selection)
    return _download_units(
        artifacts, units, force=force, hf_token=hf_token, api=api, verbose=verbose
    )


def restore_task_summaries(
    artifacts: DataArtifacts,
    *,
    hf_token: str | None = None,
) -> tuple[DownloadResult, ...]:
    """Restore the shared published task table before adding a recipe locally."""
    if artifacts.olmes_details_tasks_path().exists():
        return ()
    publishing = load_publishing_contract()
    api = HfApi(token=hf_token)
    remote = api.get_paths_info(
        publishing.target.repo_id,
        [publishing.olmes_details.tasks_remote_path],
        repo_type="dataset",
        revision=publishing.target.revision,
    )
    if not remote:
        return ()
    return _download_units(
        artifacts,
        (olmes_details_tasks_publication_unit(artifacts),),
        hf_token=hf_token,
        api=api,
    )


def _download_units(
    artifacts: DataArtifacts,
    units: tuple[PublicationUnit, ...],
    *,
    force: bool = False,
    hf_token: str | None = None,
    api: HfApi | None = None,
    verbose: bool = False,
) -> tuple[DownloadResult, ...]:
    publishing = load_publishing_contract()
    files = tuple((unit, file) for unit in units for file in unit.files)
    if not files:
        return ()

    resolved_api = api or HfApi(token=hf_token)
    target = publishing.target
    repo_info = resolved_api.repo_info(
        target.repo_id,
        repo_type="dataset",
        revision=target.revision,
    )
    commit_oid = getattr(repo_info, "sha", None)
    if not isinstance(commit_oid, str) or not commit_oid:
        raise RuntimeError(f"could not resolve {target.repo_id}@{target.revision}")

    remote_paths = [file.remote_path for _, file in files]
    remote_items = resolved_api.get_paths_info(
        target.repo_id,
        remote_paths,
        repo_type="dataset",
        revision=commit_oid,
    )
    remote_by_path = {getattr(item, "path", None): item for item in remote_items}
    missing = [path for path in remote_paths if path not in remote_by_path]
    if missing:
        raise RuntimeError(
            f"processed files are missing at immutable commit {commit_oid}: "
            f"{', '.join(missing)}"
        )

    staged: list[tuple[Path, Path]] = []
    results: list[DownloadResult] = []
    try:
        for unit, publication_file in files:
            remote_path = publication_file.remote_path
            remote = remote_by_path[remote_path]
            expected_size = getattr(remote, "size", None)
            expected_sha256 = _remote_lfs_sha256(remote)
            if not isinstance(expected_size, int) or expected_size <= 0:
                raise RuntimeError(
                    f"processed file size missing at {commit_oid} for {remote_path}"
                )
            if expected_sha256 is None:
                raise RuntimeError(
                    f"processed file LFS SHA-256 missing at {commit_oid} "
                    f"for {remote_path}"
                )

            destination = publication_file.local_path
            source_name = f"{unit.name}:{remote_path}"
            if not force and _matches_file_identity(
                destination,
                expected_size=expected_size,
                expected_sha256=expected_sha256,
            ):
                _validate_processed_file(unit, publication_file, destination)
                result = DownloadResult(source_name, destination, "reused")
                results.append(result)
                continue

            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = destination.with_name(f".{destination.name}.tmp")
            temporary_path.unlink(missing_ok=True)
            staged.append((temporary_path, destination))
            cached_path = Path(
                resolved_api.hf_hub_download(
                    target.repo_id,
                    remote_path,
                    repo_type="dataset",
                    revision=commit_oid,
                    cache_dir=artifacts.data_dir / "cache" / "huggingface",
                    force_download=force,
                )
            )
            shutil.copyfile(cached_path, temporary_path)
            _validate_file_identity(
                temporary_path,
                expected_size=expected_size,
                expected_sha256=expected_sha256,
                description=f"downloaded processed file {remote_path}",
            )
            _validate_processed_file(unit, publication_file, temporary_path)
            results.append(DownloadResult(source_name, destination, "downloaded"))

        for temporary_path, destination in staged:
            os.replace(temporary_path, destination)
        if verbose:
            for result in results:
                print(f"{result.source}: {result.status} -> {result.destination}")
    finally:
        for temporary_path, _ in staged:
            temporary_path.unlink(missing_ok=True)

    return tuple(results)


__all__ = [
    "DownloadResult",
    "download_processed_outputs",
    "download_sources",
    "restore_task_summaries",
]
