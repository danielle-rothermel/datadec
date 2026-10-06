from __future__ import annotations

from dataclasses import dataclass
from enum import UNIQUE, StrEnum, verify
import os
from pathlib import Path
import shutil

from huggingface_hub import HfApi

from datadec.data.artifacts import DataArtifacts
from datadec.data.publication import PublicationUnit, publication_units
from datadec.data.publish import verify_published_unit
from datadec.data.selection import DatasetSelection


@verify(UNIQUE)
class CleanupMode(StrEnum):
    RAW = "raw"
    ALL = "all"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class CleanupResult:
    removed_paths: tuple[Path, ...]
    would_remove_paths: tuple[Path, ...]
    missing_paths: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class _DeletionPlan:
    file_paths: tuple[Path, ...]
    tree_paths: tuple[Path, ...]

    @property
    def paths(self) -> tuple[Path, ...]:
        return (*self.file_paths, *self.tree_paths)


def _deduplicate(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    return tuple(dict.fromkeys(paths))


def _raw_deletion_plan(
    artifacts: DataArtifacts, selection: DatasetSelection
) -> _DeletionPlan:
    cache_paths = artifacts.cache_paths(selection)
    if selection.all_data:
        cache_files: tuple[Path, ...] = ()
        cache_trees = cache_paths
    else:
        cache_files = cache_paths
        cache_trees = ()
    return _DeletionPlan(
        file_paths=_deduplicate(
            (
                *artifacts.raw_paths(selection),
                *artifacts.raw_intermediate_paths(selection),
                *cache_files,
            )
        ),
        tree_paths=_deduplicate(cache_trees),
    )


def _data_deletion_plan(
    artifacts: DataArtifacts, selection: DatasetSelection
) -> _DeletionPlan:
    raw_plan = _raw_deletion_plan(artifacts, selection)
    processing_trees = artifacts.processing_intermediate_tree_paths(selection)
    tree_paths = set(processing_trees)
    return _DeletionPlan(
        file_paths=_deduplicate(
            (
                *raw_plan.file_paths,
                *artifacts.processed_paths(selection),
                *(
                    path
                    for path in artifacts.processing_intermediate_paths(selection)
                    if path not in tree_paths
                ),
            )
        ),
        tree_paths=_deduplicate((*raw_plan.tree_paths, *processing_trees)),
    )


def _validate_owned_path(data_dir: Path, path: Path) -> None:
    lexical_root = Path(os.path.abspath(data_dir))
    lexical_path = Path(os.path.abspath(path))
    try:
        relative_path = lexical_path.relative_to(lexical_root)
    except ValueError as error:
        raise ValueError(f"cleanup path escapes the data directory: {path}") from error
    if not relative_path.parts:
        raise ValueError("cleanup cannot remove the data directory itself")

    resolved_root = lexical_root.resolve(strict=False)
    resolved_path = lexical_path.resolve(strict=False)
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"cleanup path escapes through a symlink: {path}") from error

    current = lexical_root
    for part in relative_path.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"cleanup path contains a symlink: {current}")


def _validate_plan(artifacts: DataArtifacts, plan: _DeletionPlan) -> None:
    for path in plan.paths:
        _validate_owned_path(artifacts.data_dir, path)
    for path in plan.file_paths:
        if path.exists() and not path.is_file():
            raise ValueError(f"cleanup file path is not a regular file: {path}")
    for path in plan.tree_paths:
        if path.exists() and not path.is_dir():
            raise ValueError(f"cleanup directory path is not a directory: {path}")


def _existing_publication_units(
    artifacts: DataArtifacts, selection: DatasetSelection
) -> tuple[PublicationUnit, ...]:
    result: list[PublicationUnit] = []
    for unit in publication_units(artifacts, selection):
        existing_files = tuple(file for file in unit.files if file.local_path.exists())
        if existing_files:
            result.append(
                PublicationUnit(
                    name=unit.name,
                    files=existing_files,
                    commit_message=unit.commit_message,
                )
            )
    return tuple(result)


def _apply_plan(plan: _DeletionPlan, *, dry_run: bool) -> CleanupResult:
    existing_paths = tuple(path for path in plan.paths if path.exists())
    missing_paths = tuple(path for path in plan.paths if not path.exists())
    if dry_run:
        return CleanupResult(
            removed_paths=(),
            would_remove_paths=existing_paths,
            missing_paths=missing_paths,
        )

    file_paths = set(plan.file_paths)
    for path in existing_paths:
        if path in file_paths:
            path.unlink()
        else:
            shutil.rmtree(path)
    return CleanupResult(
        removed_paths=existing_paths,
        would_remove_paths=(),
        missing_paths=missing_paths,
    )


def cleanup_raw(
    artifacts: DataArtifacts,
    selection: DatasetSelection,
    *,
    dry_run: bool = False,
) -> CleanupResult:
    """Remove selected reproducible raw inputs and owned download state."""
    plan = _raw_deletion_plan(artifacts, selection)
    _validate_plan(artifacts, plan)
    return _apply_plan(plan, dry_run=dry_run)


def clean_data(
    artifacts: DataArtifacts,
    selection: DatasetSelection,
    *,
    dry_run: bool = False,
    hf_token: str | None = None,
    api: HfApi | None = None,
) -> CleanupResult:
    """Verify selected final outputs remotely, then remove selected artifacts."""
    plan = _data_deletion_plan(artifacts, selection)
    _validate_plan(artifacts, plan)
    for unit in _existing_publication_units(artifacts, selection):
        verify_published_unit(unit, hf_token=hf_token, api=api)
    return _apply_plan(plan, dry_run=dry_run)


__all__ = [
    "CleanupMode",
    "CleanupResult",
    "clean_data",
    "cleanup_raw",
]
