from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from datadec.data.artifacts import DataArtifacts
from datadec.data.download import (
    DownloadResult,
    download_processed_outputs,
    restore_task_summaries,
)
from datadec.data.publication import (
    PublicationColumn,
    PublicationFile,
    PublicationUnit,
)
from datadec.data.selection import resolve_selection


def _write_parquet(path: Path, *, column: str = "value", value: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({column: [value]}).to_parquet(path, index=False)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unit(*files: PublicationFile) -> PublicationUnit:
    return PublicationUnit("selected", tuple(files), "Publish selected outputs")


def _file(local_path: Path, remote_path: str) -> PublicationFile:
    return PublicationFile(
        local_path=local_path,
        remote_path=remote_path,
        expected_schema=(PublicationColumn("value", "int64", nullable=False),),
    )


class FakeApi:
    def __init__(
        self,
        remote_files: dict[str, Path],
        *,
        downloads: dict[str, Path] | None = None,
    ) -> None:
        self.remote_files = remote_files
        self.downloads = downloads or remote_files
        self.repo_info_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.paths_info_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.download_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def repo_info(self, *args: object, **kwargs: object) -> object:
        self.repo_info_calls.append((args, kwargs))
        return SimpleNamespace(sha="immutable-commit")

    def get_paths_info(self, *args: object, **kwargs: object) -> list[object]:
        self.paths_info_calls.append((args, kwargs))
        requested = args[1]
        assert isinstance(requested, list)
        return [
            SimpleNamespace(
                path=path,
                size=self.remote_files[path].stat().st_size,
                lfs={"sha256": _sha256(self.remote_files[path])},
            )
            for path in requested
            if path in self.remote_files
        ]

    def hf_hub_download(self, *args: object, **kwargs: object) -> str:
        self.download_calls.append((args, kwargs))
        remote_path = args[1]
        assert isinstance(remote_path, str)
        return str(self.downloads[remote_path])


def test_downloads_only_selected_files_at_one_immutable_commit(
    tmp_path: Path,
) -> None:
    artifacts = DataArtifacts(tmp_path / "data")
    cached_a = tmp_path / "cache-source/a.parquet"
    cached_b = tmp_path / "cache-source/b.parquet"
    _write_parquet(cached_a, value=1)
    _write_parquet(cached_b, value=2)
    file_a = _file(artifacts.data_dir / "processed/a.parquet", "a.parquet")
    file_b = _file(artifacts.data_dir / "processed/b.parquet", "nested/b.parquet")
    unit = _unit(file_a, file_b)
    api = FakeApi({"a.parquet": cached_a, "nested/b.parquet": cached_b})
    selection = resolve_selection(ppl=True)

    with patch(
        "datadec.data.download.publication_units", return_value=(unit,)
    ) as compose:
        results = download_processed_outputs(artifacts, selection, api=api)

    compose.assert_called_once()
    assert compose.call_args.args[:2] == (artifacts, selection)
    assert results == (
        DownloadResult("selected:a.parquet", file_a.local_path, "downloaded"),
        DownloadResult("selected:nested/b.parquet", file_b.local_path, "downloaded"),
    )
    assert pd.read_parquet(file_a.local_path)["value"].tolist() == [1]
    assert pd.read_parquet(file_b.local_path)["value"].tolist() == [2]
    assert not file_a.local_path.is_symlink()
    assert not file_b.local_path.is_symlink()
    assert api.repo_info_calls == [
        (
            ("drotherm/dd_parsed",),
            {"repo_type": "dataset", "revision": "main"},
        )
    ]
    assert api.paths_info_calls == [
        (
            ("drotherm/dd_parsed", ["a.parquet", "nested/b.parquet"]),
            {"repo_type": "dataset", "revision": "immutable-commit"},
        )
    ]
    assert [item[0][:2] for item in api.download_calls] == [
        ("drotherm/dd_parsed", "a.parquet"),
        ("drotherm/dd_parsed", "nested/b.parquet"),
    ]
    assert all(item[1]["revision"] == "immutable-commit" for item in api.download_calls)


def test_reuses_local_file_only_when_remote_identity_and_schema_match(
    tmp_path: Path,
) -> None:
    artifacts = DataArtifacts(tmp_path / "data")
    destination = artifacts.data_dir / "processed/a.parquet"
    _write_parquet(destination)
    publication_file = _file(destination, "a.parquet")
    api = FakeApi({"a.parquet": destination})

    with patch(
        "datadec.data.download.publication_units",
        return_value=(_unit(publication_file),),
    ):
        results = download_processed_outputs(
            artifacts, resolve_selection(ppl=True), api=api
        )

    assert results == (DownloadResult("selected:a.parquet", destination, "reused"),)
    assert api.download_calls == []


def test_force_downloads_a_matching_local_file(tmp_path: Path) -> None:
    artifacts = DataArtifacts(tmp_path / "data")
    destination = artifacts.data_dir / "processed/a.parquet"
    cached = tmp_path / "cache-source/a.parquet"
    _write_parquet(destination, value=1)
    _write_parquet(cached, value=1)
    publication_file = _file(destination, "a.parquet")
    api = FakeApi({"a.parquet": cached})

    with patch(
        "datadec.data.download.publication_units",
        return_value=(_unit(publication_file),),
    ):
        result = download_processed_outputs(
            artifacts,
            resolve_selection(ppl=True),
            api=api,
            force=True,
        )

    assert result[0].status == "downloaded"
    assert pd.read_parquet(destination)["value"].tolist() == [1]
    assert api.download_calls[0][1]["force_download"] is True


def test_download_failure_preserves_every_existing_output(tmp_path: Path) -> None:
    artifacts = DataArtifacts(tmp_path / "data")
    destination_a = artifacts.data_dir / "processed/a.parquet"
    destination_b = artifacts.data_dir / "processed/b.parquet"
    destination_a.parent.mkdir(parents=True)
    destination_a.write_bytes(b"existing-a")
    destination_b.write_bytes(b"existing-b")
    remote_a = tmp_path / "remote/a.parquet"
    remote_b = tmp_path / "remote/b.parquet"
    bad_b = tmp_path / "downloads/b.parquet"
    _write_parquet(remote_a, value=1)
    _write_parquet(remote_b, value=2)
    _write_parquet(bad_b, value=3)
    unit = _unit(
        _file(destination_a, "a.parquet"),
        _file(destination_b, "b.parquet"),
    )
    api = FakeApi(
        {"a.parquet": remote_a, "b.parquet": remote_b},
        downloads={"a.parquet": remote_a, "b.parquet": bad_b},
    )

    with (
        patch("datadec.data.download.publication_units", return_value=(unit,)),
        pytest.raises(ValueError, match="unexpected SHA-256"),
    ):
        download_processed_outputs(artifacts, resolve_selection(ppl=True), api=api)

    assert destination_a.read_bytes() == b"existing-a"
    assert destination_b.read_bytes() == b"existing-b"
    assert not destination_a.with_name(".a.parquet.tmp").exists()
    assert not destination_b.with_name(".b.parquet.tmp").exists()


def test_schema_mismatch_is_rejected_before_replacing_local_file(
    tmp_path: Path,
) -> None:
    artifacts = DataArtifacts(tmp_path / "data")
    destination = artifacts.data_dir / "processed/a.parquet"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"existing")
    cached = tmp_path / "cache-source/a.parquet"
    _write_parquet(cached, column="wrong")
    publication_file = _file(destination, "a.parquet")
    api = FakeApi({"a.parquet": cached})

    with (
        patch(
            "datadec.data.download.publication_units",
            return_value=(_unit(publication_file),),
        ),
        pytest.raises(ValueError, match="schema mismatch"),
    ):
        download_processed_outputs(artifacts, resolve_selection(ppl=True), api=api)

    assert destination.read_bytes() == b"existing"
    assert not destination.with_name(".a.parquet.tmp").exists()


def test_missing_remote_file_is_reported_before_any_download(tmp_path: Path) -> None:
    artifacts = DataArtifacts(tmp_path / "data")
    publication_file = _file(artifacts.data_dir / "processed/a.parquet", "a.parquet")
    api = FakeApi({})

    with (
        patch(
            "datadec.data.download.publication_units",
            return_value=(_unit(publication_file),),
        ),
        pytest.raises(RuntimeError, match="missing at immutable commit"),
    ):
        download_processed_outputs(artifacts, resolve_selection(ppl=True), api=api)

    assert api.download_calls == []


def test_processed_download_rejects_raw_only_published_figures(
    tmp_path: Path,
) -> None:
    with (
        patch("datadec.data.download.HfApi") as api_type,
        patch("datadec.data.download.publication_units") as compose,
        pytest.raises(ValueError, match="published figures are raw-only"),
    ):
        download_processed_outputs(
            DataArtifacts(tmp_path),
            resolve_selection(published_figures=True),
        )

    compose.assert_not_called()
    api_type.assert_not_called()


def test_task_summary_restore_is_a_noop_when_shared_table_exists(
    tmp_path: Path,
) -> None:
    artifacts = DataArtifacts(tmp_path)
    destination = artifacts.olmes_details_tasks_path()
    _write_parquet(destination)

    with patch("datadec.data.download.HfApi") as api_type:
        result = restore_task_summaries(artifacts)

    assert result == ()
    api_type.assert_not_called()


def test_task_summary_restore_downloads_and_verifies_the_shared_table(
    tmp_path: Path,
) -> None:
    artifacts = DataArtifacts(tmp_path / "data")
    remote = tmp_path / "remote/tasks.parquet"
    destination = artifacts.olmes_details_tasks_path()
    _write_parquet(remote)
    publication_file = _file(destination, "olmes-details/tasks.parquet")
    api = FakeApi({"olmes-details/tasks.parquet": remote})

    with (
        patch("datadec.data.download.HfApi", return_value=api),
        patch(
            "datadec.data.download.olmes_details_tasks_publication_unit",
            return_value=_unit(publication_file),
        ),
    ):
        result = restore_task_summaries(artifacts, hf_token="token")

    assert result == (
        DownloadResult(
            "selected:olmes-details/tasks.parquet", destination, "downloaded"
        ),
    )
    assert pd.read_parquet(destination)["value"].tolist() == [1]
    assert api.paths_info_calls == [
        (
            ("drotherm/dd_parsed", ["olmes-details/tasks.parquet"]),
            {"repo_type": "dataset", "revision": "main"},
        ),
        (
            ("drotherm/dd_parsed", ["olmes-details/tasks.parquet"]),
            {"repo_type": "dataset", "revision": "immutable-commit"},
        ),
    ]
    assert api.download_calls[0][1]["revision"] == "immutable-commit"
