from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from datadec.data.artifacts import DataArtifacts
from datadec.data.publication import (
    PublicationColumn,
    PublicationFile,
    PublicationUnit,
)
from datadec.data.publish import (
    publish_existing_outputs,
    publish_unit,
    validate_publication_unit,
    verify_published_unit,
)
from datadec.data.selection import resolve_selection


def _write_parquet(path: Path, values: list[int] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"value": values if values is not None else [1]}), path)


def _unit(output: Path) -> PublicationUnit:
    return PublicationUnit(
        name="test-unit",
        files=(
            PublicationFile(
                output,
                "results/output.parquet",
                (PublicationColumn("value", "int64"),),
            ),
        ),
        commit_message="Publish test output",
    )


def _api_for(
    output: Path,
    *,
    revision_oid: str = "parent-oid",
    created_path: str = "results/output.parquet",
) -> Mock:
    api = Mock()
    api.repo_info.return_value = SimpleNamespace(sha=revision_oid)
    api.get_paths_info.return_value = [
        SimpleNamespace(
            path=created_path,
            size=output.stat().st_size,
            lfs=SimpleNamespace(sha256=hashlib.sha256(output.read_bytes()).hexdigest()),
        )
    ]
    return api


@pytest.mark.parametrize(
    "case", ["missing", "directory", "empty", "invalid", "no_rows"]
)
def test_validation_fails_before_network(tmp_path: Path, case: str) -> None:
    output = tmp_path / "output.parquet"
    if case == "directory":
        output.mkdir()
    elif case == "empty":
        output.touch()
    elif case == "invalid":
        output.write_bytes(b"not parquet")
    elif case == "no_rows":
        _write_parquet(output, [])

    api = Mock()
    with (
        patch("datadec.data.publish.commit_dataset_files_to_hf") as commit,
        pytest.raises(ValueError, match="publication input"),
    ):
        publish_unit(_unit(output), api=api)

    api.repo_info.assert_not_called()
    commit.assert_not_called()


def test_public_validation_checks_schema_order_and_required_values(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output.parquet"
    pq.write_table(
        pa.table({"value": pa.array([None], type=pa.int64())}),
        output,
    )
    unit = PublicationUnit(
        name="required-column",
        files=(
            PublicationFile(
                output,
                "output.parquet",
                (PublicationColumn("value", "int64", nullable=False),),
            ),
        ),
        commit_message="Publish required column",
    )

    with pytest.raises(ValueError, match="nulls in required column 'value'"):
        validate_publication_unit(unit)


def test_public_validation_rejects_schema_mismatch_before_network(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output.parquet"
    pq.write_table(pa.table({"other": [1], "value": [2]}), output)

    with pytest.raises(ValueError, match="schema mismatch"):
        validate_publication_unit(_unit(output))


def test_public_validation_rejects_empty_and_duplicate_remote_paths(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output.parquet"
    _write_parquet(output)
    with pytest.raises(ValueError, match="has no files"):
        validate_publication_unit(
            PublicationUnit(name="empty", files=(), commit_message="empty")
        )
    with pytest.raises(ValueError, match="duplicate remote paths"):
        validate_publication_unit(
            PublicationUnit(
                name="duplicate",
                files=(
                    PublicationFile(output, "same.parquet"),
                    PublicationFile(output, "same.parquet"),
                ),
                commit_message="duplicate",
            )
        )


def test_publish_uses_immediately_resolved_parent_and_verifies_commit(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output.parquet"
    raw = tmp_path / "source.csv"
    _write_parquet(output)
    raw.write_text("source")
    api = _api_for(output)
    commit_result = SimpleNamespace(created=True, commit_oid="commit-oid")

    with patch(
        "datadec.data.publish.commit_dataset_files_to_hf",
        return_value=commit_result,
    ) as commit:
        result = publish_unit(_unit(output), api=api, hf_token="token")

    api.repo_info.assert_called_once_with(
        "drotherm/dd_parsed", repo_type="dataset", revision="main"
    )
    api.get_paths_info.assert_called_once_with(
        "drotherm/dd_parsed",
        ["results/output.parquet"],
        repo_type="dataset",
        revision="commit-oid",
    )
    assert commit.call_args.kwargs == {
        "revision": "main",
        "expected_parent": "parent-oid",
        "commit_message": "Publish test output",
        "create_pr": False,
        "hf_token": "token",
    }
    assert result.commit_oid == "commit-oid"
    assert result.created is True
    assert raw.is_file()
    assert output.is_file()
    assert not hasattr(result, "deleted_sources")


@pytest.mark.parametrize("remote_case", ["missing", "wrong_size", "wrong_hash"])
def test_remote_verification_failure_retains_all_local_files(
    tmp_path: Path, remote_case: str
) -> None:
    output = tmp_path / "output.parquet"
    raw = tmp_path / "source.csv"
    _write_parquet(output)
    raw.write_text("source")
    api = _api_for(output)
    if remote_case == "missing":
        api.get_paths_info.return_value = []
    elif remote_case == "wrong_size":
        api.get_paths_info.return_value[0].size += 1
    else:
        api.get_paths_info.return_value[0].lfs.sha256 = "0" * 64

    with (
        patch(
            "datadec.data.publish.commit_dataset_files_to_hf",
            return_value=SimpleNamespace(created=True, commit_oid="commit-oid"),
        ),
        pytest.raises(RuntimeError, match="missing|size mismatch|SHA-256 mismatch"),
    ):
        publish_unit(_unit(output), api=api)

    assert output.is_file()
    assert raw.is_file()


def test_remote_verification_requires_lfs_sha256(tmp_path: Path) -> None:
    output = tmp_path / "output.parquet"
    _write_parquet(output)
    api = _api_for(output)
    api.get_paths_info.return_value[0].lfs = None

    with (
        patch(
            "datadec.data.publish.commit_dataset_files_to_hf",
            return_value=SimpleNamespace(created=True, commit_oid="commit-oid"),
        ),
        pytest.raises(RuntimeError, match="LFS SHA-256 missing"),
    ):
        publish_unit(_unit(output), api=api)

    assert output.is_file()


def test_commit_failure_retains_all_local_files_without_retry(tmp_path: Path) -> None:
    output = tmp_path / "output.parquet"
    raw = tmp_path / "source.csv"
    _write_parquet(output)
    raw.write_text("source")
    api = _api_for(output)

    with (
        patch(
            "datadec.data.publish.commit_dataset_files_to_hf",
            side_effect=RuntimeError("stale parent"),
        ) as commit,
        pytest.raises(RuntimeError, match="stale parent"),
    ):
        publish_unit(_unit(output), api=api)

    assert commit.call_count == 1
    assert output.is_file()
    assert raw.is_file()
    api.get_paths_info.assert_not_called()


def test_standalone_verification_resolves_and_uses_one_immutable_commit(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output.parquet"
    raw = tmp_path / "source.csv"
    _write_parquet(output)
    raw.write_text("source")
    api = _api_for(output, revision_oid="immutable-oid")

    with patch("datadec.data.publish.commit_dataset_files_to_hf") as commit:
        result = verify_published_unit(_unit(output), api=api)

    commit.assert_not_called()
    api.repo_info.assert_called_once_with(
        "drotherm/dd_parsed", repo_type="dataset", revision="main"
    )
    api.get_paths_info.assert_called_once_with(
        "drotherm/dd_parsed",
        ["results/output.parquet"],
        repo_type="dataset",
        revision="immutable-oid",
    )
    assert result.commit_oid == "immutable-oid"
    assert result.created is False
    assert raw.is_file()
    assert output.is_file()


def test_standalone_verification_rejects_unresolved_revision(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output.parquet"
    _write_parquet(output)
    api = Mock()
    api.repo_info.return_value = SimpleNamespace(sha=None)

    with pytest.raises(RuntimeError, match="could not resolve"):
        verify_published_unit(_unit(output), api=api)

    api.get_paths_info.assert_not_called()


def test_multi_file_verification_requires_every_file(tmp_path: Path) -> None:
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    _write_parquet(first)
    _write_parquet(second)
    unit = PublicationUnit(
        name="two-file-unit",
        files=(
            PublicationFile(first, "first.parquet"),
            PublicationFile(second, "second.parquet"),
        ),
        commit_message="Publish both",
    )
    api = Mock()
    api.repo_info.return_value = SimpleNamespace(sha="immutable-oid")
    api.get_paths_info.return_value = [
        SimpleNamespace(
            path="first.parquet",
            size=first.stat().st_size,
            lfs={"sha256": hashlib.sha256(first.read_bytes()).hexdigest()},
        )
    ]

    with pytest.raises(RuntimeError, match="second.parquet"):
        verify_published_unit(unit, api=api)


def test_verified_no_op_keeps_immutable_commit_evidence(tmp_path: Path) -> None:
    output = tmp_path / "output.parquet"
    _write_parquet(output)
    api = _api_for(output)

    with patch(
        "datadec.data.publish.commit_dataset_files_to_hf",
        return_value=SimpleNamespace(created=False, commit_oid="existing-oid"),
    ):
        result = publish_unit(_unit(output), api=api)

    assert result.created is False
    assert result.commit_oid == "existing-oid"
    api.get_paths_info.assert_called_once_with(
        "drotherm/dd_parsed",
        ["results/output.parquet"],
        repo_type="dataset",
        revision="existing-oid",
    )


def test_existing_output_publication_uses_selection_and_stops_on_failure(
    tmp_path: Path,
) -> None:
    artifacts = DataArtifacts(tmp_path)
    selection = resolve_selection(ppl=True, olmes=True)
    units = (_unit(tmp_path / "first.parquet"), _unit(tmp_path / "second.parquet"))
    with (
        patch("datadec.data.publish.load_publishing_contract") as load_contract,
        patch("datadec.data.publish.publication_units", return_value=units) as compose,
        patch(
            "datadec.data.publish.publish_unit",
            side_effect=(
                SimpleNamespace(unit_name="first"),
                RuntimeError("second failed"),
            ),
        ) as publish,
        pytest.raises(RuntimeError, match="second failed"),
    ):
        publish_existing_outputs(artifacts, selection)

    compose.assert_called_once_with(
        artifacts,
        selection,
        contract=load_contract.return_value,
    )
    assert publish.call_count == 2
