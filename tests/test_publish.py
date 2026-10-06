from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from datadec.config import (
    PUBLISHED_RESULT_SCHEMAS,
    load_published_results_manifest,
    load_publishing_contract,
)
from datadec.data.artifacts import DataArtifacts
from datadec.data.publish import (
    PublicationColumn,
    PublicationFile,
    PublicationUnit,
    olmes_details_publication_unit,
    ppl_publication_unit,
    publish_existing_outputs,
    publish_unit,
    published_results_publication_units,
    scaling_law_publication_unit,
)


def _write_parquet(path: Path, values: list[int] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"value": values if values is not None else [1]}), path)


def _write_publication_file(file: PublicationFile) -> None:
    assert file.expected_schema is not None
    values: dict[str, pa.Array] = {}
    for column in file.expected_schema:
        if column.logical_type == "string":
            values[column.name] = pa.array(["value"], type=pa.string())
        elif column.logical_type == "int64":
            values[column.name] = pa.array([1], type=pa.int64())
        elif column.logical_type == "float64":
            values[column.name] = pa.array([1.0], type=pa.float64())
        else:
            values[column.name] = pa.array([True], type=pa.bool_())
    file.local_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(values), file.local_path)


def _unit(
    output: Path,
    *,
    cleanup_paths: tuple[Path, ...] = (),
) -> PublicationUnit:
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
        cleanup_paths=cleanup_paths,
    )


def _api_for(output: Path, *, created_path: str = "results/output.parquet") -> Mock:
    api = Mock()
    api.repo_info.return_value = SimpleNamespace(sha="parent-oid")
    api.get_paths_info.return_value = [
        SimpleNamespace(
            path=created_path,
            size=output.stat().st_size,
            lfs=SimpleNamespace(sha256=hashlib.sha256(output.read_bytes()).hexdigest()),
        )
    ]
    return api


def _commit_result(*, created: bool = True) -> SimpleNamespace:
    return SimpleNamespace(created=created, commit_oid="commit-oid")


@pytest.mark.parametrize(
    "case", ["missing", "directory", "empty", "invalid", "no_rows"]
)
def test_local_validation_fails_before_hugging_face_calls(
    tmp_path: Path, case: str
) -> None:
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
        patch(
            "datadec.data.publish.commit_dataset_files_to_hf"
        ) as commit_dataset_files,
        pytest.raises(ValueError, match="publication input"),
    ):
        publish_unit(_unit(output), api=api)

    api.repo_info.assert_not_called()
    api.get_paths_info.assert_not_called()
    commit_dataset_files.assert_not_called()


def test_local_schema_and_order_are_validated_before_network(tmp_path: Path) -> None:
    output = tmp_path / "output.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"other": [1], "value": [2]}), output)
    api = Mock()

    with pytest.raises(ValueError, match="schema mismatch"):
        publish_unit(_unit(output), api=api)

    api.repo_info.assert_not_called()


def test_contract_required_columns_reject_nulls_before_network(tmp_path: Path) -> None:
    output = tmp_path / "output.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"value": pa.array([None], type=pa.int64())}), output)
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
    api = Mock()

    with pytest.raises(ValueError, match="nulls in required column 'value'"):
        publish_unit(unit, api=api)

    api.repo_info.assert_not_called()


def test_commit_failure_preserves_outputs_and_sources_without_retry(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output.parquet"
    raw = tmp_path / "source.csv"
    _write_parquet(output)
    raw.write_text("source")
    api = _api_for(output)

    with (
        patch(
            "datadec.data.publish.commit_dataset_files_to_hf",
            side_effect=RuntimeError("stale parent"),
        ) as commit_dataset_files,
        pytest.raises(RuntimeError, match="stale parent"),
    ):
        publish_unit(_unit(output, cleanup_paths=(raw,)), api=api)

    assert commit_dataset_files.call_count == 1
    assert output.is_file()
    assert raw.is_file()
    api.get_paths_info.assert_not_called()


@pytest.mark.parametrize("remote_case", ["missing", "wrong_path", "wrong_size"])
def test_remote_verification_failure_preserves_outputs_and_sources(
    tmp_path: Path, remote_case: str
) -> None:
    output = tmp_path / "output.parquet"
    raw = tmp_path / "source.csv"
    _write_parquet(output)
    raw.write_text("source")
    api = _api_for(output)
    if remote_case == "missing":
        api.get_paths_info.return_value = []
    elif remote_case == "wrong_path":
        api.get_paths_info.return_value = [
            SimpleNamespace(path="results/other.parquet", size=output.stat().st_size)
        ]
    else:
        api.get_paths_info.return_value = [
            SimpleNamespace(
                path="results/output.parquet", size=output.stat().st_size + 1, lfs=None
            )
        ]

    with (
        patch(
            "datadec.data.publish.commit_dataset_files_to_hf",
            return_value=_commit_result(),
        ),
        pytest.raises(RuntimeError, match="missing|size mismatch"),
    ):
        publish_unit(_unit(output, cleanup_paths=(raw,)), api=api)

    assert output.is_file()
    assert raw.is_file()
    api.get_paths_info.assert_called_once_with(
        "drotherm/dd_parsed",
        ["results/output.parquet"],
        repo_type="dataset",
        revision="commit-oid",
    )


def test_partial_multi_file_remote_verification_fails_the_whole_unit(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    raw = tmp_path / "source.csv"
    _write_parquet(first)
    _write_parquet(second)
    raw.write_text("source")
    unit = PublicationUnit(
        name="two-file-unit",
        files=(
            PublicationFile(first, "first.parquet"),
            PublicationFile(second, "second.parquet"),
        ),
        commit_message="Publish both",
        cleanup_paths=(raw,),
    )
    api = Mock()
    api.repo_info.return_value = SimpleNamespace(sha="parent-oid")
    api.get_paths_info.return_value = [
        SimpleNamespace(path="first.parquet", size=first.stat().st_size, lfs=None)
    ]

    with (
        patch(
            "datadec.data.publish.commit_dataset_files_to_hf",
            return_value=_commit_result(),
        ),
        pytest.raises(RuntimeError, match="second.parquet"),
    ):
        publish_unit(unit, api=api)

    assert first.is_file()
    assert second.is_file()
    assert raw.is_file()


def test_lfs_hash_mismatch_preserves_source(tmp_path: Path) -> None:
    output = tmp_path / "output.parquet"
    raw = tmp_path / "source.csv"
    _write_parquet(output)
    raw.write_text("source")
    api = _api_for(output)
    api.get_paths_info.return_value[0].lfs = SimpleNamespace(sha256="0" * 64)

    with (
        patch(
            "datadec.data.publish.commit_dataset_files_to_hf",
            return_value=_commit_result(),
        ),
        pytest.raises(RuntimeError, match="LFS SHA-256 mismatch"),
    ):
        publish_unit(_unit(output, cleanup_paths=(raw,)), api=api)

    assert raw.is_file()


def test_missing_lfs_hash_preserves_source(tmp_path: Path) -> None:
    output = tmp_path / "output.parquet"
    raw = tmp_path / "source.csv"
    _write_parquet(output)
    raw.write_text("source")
    api = _api_for(output)
    api.get_paths_info.return_value[0].lfs = None

    with (
        patch(
            "datadec.data.publish.commit_dataset_files_to_hf",
            return_value=_commit_result(),
        ),
        pytest.raises(RuntimeError, match="LFS SHA-256 missing"),
    ):
        publish_unit(_unit(output, cleanup_paths=(raw,)), api=api)

    assert output.is_file()
    assert raw.is_file()


@pytest.mark.parametrize("keep_sources", [False, True])
def test_cleanup_occurs_only_after_verified_commit(
    tmp_path: Path, keep_sources: bool
) -> None:
    output = tmp_path / "output.parquet"
    first_raw = tmp_path / "first.csv"
    second_raw = tmp_path / "second.csv"
    _write_parquet(output)
    first_raw.write_text("first")
    second_raw.write_text("second")
    api = _api_for(output)
    api.get_paths_info.return_value[0].lfs = SimpleNamespace(
        sha256=hashlib.sha256(output.read_bytes()).hexdigest()
    )

    with patch(
        "datadec.data.publish.commit_dataset_files_to_hf",
        return_value=_commit_result(),
    ):
        result = publish_unit(
            _unit(output, cleanup_paths=(first_raw, second_raw)),
            api=api,
            keep_sources=keep_sources,
        )

    assert output.is_file()
    assert first_raw.is_file() is keep_sources
    assert second_raw.is_file() is keep_sources
    assert result.deleted_sources == (() if keep_sources else (first_raw, second_raw))


def test_verified_no_op_is_successful_and_allows_cleanup(tmp_path: Path) -> None:
    output = tmp_path / "output.parquet"
    raw = tmp_path / "source.csv"
    _write_parquet(output)
    raw.write_text("source")
    api = _api_for(output)

    with patch(
        "datadec.data.publish.commit_dataset_files_to_hf",
        return_value=_commit_result(created=False),
    ):
        result = publish_unit(_unit(output, cleanup_paths=(raw,)), api=api)

    assert result.created is False
    assert result.commit_oid == "commit-oid"
    assert not raw.exists()


def test_commit_uses_immediately_resolved_parent_and_direct_dataset_commit(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output.parquet"
    _write_parquet(output)
    api = _api_for(output)

    with patch(
        "datadec.data.publish.commit_dataset_files_to_hf",
        return_value=_commit_result(),
    ) as commit_dataset_files:
        publish_unit(_unit(output), api=api, hf_token="token")

    api.repo_info.assert_called_once_with(
        "drotherm/dd_parsed", repo_type="dataset", revision="main"
    )
    _, hf_location = commit_dataset_files.call_args.args
    assert hf_location.repo_id == "drotherm/dd_parsed"
    assert commit_dataset_files.call_args.kwargs == {
        "revision": "main",
        "expected_parent": "parent-oid",
        "commit_message": "Publish test output",
        "create_pr": False,
        "hf_token": "token",
    }


def test_scaling_factory_is_atomic_and_cleans_only_three_configured_sources(
    tmp_path: Path,
) -> None:
    unit = scaling_law_publication_unit(DataArtifacts(tmp_path))

    assert tuple(file.remote_path for file in unit.files) == (
        "scaling-law/evaluations.parquet",
        "scaling-law/checkpoint-losses.parquet",
    )
    assert len(unit.cleanup_paths) == 3
    assert unit.cleanup_paths == DataArtifacts(tmp_path).scaling_law_raw_paths()
    assert all(
        column.nullable is not None
        for file in unit.files
        for column in file.expected_schema or ()
    )


def test_published_results_factory_maps_all_manifest_units_exactly(
    tmp_path: Path,
) -> None:
    paths = DataArtifacts(tmp_path)
    manifest = load_published_results_manifest()
    units = published_results_publication_units(paths, manifest=manifest)
    expected_unit_names = (
        "cheap-decisions",
        "new-eval-intermediates",
        "outputs2",
        "per-task-arc-challenge",
        "per-task-arc-easy",
        "per-task-boolq",
        "per-task-csqa",
        "per-task-hellaswag",
        "per-task-mmlu",
        "per-task-openbookqa",
        "per-task-piqa",
        "per-task-socialiqa",
        "per-task-winogrande",
        "processed-data-current",
        "processed-data-pre-extra-real",
    )

    assert tuple(unit.name for unit in units) == tuple(
        f"published-results:{name}" for name in expected_unit_names
    )
    assert len(units) == 15
    for unit_name, unit in zip(expected_unit_names, units, strict=True):
        sources = tuple(
            source
            for source in manifest.files
            if source.category == "published_results"
            and source.publication_unit == unit_name
        )
        assert tuple(file.local_path for file in unit.files) == tuple(
            paths.published_result_output_path(source) for source in sources
        )
        assert tuple(file.remote_path for file in unit.files) == tuple(
            f"published-results/{source.parquet_relative_path().as_posix()}"
            for source in sources
        )
        assert unit.cleanup_paths == tuple(
            paths.published_result_source_path(source) for source in sources
        )
        assert unit.commit_message == f"Publish published results for {unit_name}"
        for file, source in zip(unit.files, sources, strict=True):
            assert source.schema is not None
            assert file.expected_schema == tuple(
                PublicationColumn(column.name, column.logical_type, column.nullable)
                for column in PUBLISHED_RESULT_SCHEMAS[source.schema].columns
            )
            assert file.remote_path.endswith(".parquet")


def test_published_results_factory_selects_units_in_manifest_order(
    tmp_path: Path,
) -> None:
    units = published_results_publication_units(
        DataArtifacts(tmp_path),
        units=("per-task-winogrande", "outputs2"),
    )

    assert tuple(unit.name for unit in units) == (
        "published-results:outputs2",
        "published-results:per-task-winogrande",
    )


def test_detail_factory_cleanup_is_isolated_to_one_recipe(tmp_path: Path) -> None:
    paths = DataArtifacts(tmp_path)
    c4 = olmes_details_publication_unit(paths, "c4")
    fineweb = olmes_details_publication_unit(paths, "fineweb-pro")

    assert c4.cleanup_paths == (tmp_path / "raw/olmes-details/models/c4.tar.gz",)
    assert fineweb.cleanup_paths == (
        tmp_path / "raw/olmes-details/models/fineweb-pro.tar.gz",
    )
    assert set(c4.cleanup_paths).isdisjoint(fineweb.cleanup_paths)
    assert tuple(file.remote_path for file in c4.files) == (
        "olmes-details/c4/tasks.parquet",
        "olmes-details/c4/instances.parquet",
        "olmes-details/c4/choices.parquet",
    )


def test_detail_factory_can_disable_canonical_source_cleanup(tmp_path: Path) -> None:
    unit = olmes_details_publication_unit(
        DataArtifacts(tmp_path),
        "c4",
        cleanup_source=False,
    )

    assert unit.cleanup_paths == ()


def test_factories_publish_exact_output_overrides(tmp_path: Path) -> None:
    paths = DataArtifacts(tmp_path)
    ppl_output = tmp_path / "custom/ppl.parquet"
    scaling_evaluations = tmp_path / "custom/evaluations.parquet"
    scaling_losses = tmp_path / "custom/losses.parquet"
    detail_outputs = (
        tmp_path / "custom/tasks.parquet",
        tmp_path / "custom/instances.parquet",
        tmp_path / "custom/choices.parquet",
    )

    assert ppl_publication_unit(paths, output_path=ppl_output).files[0].local_path == (
        ppl_output
    )
    scaling = scaling_law_publication_unit(
        paths,
        evaluations_output_path=scaling_evaluations,
        checkpoint_losses_output_path=scaling_losses,
    )
    assert tuple(file.local_path for file in scaling.files) == (
        scaling_evaluations,
        scaling_losses,
    )
    details = olmes_details_publication_unit(
        paths,
        "c4",
        output_tasks_path=detail_outputs[0],
        output_instances_path=detail_outputs[1],
        output_choices_path=detail_outputs[2],
    )
    assert tuple(file.local_path for file in details.files) == detail_outputs


def test_existing_final_output_can_be_republished_without_raw_or_preprocessing(
    tmp_path: Path,
) -> None:
    paths = DataArtifacts(tmp_path)
    publication_file = ppl_publication_unit(paths).files[0]
    output = publication_file.local_path
    _write_publication_file(publication_file)
    api = _api_for(output, created_path="ppl.parquet")

    with (
        patch("datadec.data.publish.HfApi", return_value=api),
        patch(
            "datadec.data.publish.commit_dataset_files_to_hf",
            return_value=_commit_result(created=False),
        ),
    ):
        results = publish_existing_outputs(paths, ppl=True)

    assert len(results) == 1
    assert results[0].unit_name == "ppl"
    assert results[0].created is False
    assert output.is_file()
    assert not (tmp_path / "raw").exists()


def test_existing_published_results_can_be_retried_without_sources_or_preprocessing(
    tmp_path: Path,
) -> None:
    paths = DataArtifacts(tmp_path)
    results = [SimpleNamespace(unit_name=f"unit-{index}") for index in range(15)]
    with patch(
        "datadec.data.publish.publish_unit", side_effect=results
    ) as publish_selected_unit:
        actual = publish_existing_outputs(paths, published_results=True)

    assert actual == results
    units = tuple(call.args[0] for call in publish_selected_unit.call_args_list)
    assert len(units) == 15
    assert all(unit.name.startswith("published-results:") for unit in units)
    assert not (tmp_path / "reference/published-results").exists()


def test_published_results_stop_after_failed_unit_without_retry(
    tmp_path: Path,
) -> None:
    units = tuple(
        PublicationUnit(
            name=f"unit-{index}",
            files=(PublicationFile(tmp_path / f"{index}.parquet", f"{index}.parquet"),),
            commit_message=f"Publish unit {index}",
            cleanup_paths=(tmp_path / f"source-{index}.csv",),
        )
        for index in range(3)
    )
    for unit in units:
        unit.cleanup_paths[0].write_text("source")

    def publish_or_fail(unit: PublicationUnit, **_: object) -> object:
        if unit is units[1]:
            raise RuntimeError("second unit failed")
        unit.cleanup_paths[0].unlink()
        return SimpleNamespace(unit_name=unit.name)

    with (
        patch(
            "datadec.data.publish.published_results_publication_units",
            return_value=units,
        ),
        patch(
            "datadec.data.publish.publish_unit", side_effect=publish_or_fail
        ) as publish_selected_unit,
        pytest.raises(RuntimeError, match="second unit failed"),
    ):
        publish_existing_outputs(DataArtifacts(tmp_path), published_results=True)

    assert publish_selected_unit.call_count == 2
    assert not units[0].cleanup_paths[0].exists()
    assert units[1].cleanup_paths[0].is_file()
    assert units[2].cleanup_paths[0].is_file()


def test_existing_output_publication_rejects_no_selection(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="select at least one output"):
        publish_existing_outputs(DataArtifacts(tmp_path))


def test_existing_detail_outputs_publish_as_recipe_isolated_units(
    tmp_path: Path,
) -> None:
    results = [SimpleNamespace(unit_name="c4"), SimpleNamespace(unit_name="fineweb")]
    with patch(
        "datadec.data.publish.publish_unit", side_effect=results
    ) as publish_selected_unit:
        actual = publish_existing_outputs(
            DataArtifacts(tmp_path),
            olmes_details=["c4", "fineweb-pro"],
        )

    assert actual == results
    units = tuple(call.args[0] for call in publish_selected_unit.call_args_list)
    assert tuple(unit.name for unit in units) == (
        "olmes-details:c4",
        "olmes-details:fineweb-pro",
    )
    assert units[0].cleanup_paths == (tmp_path / "raw/olmes-details/models/c4.tar.gz",)
    assert units[1].cleanup_paths == (
        tmp_path / "raw/olmes-details/models/fineweb-pro.tar.gz",
    )


def test_ppl_and_olmes_units_never_clean_sources(tmp_path: Path) -> None:
    from datadec.data.publish import olmes_publication_unit

    paths = DataArtifacts(tmp_path)
    contract = load_publishing_contract()
    assert ppl_publication_unit(paths, contract=contract).cleanup_paths == ()
    assert olmes_publication_unit(paths, contract=contract).cleanup_paths == ()


def test_ppl_factory_owns_exact_ordered_types_without_invented_nullability(
    tmp_path: Path,
) -> None:
    schema = ppl_publication_unit(DataArtifacts(tmp_path)).files[0].expected_schema

    assert schema is not None
    assert tuple(column.name for column in schema[:4]) == (
        "params",
        "data",
        "seed",
        "step",
    )
    assert tuple(column.logical_type for column in schema[:4]) == (
        "string",
        "string",
        "string",
        "int64",
    )
    assert all(column.nullable is None for column in schema)
