from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from datadec.config import load_published_results_manifest
from datadec.data.artifacts import DataArtifacts
from datadec.data.cleanup import CleanupMode, clean_data, cleanup_raw
from datadec.data.publication import PublicationUnit
from datadec.data.selection import resolve_selection


def _write(path: Path, content: str = "data") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_cleanup_modes_have_stable_values() -> None:
    assert tuple(CleanupMode) == (
        CleanupMode.DEFAULT,
        CleanupMode.RAW,
        CleanupMode.ALL,
        CleanupMode.NONE,
    )
    assert tuple(mode.value for mode in CleanupMode) == (
        "default",
        "raw",
        "all",
        "none",
    )


def test_default_cleanup_keeps_base_outputs_and_verifies_all_deleted_finals_first(
    tmp_path: Path,
) -> None:
    artifacts = DataArtifacts(tmp_path)
    selection = resolve_selection(all_data=True, published_figures=True)
    base = {
        artifacts.get_path("ppl_processed"),
        artifacts.get_path("olmes_processed"),
        artifacts.scaling_law_evaluations_path(),
        artifacts.scaling_law_checkpoint_losses_path(),
        *(
            artifacts.published_result_output_path(source)
            for source in load_published_results_manifest().files
            if source.category == "published_results"
        ),
    }
    finals = set(artifacts.processed_paths(selection))
    raw = artifacts.get_path("ppl_raw")
    temp = tmp_path / "processed/.ppl.parquet.tmp"
    staging = artifacts.olmes_details_staging_path("c4")
    spill = artifacts.processing_intermediate_tree_paths(
        resolve_selection(olmes_details=("c4",))
    )[0]
    for path in (*finals, raw, temp, staging, spill / "spill.bin"):
        _write(path)
    unrelated = tmp_path / "custom/keep.txt"
    _write(unrelated)
    verified: list[Path] = []

    def verify(unit: PublicationUnit, **_kwargs: object) -> Mock:
        assert all(path.exists() for path in (*finals, raw, temp, staging))
        verified.extend(file.local_path for file in unit.files)
        return Mock()

    with patch("datadec.data.cleanup.verify_published_unit", side_effect=verify):
        preview = clean_data(
            artifacts, selection, mode=CleanupMode.DEFAULT, dry_run=True
        )
        assert set(verified) == finals - base
        assert set(preview.would_remove_paths) == (finals - base) | {
            raw,
            temp,
            staging,
            spill,
        }
        result = clean_data(artifacts, selection, mode=CleanupMode.DEFAULT)

    assert len(base) == 55
    assert len(finals - base) == 75
    assert set(result.removed_paths) == set(preview.would_remove_paths)
    assert set(tmp_path.rglob("*.parquet")) == base
    assert all(path.read_text() == "data" for path in base)
    assert unrelated.read_text() == "data"
    assert not spill.exists()


def test_default_cleanup_remote_failure_retains_every_selected_artifact(
    tmp_path: Path,
) -> None:
    artifacts = DataArtifacts(tmp_path)
    selection = resolve_selection(ppl=True, olmes_details=("c4", "dolma1.7"))
    paths = (*artifacts.processed_paths(selection), artifacts.get_path("ppl_raw"))
    for path in paths:
        _write(path)
    with (
        patch(
            "datadec.data.cleanup.verify_published_unit",
            side_effect=(Mock(), RuntimeError("remote mismatch")),
        ) as verify,
        pytest.raises(RuntimeError, match="remote mismatch"),
    ):
        clean_data(artifacts, selection, mode=CleanupMode.DEFAULT)

    assert verify.call_count == 2
    assert all(path.exists() for path in paths)


def test_default_cleanup_does_not_require_retained_aggregate_remote_copies(
    tmp_path: Path,
) -> None:
    artifacts = DataArtifacts(tmp_path)
    selection = resolve_selection(ppl=True)
    raw = artifacts.get_path("ppl_raw")
    final = artifacts.get_path("ppl_processed")
    _write(raw)
    _write(final, "unpublished aggregate")
    with patch("datadec.data.cleanup.verify_published_unit") as verify:
        result = clean_data(artifacts, selection, mode=CleanupMode.DEFAULT)
    verify.assert_not_called()
    assert result.removed_paths == (raw,)
    assert final.read_text() == "unpublished aggregate"


def test_default_cleanup_preserves_unselected_recipe(tmp_path: Path) -> None:
    artifacts = DataArtifacts(tmp_path)
    selected = resolve_selection(olmes_details=("c4",))
    other = resolve_selection(olmes_details=("dolma1.7",))
    for path in (
        *artifacts.processed_paths(selected),
        *artifacts.processed_paths(other),
    ):
        _write(path)
    with patch("datadec.data.cleanup.verify_published_unit"):
        clean_data(artifacts, selected, mode=CleanupMode.DEFAULT)
    assert all(not path.exists() for path in artifacts.processed_paths(selected))
    assert all(path.exists() for path in artifacts.processed_paths(other))


@pytest.mark.parametrize("mode", [CleanupMode.RAW, CleanupMode.NONE])
def test_clean_data_handles_raw_and_none_modes_without_remote_checks(
    tmp_path: Path, mode: CleanupMode
) -> None:
    artifacts = DataArtifacts(tmp_path)
    selection = resolve_selection(ppl=True)
    raw = artifacts.get_path("ppl_raw")
    final = artifacts.get_path("ppl_processed")
    _write(raw)
    _write(final)
    with patch("datadec.data.cleanup.verify_published_unit") as verify:
        result = clean_data(artifacts, selection, mode=mode)
    verify.assert_not_called()
    assert final.exists()
    assert raw.exists() is (mode is CleanupMode.NONE)
    assert result.removed_paths == ((raw,) if mode is CleanupMode.RAW else ())


def test_unknown_cleanup_mode_rejected_before_any_deletion(tmp_path: Path) -> None:
    artifacts = DataArtifacts(tmp_path)
    selection = resolve_selection(ppl=True)
    raw = artifacts.get_path("ppl_raw")
    _write(raw)
    with pytest.raises(ValueError, match="not a valid CleanupMode"):
        clean_data(artifacts, selection, mode="unknown")  # type: ignore[arg-type]
    assert raw.exists()


def test_raw_cleanup_removes_only_raw_download_state_for_selected_unit(
    tmp_path: Path,
) -> None:
    artifacts = DataArtifacts(tmp_path)
    selection = resolve_selection(units=("new-eval-intermediates",))
    source = next(
        source
        for source in load_published_results_manifest().files
        if source.publication_unit == "new-eval-intermediates"
    )
    raw = artifacts.published_result_source_path(source)
    partial = raw.with_name(f"{raw.name}.part")
    output = artifacts.published_result_output_path(source)
    processing_temp = output.with_name(f".{output.name}.tmp")
    for path in (raw, partial, output, processing_temp):
        _write(path)

    result = cleanup_raw(artifacts, selection)

    assert raw in result.removed_paths
    assert partial in result.removed_paths
    assert not raw.exists()
    assert not partial.exists()
    assert output.exists()
    assert processing_temp.exists()


def test_raw_cleanup_removes_selected_detail_metadata_but_keeps_staging(
    tmp_path: Path,
) -> None:
    artifacts = DataArtifacts(tmp_path)
    selection = resolve_selection(olmes_details=("c4",))
    raw = artifacts.raw_paths(selection)[0]
    metadata = artifacts.cache_paths(selection)
    staging = artifacts.olmes_details_staging_path("c4")
    for path in (raw, *metadata, staging):
        _write(path)
    shared_cache = tmp_path / "cache/huggingface/shared-entry"
    _write(shared_cache)

    cleanup_raw(artifacts, selection)

    assert not raw.exists()
    assert all(not path.exists() for path in metadata)
    assert staging.exists()
    assert shared_cache.exists()


def test_raw_cleanup_all_removes_only_declared_cache_trees(tmp_path: Path) -> None:
    artifacts = DataArtifacts(tmp_path)
    selection = resolve_selection(all_data=True)
    cache_file = tmp_path / "cache/huggingface/downloads/entry"
    detail_cache_file = tmp_path / "raw/olmes-details/.cache/entry"
    legacy_raw_download = tmp_path / "raw_downloads/unknown-source"
    unrelated = tmp_path / "custom/keep.txt"
    for path in (cache_file, detail_cache_file, legacy_raw_download, unrelated):
        _write(path)

    result = cleanup_raw(artifacts, selection)

    assert tmp_path / "cache/huggingface" in result.removed_paths
    assert tmp_path / "raw/olmes-details/.cache" in result.removed_paths
    assert not cache_file.exists()
    assert not detail_cache_file.exists()
    assert legacy_raw_download.exists()
    assert unrelated.exists()


def test_clean_removes_only_selected_duckdb_spill_directory(tmp_path: Path) -> None:
    artifacts = DataArtifacts(tmp_path)
    selection = resolve_selection(olmes_details=("c4",))
    spill = artifacts.processing_intermediate_tree_paths(selection)[0]
    other_spill = Path(f"{artifacts.olmes_details_staging_path('dolma1.7')}.tmp")
    for directory in (spill, other_spill):
        _write(directory / "duckdb_temp_storage.bin")

    cleanup_raw(artifacts, selection)
    assert spill.exists()
    preview = clean_data(artifacts, selection, mode=CleanupMode.ALL, dry_run=True)
    assert preview.would_remove_paths == (spill,)
    result = clean_data(artifacts, selection, mode=CleanupMode.ALL)
    assert result.removed_paths == (spill,)
    assert not spill.exists()
    assert other_spill.exists()


def test_raw_cleanup_is_idempotent_and_reports_missing_paths(tmp_path: Path) -> None:
    artifacts = DataArtifacts(tmp_path)
    selection = resolve_selection(ppl=True)
    raw = artifacts.raw_paths(selection)[0]
    _write(raw)

    first = cleanup_raw(artifacts, selection)
    second = cleanup_raw(artifacts, selection)

    assert first.removed_paths == (raw,)
    assert first.would_remove_paths == ()
    assert first.missing_paths == ()
    assert second.removed_paths == ()
    assert second.would_remove_paths == ()
    assert second.missing_paths == (raw,)


def test_raw_cleanup_dry_run_reports_without_removing(tmp_path: Path) -> None:
    artifacts = DataArtifacts(tmp_path)
    selection = resolve_selection(ppl=True)
    raw = artifacts.raw_paths(selection)[0]
    _write(raw)

    result = cleanup_raw(artifacts, selection, dry_run=True)

    assert result.removed_paths == ()
    assert result.would_remove_paths == (raw,)
    assert result.missing_paths == ()
    assert raw.exists()


def test_clean_dry_run_uses_exact_plan_without_removing_paths(tmp_path: Path) -> None:
    artifacts = DataArtifacts(tmp_path)
    selection = resolve_selection(ppl=True)
    raw = artifacts.raw_paths(selection)[0]
    output = artifacts.processed_paths(selection)[0]
    processing_temp, missing_backup = artifacts.processing_intermediate_paths(selection)
    for path in (raw, output, processing_temp):
        _write(path)

    with patch("datadec.data.cleanup.verify_published_unit") as verify:
        result = clean_data(artifacts, selection, mode=CleanupMode.ALL, dry_run=True)

    verify.assert_called_once()
    assert result.removed_paths == ()
    assert result.would_remove_paths == (raw, output, processing_temp)
    assert result.missing_paths == (missing_backup,)
    assert all(path.exists() for path in result.would_remove_paths)


def test_clean_verifies_every_existing_unit_before_any_deletion(
    tmp_path: Path,
) -> None:
    artifacts = DataArtifacts(tmp_path)
    selection = resolve_selection(ppl=True, olmes=True)
    tracked = (*artifacts.raw_paths(selection), *artifacts.processed_paths(selection))
    for path in tracked:
        _write(path)
    checked: list[str] = []

    def verify(unit: object, **_kwargs: object) -> Mock:
        assert all(path.exists() for path in tracked)
        checked.append(getattr(unit, "name"))
        return Mock()

    with patch("datadec.data.cleanup.verify_published_unit", side_effect=verify):
        result = clean_data(artifacts, selection, mode=CleanupMode.ALL, api=Mock())

    assert checked == ["ppl", "olmes"]
    assert set(result.removed_paths) == set(tracked)
    assert all(not path.exists() for path in tracked)


def test_clean_remote_failure_blocks_all_deletion(tmp_path: Path) -> None:
    artifacts = DataArtifacts(tmp_path)
    selection = resolve_selection(ppl=True, olmes=True)
    tracked = (*artifacts.raw_paths(selection), *artifacts.processed_paths(selection))
    for path in tracked:
        _write(path)

    with (
        patch(
            "datadec.data.cleanup.verify_published_unit",
            side_effect=(Mock(), RuntimeError("remote mismatch")),
        ) as verify,
        pytest.raises(RuntimeError, match="remote mismatch"),
    ):
        clean_data(artifacts, selection, mode=CleanupMode.ALL, api=Mock())

    assert verify.call_count == 2
    assert all(path.exists() for path in tracked)


def test_clean_verifies_only_existing_files_in_partial_unit(tmp_path: Path) -> None:
    artifacts = DataArtifacts(tmp_path)
    selection = resolve_selection(olmes_details=("c4",))
    tasks = artifacts.olmes_details_tasks_path("c4")
    _write(tasks)

    with patch("datadec.data.cleanup.verify_published_unit") as verify:
        clean_data(artifacts, selection, mode=CleanupMode.ALL, dry_run=True)

    verified_unit = verify.call_args.args[0]
    assert tuple(file.local_path for file in verified_unit.files) == (tasks,)


def test_clean_tolerates_all_selected_outputs_being_absent(tmp_path: Path) -> None:
    artifacts = DataArtifacts(tmp_path)
    selection = resolve_selection(ppl=True)
    raw = artifacts.raw_paths(selection)[0]
    _write(raw)

    with patch("datadec.data.cleanup.verify_published_unit") as verify:
        result = clean_data(artifacts, selection, mode=CleanupMode.ALL)

    verify.assert_not_called()
    assert result.removed_paths == (raw,)


def test_base_cleanup_preserves_unselected_detail_artifacts(tmp_path: Path) -> None:
    artifacts = DataArtifacts(tmp_path)
    selection = resolve_selection(olmes=True)
    base_raw = artifacts.raw_paths(selection)[0]
    base_output = artifacts.processed_paths(selection)[0]
    detail_selection = resolve_selection(olmes_details=("c4",))
    detail_raw = artifacts.raw_paths(detail_selection)[0]
    detail_output = artifacts.processed_paths(detail_selection)[0]
    for path in (base_raw, base_output, detail_raw, detail_output):
        _write(path)

    with patch("datadec.data.cleanup.verify_published_unit"):
        clean_data(artifacts, selection, mode=CleanupMode.ALL)

    assert not base_raw.exists()
    assert not base_output.exists()
    assert detail_raw.exists()
    assert detail_output.exists()


def test_cleanup_rejects_symlink_escape_without_touching_target(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    artifacts = DataArtifacts(data_dir)
    selection = resolve_selection(ppl=True)
    outside = tmp_path / "outside.parquet"
    _write(outside)
    raw = artifacts.raw_paths(selection)[0]
    raw.parent.mkdir(parents=True)
    raw.symlink_to(outside)

    with pytest.raises(ValueError, match="symlink"):
        cleanup_raw(artifacts, selection)

    assert raw.is_symlink()
    assert outside.read_text() == "data"


def test_cleanup_refuses_directory_at_file_owned_path_before_deletion(
    tmp_path: Path,
) -> None:
    artifacts = DataArtifacts(tmp_path)
    selection = resolve_selection(ppl=True)
    raw = artifacts.raw_paths(selection)[0]
    raw.mkdir(parents=True)
    unrelated = raw / "keep.txt"
    _write(unrelated)

    with pytest.raises(ValueError, match="not a regular file"):
        cleanup_raw(artifacts, selection)

    assert unrelated.exists()
