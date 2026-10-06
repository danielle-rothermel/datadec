from __future__ import annotations

from pathlib import Path

import pytest

from datadec.config import load_published_results_manifest, load_source_manifest
from datadec.data.artifacts import (
    OLMES_DETAILS_STAGING_FILENAME,
    DataArtifacts,
)
from datadec.data.selection import resolve_selection, selected_published_result_sources


def test_named_paths_use_the_custom_data_root(tmp_path: Path) -> None:
    artifacts = DataArtifacts(tmp_path)

    assert artifacts.data_dir == tmp_path
    assert artifacts.get_path("ppl_raw") == tmp_path / "raw/ppl.parquet"
    assert artifacts.get_path("dwn_raw") == tmp_path / "raw/olmes.parquet"
    assert artifacts.get_path("ppl_processed") == tmp_path / "processed/ppl.parquet"
    assert artifacts.get_path("olmes_processed") == (
        tmp_path / "processed/olmes.parquet"
    )
    assert artifacts.scaling_law_evaluations_path() == (
        tmp_path / "processed/scaling-law/evaluations.parquet"
    )
    assert artifacts.scaling_law_checkpoint_losses_path() == (
        tmp_path / "processed/scaling-law/checkpoint-losses.parquet"
    )
    assert artifacts.olmes_details_tasks_path() == (
        tmp_path / "processed/olmes-details/tasks.parquet"
    )
    assert artifacts.olmes_details_instances_path("c4") == (
        tmp_path / "processed/olmes-details/instances/c4.parquet"
    )
    assert artifacts.olmes_details_choices_path("c4") == (
        tmp_path / "processed/olmes-details/choices/c4.parquet"
    )


def test_unknown_named_path_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown dataframe 'missing'"):
        DataArtifacts(tmp_path).get_path("missing")


def test_ppl_selection_owns_only_its_known_paths(tmp_path: Path) -> None:
    artifacts = DataArtifacts(tmp_path)
    selection = resolve_selection(ppl=True)

    assert artifacts.raw_paths(selection) == (tmp_path / "raw/ppl.parquet",)
    assert artifacts.processed_paths(selection) == (tmp_path / "processed/ppl.parquet",)
    assert artifacts.raw_intermediate_paths(selection) == ()
    assert artifacts.intermediate_paths(selection) == (
        tmp_path / "processed/.ppl.parquet.tmp",
        tmp_path / "processed/.ppl.parquet.backup.tmp",
    )
    assert artifacts.cache_paths(selection) == ()


def test_detail_selection_owns_archive_outputs_and_staging(tmp_path: Path) -> None:
    artifacts = DataArtifacts(tmp_path)
    selection = resolve_selection(olmes_details=("c4",))
    output_root = tmp_path / "processed/olmes-details"
    outputs = (
        output_root / "tasks.parquet",
        output_root / "instances/c4.parquet",
        output_root / "choices/c4.parquet",
    )

    assert artifacts.raw_paths(selection) == (
        tmp_path / "raw/olmes-details/models/c4.tar.gz",
    )
    assert artifacts.processed_paths(selection) == outputs
    intermediates = set(artifacts.intermediate_paths(selection))
    assert {output.with_name(f".{output.name}.tmp") for output in outputs}.issubset(
        intermediates
    )
    assert {
        output.with_name(f".{output.name}.backup.tmp") for output in outputs
    }.issubset(intermediates)
    assert artifacts.raw_intermediate_paths(selection) == ()
    staging = output_root / "staging/c4" / OLMES_DETAILS_STAGING_FILENAME
    assert {staging, Path(f"{staging}.wal"), Path(f"{staging}.tmp")}.issubset(
        intermediates
    )
    assert not any("fineweb-pro" in str(path) for path in intermediates)
    metadata_root = (
        tmp_path / "raw/olmes-details/.cache/huggingface/download/models/c4.tar.gz"
    )
    assert artifacts.cache_paths(selection) == (
        metadata_root.with_name("c4.tar.gz.lock"),
        metadata_root.with_name("c4.tar.gz.metadata"),
    )


def test_published_unit_owns_only_its_sources_outputs_and_partials(
    tmp_path: Path,
) -> None:
    artifacts = DataArtifacts(tmp_path)
    selection = resolve_selection(units=("new-eval-intermediates",))
    manifest = load_published_results_manifest()
    sources = selected_published_result_sources(("new-eval-intermediates",), manifest)
    raw = tuple(artifacts.published_result_source_path(source) for source in sources)
    processed = tuple(
        artifacts.published_result_output_path(source) for source in sources
    )

    assert artifacts.raw_paths(selection) == raw
    assert artifacts.processed_paths(selection) == processed
    intermediates = set(artifacts.intermediate_paths(selection))
    raw_intermediates = tuple(path.with_name(f"{path.name}.part") for path in raw)
    assert artifacts.raw_intermediate_paths(selection) == raw_intermediates
    assert set(raw_intermediates).issubset(intermediates)
    assert len(processed) == 2
    assert len(intermediates) == len(raw) + 2 * len(processed)
    assert not any("outputs2" in str(path) for path in intermediates)


def test_figure_selection_has_raw_and_partial_paths_only(tmp_path: Path) -> None:
    artifacts = DataArtifacts(tmp_path)
    selection = resolve_selection(published_figures=True)
    sources = tuple(
        source
        for source in load_published_results_manifest().files
        if source.category == "published_figures"
    )
    raw = tuple(artifacts.published_result_download_path(source) for source in sources)

    assert artifacts.raw_paths(selection) == raw
    assert artifacts.processed_paths(selection) == ()
    assert artifacts.intermediate_paths(selection) == tuple(
        path.with_name(f"{path.name}.part") for path in raw
    )


def test_all_data_inventory_includes_every_known_source_and_shared_cache(
    tmp_path: Path,
) -> None:
    artifacts = DataArtifacts(tmp_path)
    selection = resolve_selection(all_data=True, published_figures=True)
    source_manifest = load_source_manifest()
    result_manifest = load_published_results_manifest()

    raw = set(artifacts.raw_paths(selection))
    expected_dataset_sources = {
        tmp_path / source_manifest.ppl.output,
        tmp_path / source_manifest.olmes.output,
    }
    expected_details = {
        artifacts.olmes_detail_source_path(recipe)
        for recipe in source_manifest.olmes_details.recipes
    }
    expected_drive_sources = {
        artifacts.published_result_download_path(source)
        for source in result_manifest.files
    }
    assert raw == expected_dataset_sources | expected_details | expected_drive_sources
    assert artifacts.cache_paths(selection) == (
        tmp_path / "cache/huggingface",
        tmp_path / "raw/olmes-details/.cache",
    )

    owned = (
        raw
        | set(artifacts.processed_paths(selection))
        | set(artifacts.intermediate_paths(selection))
        | set(artifacts.cache_paths(selection))
    )
    assert tmp_path not in owned
    assert tmp_path / "raw_downloads" not in owned


def test_structured_source_path_rejects_figure(tmp_path: Path) -> None:
    figure = next(
        source
        for source in load_published_results_manifest().files
        if source.category == "published_figures"
    )

    with pytest.raises(
        ValueError, match="only structured published results have source paths"
    ):
        DataArtifacts(tmp_path).published_result_source_path(figure)
