from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path, PurePosixPath

from datadec.config import (
    DatasetSource,
    DetailSource,
    PublishedResultFile,
    load_olmes_contract,
    load_published_results_manifest,
    load_scaling_law_contract,
    load_source_manifest,
)
from datadec.data.selection import DatasetSelection

DEFAULT_DATA_DIR = "./data"
OLMES_DETAILS_STAGING_FILENAME = ".olmes-details.duckdb"


def _deduplicate(paths: Iterable[Path]) -> tuple[Path, ...]:
    return tuple(dict.fromkeys(paths))


class DataArtifacts:
    def __init__(self, data_dir: str | Path = DEFAULT_DATA_DIR) -> None:
        manifest = load_source_manifest()
        self.data_dir = Path(data_dir)
        self.dataframes = {
            "ppl_raw": manifest.ppl.output,
            "dwn_raw": manifest.olmes.output,
            "ppl_processed": "processed/ppl.parquet",
            "olmes_processed": "processed/olmes.parquet",
        }

    def get_path(self, name: str) -> Path:
        if name not in self.dataframes:
            available = ", ".join(sorted(self.dataframes))
            raise ValueError(f"Unknown dataframe '{name}'. Available: {available}")
        return self.data_dir / self.dataframes[name]

    def dataset_source_path(self, source: DatasetSource) -> Path:
        return self.data_dir / source.output

    def olmes_detail_source_path(
        self, recipe: str, source: DetailSource | None = None
    ) -> Path:
        detail_source = source or load_source_manifest().olmes_details
        detail_source.file_for_recipe(recipe)
        filename = detail_source.filename_template.format(recipe=recipe)
        return self.data_dir / detail_source.output_root / filename

    def olmes_detail_metadata_paths(
        self, recipe: str, source: DetailSource | None = None
    ) -> tuple[Path, Path]:
        detail_source = source or load_source_manifest().olmes_details
        detail_source.file_for_recipe(recipe)
        filename = detail_source.filename_template.format(recipe=recipe)
        metadata_path = (
            self.data_dir
            / detail_source.output_root
            / ".cache"
            / "huggingface"
            / "download"
            / filename
        )
        return (
            metadata_path.with_name(f"{metadata_path.name}.lock"),
            metadata_path.with_name(f"{metadata_path.name}.metadata"),
        )

    def scaling_law_raw_paths(self) -> tuple[Path, ...]:
        contract = load_scaling_law_contract()
        raw_directory = self.data_dir / contract.raw_directory
        return tuple(
            raw_directory / filename for filename in contract.source_precedence
        )

    def scaling_law_evaluations_path(self) -> Path:
        contract = load_scaling_law_contract()
        return self.data_dir / contract.tables.evaluations.path

    def scaling_law_checkpoint_losses_path(self) -> Path:
        contract = load_scaling_law_contract()
        return self.data_dir / contract.tables.checkpoint_losses.path

    def published_result_download_path(self, source: PublishedResultFile) -> Path:
        relative_path = PurePosixPath(source.path)
        if source.category == "scaling_law":
            return self.data_dir / "raw" / "scaling-law" / relative_path.name
        return (
            self.data_dir
            / "reference"
            / "published-results"
            / Path(*relative_path.parts)
        )

    def published_result_source_path(self, source: PublishedResultFile) -> Path:
        if source.category != "published_results":
            raise ValueError("only structured published results have source paths")
        return self.published_result_download_path(source)

    def published_result_output_path(self, source: PublishedResultFile) -> Path:
        relative_path = source.parquet_relative_path()
        return self.data_dir / "processed" / "published-results" / relative_path

    def olmes_details_tasks_path(self, recipe: str) -> Path:
        return self._olmes_details_table_path("detailed_tasks", recipe)

    def olmes_details_instances_path(self, recipe: str) -> Path:
        return self._olmes_details_table_path("detailed_instances", recipe)

    def olmes_details_choices_path(self, recipe: str) -> Path:
        return self._olmes_details_table_path("detailed_choices", recipe)

    def olmes_details_staging_path(self, recipe: str) -> Path:
        return (
            self.olmes_details_tasks_path(recipe).parent
            / OLMES_DETAILS_STAGING_FILENAME
        )

    def _olmes_details_table_path(self, table_name: str, recipe: str) -> Path:
        contract = load_olmes_contract()
        table = getattr(contract.tables, table_name)
        template = table.path_template
        if template is None:
            raise ValueError(f"OLMES {table_name} table has no path_template")
        return self.data_dir / template.format(recipe=recipe)

    def raw_paths(self, selection: DatasetSelection) -> tuple[Path, ...]:
        manifest = load_source_manifest()
        paths: list[Path] = []
        if selection.ppl:
            paths.append(self.dataset_source_path(manifest.ppl))
        if selection.olmes:
            paths.append(self.dataset_source_path(manifest.olmes))
        paths.extend(
            self.olmes_detail_source_path(recipe, manifest.olmes_details)
            for recipe in selection.olmes_details
        )
        paths.extend(
            self.published_result_download_path(source)
            for source in self._selected_drive_sources(selection)
        )
        return _deduplicate(paths)

    def processed_paths(self, selection: DatasetSelection) -> tuple[Path, ...]:
        paths: list[Path] = []
        if selection.ppl:
            paths.append(self.get_path("ppl_processed"))
        if selection.olmes:
            paths.append(self.get_path("olmes_processed"))
        for recipe in selection.olmes_details:
            paths.extend(
                (
                    self.olmes_details_tasks_path(recipe),
                    self.olmes_details_instances_path(recipe),
                    self.olmes_details_choices_path(recipe),
                )
            )
        if selection.scaling_law:
            paths.extend(
                (
                    self.scaling_law_evaluations_path(),
                    self.scaling_law_checkpoint_losses_path(),
                )
            )
        selected_units = set(selection.published_results)
        if selected_units:
            paths.extend(
                self.published_result_output_path(source)
                for source in load_published_results_manifest().files
                if source.category == "published_results"
                and source.publication_unit in selected_units
            )
        return _deduplicate(paths)

    def aggregate_processed_paths(
        self, selection: DatasetSelection
    ) -> tuple[Path, ...]:
        """Base outputs retained by default cleanup, excluding recipe detail tables."""
        return self.processed_paths(replace(selection, olmes_details=()))

    def intermediate_paths(self, selection: DatasetSelection) -> tuple[Path, ...]:
        return _deduplicate(
            (
                *self.raw_intermediate_paths(selection),
                *self.processing_intermediate_paths(selection),
            )
        )

    def raw_intermediate_paths(self, selection: DatasetSelection) -> tuple[Path, ...]:
        return tuple(
            source_path.with_name(f"{source_path.name}.part")
            for source_path in (
                self.published_result_download_path(source)
                for source in self._selected_drive_sources(selection)
            )
        )

    def processing_intermediate_paths(
        self, selection: DatasetSelection
    ) -> tuple[Path, ...]:
        paths: list[Path] = []
        for output_path in self.processed_paths(selection):
            paths.extend(
                (
                    output_path.with_name(f".{output_path.name}.tmp"),
                    output_path.with_name(f".{output_path.name}.backup.tmp"),
                )
            )
        for recipe in selection.olmes_details:
            staging_path = self.olmes_details_staging_path(recipe)
            paths.extend(
                (
                    staging_path,
                    staging_path.with_name(f"{staging_path.name}.wal"),
                    Path(f"{staging_path}.tmp"),
                )
            )
        return _deduplicate(paths)

    def cache_paths(self, selection: DatasetSelection) -> tuple[Path, ...]:
        if selection.all_data:
            return (
                self.data_dir / "cache" / "huggingface",
                self.data_dir / "raw" / "olmes-details" / ".cache",
            )
        return tuple(
            path
            for recipe in selection.olmes_details
            for path in self.olmes_detail_metadata_paths(recipe)
        )

    def processing_intermediate_tree_paths(
        self, selection: DatasetSelection
    ) -> tuple[Path, ...]:
        """DuckDB spill directories owned by selected detail processing."""
        return tuple(
            Path(f"{self.olmes_details_staging_path(recipe)}.tmp")
            for recipe in selection.olmes_details
        )

    def _selected_drive_sources(
        self, selection: DatasetSelection
    ) -> tuple[PublishedResultFile, ...]:
        selected_units = set(selection.published_results)
        include_figures = selection.published_figures
        return tuple(
            source
            for source in load_published_results_manifest().files
            if (selection.scaling_law and source.category == "scaling_law")
            or (
                source.category == "published_results"
                and source.publication_unit in selected_units
            )
            or (include_figures and source.category == "published_figures")
        )


__all__ = [
    "DEFAULT_DATA_DIR",
    "DataArtifacts",
    "OLMES_DETAILS_STAGING_FILENAME",
]
