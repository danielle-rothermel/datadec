from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from datadec.config import (
    DetailSource,
    PublishedResultUnit,
    PublishedResultsManifest,
    load_published_results_manifest,
    load_source_manifest,
)


@dataclass(frozen=True, slots=True)
class DatasetSelection:
    ppl: bool
    olmes: bool
    olmes_details: tuple[str, ...]
    scaling_law: bool
    published_results: tuple[PublishedResultUnit, ...]
    published_figures: bool
    all_data: bool


def resolve_olmes_detail_recipes(
    requested: Sequence[str], source: DetailSource
) -> tuple[str, ...]:
    allowed = set(source.recipes)
    unknown = [
        recipe for recipe in requested if recipe != "all" and recipe not in allowed
    ]
    if unknown:
        names = ", ".join(dict.fromkeys(unknown))
        raise ValueError(f"unknown OLMES detail recipe: {names}")
    if "all" in requested:
        return source.recipes
    return tuple(dict.fromkeys(requested))


def published_result_units(
    manifest: PublishedResultsManifest,
) -> tuple[PublishedResultUnit, ...]:
    units: list[PublishedResultUnit] = []
    for source in manifest.files:
        unit = source.publication_unit
        if source.category == "published_results" and unit is not None:
            if unit not in units:
                units.append(unit)
    return tuple(units)


def resolve_published_result_units(
    requested: Sequence[str], manifest: PublishedResultsManifest
) -> tuple[PublishedResultUnit, ...]:
    available = published_result_units(manifest)
    allowed = set(available)
    unknown = [unit for unit in requested if unit != "all" and unit not in allowed]
    if unknown:
        names = ", ".join(dict.fromkeys(unknown))
        raise ValueError(f"unknown published-results unit: {names}")
    if not requested or "all" in requested:
        return available
    requested_set = set(requested)
    return tuple(unit for unit in available if unit in requested_set)


def resolve_selection(
    *,
    ppl: bool = False,
    olmes: bool = False,
    olmes_details: Sequence[str] = (),
    scaling_law: bool = False,
    published_results: bool = False,
    units: Sequence[str] = (),
    published_figures: bool = False,
    all_data: bool = False,
) -> DatasetSelection:
    source_manifest = load_source_manifest()
    published_results_manifest = load_published_results_manifest()

    requested_details = resolve_olmes_detail_recipes(
        olmes_details, source_manifest.olmes_details
    )
    selected_details = (
        source_manifest.olmes_details.recipes if all_data else requested_details
    )
    requested_units = (
        resolve_published_result_units(units, published_results_manifest)
        if units
        else ()
    )
    selected_units = (
        published_result_units(published_results_manifest)
        if all_data or (published_results and not units)
        else requested_units
    )

    selection = DatasetSelection(
        ppl=ppl or all_data,
        olmes=olmes or all_data,
        olmes_details=selected_details,
        scaling_law=scaling_law or all_data,
        published_results=selected_units,
        published_figures=published_figures,
        all_data=all_data,
    )
    if not any(
        (
            selection.ppl,
            selection.olmes,
            selection.olmes_details,
            selection.scaling_law,
            selection.published_results,
            selection.published_figures,
        )
    ):
        raise ValueError("select at least one dataset")
    return selection


__all__ = [
    "DatasetSelection",
    "published_result_units",
    "resolve_olmes_detail_recipes",
    "resolve_published_result_units",
    "resolve_selection",
]
