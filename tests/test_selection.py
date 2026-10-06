from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from datadec.config import load_published_results_manifest, load_source_manifest
from datadec.data.selection import (
    DatasetSelection,
    published_result_units,
    resolve_olmes_detail_recipes,
    resolve_published_result_units,
    resolve_selection,
)


def test_resolve_selection_preserves_explicit_groups_and_deduplicates() -> None:
    selection = resolve_selection(
        ppl=True,
        olmes_details=("fineweb-pro", "c4", "fineweb-pro"),
        units=("per-task-arc-easy", "outputs2", "outputs2"),
    )

    assert selection == DatasetSelection(
        ppl=True,
        olmes=False,
        olmes_details=("fineweb-pro", "c4"),
        scaling_law=False,
        published_results=("outputs2", "per-task-arc-easy"),
        published_figures=False,
        all_data=False,
    )


def test_units_alone_select_published_results() -> None:
    selection = resolve_selection(units=("new-eval-intermediates",))

    assert selection.published_results == ("new-eval-intermediates",)


def test_all_expands_processable_groups_but_not_figures() -> None:
    manifest = load_source_manifest()
    result_manifest = load_published_results_manifest()

    selection = resolve_selection(all_data=True)

    assert selection == DatasetSelection(
        ppl=True,
        olmes=True,
        olmes_details=manifest.olmes_details.recipes,
        scaling_law=True,
        published_results=published_result_units(result_manifest),
        published_figures=False,
        all_data=True,
    )


def test_figures_are_a_valid_explicit_selection() -> None:
    selection = resolve_selection(published_figures=True)

    assert selection.published_figures is True
    assert selection.published_results == ()


def test_empty_selection_is_rejected() -> None:
    with pytest.raises(ValueError, match="select at least one dataset"):
        resolve_selection()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"olmes_details": ("missing",)}, "unknown OLMES detail recipe: missing"),
        ({"units": ("missing",)}, "unknown published-results unit: missing"),
        (
            {"all_data": True, "olmes_details": ("missing",)},
            "unknown OLMES detail recipe: missing",
        ),
        (
            {"all_data": True, "units": ("missing",)},
            "unknown published-results unit: missing",
        ),
    ],
)
def test_unknown_selection_values_are_rejected(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_selection(**kwargs)


def test_recipe_all_and_unit_all_follow_manifest_order() -> None:
    source = load_source_manifest().olmes_details
    manifest = load_published_results_manifest()

    assert resolve_olmes_detail_recipes(("c4", "all", "c4"), source) == (source.recipes)
    assert resolve_published_result_units(("outputs2", "all"), manifest) == (
        published_result_units(manifest)
    )


def test_resolved_selection_is_immutable() -> None:
    selection = resolve_selection(ppl=True)

    with pytest.raises(FrozenInstanceError):
        selection.ppl = False  # type: ignore[misc]
