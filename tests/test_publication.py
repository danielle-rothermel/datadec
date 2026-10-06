from __future__ import annotations

from pathlib import Path

from datadec.config import (
    PUBLISHED_RESULT_SCHEMAS,
    load_published_results_manifest,
    load_publishing_contract,
)
from datadec.data.artifacts import DataArtifacts
from datadec.data.publication import (
    olmes_details_publication_unit,
    ppl_publication_unit,
    publication_units,
    published_results_publication_units,
    scaling_law_publication_unit,
)
from datadec.data.selection import resolve_selection


def test_publication_units_follow_resolved_selection_in_pipeline_order(
    tmp_path: Path,
) -> None:
    artifacts = DataArtifacts(tmp_path)
    selection = resolve_selection(
        ppl=True,
        olmes=True,
        olmes_details=("c4", "fineweb-pro"),
        scaling_law=True,
        units=("outputs2", "per-task-arc-easy"),
    )

    units = publication_units(artifacts, selection)

    assert tuple(unit.name for unit in units) == (
        "ppl",
        "olmes",
        "olmes-details:c4",
        "olmes-details:fineweb-pro",
        "scaling-law",
        "published-results:outputs2",
        "published-results:per-task-arc-easy",
    )
    assert tuple(
        file.local_path for unit in units for file in unit.files
    ) == artifacts.processed_paths(selection)
    assert all(not hasattr(unit, "cleanup_paths") for unit in units)


def test_publication_factories_apply_explicit_output_overrides(
    tmp_path: Path,
) -> None:
    artifacts = DataArtifacts(tmp_path)
    ppl = tmp_path / "custom-ppl.parquet"
    evaluations = tmp_path / "custom-evaluations.parquet"
    losses = tmp_path / "custom-losses.parquet"
    tasks = tmp_path / "custom-tasks.parquet"
    instances = tmp_path / "custom-instances.parquet"
    choices = tmp_path / "custom-choices.parquet"

    assert ppl_publication_unit(artifacts, output_path=ppl).files[0].local_path == ppl
    assert tuple(
        file.local_path
        for file in scaling_law_publication_unit(
            artifacts,
            evaluations_output_path=evaluations,
            checkpoint_losses_output_path=losses,
        ).files
    ) == (evaluations, losses)
    assert tuple(
        file.local_path
        for file in olmes_details_publication_unit(
            artifacts,
            "c4",
            output_tasks_path=tasks,
            output_instances_path=instances,
            output_choices_path=choices,
        ).files
    ) == (tasks, instances, choices)


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


def test_multi_file_factories_are_atomic_and_contract_typed(tmp_path: Path) -> None:
    artifacts = DataArtifacts(tmp_path)
    scaling_law = scaling_law_publication_unit(artifacts)
    details = olmes_details_publication_unit(artifacts, "c4")

    assert tuple(file.remote_path for file in scaling_law.files) == (
        "scaling-law/evaluations.parquet",
        "scaling-law/checkpoint-losses.parquet",
    )
    assert tuple(file.remote_path for file in details.files) == (
        "olmes-details/c4/tasks.parquet",
        "olmes-details/c4/instances.parquet",
        "olmes-details/c4/choices.parquet",
    )
    assert all(
        column.nullable is not None
        for unit in (scaling_law, details)
        for file in unit.files
        for column in file.expected_schema or ()
    )


def test_published_result_units_map_manifest_paths_and_schemas_exactly(
    tmp_path: Path,
) -> None:
    artifacts = DataArtifacts(tmp_path)
    manifest = load_published_results_manifest()
    publishing = load_publishing_contract()

    units = published_results_publication_units(
        artifacts,
        contract=publishing,
        manifest=manifest,
    )

    files_by_path = {file.local_path: file for unit in units for file in unit.files}
    structured_sources = tuple(
        source for source in manifest.files if source.category == "published_results"
    )
    assert set(files_by_path) == {
        artifacts.published_result_output_path(source) for source in structured_sources
    }
    for source in structured_sources:
        file = files_by_path[artifacts.published_result_output_path(source)]
        assert file.remote_path == (
            f"{publishing.published_results.remote_root}/"
            f"{source.parquet_relative_path()}"
        )
        schema_name = source.schema
        assert schema_name is not None
        assert tuple(
            (column.name, column.logical_type, column.nullable)
            for column in file.expected_schema or ()
        ) == tuple(
            (column.name, column.logical_type, column.nullable)
            for column in PUBLISHED_RESULT_SCHEMAS[schema_name].columns
        )


def test_published_result_units_select_in_manifest_order(tmp_path: Path) -> None:
    units = published_results_publication_units(
        DataArtifacts(tmp_path),
        units=("per-task-winogrande", "outputs2"),
    )

    assert tuple(unit.name for unit in units) == (
        "published-results:outputs2",
        "published-results:per-task-winogrande",
    )


def test_publication_units_do_not_include_unselected_outputs(tmp_path: Path) -> None:
    artifacts = DataArtifacts(tmp_path)

    units = publication_units(artifacts, resolve_selection(olmes=True))

    assert tuple(unit.name for unit in units) == ("olmes",)
