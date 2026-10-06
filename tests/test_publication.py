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
        "olmes-detail-tasks",
        "olmes-details:c4",
        "olmes-details:fineweb-pro",
        "scaling-law",
        "published-results:target_pairs",
        "published-results:transformed",
        "published-results:prediction_model_scale",
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
        "olmes-details/tasks.parquet",
        "olmes-details/instances/c4.parquet",
        "olmes-details/choices/c4.parquet",
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

    files_by_name = {unit.name: unit.files[0] for unit in units}
    assert set(files_by_name) == {
        f"published-results:{schema_name}" for schema_name in PUBLISHED_RESULT_SCHEMAS
    }
    for schema_name, schema in PUBLISHED_RESULT_SCHEMAS.items():
        file = files_by_name[f"published-results:{schema_name}"]
        assert file.local_path == artifacts.published_result_table_path(schema_name)
        assert file.remote_path == (
            f"{publishing.published_results.remote_root}/{schema_name}.parquet"
        )
        assert tuple(
            (column.name, column.logical_type, column.nullable)
            for column in file.expected_schema or ()
        ) == (
            ("source_file", "string", False),
            ("source_unit", "string", False),
            *(
                (column.name, column.logical_type, column.nullable)
                for column in schema.columns
            ),
        )


def test_published_result_units_select_in_manifest_order(tmp_path: Path) -> None:
    units = published_results_publication_units(
        DataArtifacts(tmp_path),
        units=("per-task-winogrande", "outputs2"),
    )

    assert tuple(unit.name for unit in units) == (
        "published-results:target_pairs",
        "published-results:transformed",
        "published-results:prediction_model_scale",
    )


def test_publication_units_do_not_include_unselected_outputs(tmp_path: Path) -> None:
    artifacts = DataArtifacts(tmp_path)

    units = publication_units(artifacts, resolve_selection(olmes=True))

    assert tuple(unit.name for unit in units) == ("olmes",)
