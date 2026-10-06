from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
from unittest.mock import patch

from typer.testing import CliRunner

from datadec.data.artifacts import DataArtifacts

SCRIPT_PATH = Path(__file__).parents[1] / "scripts/preprocess_olmes_details.py"
SCRIPT_MODULE = "datadec_preprocess_olmes_details_script"
spec = importlib.util.spec_from_file_location(SCRIPT_MODULE, SCRIPT_PATH)
assert spec is not None and spec.loader is not None
script = importlib.util.module_from_spec(spec)
sys.modules[SCRIPT_MODULE] = script
spec.loader.exec_module(script)

DEFAULT_DATA_DIR = script.DEFAULT_DATA_DIR
app = script.app

runner = CliRunner()


def _preprocess_result(tmp_path: Path, recipe: str) -> SimpleNamespace:
    return SimpleNamespace(
        output_tasks_path=tmp_path / f"{recipe}-tasks.parquet",
        output_instances_path=tmp_path / f"{recipe}-instances.parquet",
        output_choices_path=tmp_path / f"{recipe}-choices.parquet",
    )


def test_cli_default_is_repo_data_independent_of_cwd(
    tmp_path: Path, monkeypatch
) -> None:
    paths = DataArtifacts(tmp_path)
    monkeypatch.chdir(tmp_path)
    with (
        patch.object(script, "DataArtifacts", return_value=paths) as path_type,
        patch.object(
            script,
            "preprocess_olmes_details",
            return_value=_preprocess_result(tmp_path, "c4"),
        ) as preprocess,
        patch.object(script, "publish_unit") as publish,
    ):
        result = runner.invoke(app, ["--recipe", "c4"])

    assert result.exit_code == 0
    path_type.assert_called_once_with(DEFAULT_DATA_DIR)
    preprocess.assert_called_once_with(
        paths,
        "c4",
        input_path=None,
        output_tasks_path=None,
        output_instances_path=None,
        output_choices_path=None,
        verbose=True,
    )
    unit = publish.call_args.args[0]
    assert unit.name == "olmes-details:c4"
    assert tuple(file.local_path for file in unit.files) == (
        tmp_path / "c4-tasks.parquet",
        tmp_path / "c4-instances.parquet",
        tmp_path / "c4-choices.parquet",
    )
    assert not hasattr(unit, "cleanup_paths")
    assert DEFAULT_DATA_DIR == Path(__file__).resolve().parents[1] / "data"


def test_cli_data_dir_invokes_preprocess_for_each_recipe(tmp_path: Path) -> None:
    paths = object()
    with (
        patch.object(script, "DataArtifacts", return_value=paths),
        patch.object(script, "preprocess_olmes_details") as preprocess,
        patch("datadec.data.download.download_sources") as download_sources,
    ):
        result = runner.invoke(
            app,
            [
                "--recipe",
                "c4",
                "--recipe",
                "fineweb-pro",
                "--data-dir",
                str(tmp_path),
                "--no-upload",
            ],
        )

    assert result.exit_code == 0
    assert preprocess.call_count == 2
    preprocess.assert_any_call(
        paths,
        "c4",
        input_path=None,
        output_tasks_path=None,
        output_instances_path=None,
        output_choices_path=None,
        verbose=True,
    )
    preprocess.assert_any_call(
        paths,
        "fineweb-pro",
        input_path=None,
        output_tasks_path=None,
        output_instances_path=None,
        output_choices_path=None,
        verbose=True,
    )
    download_sources.assert_not_called()


def test_cli_preprocesses_and_publishes_each_recipe_sequentially(
    tmp_path: Path,
) -> None:
    paths = DataArtifacts(tmp_path)
    events: list[str] = []

    def preprocess_side_effect(
        paths_arg: object, recipe: str, **kwargs: object
    ) -> SimpleNamespace:
        assert paths_arg is paths
        events.append(f"preprocess:{recipe}")
        return _preprocess_result(tmp_path, recipe)

    def publish_side_effect(unit: object, **kwargs: object) -> None:
        events.append(f"publish:{getattr(unit, 'name')}")

    with (
        patch.object(script, "DataArtifacts", return_value=paths),
        patch.object(
            script,
            "preprocess_olmes_details",
            side_effect=preprocess_side_effect,
        ),
        patch.object(script, "publish_unit", side_effect=publish_side_effect),
    ):
        result = runner.invoke(
            app,
            ["--recipe", "c4", "--recipe", "fineweb-pro"],
        )

    assert result.exit_code == 0
    assert events == [
        "preprocess:c4",
        "publish:olmes-details:c4",
        "preprocess:fineweb-pro",
        "publish:olmes-details:fineweb-pro",
    ]


def test_cli_path_overrides_are_forwarded(tmp_path: Path) -> None:
    paths = DataArtifacts(tmp_path)
    input_path = tmp_path / "custom-input.tar.gz"
    output_tasks = tmp_path / "tasks.parquet"
    output_instances = tmp_path / "instances.parquet"
    output_choices = tmp_path / "choices.parquet"
    input_path.write_text("source")
    with (
        patch.object(script, "DataArtifacts", return_value=paths),
        patch.object(
            script,
            "preprocess_olmes_details",
            return_value=SimpleNamespace(
                output_tasks_path=output_tasks,
                output_instances_path=output_instances,
                output_choices_path=output_choices,
            ),
        ) as preprocess,
        patch.object(script, "publish_unit") as publish,
    ):
        result = runner.invoke(
            app,
            [
                "--recipe",
                "c4",
                "--input",
                str(input_path),
                "--output-tasks",
                str(output_tasks),
                "--output-instances",
                str(output_instances),
                "--output-choices",
                str(output_choices),
            ],
        )

    assert result.exit_code == 0
    preprocess.assert_called_once_with(
        paths,
        "c4",
        input_path=input_path,
        output_tasks_path=output_tasks,
        output_instances_path=output_instances,
        output_choices_path=output_choices,
        verbose=True,
    )
    unit = publish.call_args.args[0]
    assert tuple(file.local_path for file in unit.files) == (
        output_tasks,
        output_instances,
        output_choices,
    )
    assert not hasattr(unit, "cleanup_paths")
    assert input_path.is_file()


def test_cli_rejects_unknown_recipe() -> None:
    result = runner.invoke(app, ["--recipe", "missing"])
    assert result.exit_code != 0
    assert "unknown OLMES detail recipe" in result.output


def test_cli_rejects_path_override_for_multiple_recipes(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "--recipe",
            "c4",
            "--recipe",
            "fineweb-pro",
            "--output-tasks",
            str(tmp_path / "tasks.parquet"),
        ],
    )

    assert result.exit_code == 2
    assert "path overrides require exactly one --recipe" in result.output


def test_cli_publish_failure_stops_before_next_recipe(tmp_path: Path) -> None:
    paths = DataArtifacts(tmp_path)

    def preprocess_side_effect(
        paths_arg: object, recipe: str, **kwargs: object
    ) -> SimpleNamespace:
        assert paths_arg is paths
        return _preprocess_result(tmp_path, recipe)

    with (
        patch.object(script, "DataArtifacts", return_value=paths),
        patch.object(
            script,
            "preprocess_olmes_details",
            side_effect=preprocess_side_effect,
        ) as preprocess,
        patch.object(
            script,
            "publish_unit",
            side_effect=RuntimeError("stale parent"),
        ) as publish,
    ):
        result = runner.invoke(
            app,
            ["--recipe", "c4", "--recipe", "fineweb-pro"],
        )

    assert result.exit_code == 1
    assert isinstance(result.exception, RuntimeError)
    assert preprocess.call_count == 1
    assert publish.call_count == 1
    unit = publish.call_args.args[0]
    assert not hasattr(unit, "cleanup_paths")
