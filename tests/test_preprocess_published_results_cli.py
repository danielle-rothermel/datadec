from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
from unittest.mock import patch

from typer.testing import CliRunner

SCRIPT_PATH = Path(__file__).parents[1] / "scripts/preprocess_published_results.py"
SCRIPT_MODULE = "datadec_preprocess_published_results_script"
spec = importlib.util.spec_from_file_location(SCRIPT_MODULE, SCRIPT_PATH)
assert spec is not None and spec.loader is not None
script = importlib.util.module_from_spec(spec)
sys.modules[SCRIPT_MODULE] = script
spec.loader.exec_module(script)

DEFAULT_DATA_DIR = script.DEFAULT_DATA_DIR
app = script.app
runner = CliRunner()


def test_cli_defaults_to_all_manifest_units_independent_of_cwd(
    tmp_path: Path, monkeypatch
) -> None:
    paths = object()
    manifest = object()
    units = ("outputs2", "per-task-arc-easy")
    preprocess_results = tuple(SimpleNamespace(publication_unit=unit) for unit in units)
    publication_units = tuple(object() for _ in units)
    monkeypatch.chdir(tmp_path)
    with (
        patch.object(script, "load_published_results_manifest", return_value=manifest),
        patch.object(
            script, "resolve_published_result_units", return_value=units
        ) as resolve,
        patch.object(script, "DataArtifacts", return_value=paths) as path_type,
        patch.object(
            script,
            "preprocess_published_results",
            return_value=preprocess_results,
        ) as preprocess,
        patch.object(
            script,
            "published_results_publication_units",
            return_value=publication_units,
        ) as compose,
        patch.object(script, "load_publishing_contract") as load_publishing,
        patch.object(
            script,
            "publish_unit",
            side_effect=(
                SimpleNamespace(
                    unit_name=f"published-results:{unit}",
                    created=True,
                    commit_oid=f"commit-{unit}",
                )
                for unit in units
            ),
        ) as publish,
    ):
        load_publishing.return_value.target = object()
        result = runner.invoke(app, [])

    assert result.exit_code == 0
    resolve.assert_called_once_with((), manifest)
    path_type.assert_called_once_with(DEFAULT_DATA_DIR)
    preprocess.assert_called_once_with(
        paths, units=units, manifest=manifest, verbose=True
    )
    compose.assert_called_once_with(
        paths,
        units=units,
        contract=load_publishing.return_value,
        manifest=manifest,
    )
    assert [call.args[0] for call in publish.call_args_list] == list(publication_units)
    assert all("keep_sources" not in call.kwargs for call in publish.call_args_list)
    assert DEFAULT_DATA_DIR == Path(__file__).resolve().parents[1] / "data"


def test_cli_forwards_repeatable_units_and_data_dir(tmp_path: Path) -> None:
    manifest = object()
    with (
        patch.object(script, "load_published_results_manifest", return_value=manifest),
        patch.object(
            script,
            "resolve_published_result_units",
            return_value=("outputs2", "per-task-arc-easy"),
        ) as resolve,
        patch.object(
            script,
            "preprocess_published_results",
            return_value=(SimpleNamespace(publication_unit="outputs2"),),
        ) as preprocess,
        patch.object(script, "published_results_publication_units", return_value=()),
    ):
        result = runner.invoke(
            app,
            [
                "--unit",
                "per-task-arc-easy",
                "--unit",
                "outputs2",
                "--data-dir",
                str(tmp_path),
            ],
        )

    assert result.exit_code == 0
    resolve.assert_called_once_with(["per-task-arc-easy", "outputs2"], manifest)
    preprocess.assert_called_once()
    assert preprocess.call_args.kwargs == {
        "units": ("outputs2", "per-task-arc-easy"),
        "manifest": manifest,
        "verbose": True,
    }


def test_cli_no_upload_preprocesses_without_hf_work_or_cleanup(tmp_path: Path) -> None:
    manifest = object()
    preprocess_results = (SimpleNamespace(publication_unit="outputs2"),)
    with (
        patch.object(script, "load_published_results_manifest", return_value=manifest),
        patch.object(
            script, "resolve_published_result_units", return_value=("outputs2",)
        ),
        patch.object(
            script,
            "preprocess_published_results",
            return_value=preprocess_results,
        ) as preprocess,
        patch.object(script, "load_publishing_contract") as load_publishing,
        patch.object(script, "published_results_publication_units") as compose,
        patch.object(script, "publish_unit") as publish,
    ):
        result = runner.invoke(app, ["--no-upload", "--data-dir", str(tmp_path)])

    assert result.exit_code == 0
    preprocess.assert_called_once()
    load_publishing.assert_not_called()
    compose.assert_not_called()
    publish.assert_not_called()


def test_cli_publishes_only_returned_units_without_cleanup_policy(
    tmp_path: Path,
) -> None:
    manifest = object()
    paths = object()
    preprocess_results = (
        SimpleNamespace(publication_unit="outputs2"),
        SimpleNamespace(publication_unit="per-task-winogrande"),
    )
    publication_units = (object(), object())
    publishing = SimpleNamespace(target=object())
    with (
        patch.object(script, "load_published_results_manifest", return_value=manifest),
        patch.object(
            script,
            "resolve_published_result_units",
            return_value=("outputs2", "per-task-winogrande"),
        ),
        patch.object(script, "DataArtifacts", return_value=paths),
        patch.object(
            script,
            "preprocess_published_results",
            return_value=preprocess_results,
        ),
        patch.object(script, "load_publishing_contract", return_value=publishing),
        patch.object(
            script,
            "published_results_publication_units",
            return_value=publication_units,
        ) as compose,
        patch.object(
            script,
            "publish_unit",
            side_effect=(
                SimpleNamespace(
                    unit_name=f"unit-{index}",
                    created=False,
                    commit_oid=f"commit-{index}",
                )
                for index in range(2)
            ),
        ) as publish,
    ):
        result = runner.invoke(
            app,
            [
                "--unit",
                "outputs2",
                "--unit",
                "per-task-winogrande",
                "--data-dir",
                str(tmp_path),
            ],
        )

    assert result.exit_code == 0
    compose.assert_called_once_with(
        paths,
        units=("outputs2", "per-task-winogrande"),
        contract=publishing,
        manifest=manifest,
    )
    assert publish.call_count == 2
    assert all("keep_sources" not in call.kwargs for call in publish.call_args_list)


def test_cli_does_not_upload_when_preprocessing_fails() -> None:
    with (
        patch.object(
            script,
            "preprocess_published_results",
            side_effect=ValueError("conversion failed"),
        ),
        patch.object(script, "load_publishing_contract") as load_publishing,
        patch.object(script, "publish_unit") as publish,
    ):
        result = runner.invoke(app, [])

    assert result.exit_code == 1
    load_publishing.assert_not_called()
    publish.assert_not_called()


def test_cli_stops_after_failed_publication_without_retry() -> None:
    preprocess_results = tuple(
        SimpleNamespace(publication_unit=f"unit-{index}") for index in range(3)
    )
    publication_units = tuple(object() for _ in range(3))
    with (
        patch.object(
            script,
            "resolve_published_result_units",
            return_value=tuple(
                result.publication_unit for result in preprocess_results
            ),
        ),
        patch.object(
            script,
            "preprocess_published_results",
            return_value=preprocess_results,
        ),
        patch.object(
            script,
            "published_results_publication_units",
            return_value=publication_units,
        ),
        patch.object(
            script,
            "publish_unit",
            side_effect=(
                SimpleNamespace(unit_name="first", created=True, commit_oid="1"),
                RuntimeError("second failed"),
            ),
        ) as publish,
    ):
        result = runner.invoke(app, [])

    assert result.exit_code == 1
    assert publish.call_count == 2
    assert [call.args[0] for call in publish.call_args_list] == list(
        publication_units[:2]
    )


def test_cli_reports_unknown_unit_as_usage_error() -> None:
    with patch.object(
        script,
        "resolve_published_result_units",
        side_effect=ValueError("unknown published-results unit: missing"),
    ):
        result = runner.invoke(app, ["--unit", "missing"])

    assert result.exit_code == 2
    assert "unknown published-results unit: missing" in result.output
