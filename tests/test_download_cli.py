from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch

from typer.testing import CliRunner

SCRIPT_PATH = Path(__file__).parents[1] / "scripts/download.py"
SCRIPT_MODULE = "datadec_download_script"
spec = importlib.util.spec_from_file_location(SCRIPT_MODULE, SCRIPT_PATH)
assert spec is not None and spec.loader is not None
script = importlib.util.module_from_spec(spec)
sys.modules[SCRIPT_MODULE] = script
spec.loader.exec_module(script)

DEFAULT_DATA_DIR = script.DEFAULT_DATA_DIR
app = script.app

runner = CliRunner()


def test_cli_requires_an_explicit_selection() -> None:
    result = runner.invoke(app, [])

    assert result.exit_code == 2
    assert "select --ppl, --olmes, --olmes-details, --scaling-law" in result.output
    assert "--published-results" in result.output
    assert "--published-figures" in result.output


def test_cli_forwards_mixed_repeatable_options_and_force(tmp_path: Path) -> None:
    paths = object()
    selection = object()
    with (
        patch.object(script, "DataArtifacts", return_value=paths) as path_type,
        patch.object(
            script, "resolve_selection", return_value=selection
        ) as resolve_selection,
        patch.object(script, "download_sources") as download_sources,
    ):
        result = runner.invoke(
            app,
            [
                "--ppl",
                "--olmes",
                "--olmes-details",
                "fineweb-pro",
                "--olmes-details",
                "c4",
                "--scaling-law",
                "--published-results",
                "--unit",
                "outputs2",
                "--published-figures",
                "--force",
                "--data-dir",
                str(tmp_path),
            ],
        )

    assert result.exit_code == 0
    path_type.assert_called_once_with(tmp_path)
    resolve_selection.assert_called_once_with(
        ppl=True,
        olmes=True,
        olmes_details=["fineweb-pro", "c4"],
        scaling_law=True,
        published_results=True,
        units=["outputs2"],
        published_figures=True,
    )
    download_sources.assert_called_once_with(
        paths,
        selection,
        force=True,
        verbose=True,
    )


def test_cli_default_is_repo_data_independent_of_cwd(
    tmp_path: Path, monkeypatch
) -> None:
    paths = object()
    monkeypatch.chdir(tmp_path)
    with (
        patch.object(script, "DataArtifacts", return_value=paths) as path_type,
        patch.object(script, "download_sources"),
    ):
        result = runner.invoke(app, ["--ppl"])

    assert result.exit_code == 0
    path_type.assert_called_once_with(DEFAULT_DATA_DIR)
    assert DEFAULT_DATA_DIR == Path(__file__).resolve().parents[1] / "data"


def test_cli_reports_unknown_detail_recipe_as_usage_error() -> None:
    with (
        patch.object(
            script,
            "resolve_selection",
            side_effect=ValueError("unknown OLMES detail recipe: missing"),
        ),
    ):
        result = runner.invoke(app, ["--olmes-details", "missing"])

    assert result.exit_code == 2
    assert "unknown OLMES detail recipe: missing" in result.output
