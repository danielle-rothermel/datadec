from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
from unittest.mock import patch

from typer.testing import CliRunner

SCRIPT_PATH = Path(__file__).parents[1] / "scripts/publish.py"
SCRIPT_MODULE = "datadec_publish_script"
spec = importlib.util.spec_from_file_location(SCRIPT_MODULE, SCRIPT_PATH)
assert spec is not None and spec.loader is not None
script = importlib.util.module_from_spec(spec)
sys.modules[SCRIPT_MODULE] = script
spec.loader.exec_module(script)

app = script.app
runner = CliRunner()


def test_cli_requires_an_explicit_selection() -> None:
    result = runner.invoke(app, [])

    assert result.exit_code == 2
    assert "select --ppl, --olmes, --olmes-details" in result.output
    assert "--scaling-law" in result.output
    assert "--all" in result.output


def test_cli_forwards_repeatable_detail_selection(
    tmp_path: Path,
) -> None:
    paths = object()
    with (
        patch.object(script, "DataArtifacts", return_value=paths) as path_type,
        patch.object(script, "publish_existing_outputs", return_value=[]) as publish,
    ):
        result = runner.invoke(
            app,
            [
                "--ppl",
                "--olmes-details",
                "c4",
                "--olmes-details",
                "fineweb-pro",
                "--data-dir",
                str(tmp_path),
            ],
        )

    assert result.exit_code == 0
    path_type.assert_called_once_with(tmp_path)
    selection = publish.call_args.args[1]
    assert publish.call_args.args[0] is paths
    assert selection.ppl is True
    assert selection.olmes is False
    assert selection.olmes_details == ("c4", "fineweb-pro")
    assert selection.scaling_law is False
    assert selection.published_results == ()


def test_cli_all_expands_all_supported_existing_outputs(tmp_path: Path) -> None:
    result_item = SimpleNamespace(
        unit_name="ppl", created=False, commit_oid="commit-oid"
    )
    with patch.object(
        script, "publish_existing_outputs", return_value=[result_item]
    ) as publish:
        result = runner.invoke(app, ["--all", "--data-dir", str(tmp_path)])

    assert result.exit_code == 0
    publish.assert_called_once()
    selection = publish.call_args.args[1]
    assert selection.all_data is True
    assert selection.ppl is True
    assert selection.olmes is True
    assert selection.olmes_details
    assert selection.scaling_law is True
    assert selection.published_results
    assert result.output == "ppl: verified no-op at commit-oid\n"


def test_cli_selects_all_existing_published_results(tmp_path: Path) -> None:
    with patch.object(script, "publish_existing_outputs", return_value=[]) as publish:
        result = runner.invoke(
            app, ["--published-results", "--data-dir", str(tmp_path)]
        )

    assert result.exit_code == 0
    selection = publish.call_args.args[1]
    assert selection.ppl is False
    assert selection.olmes is False
    assert selection.olmes_details == ()
    assert selection.scaling_law is False
    assert selection.published_results


def test_cli_reports_unknown_recipe_as_usage_error() -> None:
    with patch.object(
        script,
        "resolve_selection",
        side_effect=ValueError("unknown OLMES detail recipe: missing"),
    ):
        result = runner.invoke(app, ["--olmes-details", "missing"])

    assert result.exit_code == 2
    assert "unknown OLMES detail recipe: missing" in result.output
