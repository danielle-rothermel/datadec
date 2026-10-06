from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from datadec.cli import app
from datadec.data.cleanup import CleanupMode, CleanupResult

runner = CliRunner()


def _pipeline_result(
    *,
    diagnostics: tuple[object, ...] = (),
    cleanup: CleanupResult | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        verification=SimpleNamespace(diagnostic_checks=diagnostics),
        cleanup=cleanup,
    )


def test_cli_exposes_the_consolidated_commands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("run", "download", "publish", "raw-clean", "clean"):
        assert command in result.output


@pytest.mark.parametrize(
    "command", ("run", "download", "publish", "raw-clean", "clean")
)
def test_commands_require_an_explicit_selection(command: str) -> None:
    result = runner.invoke(app, [command])

    assert result.exit_code == 2
    assert "select at least one dataset" in result.output


def test_run_dispatches_typed_request_with_cli_defaults(tmp_path: Path) -> None:
    with patch(
        "datadec.cli.run_pipeline", return_value=_pipeline_result()
    ) as run_pipeline:
        result = runner.invoke(
            app,
            [
                "run",
                "--ppl",
                "--data-dir",
                str(tmp_path),
                "--force",
                "--verbose",
            ],
        )

    assert result.exit_code == 0
    request = run_pipeline.call_args.args[0]
    assert request.artifacts.data_dir == tmp_path
    assert request.selection.ppl is True
    assert request.selection.olmes is False
    assert request.upload is True
    assert request.cleanup is None
    assert request.force is True
    assert run_pipeline.call_args.kwargs == {"verbose": True}


def test_run_preserves_repeatable_selector_semantics(tmp_path: Path) -> None:
    with patch(
        "datadec.cli.run_pipeline", return_value=_pipeline_result()
    ) as run_pipeline:
        result = runner.invoke(
            app,
            [
                "run",
                "--olmes-details",
                "fineweb-pro",
                "--olmes-details",
                "c4",
                "--olmes-details",
                "fineweb-pro",
                "--unit",
                "per-task-arc-easy",
                "--unit",
                "outputs2",
                "--data-dir",
                str(tmp_path),
            ],
        )

    assert result.exit_code == 0
    selection = run_pipeline.call_args.args[0].selection
    assert selection.ppl is False
    assert selection.olmes is False
    assert selection.olmes_details == ("fineweb-pro", "c4")
    assert selection.scaling_law is False
    assert selection.published_results == ("outputs2", "per-task-arc-easy")


def test_run_no_upload_preserves_implicit_no_cleanup_policy() -> None:
    with patch(
        "datadec.cli.run_pipeline", return_value=_pipeline_result()
    ) as run_pipeline:
        result = runner.invoke(app, ["run", "--ppl", "--no-upload"])

    assert result.exit_code == 0
    request = run_pipeline.call_args.args[0]
    assert request.artifacts.data_dir == Path("data")
    assert request.upload is False
    assert request.cleanup is None


@pytest.mark.parametrize("cleanup", ("raw", "all"))
def test_run_rejects_cleanup_that_contradicts_no_upload(cleanup: str) -> None:
    with patch("datadec.cli.run_pipeline") as run_pipeline:
        result = runner.invoke(
            app,
            ["run", "--ppl", "--no-upload", "--cleanup", cleanup],
        )

    assert result.exit_code == 2
    assert f"--cleanup {cleanup} requires upload" in result.output
    run_pipeline.assert_not_called()


def test_run_reports_pipeline_diagnostics_and_cleanup_paths(tmp_path: Path) -> None:
    removed = tmp_path / "raw.csv"
    missing = tmp_path / "raw.csv.part"
    diagnostic = SimpleNamespace(
        name="derivations:raw scaling-law",
        detail="checked 12 raw rows; 3 exact-parameter compute differences",
    )
    pipeline_result = _pipeline_result(
        diagnostics=(diagnostic,),
        cleanup=CleanupResult(
            removed_paths=(removed,),
            would_remove_paths=(),
            missing_paths=(missing,),
        ),
    )
    with patch("datadec.cli.run_pipeline", return_value=pipeline_result):
        result = runner.invoke(app, ["run", "--scaling-law"])

    assert result.exit_code == 0
    assert result.output == (
        "diagnostic: derivations:raw scaling-law: checked 12 raw rows; "
        "3 exact-parameter compute differences\n"
        f"removed: {removed}\n"
        f"missing: {missing}\n"
    )


def test_download_defaults_to_processed_outputs(tmp_path: Path) -> None:
    with (
        patch("datadec.cli.download_processed_outputs") as download_processed,
        patch("datadec.cli.download_sources") as download_raw,
    ):
        result = runner.invoke(
            app,
            [
                "download",
                "--ppl",
                "--force",
                "--verbose",
                "--data-dir",
                str(tmp_path),
            ],
        )

    assert result.exit_code == 0
    artifacts, selection = download_processed.call_args.args
    assert artifacts.data_dir == tmp_path
    assert selection.ppl is True
    assert download_processed.call_args.kwargs == {"force": True, "verbose": True}
    download_raw.assert_not_called()


def test_download_raw_supports_published_figures(tmp_path: Path) -> None:
    with (
        patch("datadec.cli.download_sources") as download_raw,
        patch("datadec.cli.download_processed_outputs") as download_processed,
    ):
        result = runner.invoke(
            app,
            [
                "download",
                "--published-figures",
                "--raw",
                "--data-dir",
                str(tmp_path),
            ],
        )

    assert result.exit_code == 0
    selection = download_raw.call_args.args[1]
    assert selection.published_figures is True
    assert selection.published_results == ()
    download_processed.assert_not_called()


def test_download_rejects_processed_published_figures_before_dispatch() -> None:
    with (
        patch("datadec.cli.download_sources") as download_raw,
        patch("datadec.cli.download_processed_outputs") as download_processed,
    ):
        result = runner.invoke(app, ["download", "--published-figures"])

    assert result.exit_code == 2
    assert "--published-figures requires --raw" in result.output
    download_raw.assert_not_called()
    download_processed.assert_not_called()


def test_publish_dispatches_selected_units_and_reports_results(tmp_path: Path) -> None:
    publication = SimpleNamespace(
        unit_name="published-results:outputs2",
        created=False,
        commit_oid="abc123",
    )
    with patch(
        "datadec.cli.publish_existing_outputs", return_value=(publication,)
    ) as publish:
        result = runner.invoke(
            app,
            [
                "publish",
                "--unit",
                "outputs2",
                "--data-dir",
                str(tmp_path),
            ],
        )

    assert result.exit_code == 0
    artifacts, selection = publish.call_args.args
    assert artifacts.data_dir == tmp_path
    assert selection.published_results == ("outputs2",)
    assert result.output == ("published-results:outputs2: verified no-op at abc123\n")


def test_raw_clean_dry_run_reports_exact_shared_result(tmp_path: Path) -> None:
    would_remove = tmp_path / "figure.pdf"
    missing = tmp_path / "figure.pdf.part"
    cleanup_result = CleanupResult(
        removed_paths=(),
        would_remove_paths=(would_remove,),
        missing_paths=(missing,),
    )
    with patch("datadec.cli.cleanup_raw", return_value=cleanup_result) as clean_raw:
        result = runner.invoke(
            app,
            [
                "raw-clean",
                "--published-figures",
                "--dry-run",
                "--data-dir",
                str(tmp_path),
            ],
        )

    assert result.exit_code == 0
    artifacts, selection = clean_raw.call_args.args
    assert artifacts.data_dir == tmp_path
    assert selection.published_figures is True
    assert clean_raw.call_args.kwargs == {"dry_run": True}
    assert result.output == (f"would remove: {would_remove}\nmissing: {missing}\n")


def test_clean_forwards_all_and_dry_run(tmp_path: Path) -> None:
    cleanup_result = CleanupResult((), (), ())
    with patch("datadec.cli.clean_data", return_value=cleanup_result) as clean_data:
        result = runner.invoke(
            app,
            ["clean", "--all", "--dry-run", "--data-dir", str(tmp_path)],
        )

    assert result.exit_code == 0
    artifacts, selection = clean_data.call_args.args
    assert artifacts.data_dir == tmp_path
    assert selection.all_data is True
    assert selection.ppl is True
    assert selection.olmes is True
    assert selection.published_figures is False
    assert clean_data.call_args.kwargs == {"dry_run": True}


def test_operation_failure_is_reported_without_a_traceback() -> None:
    with patch(
        "datadec.cli.download_processed_outputs",
        side_effect=RuntimeError("remote commit is unavailable"),
    ):
        result = runner.invoke(app, ["download", "--ppl"])

    assert result.exit_code == 1
    assert "Error: remote commit is unavailable" in result.output
    assert "Traceback" not in result.output


def test_unknown_selector_is_a_usage_error() -> None:
    result = runner.invoke(app, ["download", "--olmes-details", "missing", "--raw"])

    assert result.exit_code == 2
    assert "unknown OLMES detail recipe: missing" in result.output


def test_cleanup_mode_values_are_accepted_by_run() -> None:
    with patch(
        "datadec.cli.run_pipeline", return_value=_pipeline_result()
    ) as run_pipeline:
        result = runner.invoke(app, ["run", "--ppl", "--cleanup", "none"])

    assert result.exit_code == 0
    assert run_pipeline.call_args.args[0].cleanup is CleanupMode.NONE
