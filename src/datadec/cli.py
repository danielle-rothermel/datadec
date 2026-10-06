from __future__ import annotations

from pathlib import Path
from typing import Annotated, NoReturn

import typer

from datadec.data.artifacts import DataArtifacts
from datadec.data.cleanup import CleanupMode, CleanupResult, clean_data, cleanup_raw
from datadec.data.download import download_processed_outputs, download_sources
from datadec.data.pipeline import PipelineRequest, run_pipeline
from datadec.data.publish import publish_existing_outputs
from datadec.data.selection import DatasetSelection, resolve_selection

DEFAULT_DATA_DIR = Path("data")

app = typer.Typer(help="Manage DataDecide data artifacts.", no_args_is_help=True)


def _resolve_cli_selection(
    *,
    ppl: bool,
    olmes: bool,
    olmes_details: list[str] | None,
    scaling_law: bool,
    published_results: bool,
    units: list[str] | None,
    published_figures: bool = False,
    all_data: bool,
) -> DatasetSelection:
    try:
        return resolve_selection(
            ppl=ppl,
            olmes=olmes,
            olmes_details=olmes_details or (),
            scaling_law=scaling_law,
            published_results=published_results,
            units=units or (),
            published_figures=published_figures,
            all_data=all_data,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error


def _report_cleanup(result: CleanupResult) -> None:
    for path in result.removed_paths:
        typer.echo(f"removed: {path}")
    for path in result.would_remove_paths:
        typer.echo(f"would remove: {path}")
    for path in result.missing_paths:
        typer.echo(f"missing: {path}")


def _fail(error: Exception) -> NoReturn:
    typer.echo(f"Error: {error}", err=True)
    raise typer.Exit(code=1) from error


@app.command()
def run(
    data_dir: Annotated[Path, typer.Option("--data-dir")] = DEFAULT_DATA_DIR,
    ppl: Annotated[bool, typer.Option("--ppl")] = False,
    olmes: Annotated[bool, typer.Option("--olmes")] = False,
    olmes_details: Annotated[list[str] | None, typer.Option("--olmes-details")] = None,
    scaling_law: Annotated[bool, typer.Option("--scaling-law")] = False,
    published_results: Annotated[bool, typer.Option("--published-results")] = False,
    unit: Annotated[list[str] | None, typer.Option("--unit")] = None,
    all_data: Annotated[bool, typer.Option("--all")] = False,
    upload: Annotated[bool, typer.Option("--upload/--no-upload")] = True,
    cleanup: Annotated[CleanupMode | None, typer.Option("--cleanup")] = None,
    force: Annotated[bool, typer.Option("--force")] = False,
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
) -> None:
    """Download, process, verify, publish, and clean selected datasets."""
    selection = _resolve_cli_selection(
        ppl=ppl,
        olmes=olmes,
        olmes_details=olmes_details,
        scaling_law=scaling_law,
        published_results=published_results,
        units=unit,
        all_data=all_data,
    )
    if not upload and cleanup in {CleanupMode.RAW, CleanupMode.ALL}:
        raise typer.BadParameter(
            f"--cleanup {cleanup.value} requires upload; use --cleanup none with "
            "--no-upload"
        )

    request = PipelineRequest(
        artifacts=DataArtifacts(data_dir),
        selection=selection,
        upload=upload,
        cleanup=cleanup,
        force=force,
    )
    try:
        result = run_pipeline(request, verbose=verbose)
    except Exception as error:
        _fail(error)

    for check in result.verification.diagnostic_checks:
        typer.echo(f"diagnostic: {check.name}: {check.detail}")
    if result.cleanup is not None:
        _report_cleanup(result.cleanup)


@app.command()
def download(
    data_dir: Annotated[Path, typer.Option("--data-dir")] = DEFAULT_DATA_DIR,
    ppl: Annotated[bool, typer.Option("--ppl")] = False,
    olmes: Annotated[bool, typer.Option("--olmes")] = False,
    olmes_details: Annotated[list[str] | None, typer.Option("--olmes-details")] = None,
    scaling_law: Annotated[bool, typer.Option("--scaling-law")] = False,
    published_results: Annotated[bool, typer.Option("--published-results")] = False,
    unit: Annotated[list[str] | None, typer.Option("--unit")] = None,
    published_figures: Annotated[bool, typer.Option("--published-figures")] = False,
    all_data: Annotated[bool, typer.Option("--all")] = False,
    raw: Annotated[bool, typer.Option("--raw")] = False,
    force: Annotated[bool, typer.Option("--force")] = False,
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
) -> None:
    """Download selected processed outputs, or original sources with --raw."""
    selection = _resolve_cli_selection(
        ppl=ppl,
        olmes=olmes,
        olmes_details=olmes_details,
        scaling_law=scaling_law,
        published_results=published_results,
        units=unit,
        published_figures=published_figures,
        all_data=all_data,
    )
    if selection.published_figures and not raw:
        raise typer.BadParameter("--published-figures requires --raw")

    downloader = download_sources if raw else download_processed_outputs
    try:
        downloader(
            DataArtifacts(data_dir),
            selection,
            force=force,
            verbose=verbose,
        )
    except Exception as error:
        _fail(error)


@app.command()
def publish(
    data_dir: Annotated[Path, typer.Option("--data-dir")] = DEFAULT_DATA_DIR,
    ppl: Annotated[bool, typer.Option("--ppl")] = False,
    olmes: Annotated[bool, typer.Option("--olmes")] = False,
    olmes_details: Annotated[list[str] | None, typer.Option("--olmes-details")] = None,
    scaling_law: Annotated[bool, typer.Option("--scaling-law")] = False,
    published_results: Annotated[bool, typer.Option("--published-results")] = False,
    unit: Annotated[list[str] | None, typer.Option("--unit")] = None,
    all_data: Annotated[bool, typer.Option("--all")] = False,
) -> None:
    """Publish selected existing final outputs."""
    selection = _resolve_cli_selection(
        ppl=ppl,
        olmes=olmes,
        olmes_details=olmes_details,
        scaling_law=scaling_law,
        published_results=published_results,
        units=unit,
        all_data=all_data,
    )
    try:
        results = publish_existing_outputs(DataArtifacts(data_dir), selection)
    except Exception as error:
        _fail(error)

    for result in results:
        status = "created" if result.created else "verified no-op"
        typer.echo(f"{result.unit_name}: {status} at {result.commit_oid}")


@app.command("raw-clean")
def raw_clean(
    data_dir: Annotated[Path, typer.Option("--data-dir")] = DEFAULT_DATA_DIR,
    ppl: Annotated[bool, typer.Option("--ppl")] = False,
    olmes: Annotated[bool, typer.Option("--olmes")] = False,
    olmes_details: Annotated[list[str] | None, typer.Option("--olmes-details")] = None,
    scaling_law: Annotated[bool, typer.Option("--scaling-law")] = False,
    published_results: Annotated[bool, typer.Option("--published-results")] = False,
    unit: Annotated[list[str] | None, typer.Option("--unit")] = None,
    published_figures: Annotated[bool, typer.Option("--published-figures")] = False,
    all_data: Annotated[bool, typer.Option("--all")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Remove selected reproducible raw downloads."""
    selection = _resolve_cli_selection(
        ppl=ppl,
        olmes=olmes,
        olmes_details=olmes_details,
        scaling_law=scaling_law,
        published_results=published_results,
        units=unit,
        published_figures=published_figures,
        all_data=all_data,
    )
    try:
        result = cleanup_raw(DataArtifacts(data_dir), selection, dry_run=dry_run)
    except Exception as error:
        _fail(error)
    _report_cleanup(result)


@app.command()
def clean(
    data_dir: Annotated[Path, typer.Option("--data-dir")] = DEFAULT_DATA_DIR,
    ppl: Annotated[bool, typer.Option("--ppl")] = False,
    olmes: Annotated[bool, typer.Option("--olmes")] = False,
    olmes_details: Annotated[list[str] | None, typer.Option("--olmes-details")] = None,
    scaling_law: Annotated[bool, typer.Option("--scaling-law")] = False,
    published_results: Annotated[bool, typer.Option("--published-results")] = False,
    unit: Annotated[list[str] | None, typer.Option("--unit")] = None,
    published_figures: Annotated[bool, typer.Option("--published-figures")] = False,
    all_data: Annotated[bool, typer.Option("--all")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Verify and remove selected raw, intermediate, and final artifacts."""
    selection = _resolve_cli_selection(
        ppl=ppl,
        olmes=olmes,
        olmes_details=olmes_details,
        scaling_law=scaling_law,
        published_results=published_results,
        units=unit,
        published_figures=published_figures,
        all_data=all_data,
    )
    try:
        result = clean_data(DataArtifacts(data_dir), selection, dry_run=dry_run)
    except Exception as error:
        _fail(error)
    _report_cleanup(result)


__all__ = ["app"]
