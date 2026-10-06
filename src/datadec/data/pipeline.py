from __future__ import annotations

from dataclasses import dataclass

from datadec.data.artifacts import DataArtifacts
from datadec.data.cleanup import CleanupMode, CleanupResult, clean_data, cleanup_raw
from datadec.data.download import DownloadResult, download_sources
from datadec.data.preprocess import (
    preprocess_olmes,
    preprocess_olmes_details,
    preprocess_ppl,
    preprocess_published_results,
    preprocess_scaling_law,
)
from datadec.data.publication import publication_units
from datadec.data.publish import (
    PublicationResult,
    publish_unit,
    validate_publication_unit,
)
from datadec.data.selection import DatasetSelection
from datadec.data.verify import VerificationReport, verify_selected_outputs


@dataclass(frozen=True, slots=True)
class PipelineRequest:
    artifacts: DataArtifacts
    selection: DatasetSelection
    upload: bool = True
    cleanup: CleanupMode | None = None
    force: bool = False


@dataclass(frozen=True, slots=True)
class PipelineResult:
    downloads: tuple[DownloadResult, ...]
    verification: VerificationReport
    publications: tuple[PublicationResult, ...]
    cleanup: CleanupResult | None


def _cleanup_mode(request: PipelineRequest) -> CleanupMode:
    cleanup = request.cleanup
    if cleanup is None:
        return CleanupMode.DEFAULT if request.upload else CleanupMode.NONE
    if not request.upload and cleanup is not CleanupMode.NONE:
        raise ValueError("cleanup requires upload; use cleanup='none' with no-upload runs")
    return cleanup


def _process_selected(request: PipelineRequest, *, verbose: bool) -> None:
    artifacts = request.artifacts
    selection = request.selection
    if selection.ppl:
        preprocess_ppl(artifacts, verbose=verbose)
    if selection.olmes:
        preprocess_olmes(artifacts, verbose=verbose)
    for recipe in selection.olmes_details:
        preprocess_olmes_details(artifacts, recipe, verbose=verbose)
    if selection.scaling_law:
        preprocess_scaling_law(artifacts, verbose=verbose)
    if selection.published_results:
        preprocess_published_results(
            artifacts,
            units=selection.published_results,
            verbose=verbose,
        )


def _print_verification(report: VerificationReport) -> None:
    for check in report.checks:
        print(f"verification {check.outcome.value}: {check.name} — {check.detail}")


def run_pipeline(
    request: PipelineRequest,
    *,
    hf_token: str | None = None,
    verbose: bool = False,
) -> PipelineResult:
    """Run the selected local processing lifecycle and optional publication."""
    cleanup_mode = _cleanup_mode(request)
    if request.selection.published_figures:
        raise ValueError("published figures are raw-only and cannot be run or published")

    if verbose:
        print("pipeline stage: download raw sources")
    downloads = download_sources(
        request.artifacts,
        request.selection,
        force=request.force,
        verbose=verbose,
    )

    if verbose:
        print("pipeline stage: process selected datasets")
    _process_selected(request, verbose=verbose)

    units = publication_units(request.artifacts, request.selection)
    if verbose:
        print("pipeline stage: validate publication schemas")
    for unit in units:
        validate_publication_unit(unit)

    if verbose:
        print("pipeline stage: verify selected outputs")
    verification = verify_selected_outputs(request.artifacts, request.selection)
    if verbose:
        _print_verification(verification)

    publications: list[PublicationResult] = []
    if request.upload:
        if verbose:
            print("pipeline stage: publish and verify immutable outputs")
        for unit in units:
            result = publish_unit(unit, hf_token=hf_token)
            publications.append(result)
            if verbose:
                print(
                    f"published {result.unit_name} at immutable commit "
                    f"{result.commit_oid}"
                )

    cleanup_result: CleanupResult | None = None
    if cleanup_mode is not CleanupMode.NONE:
        if verbose:
            print(f"pipeline stage: cleanup {cleanup_mode.value}")
        if cleanup_mode is CleanupMode.RAW:
            cleanup_result = cleanup_raw(request.artifacts, request.selection)
        else:
            cleanup_result = clean_data(
                request.artifacts,
                request.selection,
                mode=cleanup_mode,
                hf_token=hf_token,
            )
        if verbose:
            print(f"cleanup removed {len(cleanup_result.removed_paths)} paths")

    return PipelineResult(
        downloads=downloads,
        verification=verification,
        publications=tuple(publications),
        cleanup=cleanup_result,
    )


__all__ = ["PipelineRequest", "PipelineResult", "run_pipeline"]
