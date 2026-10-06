from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import Mock

import pytest

from datadec.data import pipeline
from datadec.data.artifacts import DataArtifacts
from datadec.data.cleanup import CleanupMode, CleanupResult
from datadec.data.download import DownloadResult
from datadec.data.pipeline import PipelineRequest, run_pipeline
from datadec.data.publication import PublicationUnit
from datadec.data.publish import PublicationResult
from datadec.data.selection import DatasetSelection
from datadec.data.verify import (
    DerivationVerificationResult,
    VerificationCheck,
    VerificationOutcome,
    VerificationReport,
)


def _selection(
    *,
    ppl: bool = False,
    olmes: bool = False,
    olmes_details: tuple[str, ...] = (),
    scaling_law: bool = False,
    published_results: tuple[str, ...] = (),
    published_figures: bool = False,
) -> DatasetSelection:
    return DatasetSelection(
        ppl=ppl,
        olmes=olmes,
        olmes_details=olmes_details,
        scaling_law=scaling_law,
        published_results=published_results,  # type: ignore[arg-type]
        published_figures=published_figures,
        all_data=False,
    )


def _report(*checks: VerificationCheck) -> VerificationReport:
    return VerificationReport(
        checks=checks,
        derivations=DerivationVerificationResult((), None, None, ()),
        olmes_details=(),
    )


def _unit(name: str) -> PublicationUnit:
    return PublicationUnit(name=name, files=(), commit_message=f"Publish {name}")


def _publication(name: str) -> PublicationResult:
    return PublicationResult(
        unit_name=name,
        created=True,
        commit_oid=f"{name}-commit",
        remote_paths=(f"{name}.parquet",),
    )


def test_request_is_immutable(tmp_path: Path) -> None:
    request = PipelineRequest(DataArtifacts(tmp_path), _selection(ppl=True))

    with pytest.raises(FrozenInstanceError):
        request.upload = False  # type: ignore[misc]


def test_pipeline_orders_selected_stages_and_defaults_to_raw_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = DataArtifacts(tmp_path)
    selection = _selection(
        ppl=True,
        olmes=True,
        olmes_details=("c4",),
        scaling_law=True,
        published_results=("outputs2",),
    )
    events: list[str] = []
    downloads = (DownloadResult("source", tmp_path / "raw", "downloaded"),)
    units = (_unit("one"), _unit("two"))
    report = _report()
    cleanup = CleanupResult((tmp_path / "raw",), (), ())

    monkeypatch.setattr(
        pipeline,
        "download_sources",
        lambda *_args, **_kwargs: events.append("download") or downloads,
    )
    monkeypatch.setattr(
        pipeline,
        "preprocess_ppl",
        lambda *_args, **_kwargs: events.append("process:ppl"),
    )
    monkeypatch.setattr(
        pipeline,
        "preprocess_olmes",
        lambda *_args, **_kwargs: events.append("process:olmes"),
    )
    monkeypatch.setattr(
        pipeline,
        "preprocess_olmes_details",
        lambda _artifacts, recipe, **_kwargs: events.append(
            f"process:olmes-details:{recipe}"
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "preprocess_scaling_law",
        lambda *_args, **_kwargs: events.append("process:scaling-law"),
    )
    monkeypatch.setattr(
        pipeline,
        "preprocess_published_results",
        lambda *_args, **_kwargs: events.append("process:published-results"),
    )
    monkeypatch.setattr(
        pipeline,
        "publication_units",
        lambda *_args: events.append("construct-units") or units,
    )
    monkeypatch.setattr(
        pipeline,
        "validate_publication_unit",
        lambda unit: events.append(f"validate:{unit.name}"),
    )
    monkeypatch.setattr(
        pipeline,
        "verify_selected_outputs",
        lambda *_args: events.append("verify") or report,
    )
    monkeypatch.setattr(
        pipeline,
        "publish_unit",
        lambda unit, **_kwargs: events.append(f"publish:{unit.name}")
        or _publication(unit.name),
    )
    monkeypatch.setattr(
        pipeline,
        "cleanup_raw",
        lambda *_args: events.append("cleanup:raw") or cleanup,
    )

    result = run_pipeline(
        PipelineRequest(artifacts, selection), hf_token="token", verbose=False
    )

    assert events == [
        "download",
        "process:ppl",
        "process:olmes",
        "process:olmes-details:c4",
        "process:scaling-law",
        "process:published-results",
        "construct-units",
        "validate:one",
        "validate:two",
        "verify",
        "publish:one",
        "publish:two",
        "cleanup:raw",
    ]
    assert result.downloads == downloads
    assert result.verification is report
    assert result.publications == (_publication("one"), _publication("two"))
    assert result.cleanup is cleanup


def test_all_publication_units_validate_before_any_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    units = (_unit("valid"), _unit("invalid"))
    publish = Mock()
    cleanup = Mock()
    monkeypatch.setattr(pipeline, "download_sources", Mock(return_value=()))
    monkeypatch.setattr(pipeline, "preprocess_ppl", Mock())
    monkeypatch.setattr(pipeline, "publication_units", Mock(return_value=units))
    monkeypatch.setattr(
        pipeline,
        "validate_publication_unit",
        Mock(side_effect=(None, ValueError("invalid schema"))),
    )
    monkeypatch.setattr(pipeline, "verify_selected_outputs", Mock())
    monkeypatch.setattr(pipeline, "publish_unit", publish)
    monkeypatch.setattr(pipeline, "cleanup_raw", cleanup)

    with pytest.raises(ValueError, match="invalid schema"):
        run_pipeline(PipelineRequest(DataArtifacts(tmp_path), _selection(ppl=True)))

    publish.assert_not_called()
    cleanup.assert_not_called()


@pytest.mark.parametrize("failed_stage", ["download", "process", "verify"])
def test_failures_retain_local_artifacts_and_skip_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_stage: str,
) -> None:
    artifacts = DataArtifacts(tmp_path)
    retained = tmp_path / "retained.parquet"
    retained.write_text("retry input")
    failure = RuntimeError(f"{failed_stage} failed")
    download = Mock(return_value=())
    process = Mock()
    verify = Mock(return_value=_report())
    if failed_stage == "download":
        download.side_effect = failure
    elif failed_stage == "process":
        process.side_effect = failure
    else:
        verify.side_effect = failure
    cleanup = Mock()

    monkeypatch.setattr(pipeline, "download_sources", download)
    monkeypatch.setattr(pipeline, "preprocess_ppl", process)
    monkeypatch.setattr(pipeline, "publication_units", Mock(return_value=(_unit("ppl"),)))
    monkeypatch.setattr(pipeline, "validate_publication_unit", Mock())
    monkeypatch.setattr(pipeline, "verify_selected_outputs", verify)
    monkeypatch.setattr(pipeline, "publish_unit", Mock(return_value=_publication("ppl")))
    monkeypatch.setattr(pipeline, "cleanup_raw", cleanup)

    with pytest.raises(RuntimeError, match=f"{failed_stage} failed"):
        run_pipeline(PipelineRequest(artifacts, _selection(ppl=True)))

    assert retained.read_text() == "retry input"
    cleanup.assert_not_called()


def test_partial_upload_failure_retains_data_and_skips_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    retained = tmp_path / "output.parquet"
    retained.write_text("retry output")
    units = (_unit("first"), _unit("second"))
    publish = Mock(
        side_effect=(_publication("first"), RuntimeError("second upload failed"))
    )
    cleanup = Mock()
    monkeypatch.setattr(pipeline, "download_sources", Mock(return_value=()))
    monkeypatch.setattr(pipeline, "preprocess_ppl", Mock())
    monkeypatch.setattr(pipeline, "publication_units", Mock(return_value=units))
    monkeypatch.setattr(pipeline, "validate_publication_unit", Mock())
    monkeypatch.setattr(
        pipeline, "verify_selected_outputs", Mock(return_value=_report())
    )
    monkeypatch.setattr(pipeline, "publish_unit", publish)
    monkeypatch.setattr(pipeline, "cleanup_raw", cleanup)

    with pytest.raises(RuntimeError, match="second upload failed"):
        run_pipeline(
            PipelineRequest(DataArtifacts(tmp_path), _selection(ppl=True))
        )

    assert publish.call_count == 2
    assert retained.read_text() == "retry output"
    cleanup.assert_not_called()


def test_no_upload_defaults_to_no_cleanup_but_still_validates_locally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unit = _unit("ppl")
    report = _report()
    validate = Mock()
    verify = Mock(return_value=report)
    publish = Mock()
    cleanup_raw = Mock()
    clean_data = Mock()
    monkeypatch.setattr(pipeline, "download_sources", Mock(return_value=()))
    monkeypatch.setattr(pipeline, "preprocess_ppl", Mock())
    monkeypatch.setattr(pipeline, "publication_units", Mock(return_value=(unit,)))
    monkeypatch.setattr(pipeline, "validate_publication_unit", validate)
    monkeypatch.setattr(pipeline, "verify_selected_outputs", verify)
    monkeypatch.setattr(pipeline, "publish_unit", publish)
    monkeypatch.setattr(pipeline, "cleanup_raw", cleanup_raw)
    monkeypatch.setattr(pipeline, "clean_data", clean_data)

    result = run_pipeline(
        PipelineRequest(
            DataArtifacts(tmp_path),
            _selection(ppl=True),
            upload=False,
        )
    )

    validate.assert_called_once_with(unit)
    verify.assert_called_once()
    publish.assert_not_called()
    cleanup_raw.assert_not_called()
    clean_data.assert_not_called()
    assert result.publications == ()
    assert result.cleanup is None


@pytest.mark.parametrize("cleanup_mode", [CleanupMode.RAW, CleanupMode.ALL])
def test_no_upload_rejects_cleanup_before_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup_mode: CleanupMode,
) -> None:
    download = Mock()
    monkeypatch.setattr(pipeline, "download_sources", download)

    with pytest.raises(ValueError, match="cleanup requires upload"):
        run_pipeline(
            PipelineRequest(
                DataArtifacts(tmp_path),
                _selection(ppl=True),
                upload=False,
                cleanup=cleanup_mode,
            )
        )

    download.assert_not_called()


def test_published_figures_are_rejected_before_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    download = Mock()
    monkeypatch.setattr(pipeline, "download_sources", download)

    with pytest.raises(ValueError, match="published figures are raw-only"):
        run_pipeline(
            PipelineRequest(DataArtifacts(tmp_path), _selection(published_figures=True))
        )

    download.assert_not_called()


def test_base_only_selection_never_processes_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preprocess_olmes = Mock()
    preprocess_details = Mock()
    monkeypatch.setattr(pipeline, "download_sources", Mock(return_value=()))
    monkeypatch.setattr(pipeline, "preprocess_olmes", preprocess_olmes)
    monkeypatch.setattr(pipeline, "preprocess_olmes_details", preprocess_details)
    monkeypatch.setattr(pipeline, "publication_units", Mock(return_value=()))
    monkeypatch.setattr(pipeline, "verify_selected_outputs", Mock(return_value=_report()))

    run_pipeline(
        PipelineRequest(
            DataArtifacts(tmp_path),
            _selection(olmes=True),
            upload=False,
        )
    )

    preprocess_olmes.assert_called_once()
    preprocess_details.assert_not_called()


@pytest.mark.parametrize(
    ("cleanup_mode", "expected"),
    [(CleanupMode.ALL, "all"), (CleanupMode.NONE, None)],
)
def test_explicit_cleanup_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup_mode: CleanupMode,
    expected: str | None,
) -> None:
    artifacts = DataArtifacts(tmp_path)
    selection = _selection(ppl=True)
    raw_result = CleanupResult((), (), ())
    all_result = CleanupResult((tmp_path / "processed"), (), ())
    cleanup_raw = Mock(return_value=raw_result)
    clean_data = Mock(return_value=all_result)
    monkeypatch.setattr(pipeline, "download_sources", Mock(return_value=()))
    monkeypatch.setattr(pipeline, "preprocess_ppl", Mock())
    monkeypatch.setattr(pipeline, "publication_units", Mock(return_value=()))
    monkeypatch.setattr(pipeline, "verify_selected_outputs", Mock(return_value=_report()))
    monkeypatch.setattr(pipeline, "cleanup_raw", cleanup_raw)
    monkeypatch.setattr(pipeline, "clean_data", clean_data)

    result = run_pipeline(
        PipelineRequest(
            artifacts,
            selection,
            cleanup=cleanup_mode,
        ),
        hf_token="token",
    )

    cleanup_raw.assert_not_called()
    if expected == "all":
        clean_data.assert_called_once_with(
            artifacts,
            selection,
            hf_token="token",
        )
        assert result.cleanup is all_result
    else:
        clean_data.assert_not_called()
        assert result.cleanup is None


def test_verbose_output_reports_diagnostics_without_blocking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = _report(
        VerificationCheck(
            "derivations:raw scaling-law",
            VerificationOutcome.DIAGNOSTIC,
            "489258 exact-parameter compute differences",
        )
    )
    monkeypatch.setattr(pipeline, "download_sources", Mock(return_value=()))
    monkeypatch.setattr(pipeline, "preprocess_scaling_law", Mock())
    monkeypatch.setattr(pipeline, "publication_units", Mock(return_value=()))
    monkeypatch.setattr(
        pipeline, "verify_selected_outputs", Mock(return_value=report)
    )

    result = run_pipeline(
        PipelineRequest(
            DataArtifacts(tmp_path),
            _selection(scaling_law=True),
            upload=False,
        ),
        verbose=True,
    )

    output = capsys.readouterr().out
    assert "verification diagnostic: derivations:raw scaling-law" in output
    assert "489258 exact-parameter compute differences" in output
    assert result.verification.diagnostic_checks == report.checks
