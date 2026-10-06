from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from datadec.data.artifacts import DataArtifacts
from datadec.data.selection import DatasetSelection
from datadec.data.verify.derivations import DerivationVerificationResult
from datadec.data.verify.olmes_details import OlmesVerificationResult
from datadec.data.verify.selection import (
    VerificationOutcome,
    verify_selected_outputs,
)


def _selection(*, olmes: bool) -> DatasetSelection:
    return DatasetSelection(
        ppl=False,
        olmes=olmes,
        olmes_details=("c4",),
        scaling_law=False,
        published_results=(),
        published_figures=False,
        all_data=False,
    )


def _empty_derivations() -> DerivationVerificationResult:
    return DerivationVerificationResult((), None, None, ())


def test_detail_only_selection_skips_cross_source_inputs(tmp_path: Path) -> None:
    artifacts = DataArtifacts(tmp_path)
    detail_result = OlmesVerificationResult(None, None, 12)
    with (
        patch(
            "datadec.data.verify.selection.verify_preprocessed_derivations",
            return_value=_empty_derivations(),
        ),
        patch(
            "datadec.data.verify.selection.verify_olmes_details",
            return_value=detail_result,
        ) as verify_details,
    ):
        report = verify_selected_outputs(artifacts, _selection(olmes=False))

    verify_details.assert_called_once_with(
        recipe="c4",
        paths=artifacts,
        detail_archive=None,
        aggregate_path=None,
    )
    assert [check.outcome for check in report.checks] == [
        VerificationOutcome.PASSED,
        VerificationOutcome.SKIPPED,
    ]
    assert report.skipped_checks[0].name == "olmes-details:c4:aggregate-parity"


def test_selected_aggregate_enables_parity_for_selected_detail(tmp_path: Path) -> None:
    artifacts = DataArtifacts(tmp_path)
    detail_result = OlmesVerificationResult(3, 15, 12)
    with (
        patch(
            "datadec.data.verify.selection.verify_preprocessed_derivations",
            return_value=_empty_derivations(),
        ),
        patch(
            "datadec.data.verify.selection.verify_olmes_details",
            return_value=detail_result,
        ) as verify_details,
    ):
        report = verify_selected_outputs(artifacts, _selection(olmes=True))

    verify_details.assert_called_once_with(
        recipe="c4",
        paths=artifacts,
        detail_archive=artifacts.olmes_detail_source_path("c4"),
        aggregate_path=artifacts.get_path("olmes_processed"),
    )
    assert not report.skipped_checks
    assert report.passed_checks[-1].detail == (
        "verified 15 rows across 3 checkpoints"
    )
