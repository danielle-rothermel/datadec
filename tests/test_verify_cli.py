import runpy
from pathlib import Path
from unittest.mock import Mock

import pytest
from click.testing import Result
from typer.testing import CliRunner

from datadec.data import verify
from datadec.data.verify import (
    DerivationVerificationResult,
    VerificationCheck,
    VerificationOutcome,
    VerificationReport,
)


def _invoke(monkeypatch: pytest.MonkeyPatch, verifier: Mock) -> Result:
    monkeypatch.setattr(verify, "verify_selected_outputs", verifier)
    namespace = runpy.run_path(
        str(Path(__file__).parents[1] / "scripts/verify_preprocessed_derivations.py")
    )
    return CliRunner().invoke(namespace["app"], [])


def test_manual_verifier_reports_raw_compute_diagnostics_without_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = VerificationReport(
        checks=(
            VerificationCheck(
                "raw scaling-law",
                VerificationOutcome.DIAGNOSTIC,
                "489258 exact-compute differences; nominal compute matches",
            ),
        ),
        derivations=DerivationVerificationResult((), None, None, ()),
        olmes_details=(),
    )
    verifier = Mock(return_value=report)
    result = _invoke(monkeypatch, verifier)
    assert result.exit_code == 0
    assert "diagnostic: raw scaling-law" in result.output
    selection = verifier.call_args.args[1]
    assert selection.ppl and selection.olmes and selection.scaling_law
    assert not selection.olmes_details


def test_manual_verifier_fails_on_generated_output_contradictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _invoke(
        monkeypatch,
        Mock(side_effect=AssertionError("checkpoint derivation contradictions")),
    )
    assert result.exit_code == 1
    assert "verification failed: checkpoint derivation contradictions" in result.output
