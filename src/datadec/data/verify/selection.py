from __future__ import annotations

from dataclasses import dataclass
from enum import UNIQUE, StrEnum, auto, verify

from datadec.data.artifacts import DataArtifacts
from datadec.data.selection import DatasetSelection
from datadec.data.verify.derivations import (
    DerivationVerificationResult,
    verify_preprocessed_derivations,
)
from datadec.data.verify.olmes_details import (
    OlmesVerificationResult,
    verify_olmes_details,
)


@verify(UNIQUE)
class VerificationOutcome(StrEnum):
    PASSED = auto()
    SKIPPED = auto()
    DIAGNOSTIC = auto()


@dataclass(frozen=True, slots=True)
class VerificationCheck:
    name: str
    outcome: VerificationOutcome
    detail: str


@dataclass(frozen=True, slots=True)
class OlmesDetailVerification:
    recipe: str
    result: OlmesVerificationResult


@dataclass(frozen=True, slots=True)
class VerificationReport:
    checks: tuple[VerificationCheck, ...]
    derivations: DerivationVerificationResult
    olmes_details: tuple[OlmesDetailVerification, ...]

    @property
    def passed_checks(self) -> tuple[VerificationCheck, ...]:
        return tuple(
            check
            for check in self.checks
            if check.outcome is VerificationOutcome.PASSED
        )

    @property
    def skipped_checks(self) -> tuple[VerificationCheck, ...]:
        return tuple(
            check
            for check in self.checks
            if check.outcome is VerificationOutcome.SKIPPED
        )

    @property
    def diagnostic_checks(self) -> tuple[VerificationCheck, ...]:
        return tuple(
            check
            for check in self.checks
            if check.outcome is VerificationOutcome.DIAGNOSTIC
        )


def _require_no_contradictions(result: DerivationVerificationResult) -> None:
    contradictions = [
        (verification.name, verification.contradiction_count)
        for verification in result.processed_outputs
        if verification.contradiction_count
    ]
    if result.raw_olmes is not None and result.raw_olmes.contradiction_count:
        contradictions.append(
            (result.raw_olmes.name, result.raw_olmes.contradiction_count)
        )
    if result.raw_scaling_law is not None:
        raw = result.raw_scaling_law
        # Raw scaling-law compute records nominal parameter counts; processors
        # derive exact architectural compute for the supported output tables.
        invalid_inputs = raw.contradiction_count - raw.exact_compute_mismatch_count
        if invalid_inputs:
            contradictions.append(("raw scaling-law", invalid_inputs))
    contradictions.extend(
        (f"OLMES detail tasks {verification.path.parent.name}", count)
        for verification in result.detail_tasks
        if (count := verification.contradiction_count)
    )
    if contradictions:
        summary = ", ".join(f"{name}={count}" for name, count in contradictions)
        raise AssertionError(f"checkpoint derivation contradictions: {summary}")


def verify_selected_outputs(
    artifacts: DataArtifacts,
    selection: DatasetSelection,
) -> VerificationReport:
    """Verify only outputs and raw prerequisites named by ``selection``.

    Output mismatches raise immediately. Historical raw scaling-law compute
    differences are reported as diagnostics because processing corrects them.
    """
    derivations = verify_preprocessed_derivations(artifacts, selection)
    _require_no_contradictions(derivations)

    checks = [
        VerificationCheck(
            name=f"derivations:{verification.name}",
            outcome=VerificationOutcome.PASSED,
            detail=f"verified {verification.row_count} rows",
        )
        for verification in derivations.processed_outputs
    ]
    if derivations.raw_olmes is not None:
        checks.append(
            VerificationCheck(
                name="derivations:raw aggregate OLMES",
                outcome=VerificationOutcome.PASSED,
                detail=f"verified {derivations.raw_olmes.row_count} rows",
            )
        )
    if derivations.raw_scaling_law is not None:
        checks.append(
            VerificationCheck(
                name="derivations:raw scaling-law",
                outcome=(
                    VerificationOutcome.DIAGNOSTIC
                    if derivations.raw_scaling_law.exact_compute_mismatch_count
                    else VerificationOutcome.PASSED
                ),
                detail=(
                    f"checked {derivations.raw_scaling_law.row_count} raw rows; "
                    f"{derivations.raw_scaling_law.exact_compute_mismatch_count} "
                    "exact-parameter compute differences and "
                    f"{derivations.raw_scaling_law.nominal_compute_mismatch_count} "
                    "nominal-parameter compute differences; final outputs "
                    "are checked against exact architectural compute"
                ),
            )
        )

    detail_results: list[OlmesDetailVerification] = []
    aggregate_path = artifacts.get_path("olmes_processed") if selection.olmes else None
    for recipe in selection.olmes_details:
        result = verify_olmes_details(
            recipe=recipe,
            paths=artifacts,
            detail_archive=(
                artifacts.olmes_detail_source_path(recipe)
                if aggregate_path is not None
                else None
            ),
            aggregate_path=aggregate_path,
        )
        detail_results.append(OlmesDetailVerification(recipe, result))
        checks.append(
            VerificationCheck(
                name=f"olmes-details:{recipe}:reconstruction",
                outcome=VerificationOutcome.PASSED,
                detail=(
                    f"verified {result.reconstructed_task_count} reconstructed "
                    "task groups"
                ),
            )
        )
        if result.parity_row_count is None:
            checks.append(
                VerificationCheck(
                    name=f"olmes-details:{recipe}:aggregate-parity",
                    outcome=VerificationOutcome.SKIPPED,
                    detail="aggregate OLMES is outside the selected datasets",
                )
            )
        else:
            checks.append(
                VerificationCheck(
                    name=f"olmes-details:{recipe}:aggregate-parity",
                    outcome=VerificationOutcome.PASSED,
                    detail=(
                        f"verified {result.parity_row_count} rows across "
                        f"{result.overlapping_checkpoint_count} checkpoints"
                    ),
                )
            )

    return VerificationReport(tuple(checks), derivations, tuple(detail_results))


__all__ = [
    "OlmesDetailVerification",
    "VerificationCheck",
    "VerificationOutcome",
    "VerificationReport",
    "verify_selected_outputs",
]
