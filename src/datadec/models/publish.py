"""Convert, verify, and publish one DataDecide checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from dr_hf import ModelPin, ModelProvenance, publish_model, resolve_model_pin

from datadec.config.checkpoints import load_checkpoint_contract
from datadec.models.card import CheckpointCardFacts, render_model_card
from datadec.models.convert import (
    ConversionRecord,
    convert_hf_olmo_to_native,
    download_source_checkpoint,
)
from datadec.models.final_step import (
    FinalStepDecision,
    choose_final_step,
    final_step_evidence,
)
from datadec.models.strict_check import run_strict_check

if TYPE_CHECKING:
    from pathlib import Path

    from datadec.data.artifacts import DataArtifacts
    from datadec.models.checkpoints import DataDecideCheckpoint

__all__ = ["PublishedCheckpoint", "convert_and_publish"]


@dataclass(frozen=True, slots=True)
class PublishedCheckpoint:
    pin: ModelPin
    tags: tuple[str, ...]
    source: ModelPin
    conversion: ConversionRecord
    verification: dict[str, object]
    final: FinalStepDecision


def convert_and_publish(
    checkpoint: DataDecideCheckpoint,
    *,
    work_dir: Path,
    artifacts: DataArtifacts,
    strict_check: bool = True,
) -> PublishedCheckpoint:
    """Resolve, convert, verify, and publish one checkpoint privately.

    The converted files are written to
    ``work_dir/<recipe>-<size>-step<N>-seed<k>`` (which must not exist or
    be empty). With ``strict_check=False`` the verification record states
    that the strict logit check was skipped. The final tag is applied only
    when the checkpoint step is the authors' final step.
    """
    contract = load_checkpoint_contract()
    source = resolve_model_pin(checkpoint.source_repo_id, ref=checkpoint.source_branch)
    final = choose_final_step(final_step_evidence(checkpoint, artifacts))
    source_dir = download_source_checkpoint(source.repo_id, source.revision)
    out_dir = work_dir / (
        f"{checkpoint.recipe}-{checkpoint.size}-step{checkpoint.step}-"
        f"seed{checkpoint.seed_ordinal}"
    )
    conversion = convert_hf_olmo_to_native(
        source.repo_id, source.revision, out_dir, source_dir=source_dir
    )
    verification: dict[str, object] = (
        run_strict_check(source_dir, out_dir).as_json()
        if strict_check
        else {"kind": "strict_logit_check", "skipped": True}
    )
    sequence_length = conversion.source_config["max_sequence_length"]
    if not isinstance(sequence_length, int):
        raise TypeError("source max_sequence_length must be an int")
    facts = CheckpointCardFacts(
        checkpoint=checkpoint,
        source=source,
        conversion=conversion,
        verification=verification,
        final=final,
        sequence_length=sequence_length,
    )
    tags = (checkpoint.final_tag,) if facts.is_final else ()
    pin = publish_model(
        out_dir,
        repo_id=checkpoint.target_repo_id,
        branch=checkpoint.target_branch,
        tags=tags,
        card_markdown=render_model_card(facts),
        provenance=ModelProvenance(
            source_repo_id=source.repo_id,
            source_revision=source.revision,
            conversion_tool=conversion.conversion_tool,
            verification=verification,
            notes=(
                f"hf_olmo -> {conversion.target_architecture}; "
                f"source branch {checkpoint.source_branch}"
            ),
        ),
        private=contract.target.private,
        commit_message=(
            f"Convert {source.repo_id}@{source.revision[:7]} "
            f"({checkpoint.source_branch}) to "
            f"{conversion.target_architecture}"
        ),
    )
    return PublishedCheckpoint(
        pin=pin,
        tags=tags,
        source=source,
        conversion=conversion,
        verification=verification,
        final=final,
    )
