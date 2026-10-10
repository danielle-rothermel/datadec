from __future__ import annotations

from dr_hf import ModelPin

from datadec.models import (
    CheckpointCardFacts,
    ConversionRecord,
    DataDecideCheckpoint,
    FinalStepEvidence,
    choose_final_step,
    render_model_card,
)


def test_card_states_the_checkpoint_facts() -> None:
    checkpoint = DataDecideCheckpoint(
        recipe="dclm-baseline", size="150M", seed_ordinal=0, step=38157
    )
    final = choose_final_step(
        FinalStepEvidence(
            eval_results_last_step=37500,
            final_paths_step=38157,
            final_paths_commit="b" * 40,
            paper_training_steps=38157,
            paper_batch_size=192,
            paper_source="paper table",
            source_branch_steps=(37500, 38157),
        )
    )
    conversion = ConversionRecord(
        source_repo_id=checkpoint.source_repo_id,
        source_revision="a" * 40,
        source_model_type="hf_olmo",
        target_model_type="llama",
        target_architecture="LlamaForCausalLM",
        conversion_tool="datadec 0.1.1 datadec.models.convert",
        transformers_version="5.19.0",
        source_config={},
        target_config={},
        tensors=(),
        tokenizer_sha256={"tokenizer.json": "c" * 64},
        weights_sha256="d" * 64,
    )

    card = render_model_card(
        CheckpointCardFacts(
            checkpoint=checkpoint,
            source=ModelPin(repo_id=checkpoint.source_repo_id, revision="a" * 40),
            conversion=conversion,
            verification={"max_abs_logit_diff": 0.0},
            final=final,
            sequence_length=2048,
        )
    )

    assert "| Seed (authors' name and value) | default, 6198 |" in card
    assert "15,003,942,912 = 38157 steps x 192 sequences x 2048 tokens" in card
    assert "| Run completed | yes" in card
    assert "This commit carries the tag `final-seed0`." in card
    assert "- DataDecide-eval-results ends at step 37500" in card
    assert f"| `tokenizer.json` | `{'c' * 64}` |" in card
    assert '"max_abs_logit_diff": 0.0' in card
