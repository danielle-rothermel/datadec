from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dr_hf import ModelPin

    from datadec.models.checkpoints import DataDecideCheckpoint
    from datadec.models.convert import ConversionRecord
    from datadec.models.final_step import FinalStepDecision

__all__ = ["CheckpointCardFacts", "render_model_card"]


@dataclass(frozen=True, slots=True)
class CheckpointCardFacts:
    checkpoint: DataDecideCheckpoint
    source: ModelPin
    conversion: ConversionRecord
    verification: dict[str, object]
    final: FinalStepDecision
    sequence_length: int

    @property
    def batch_size(self) -> int:
        return self.final.evidence.paper_batch_size

    @property
    def tokens_seen(self) -> int:
        return self.checkpoint.step * self.batch_size * self.sequence_length

    @property
    def is_final(self) -> bool:
        return self.checkpoint.step == self.final.step


def render_model_card(facts: CheckpointCardFacts) -> str:
    checkpoint = facts.checkpoint
    seed = checkpoint.seed
    evidence = facts.final.evidence
    conversion = facts.conversion
    tokenizer_rows = "\n".join(
        f"| `{name}` | `{digest}` |"
        for name, digest in sorted(conversion.tokenizer_sha256.items())
    )
    disagreements = (
        "\n".join(f"- {item}" for item in facts.final.disagreements)
        if facts.final.disagreements
        else "- none"
    )
    tag_line = (
        f"This commit carries the tag `{checkpoint.final_tag}`."
        if facts.is_final
        else f"The tag `{checkpoint.final_tag}` marks step {facts.final.step}."
    )
    run_completed = "yes" if facts.final.run_completed else "no"
    verification = json.dumps(facts.verification, indent=2, sort_keys=True)
    return f"""---
license: apache-2.0
language:
- en
base_model: {facts.source.repo_id}
tags:
- datadecide
- olmo
---

# DataDecide {checkpoint.recipe_display_name} {checkpoint.size}, \
step {checkpoint.step}, seed {seed.ordinal}

A conversion of the DataDecide checkpoint `{facts.source.repo_id}` at commit
`{facts.source.revision}` (branch `{checkpoint.source_branch}`) from the
`hf_olmo` model type to the native transformers class
`{conversion.target_architecture}`. It loads with `transformers` alone; no
`ai2-olmo` package or remote code is needed.

## Checkpoint

| Field | Value |
| --- | --- |
| Recipe | {checkpoint.recipe_display_name} (`{checkpoint.recipe}`) |
| Size | {checkpoint.size} |
| Seed ordinal | {seed.ordinal} |
| Seed (authors' name and value) | {seed.name}, {seed.value} |
| Step | {checkpoint.step} |
| Tokens seen | {facts.tokens_seen:,} = {checkpoint.step} steps x \
{facts.batch_size} sequences x {facts.sequence_length} tokens |
| Run completed | {run_completed} (paper lists \
{evidence.paper_training_steps} training steps for {checkpoint.size}) |
| Source | `{facts.source.repo_id}` @ `{facts.source.revision}` |
| Conversion tool | {conversion.conversion_tool} |
| Converted with transformers | {conversion.transformers_version} |

Batch size is the paper's sequences per batch for this size; sequence length
is the source checkpoint's `max_sequence_length`.

## Final step

The authors' final step for this run is {facts.final.step}, chosen from the
{facts.final.basis}. {tag_line}

| Source | Step |
| --- | --- |
| allenai/DataDecide `checkpoints/final_defualt_seed_paths.jsonl` @ \
`{evidence.final_paths_commit}` | {_or_absent(evidence.final_paths_step)} |
| {evidence.paper_source} | {evidence.paper_training_steps} |
| Last step in `allenai/DataDecide-eval-results` (datadec processed \
aggregate table) | {_or_absent(evidence.eval_results_last_step)} |

Sources that disagree with the choice:

{disagreements}

## Conversion

Every DataDecide size uses pre-norm blocks with RMSNorm carrying a learned
scale, SwiGLU, RoPE, no biases, no QKV clipping, and untied embeddings. That
is exactly the Llama layout; transformers' native `olmo` class uses a
non-parametric LayerNorm and cannot hold these weights. The fused attention
projection is split into query, key and value; the fused feed-forward
projection is split into up (first half) and gate (second half). The full
tensor and config mapping is in `conversion.json`.

The tokenizer files are copied verbatim from the source commit:

| File | sha256 |
| --- | --- |
{tokenizer_rows}

## Verification

```json
{verification}
```
"""


def _or_absent(value: int | None) -> str:
    return "absent" if value is None else str(value)
