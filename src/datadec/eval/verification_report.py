from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from datadec.config.eval import load_eval_contract

if TYPE_CHECKING:
    from dr_hf import DatasetPin, ModelPin, NamedSubset

    from datadec.eval.verify_converted import Comparison, PublishedKey

__all__ = ["VerificationRun", "render_verification_report"]


@dataclass(frozen=True, slots=True)
class VerificationRun:
    dataset: DatasetPin
    subset: NamedSubset
    model: ModelPin
    model_ref: str
    config_hash: str
    device: str
    device_detail: str
    dtype: str
    batch_size: int
    torch_version: str
    transformers_version: str
    wall_time_seconds: float
    score_requests: int
    published: PublishedKey
    comparison: Comparison
    run_date: str


def render_verification_report(run: VerificationRun) -> str:
    comparison = run.comparison
    thresholds = load_eval_contract().verification.acceptance
    rule_rows = "\n".join(
        f"| {item.rule.value} | {item.agreement:.4f} | "
        f"{item.accuracy_ours:.4f} | {item.accuracy_published:.4f} |"
        for item in comparison.rules
    )
    mismatch_lines = "\n".join(
        f"- {item.rule.value}: " + "; ".join(item.first_mismatches)
        for item in comparison.rules
        if item.first_mismatches
    )
    acceptance_rows = "\n".join(
        f"| {name} | {'pass' if passed else 'FAIL'} |"
        for name, passed in comparison.acceptance.items()
    )
    published = run.published
    verdict = "accepted" if comparison.accepted else "NOT accepted"
    return f"""# Verification: {run.model.repo_id}

Run on {run.run_date}. Verdict: **{verdict}**.

The converted checkpoint was scored on ARC-Challenge with requests built by
`datadec.eval.olmes_rc` and compared with the DataDecide instance-level
results published for the original checkpoint. OLMES was not used.

## Inputs

| Input | Value |
| --- | --- |
| Dataset | `{run.dataset.repo_id}` config `{run.dataset.config}` @ \
`{run.dataset.revision}` |
| Subset | `{run.subset.key}` ({len(run.subset.native_ids)} items, content \
hash `{run.subset.content_hash[:16]}`) |
| Model | `{run.model.repo_id}` @ `{run.model.revision}` (resolved from \
`{run.model_ref}`) |
| Provider config hash | `{run.config_hash}` |
| Scoring | dr-providers `LocalModelProvider`, device {run.device} \
({run.device_detail}), {run.dtype}, batch size {run.batch_size}, flat text, \
no BOS |
| Libraries | torch {run.torch_version}, transformers \
{run.transformers_version} |
| Published values | datadec `olmes-details` instances and choices, recipe \
`{published.recipe}`, params `{published.params}`, seed `{published.seed}`, \
step {published.step}, task `{published.task}` |

Requests: one score request per context, the 5-shot prompt and the
unconditional `Answer:` context, {run.score_requests} requests in total.
Wall time for scoring: {run.wall_time_seconds:.1f} s.

## Agreement per decision rule

| Rule | Predicted-index agreement | Accuracy (ours) | Accuracy (published) |
| --- | --- | --- | --- |
{rule_rows}

The published per-byte prediction is derived from the published per-choice
`logits_per_byte`; the other published predictions are the authors' own
instance columns (`pmi` is OLMES `acc_uncond`).

## Per-choice values

| Quantity | Mean abs diff | Max abs diff |
| --- | --- | --- |
| `sum_logits` (5-shot context) | {comparison.sum_logits_mean_abs_diff:.6f} \
| {comparison.sum_logits_max_abs_diff:.6f} |
| `sum_logits_uncond` (`Answer:` context) | \
{comparison.sum_logits_uncond_mean_abs_diff:.6f} | \
{comparison.sum_logits_uncond_max_abs_diff:.6f} |

Items: {comparison.items}; choices: {comparison.choices}.
Continuation token counts equal to published `num_tokens`:
{comparison.continuation_token_agreement:.4f} of choices. Items whose total
scored input tokens equal the published `num_tokens_all` (minus one per
choice): {comparison.context_token_agreement:.4f}. Provider conformance
warnings: {comparison.warnings}.

## Acceptance

Thresholds: predicted-index agreement above \
{thresholds.min_predicted_index_agreement_pmi} under PMI and per-char, mean absolute
`sum_logits` difference below {thresholds.max_mean_abs_sum_logits_diff_nats} \
nats.

| Criterion | Result |
| --- | --- |
{acceptance_rows}

## First mismatching items

{mismatch_lines or "None."}
"""
