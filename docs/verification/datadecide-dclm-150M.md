# Verification: drotherm/DataDecide-dclm-baseline-150M

Run on 2026-10-08. Verdict: **accepted**.

The converted checkpoint was scored on ARC-Challenge with requests built by
`datadec.eval.olmes_rc` and compared with the DataDecide instance-level
results published for the original checkpoint. OLMES was not used.

## Inputs

| Input | Value |
| --- | --- |
| Dataset | `allenai/ai2_arc` config `ARC-Challenge` @ `210d026faf9955653af8916fad021475a3f00453` |
| Subset | `test@origin-210d026` (1172 items, content hash `a4a73a10e59eac54`) |
| Model | `drotherm/DataDecide-dclm-baseline-150M` @ `3b714eb60054c7b3628d66e38d8b22024b4a91c4` (resolved from `step38157-seed0`) |
| Provider config hash | `c7e8829ef017574a290569fc32a32d5cd20cec85281b0ec2b30418b16c1eb56f` |
| Scoring | dr-providers `LocalModelProvider`, device mps (arm64 macOS-26.5.2), float32, batch size 8, flat text, no BOS |
| Libraries | torch 2.14.1, transformers 5.19.0 |
| Published values | datadec `olmes-details` instances and choices, recipe `dclm-baseline`, params `150M`, seed `default`, step 38157, task `arc_challenge` |

Requests: one score request per context, the 5-shot prompt and the
unconditional `Answer:` context, 2344 requests in total.
Wall time for scoring: 93.2 s.

## Agreement per decision rule

| Rule | Predicted-index agreement | Accuracy (ours) | Accuracy (published) |
| --- | --- | --- | --- |
| raw | 1.0000 | 0.2253 | 0.2253 |
| per_token | 1.0000 | 0.2568 | 0.2568 |
| per_char | 1.0000 | 0.2628 | 0.2628 |
| per_byte | 1.0000 | 0.2628 | 0.2628 |
| pmi | 1.0000 | 0.2841 | 0.2841 |

The published per-byte prediction is derived from the published per-choice
`logits_per_byte`; the other published predictions are the authors' own
instance columns (`pmi` is OLMES `acc_uncond`).

## Per-choice values

| Quantity | Mean abs diff | Max abs diff |
| --- | --- | --- |
| `sum_logits` (5-shot context) | 0.000008 | 0.000046 |
| `sum_logits_uncond` (`Answer:` context) | 0.000007 | 0.000050 |

Items: 1172; choices: 4687.
Continuation token counts equal to published `num_tokens`:
1.0000 of choices. Items whose total
scored input tokens equal the published `num_tokens_all` (minus one per
choice): 1.0000. Provider conformance
warnings: 0.

## Acceptance

Thresholds: predicted-index agreement above 0.98 under PMI and per-char, mean absolute
`sum_logits` difference below 0.05 nats.

| Criterion | Result |
| --- | --- |
| pmi_agreement | pass |
| per_char_agreement | pass |
| mean_abs_sum_logits_diff | pass |

## First mismatching items

None.
