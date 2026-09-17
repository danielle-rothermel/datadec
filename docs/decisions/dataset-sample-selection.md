# Dataset sample selection

Which items we evaluate on, how many, and how they are chosen. Log, most recent first.

## 2026-09-17 10:15 — Decision: 300M noise checkpoints run before the Qwen full passes

Insert six extra 300M full-set runs (canonical prompt, RC+MC) into the queue ahead of
Qwen: seed-default steps 40000, 41250, 42500, 43750 (45000 already run) and the final
checkpoints of seeds small-aux-2 and small-aux-3 (step 127166 on HF). Purpose: a
per-item noise term (checkpoint sd, seed sd) to rank items by signal-to-noise, after
Heineman 2025. The last five default checkpoints span the final 11% of training with no
monotone trend (aggregate acc_per_char range 0.002; per-char likelihood sd 0.0002), so
they measure jitter rather than learning. Analysis script: `scripts/po_item_noise.py`.

## 2026-09-17 10:10 — Direction: cross-model item prior from clean baselines

With only the clean full-set baselines, build a per-item table across models
(150M, 300M, 1B, Qwen base, Qwen instruct) and drop items that stay at chance or at
certainty across the whole size/family span; the middle band is the candidate pool. This
is a filter and a difficulty stratification, not a ranking of prompt-discriminative
items, because model variation is only a proxy for prompt variation. Testing that link
needs a prompt-by-item matrix (see 10:00).

## 2026-09-17 10:05 — Decision: "what we have" means the clean full-set baselines only

Everything produced by the 2026-09-16 pipeline (seed sweeps, GEPA runs, their per-item
tables) is contaminated or potentially contaminated by the instruction-pool and
format-conditioning mistakes and is not used for design decisions. The usable evidence
is the OLMES-default full ARC-Easy test-set runs (150M, 300M, 1B done; Qwen queued) and
the 100-item baselines moved to `baselines/`.

## 2026-09-17 10:00 — Direction: p1 / Signal-and-Noise item selection, minimum setup

Our scoring is deterministic, so p1's response-noise term is zero and its filter reduces
to between-prompt variance per item; the only sampling noise is item sampling. Minimum
design for prompt-discriminative item selection: a matrix of ~16 instructions x 300-500
pool items (outside the held-out 100 and the fewshot ids) at a fixed format, one sample
per cell, validated on instructions not used for selection; the same matrix with
formats varying for the format axis. Not started.

## 2026-09-17 09:45 — Decision: evaluate the full ARC-Easy test set for the baselines

The 100-item seed-0 subset is 4.2% of ARC-Easy test (8.5% of ARC-Challenge). Run 300M,
then 1B, Qwen3-1.7B-Base and Qwen3-1.7B on all 2376 items (RC+MC, OLMES defaults) and
subsample at n = 25 ... 2376 to show what a random n-item subset reports, with paired
power for RC-vs-MC and model-vs-model. ARC-Easy only for now. Result:
`../results/2026-09-17-1040-arc-easy-subset-size.md`.

## 2026-09-17 09:05 — Decision: baselines live outside the sweeps

OLMES-default runs (canonical format `daa93775`, no instruction) for every model and
formulation were moved to `pipeline/arc-20260916/baselines/<sweep>/<rc|mc>/` with a
`source.json` each and a `manifest.jsonl`. `run_sweep` counts moved-out baselines when
judging chunk completeness; `load_sweep` merges them back. The MC baselines came out of
`broken-mc/` and are valid because the no-instruction pair is untouched by the RC-pool
mistake. Item ids of the 100-item subsets are copied to
`../results/assets/2026-09-17-arc-easy-baselines/`.
