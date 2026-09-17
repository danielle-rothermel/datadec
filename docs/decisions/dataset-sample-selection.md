# Dataset sample selection

Which items we evaluate on, how many, and how they are chosen. Log, most recent first.

## 2026-09-17 12:55 — Decision: SNR-filtered 300/300/300 train/dev/test subsets for ARC-Easy

Select on RC, evaluate on both RC and MC (MC carries no DD signal, so an MC-based selection
would fit noise; Qwen contrasts on MC are resolvable on any subset). Per item: signal =
dispersion of the RC per-char share across the DataDecide finals only (150M, 300M, 1B;
including Qwen makes the filter inert because Qwen-vs-DD dominates, 1 item excluded vs
84), noise = checkpoint sd from the 300M trailing checkpoints (1B to be added when its
checkpoints land). Exclude SNR < 2 (84 items, 3.5%) and the OLMES few-shot ids (none in the
test split). Shuffle the 2292 eligible items with seed 0 and take three disjoint 300-item
draws: train, dev, test (test untouched for now). Files:
`configs/po/subsets/arc_easy-test-snr-{train,dev,test}-n300-seed0.json`, pool ranking in
`arc_easy-test-snr-pool-seed0.csv`; copies under
`../results/assets/2026-09-17-arc-easy-snr-subsets/`. Script: `scripts/po_select_items.py`.
Power at n=300 on the per-char share (paired sd ~0.03): minimum detectable paired
difference ~0.005. Prompt-discriminative selection (p1) is a later second pass over the
same pool. ARC-Challenge subsets wait for its full-set runs.

## 2026-09-17 11:25 — Result-backed decision: item noise floors and the SNR criterion

From `../results/2026-09-17-1125-arc-easy-item-noise-dd300m.md`: per-item checkpoint sd
on the RC per-char share is 0.0026 (median), seed sd 0.010, cross-model dispersion
0.030; 93% of items have SNR above 3 and under 1% below 1, so on RC nearly every
ARC-Easy item is usable and the exclusion set is small. On MC half the items flip
between adjacent checkpoints; MC item selection is moot for DD models. Because
cross-model dispersion correlates 0.53 with seed noise, rank items by signal-to-noise,
not by signal alone. Per-item noise for 1B (checkpoints only) is queued.

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
