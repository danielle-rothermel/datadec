# GEPA runs

How the reflective optimizer is configured: what it optimizes, what it sees, how it is
validated, and how its results are read. Log, most recent first.

## 2026-09-17 13:20 — Direction: surrogate-plus-confirmation as an alternative to accept-on-minibatch

MIPRO (Opsahl-Ong 2024) scores candidate configurations on random mini-batches, feeds the
scores to a Tree-structured Parzen Estimator over the discrete candidate set, and every S
trials confirms the best-by-mean configuration on the full training set; the best
fully-confirmed one is returned (about 300 mini-batch trials per 50 full evaluations).
Recorded as an alternative to GEPA's strict-improvement-on-minibatch acceptance for our
noise regime: the surrogate pools evidence across trials instead of deciding on one
minibatch, and the confirmation step is a built-in guard against the winner's curse. Not
adopted yet; would apply to selecting among a fixed candidate pool (seed formats and
instructions) rather than to reflective rewriting. Demonstration selection is not an
axis for us: the fixed OLMES demonstrations are part of the evaluation design.

## 2026-09-17 11:00 — Decision: what the reflection model is shown

- Do not show the proposer raw per-candidate scores as the thing to improve.
  Promptbreeder found the LLM copied top-scoring entries when fitness values were
  listed. Our reflective records currently include every metric per item in the
  feedback string; keep the per-item correctness and the probability tables (they are
  the diagnostic signal), but drop aggregate scores and any ranking of candidates from
  the reflection prompt, and check whether the reflection output anchors on the numbers.
- Reflective records must show the question and choices rendered in the scored
  prompt's format (the same render as the demonstrations and query), not a fixed
  "Question: ... Choices: 0: ..." rendering (audit mismatch 6).
- The model card (size, recipe, architecture, likelihood scoring) stays in the
  reflection template; whether it helps is measured with the model-aware meta-prompt
  factor in instruction generation.

## 2026-09-17 10:45 — Direction: item selection for minibatches and validation

Minibatch and validation items should be chosen for discriminative power, not at random,
once a prompt-by-item matrix exists (see dataset-sample-selection 10:00): between-prompt
variance per item with response noise zero, validated on held-out prompts. Until then,
random draws from a pool that excludes items dead across the model-size span
(dataset-sample-selection 10:10). Per-instance Pareto tracking is what lets GEPA
tolerate heterogeneous items; keep it on.

## 2026-09-17 10:40 — Reference numbers the run design must respect (clean baselines)

- A 150M-to-300M size doubling is worth +0.009 on the ARC-Easy RC per-char share
  (paired item sd 0.029); 1B over 300M is +0.018. Prompt effects on DD models are
  smaller than a size doubling.
- Detecting a 0.005 paired difference at 80% power needs ~260 items; a 16-item
  minibatch resolves ~0.02, a 100-item validation set ~0.008.
- Checkpoint-to-checkpoint jitter on the full set is 0.0002-0.0006 on the per-char
  share and 0.001-0.005 on accuracy (300M, last five checkpoints, DataDecide's
  published trajectory); seed-to-seed at the final checkpoint is 0.0006 / 0.002. Any
  claimed prompt effect on a DD model must clear this floor, not just the item-sampling
  interval. Per-item version pending from the 300M checkpoint runs.
- The default GEPA settings (3-item minibatch, 40 train-split validation items, 600
  calls) could not resolve DD-scale effects; the 2026-09-16 GEPA results are
  disregarded with the rest of that pipeline.

## 2026-09-17 09:25 — Decision: score metric is a job field; validation from the test split

- `score_metric` in the job file is `primary` or `primary_likelihood`, resolved by the
  adapter against the task's OLMES primary; default `primary_likelihood`. Open: which one
  GEPA should optimize; decide after the per-item noise floor is known.
- Validation items come from the same split as the held-out evaluation (test split,
  disjoint ids: `arc_easy-test-n100-seed3.json` vs held-out seed 0), because the
  train-split validation set scored 0.35-0.48 where test scored 0.51-0.59.
- If Batch Calibration is ever used inside GEPA, the label-position means must be fixed
  from a reference run over the training pool, not re-estimated per minibatch.

## 2026-09-17 09:00 — Standing settings carried into this round

- Reflection model GPT-5.1 via OpenRouter (dr-providers), temperature 1.0, reasoning
  medium, evidence archived per call; cost is not a constraint.
- Powered configuration for Qwen-scale effects: minibatch 16, validation 100 test-split
  items, budget 1500 metric calls, 2 best + 2 worst seeds per (model, formulation).
  Not yet run.
- Seeds with no instruction start GEPA from the empty string.
