# User input selection

Which training examples an optimizer scores candidates on, how many, and how
they are chosen. Companion to
[initial-population-generation-and-impact.md](initial-population-generation-and-impact.md);
sections use the same year, first author, title format and link to the saved
full text under `docs/papers/`.

## Note from Danielle

Use statistical power to drive the search. The question for any candidate
comparison is whether the chosen inputs, at the chosen number of samples per
input, can distinguish two system prompts that actually differ by the amount
we care about. Selection of inputs and allocation of rollouts should follow
from that, rather than from a fixed dev-set size.

## TODO

- Look into which examples are always correct or always wrong across
  candidate system prompts. Those contribute nothing to discriminating
  candidates and are the first thing to drop, but check whether they carry
  information about generalization that a variance criterion would discard.
- Work out what an "inter-user-input variance" concept would look like, as
  opposed to p1's inter-system-prompt variance. p1 asks which inputs make
  system prompts look different from each other. The complementary question is
  which system prompts make inputs look different from each other, and whether
  a joint decomposition (input effect, system-prompt effect, interaction,
  response noise) gives a better selection rule than either marginal alone.

## 2025, Heineman: Signal and Noise: A Framework for Reducing Uncertainty in Language Model Evaluation

Saved as [2508.13144v1.md](../2508.13144v1.md). About choosing benchmarks and
metrics for comparing *models* (data recipes at small scale predicting large
scale), not prompts, but the framework transfers directly to choosing items
and metrics for comparing prompts.

Defines signal as relative dispersion of final-checkpoint scores across a
population of comparable models (max pairwise gap over the mean) and noise as
the relative standard deviation of a single model's score over its final n
training checkpoints; the ratio predicts decision accuracy (whether a small
scale ranking holds at large scale) better than either alone. Checkpoint
noise correlates with seed and data-order noise (R^2 0.82 to 0.95), which is
why they use the cheap one. Three interventions all improve decision
accuracy: (1) keep only the sub-tasks with the highest SNR, greedily added in
SNR order, which beat the full benchmark with 16 of 57 MMLU sub-tasks and 6
AutoBencher sub-tasks; low-SNR sub-tasks overlap with the ones MMLU-Redux
flagged as mislabeled; (2) average the last k checkpoints; (3) score with
bits-per-byte of the gold continuation instead of accuracy, which raised SNR on
almost every benchmark (ARC-Easy 21 to 65, ARC-Challenge 6.6 to 45) and
decision accuracy on 90 percent of them. Appendix B.2: random item subsets
reach diminishing returns around 1K items and a 300-item ARC-Easy subset is
less noisy than 30K AutoBencher items, so quality of items beats count.

Mapping to our setting: the population of "models" is our population of
prompts; signal is the spread of prompt scores; the checkpoint-noise term has
no direct analogue because our scoring is deterministic, so noise is item
sampling (and, if we wanted a per-item stability measure, the score of the
same item across the last n DataDecide checkpoints, which exist for every DD
model). Their sub-task filter is p1's input filter at coarser granularity, and
their BPB result is the same conclusion as DataDecide's and ours: continuous
likelihood metrics have more signal than accuracy where small models are near
chance.

Era caveat: 375 open-weight models from 60M to 32B (OLMo, DataDecide,
Qwen, Llama and others) on 30 current benchmarks, evaluated with OLMES in
2025, so the models and tasks are current; the caveat is scope, since every
result is about ranking models, and prompt effects are one to two orders of
magnitude smaller than the model differences they study.

## 2026, Gao: p1: Better Prompt Optimization with Fewer Prompts

Saved as [2604.08801v2.md](../2604.08801v2.md). The only paper here whose
central contribution is input selection.

Decomposes expected reward variance across candidate system prompts into a
response-noise term, scaling as 1/(KM) for K inputs and M samples each, and an
among-system-prompt term. Adding inputs shrinks the second term on
heterogeneous tasks because different inputs favor different system prompts
and the preferences cancel in the average. The filter scores every K_top-sized
subset of inputs by total variance minus estimated response noise, using a
success-probability matrix over N = 16 cold-sampled candidates and M samples
per cell, and trains on the argmax subset (K_top = 2 by default). Two AIME 24
questions chosen this way beat full-dataset RL and GEPA on held-out reasoning
benchmarks; on the homogeneous IFBench the full set was better.

Design choices they justify: subtract response noise rather than rank by raw
variance (raw variance favors inputs the model solves half the time), and use
the variance difference rather than a signal-to-noise ratio (the ratio is
unstable and favors always-right or always-wrong inputs).

Limits: the ranking depends on the specific candidate draw; N, M, and the
meta-prompt are not ablated for the filtering stage; the filter costs N x K x M
rollouts up front; and the filtered subset neither helped nor hurt GEPA.

## 2025, Zhao: PMPO: Probabilistic Metric Prompt Optimization for Small and Large Language Models

Saved as [2505.16307v2.md](../2505.16307v2.md). Selects inputs by difficulty
under the current prompt rather than by discriminative power across prompts.

Each iteration computes the loss of every training example under the
incumbent prompt and takes the top-k = 3 highest-loss examples as the
rewrite targets ("bad case analysis"). Each hard example is shown to the
rewriter together with the mask analysis and yields 4 variants. Candidate
selection then uses the loss over the full 50-example training set, so hard
examples steer proposal but not acceptance. Removing this selection cost
about 1.1 points on BBH in the cumulative ablation.

Two points of contact with the TODO above:

- This is the always-wrong end of the spectrum p1 warns about: the
  highest-loss examples under one prompt need not be the ones that separate
  prompts. PMPO tolerates that because the loss is continuous and
  deterministic, so an example the model gets wrong still yields a usable
  gradient-like signal, and because acceptance is decided on the whole set.
- Their single-example failure mode (a prompt that hard-codes one review's
  details) and their recommendation of at least 3 to 5 examples is empirical
  support for needing a small but diverse selected set rather than the single
  most informative input.

## 2022, Zhou: Large Language Models are Human-Level Prompt Engineers

Saved as [2211.01910v2.md](../2211.01910v2.md). Input selection appears as
budget allocation for scoring rather than as a training-set choice.

Adaptive filtering: every candidate is scored on a small random subset of
training inputs; candidates above a threshold get a fresh non-overlapping
subset and a running average; the process repeats until few candidates remain,
which are scored on everything. Promising candidates get exact scores and poor
ones are cut early. Selection scores are compared on 10 to 50 test points per
candidate in the analyses. Execution accuracy correlated better with test
performance than log-probability across 24 tasks, and a metric can be gamed by
the candidate set (the rhyming task selected instructions that echo the
input).

Era caveat: all results are on the 2022 OpenAI API family (ada through
text-davinci-002, InstructGPT 175B) and on instruction-induction tasks such
as pluralization, antonyms, and first-letter extraction, plus a 2022 subset
of BIG-Bench and TruthfulQA. These are short single-step tasks that modern
models, including small ones, solve near-perfectly zero-shot, and the
proposers are far weaker than current small instruct models. Treat the
qualitative shapes (diminishing returns in sample count, meta-prompt
sensitivity, proposer-executor mismatch) as the transferable content; the
specific saturation points (64 samples, 3 resampling rounds) and the
transfer failures between GPT-3 and InstructGPT are era-specific.

## 2025, Agrawal: GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning

Saved as [2507.19457v2.md](../2507.19457v2.md). Splits the training set into a
feedback split and a Pareto split of fixed size.

Each round draws a minibatch from the feedback split for reflection and
accept/reject; a candidate that improves on the minibatch is then scored on the
Pareto split, where per-instance scores are tracked and any candidate that is
best on at least one instance survives. Inputs are therefore used in two roles:
a few examples per round to generate a targeted edit, and a fixed held-in set
whose per-instance structure drives parent selection. Per-instance tracking is
what lets GEPA tolerate heterogeneous inputs without the averaging problem p1
describes, which is a plausible reason the p1 filter did not change GEPA's
results.

## 2026, Liu: Reflection in the Dark: Exposing and Escaping the Black Box in Reflective Prompt Optimization

Saved as [2603.18388v2.md](../2603.18388v2.md). Uses a minibatch of size b per
round to verify K parallel rewrites, selecting the rewrite with the largest
accuracy gain over the current prompt on that minibatch, then evaluates the
winner on a validation split for Pareto admission. Inputs also feed the
hypothesis agent as failure cases. No selection of inputs beyond random
minibatches; the ablation shows K = 5 parallel candidates on the same
minibatch underperform K = 3, which reads as the minibatch being too small to
rank five candidates reliably.

## 2026, Agrawal: optimize_anything: A Universal API for Optimizing any Text Parameter

Saved as [2605.19633v1.md](../2605.19633v1.md). Inherits GEPA's minibatch
plus per-example Pareto tracking and states the minibatch is deliberately
small, 2 to 3 examples, so each reflection targets a focused failure. Adds
multi-task mode, where each dataset element is an independent problem and the
front is shared across problems; the discussion notes this helps when problems
share transferable structure and hurts when they are independent (circle
packing for different N), which is the same heterogeneity concern as p1 seen
from the other side.

## 2023, Guo: EvoPrompt: Connecting LLMs with Evolutionary Algorithms Yields Powerful Prompt Optimizers

Saved as [2309.08532v3.md](../2309.08532v3.md). No input selection: every
candidate is scored on a fixed development set of 200 examples for
classification tasks and a sampled dev subset for BBH, with the full test set
reported. Relevant only as the baseline design the other papers depart from.

Era caveat: the task executor is Alpaca-7B (a 2023 LLaMA-1 fine-tune) for
classification and generation, with GPT-3.5 as the optimizer and for BBH;
tasks are SST-2/SST-5, Subj, AGNews-style classification, SAMSum, ASSET,
and 3-shot BBH. The initialization and operator ablations are on SST-5
only. Modern small models are near ceiling on these classification sets
and far more robust to prompt wording, so the absolute effect sizes (1 to 6
points) and the population-size curve should not be assumed to carry over.
The relative findings (bad-only seeds hurt, differing-parts mutation beats
whole-prompt mutation) are the part worth keeping.

