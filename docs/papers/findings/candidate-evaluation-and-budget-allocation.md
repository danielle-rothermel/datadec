# Candidate evaluation and budget allocation

How a proposed candidate earns a full evaluation, how rollouts are counted,
and where the cost goes. Companion to the other files in this directory;
sections use the year, first author, title format and link to the saved full
text under `docs/papers/`.

Summary:

- The dominant pattern is a two-stage gate: a small minibatch decides whether
  a proposal is worth a full evaluation, and only accepted proposals are
  scored on the larger set (Agrawal 2025, Liu 2026, Agrawal 2026). Zhou 2022
  generalizes this to a multi-stage successive-halving scheme over many
  candidates at once.
- Budgets are counted in rollouts or metric calls, and in every paper that
  reports it the evaluator, not the proposer, dominates cost. GEPA's
  reflection model was called only 24 to 90 times per run.
- Gao 2026 is the outlier: it spends its budget on repeated samples of the
  same inputs to reduce reward noise, and its filtering stage is a large
  fixed cost paid before optimization.
- No paper adapts the number of samples per candidate to the observed
  variance between candidates. That is the gap the statistical-power note in
  [user-input-selection.md](user-input-selection.md) points at.
- Zhao 2025 sidesteps the gate entirely: likelihood scoring is one prefill
  per example, so all 13 candidates per iteration are scored on the whole
  50-example training set and a 20-iteration run takes about 20 minutes.
- Sclar 2023 is the exception on allocation: a Thompson-sampling bandit
  over candidates with Beta posteriors on per-example accuracy found the
  best and worst of 320 formats within 1 point using about 5 percent of
  exhaustive cost, beating UCB and uniform allocation.

## 2025, Agrawal: GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning

Saved as [2507.19457v2.md](../2507.19457v2.md).

Each iteration: select a parent from the Pareto front, run it on a minibatch
of 3 from the feedback split, reflect, rerun the edited system on the same
minibatch, and accept only if the average score improved. Accepted candidates
are then scored on every instance of the Pareto split (the validation set),
which is the expensive step and the one that produces the per-instance score
vector used for selection. Merge is attempted at most 5 times per run.

Budget is a total rollout count. For fair comparison it was set per benchmark
to match MIPROv2's usage, between 2,270 and 6,926 rollouts, and stayed within
about 10 percent of it. The GRPO baseline used a fixed 24,000 rollouts.
Reflection-model calls per run were 24 to 34 with GPT-4.1-mini and up to 90
with Qwen3-8B, so proposal cost is negligible next to evaluation. The final
answer is the pool member with the best average on the Pareto split.

## 2022, Zhou: Large Language Models are Human-Level Prompt Engineers

Saved as [2211.01910v2.md](../2211.01910v2.md).

All roughly 50 candidates are generated up front. Scoring is multi-stage:
evaluate everyone on a small subset, keep candidates above a threshold, give
them a fresh non-overlapping subset and update a running average, repeat until
few remain, then score those on the full training set. Poor candidates are
cut after a handful of examples; good ones get exact scores. Increasing the
candidate count from 4 to 128 improved the selected instruction with
diminishing returns, and iterative resampling rounds plateaued after about
three. Larger aligned proposers produce shorter instructions, which reduced
scoring cost enough to dominate the accuracy-cost frontier.

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

## 2025, Zhao: PMPO: Probabilistic Metric Prompt Optimization for Small and Large Language Models

Saved as [2505.16307v2.md](../2505.16307v2.md). Removes the gate by making
evaluation cheap enough not to need one.

Every candidate is scored on the full training set of at most 50 examples
with one forward pass per example and no decoding. Per iteration the
candidate set is the incumbent plus 3 hard examples times 4 variants, so 13
prompts times 50 examples of teacher-forced scoring, plus the masking
analysis (up to 5 extra scorings of the incumbent with one segment removed).
Twenty iterations took about 20 minutes on a single H800 with a 14B model.
The paper's efficiency argument is that generation-based methods pay for
autoregressive decoding on every candidate and every example, whereas
likelihood scoring costs one prefill, so many more candidates fit in a fixed
budget. It does not report rollout or dollar budgets comparable to the other
papers, and the baselines were run under their own default settings rather
than a matched budget.

## 2023, Sclar: Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design or: How I learned to start worrying about prompt formatting

Saved as [2310.11324v2.md](../2310.11324v2.md). The most
explicit treatment in the set of allocating a fixed evaluation budget across
a candidate set, which is the statistical-power question in
[user-input-selection.md](user-input-selection.md) applied to candidates
rather than inputs.

Setup. Given n sampled formats (arms), a dataset, and a budget of E
example-evaluations in minibatches of B = 20, the goal is to find the best
and worst arm; half the budget goes to each search and evaluations from the
first are reused as priors for the second. Because per-example accuracy is
a Bernoulli trial, each arm's accuracy is modeled as a Beta posterior and
arms are drawn by Thompson sampling: sample a value from each posterior,
evaluate the arm with the highest draw on B fresh examples, update. Priors
can be informative. UCB with c = 2 (the setting Pryzant et al. 2023 found
best for prompt optimization) and naive uniform allocation are the
comparisons.

Results. With 320 formats and a budget of 51,200 evaluations, Thompson
sampling found a spread within 1 accuracy point of the truth, naive
allocation within 4, and UCB within 11. Exploring about 5 percent of the
full format-by-example space estimated spread within 2 points. FormatSpread
needs no model weights, so it runs on API models; the GPT-3.5 study cost
under 10 dollars per task.

Relevance. This is a cheaper and more principled version of Zhou 2022's
successive halving for the same job, and it comes with a noise model. The
same machinery ranks candidate system prompts under a budget when the score
is per-example correctness; the Beta posterior makes "how many more
examples does this pair need" an answerable question rather than a fixed
minibatch size.

Era caveat: models are LLaMA-2 7B to 70B (the 70B at 4-bit), Falcon 7B and
7B-Instruct, and GPT-3.5-Turbo; tasks are 53 classification and
multiple-choice tasks from Super-NaturalInstructions with 1 or 5 shots.
The headline 76-point spread is LLaMA-2-13B on a stereotype-classification
task, and GPT-3.5's median spread of 6.4 points was already much lower than
the open models'. Current models are markedly more format-robust than
LLaMA-2, so the magnitudes here are upper-end estimates for older base
models; the structural findings (non-monotonic edit space, model-specific
rankings, the grammar itself, the bandit) are the part that transfers. No
reasoning-heavy benchmark was tested.

## 2026, Liu: Reflection in the Dark: Exposing and Escaping the Black Box in Reflective Prompt Optimization

Saved as [2603.18388v2.md](../2603.18388v2.md).

Per round, K = 3 rewrites are each scored on the same minibatch of size b;
the round costs b times K rollouts plus the validation-set evaluation of the
winner. A restart round costs b plus one look-ahead rollout. Budget is
decremented by exactly these amounts and the run stops when it is exhausted.
Their ablation found K = 5 worse than K = 3, which they attribute to the
minibatch being too noisy to rank more candidates; that is a budget-allocation
result as much as a diversity one. Reported costs: about $0.20 per run on
GSM8K with a local base model and an API reflector, $4 to $6 on AIME with an
API base model, and the hypothesis agent adds $0.08 to $0.40 over GEPA.

## 2026, Agrawal: optimize_anything: A Universal API for Optimizing any Text Parameter

Saved as [2605.19633v1.md](../2605.19633v1.md).

Same minibatch gate as GEPA, with the minibatch deliberately small (2 to 3
examples) so reflection is targeted. Adds content-addressed caching so
identical candidates are never re-evaluated, and a refiner step that fixes
malformed code or syntax before evaluation so budget is not spent on
candidates that fail for trivial reasons. In the controlled circle-packing
comparison it reached its best score in 63 evaluations versus 200 for
OpenEvolve without matching it. Total run costs ranged from about $1 to $145,
with reflection a small fraction and the evaluator dominating.

## 2026, Gao: p1: Better Prompt Optimization with Fewer Prompts

Saved as [2604.08801v2.md](../2604.08801v2.md).

Per RL step: N = 16 candidate system prompts, each scored on all K training
inputs with M responses, so K times N times M rollouts per step. The paper
holds K times M roughly constant when comparing configurations, trading
inputs for repeated samples. With M = 2 on the full AIME set a step took about
1,400 seconds. The filtering stage is a separate up-front cost of N times K
times M rollouts at a large M, which at analysis settings (16 by 30 by 128) is
tens of thousands of long generations before any optimization. All runs had a
three-day cap on four H100s.

## 2023, Guo: EvoPrompt: Connecting LLMs with Evolutionary Algorithms Yields Powerful Prompt Optimizers

Saved as [2309.08532v3.md](../2309.08532v3.md).

No gate: every new candidate is scored on the full development set (200
examples for classification, 100 for generation, 50 for BBH). With a
population of 10 and 10 iterations, the genetic variant scores 10 new prompts
per iteration and the differential-evolution variant scores one challenger
per population member. Their population-size study is the budget result: on
an easy task a population of 6 matched 10 at less than half the cost; on
harder classification tasks scores kept rising up to 12.

Era caveat: the task executor is Alpaca-7B (a 2023 LLaMA-1 fine-tune) for
classification and generation, with GPT-3.5 as the optimizer and for BBH;
tasks are SST-2/SST-5, Subj, AGNews-style classification, SAMSum, ASSET,
and 3-shot BBH. The initialization and operator ablations are on SST-5
only. Modern small models are near ceiling on these classification sets
and far more robust to prompt wording, so the absolute effect sizes (1 to 6
points) and the population-size curve should not be assumed to carry over.
The relative findings (bad-only seeds hurt, differing-parts mutation beats
whole-prompt mutation) are the part worth keeping.

