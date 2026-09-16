# Scoring function and reward noise

What each candidate is scored with, how noisy that score is per rollout, and
how the papers handle or exploit the noise. Companion to the other files in
this directory; sections use the year, first author, title format and link to
the saved full text under `docs/papers/`.

Summary:

- Every paper scores candidates with a task metric averaged over examples;
  most use binary correctness. Only Gao 2026 models the noise in that average
  explicitly, and its decomposition (response noise scaling as 1/(KM) versus
  true between-candidate variance) is the right frame for the rest.
- Two papers replace or augment the scalar with richer signal: text feedback
  and traces (Agrawal 2025, Agrawal 2026) and per-aspect sub-scores
  (Agrawal 2026). Where ablated, the richer signal mattered more than any
  other design choice.
- Execution accuracy beat log-probability as a selection score in Zhou 2022,
  and the same paper shows selection can be gamed by the candidate set.
- Local note: our OLMES setup evaluates the same ARC-Easy items under two
  scoring functions, RC (per-character-normalized log-likelihood over answer
  text) and MC (letter choice). For the 150M DataDecide model MC is at chance
  (0.26) while RC is 0.51, so the choice of formulation changes whether there
  is any signal at all. Per-item results for both live under
  `~/drotherm/data/runs/olmes/` and can be used to measure response noise
  directly.

## 2026, Gao: p1: Better Prompt Optimization with Fewer Prompts

Saved as [2604.08801v2.md](../2604.08801v2.md).

Reward is binary per response: all constraints satisfied (IFBench) or
math-verify correct with over-length generations scored 0 (AIME). A system
prompt's score is the mean over K inputs and M sampled responses. The paper's
core result is the decomposition of the expected variance of these scores
across N candidate system prompts into a response-noise term, which scales as
1/(KM), and an among-system-prompt term, which is the true signal. The
optimizer can only learn when the second term is not swamped by the first.

Empirically, IFBench has much larger between-prompt variance than response
noise and optimizes easily; AIME is the reverse and does not move under
full-dataset RL or GEPA. Reward collection at M = 2 took about 1,400 seconds
per training step, which is why M is small in training and large (32 to 128)
only in analysis. They estimate the noise term with the plug-in Bernoulli
formula p(1-p)/M using the empirical success rate as p.

Design notes: the RL update is an on-policy GRPO variant with no KL term and
no standard-deviation advantage normalization, so the raw reward scale feeds
straight into the gradient; noise in the reward therefore directly becomes
noise in the update.

## 2022, Zhou: Large Language Models are Human-Level Prompt Engineers

Saved as [2211.01910v2.md](../2211.01910v2.md).

Two scores are compared: execution accuracy (0-1 match, sometimes
order-invariant) and log-probability of the gold answer under the target
model. On 24 tasks, execution accuracy had the higher Spearman correlation
with test performance, and it is the default. Log-probability was hypothesized
to help when all candidates are poor, but did not win overall.

Noise handling is budgetary rather than statistical: candidates are scored on
small random subsets with a running average, and only survivors get more
data. Selection uses 10 to 50 examples in the analyses.

Gaming: on the rhyming task, four of five selected instructions told the
model to echo the input word, which scores near-perfect under the metric.
Any scalar score exposed to a large candidate set will be exploited where the
metric and the intent diverge.

## 2026, Agrawal: optimize_anything: A Universal API for Optimizing any Text Parameter

Saved as [2605.19633v1.md](../2605.19633v1.md).

The evaluator contract returns a score plus a side-information dictionary:
free text (errors, critiques), structured data (per-test results, sub-scores),
or images. The proposer reads it during reflection. Their framing is that
side information is the text-optimization analogue of a gradient: it says why
a candidate failed, not just that it did.

Ablation on a prompt-optimization dataset: with per-aspect sub-scores the
validation score reached 0.80 in about 100 rollouts versus about 600 without,
and final test score was 86.3 versus 82.5. Across domains, score-only feedback
reached 94 percent of the best circle-packing solution and produced zero
kernels above 1.1x speedup in multi-task mode versus 40 percent with side
information. Sub-scores are also tracked individually on the Pareto front, so
a candidate that is best on any single aspect survives.

## 2025, Agrawal: GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning

Saved as [2507.19457v2.md](../2507.19457v2.md).

Distinguishes the evaluation metric, a scalar used for accept/reject and for
the Pareto scores, from a feedback function that returns the same scalar plus
text extracted during evaluation (compiler messages, failed rubric items,
per-hop feedback in multi-hop systems). Execution traces of the system and
evaluation traces from the metric are both fed to the reflection model.
Human-written explanations attached to training instances can be used the
same way. The minibatch accept test compares average metric before and after
the edit on three examples, so a single rollout's noise can accept or reject
an edit; the Pareto split evaluation afterward is what protects the pool.

## 2026, Liu: Reflection in the Dark: Exposing and Escaping the Black Box in Reflective Prompt Optimization

Saved as [2603.18388v2.md](../2603.18388v2.md).

Score is accuracy on a minibatch; the selection statistic is the accuracy
gain of each rewrite over the current prompt on the same minibatch, and only
positive-gain rewrites are eligible. Parse errors on the raw output are used
as a separate signal that drives the restart procedure. The paper's diagnosis
of GEPA is that the scalar plus unlabeled reflection never surfaces structural
root causes, which is an argument that the score's information content, not
its noise, was the binding constraint on their defective-seed task.

## 2023, Guo: EvoPrompt: Connecting LLMs with Evolutionary Algorithms Yields Powerful Prompt Optimizers

Saved as [2309.08532v3.md](../2309.08532v3.md).

Fitness is the task metric on a fixed development set (accuracy, ROUGE,
SARI), with the task model decoded greedily or at temperature 0, so each
candidate's score is deterministic and the only noise is dev-set sampling.
Scores feed roulette-wheel parent selection in the genetic variant and
pairwise replacement in the differential-evolution variant. Results are
reported as mean and standard deviation over three seeds, and many of the
reported differences are within one standard deviation.
