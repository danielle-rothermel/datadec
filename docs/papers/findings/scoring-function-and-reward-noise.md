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
- Zhao 2025 goes the other way and scores prompts by gold-output
  cross-entropy alone, one forward pass per example, no sampling. It is the
  cheapest and least noisy score in the set, but requires gold outputs, full
  log-probabilities, and at least a few examples to avoid lexical
  overfitting.
- Voronov 2024 measures how much of a likelihood-based score is format
  noise: up to 35 percent relative spread across templates on strong models,
  reported method gains inside that spread, and a 5-template probability
  ensemble that raises the mean and cuts the variance.

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

## 2025, Zhao: PMPO: Probabilistic Metric Prompt Optimization for Small and Large Language Models

Saved as [2505.16307v2.md](../2505.16307v2.md). The one paper here whose
score is not a task metric at all.

The score of a prompt is the mean token-level cross-entropy of the gold
output under the frozen model, computed in a single forward pass per example
with no sampling. For preference data it is a pairwise logistic loss on the
log-probability gap between preferred and rejected outputs (beta = 1), the
same form as DPO. The stated advantages are that the score is continuous,
deterministic, batchable, and available to models too small to critique
their own outputs. Optimization ran on 50 training examples for 20 iterations
in about 20 minutes on one GPU and beat OPRO, EvoPrompt, PromptWizard and
others on BBH average, GSM8K, AQuA-RAT, and AlpacaEval 2.0 with Qwen2.5-14B
as both optimizer and executor.

Caveats that matter for us:

- Zhou 2022 found log-probability correlated worse with test accuracy than
  execution accuracy. PMPO does not test that correlation; its ablation shows
  the preference-loss ranking beats its own no-loss baseline by about 1.2
  points, and Algorithm 1 selects the winning variant "measured by loss or
  accuracy" without saying which was used where.
- Cross-entropy of a gold output requires a gold output. On reasoning tasks
  the target is the dataset's reference solution, so the score rewards
  matching that solution's wording, not just its answer.
- With a single training example the loss overfits to lexical artifacts; the
  paper reports a prompt that hard-coded one review's details. They recommend
  at least 3 to 5 examples.
- It needs full-sequence log-probabilities, so it cannot run against
  proprietary APIs, though prompts optimized on Qwen2.5-32B transferred to
  GPT-3.5, Claude 3.5 Haiku, and GPT-4o on one BBH task.

Local relevance: the OLMES RC formulation is a likelihood score over answer
text, so our per-item RC outputs already contain the PMPO-style signal for
every candidate we evaluate, and the MC and RC accuracies on the same items
let us measure directly how well a loss-based ranking agrees with an
accuracy-based one.

## 2024, Voronov: Mind Your Format: Towards Consistent Evaluation of In-Context Learning Improvements

Saved as [2401.06766v3.md](../2401.06766v3.md). An evaluation
study rather than an optimizer, but the clearest measurement in the set of
how much score variance comes from prompt format alone.

Setup: 21 base and instruct models (0.77B to 70B), 4 classification datasets,
templates built from a grid of input verbalizer, output verbalizer,
intra-separator, and inter-separator (168 to 216 combinations), scored by
label-token likelihood (their Direct method) with 2 or 4 demonstrations,
10 random templates per seed and 3 demonstration seeds.

Findings that bear on scoring:

- Format alone moves accuracy by a large fraction of its value. Even Llama 2
  70B and Falcon 40B had standard deviations across templates up to 35
  percent of the mean, and a poor template took strong models to chance.
  Reported gains of several in-context-learning methods were within the
  spread of templates for the plain baseline, so a method-versus-method
  comparison on one template is mostly template luck.
- The prediction method changes both mean and variance. Channel (score the
  input given the label) and Calibration (correct for label prior) usually
  beat Direct on average, but Calibration was the most template-sensitive,
  and for many settings Direct's best templates matched the others. Which
  scoring rule you use changes which prompts look best.
- Template ensembles reduce the noise. Averaging label probabilities across
  5 random templates raised mean accuracy for every model and prediction
  method and cut the template-induced variance; majority voting over labels
  worked poorly on many-class tasks. Gains saturate at 4 or 5 templates;
  smaller ensembles can drop if one bad template is drawn. Cost is N times
  the forward passes.
- No component is safely prunable. Decomposing templates into parts showed
  every part with high variance, part rankings that flip between models, and
  optimal templates built from individually suboptimal parts.

Local relevance: their Direct method is the same likelihood-over-choices
scoring family as OLMES RC and MC, and their 4-shot classification setting is
close to OLMES's 5-shot ARC. Their 35 percent relative spread is a benchmark
for how much of our per-prompt score variance to expect from format rather
than content when we compare candidate system prompts.

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
