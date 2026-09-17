# Population maintenance and search diversity

How the pool of candidates is kept from collapsing onto one lineage, and how
exploration is balanced against exploitation during the run. Companion to
[initial-population-generation-and-impact.md](initial-population-generation-and-impact.md),
which covers diversity at initialization. Sections use the year, first author,
title format and link to the saved full text under `docs/papers/`.

Summary:

- Per-instance Pareto retention is the mechanism three papers rely on
  (Agrawal 2025, Agrawal 2026, Liu 2026): any candidate that is best on at
  least one instance or metric survives, and parents are sampled by how many
  instances they lead. GEPA's ablation shows greedy best-candidate selection
  gets trapped refining one strategy.
- Fixed-size populations with replacement rules (Guo 2023) get their
  diversity from the operator rather than the retention rule; differential
  mutation kept the population more varied than crossover-plus-mutation.
- Explicit exploration knobs appear only in Liu 2026 (epsilon-greedy
  hypotheses, random restart) and matter less than the taxonomy they
  sample from.
- Ferreira 2025 keeps diversity by construction: a MAP-Elites archive holds
  the best prompt per structural cell (shots x length x reasoning depth), so
  the population spans the structure axes regardless of which cell scores
  best; random sampling from the same grammar filled far fewer cells with
  competent prompts.
- Two papers find that more parallel candidates per round eventually adds
  noise rather than diversity: Liu 2026 at K = 5, and Zhou 2022's iterative
  resampling that raises the distribution without changing the best.
- Gao 2026 maintains no population at all; each RL step draws fresh samples
  from the current policy, and diversity is whatever the policy's temperature
  1 sampling provides.
- Zhao 2025 keeps a single incumbent and accepts a variant only if its loss
  improves; it is the greedy baseline that GEPA's Pareto ablation argues
  against, kept viable by cheap scoring rather than by any diversity
  mechanism.
- Voronov 2024 reframes the end of the search: averaging predictions over
  4 or 5 retained candidates beat picking one, with lower variance, at N
  times the inference cost.

## 2025, Ferreira: Diverse Prompts: Illuminating the Prompt Space of Large Language Models with MAP-Elites

Saved as [2504.14367v1.md](../2504.14367v1.md). Quality-diversity search over
prompt *structure*: a context-free grammar generates prompts from a genotype
(rule choices for role-context, number of shots, reasoning-depth instruction,
task request, task entry, answer instruction), and MAP-Elites keeps an archive
binned by phenotype (number of examples, prompt length in words, reasoning
depth; bin sizes 2 / 25 / 2), replacing a bin's occupant only when a new
prompt scores higher there. Population 50, 10 iterations, mutation only (40%
of individuals, 40% per-property chance), fitness = accuracy on 50 task
instances. Baseline is random sampling from the same grammar.

Findings: MAP-Elites covered more than 60% of the phenotype bins with
high-performing prompts (accuracy above 0.55) in 21 of 28 runs versus 6 for
random, with statistical significance on 2 of 7 tasks. The structural
features correlate only weakly with accuracy (|r| about 0.2 at best);
role-context and reasoning depth had no measurable effect, which the authors
attribute to the ten generic, task-agnostic role and thought templates.
Which structure wins is task-specific: zero-shot dominates for LD3, KU and
StrategyQA (every high performer on LD3 was zero-shot), while few- and
many-shot dominate for Winowhy, PDSD and SSB.

For us the useful part is the archive discipline, not the grammar: keep the
best prompt per structural cell rather than the best prompt overall, so the
seed population spans the structural axes by construction. Their evidence
that structure matters more than the generic text slots is the same
conclusion as our seed round (format effect sd equal to instruction effect
sd) seen from the search side.

Era caveat: four 3.5B-8B instruction-tuned models (Starling-7B, Llama 3.1 8B
Instruct, Phi-3.5 Mini, Qwen2.5 7B Instruct) accessed through inference
endpoints, temperature 0, output capped at three tokens, seven BigBench Lite
tasks that are mostly binary and evaluated on 50 instances per fitness call.
Effects are small and the 50-instance fitness has a binomial sd of about 0.07,
so most of the "high performer" archive is within noise of the rest; treat the
coverage result and the task dependence as the transferable content.

## 2025, Agrawal: GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning

Saved as [2507.19457v2.md](../2507.19457v2.md).

Pareto-based candidate selection (Algorithm 2): for each instance of the
Pareto split, record the best score any candidate achieved and the set of
candidates achieving it; take the union of those sets, prune strictly
dominated members, and sample a parent with probability proportional to the
number of instances it leads. The pool itself is never pruned; only selection
is. This is described as an illumination strategy in the sense of
quality-diversity search.

The ablation replaces it with always selecting the best-average candidate.
Figure 6 shows that variant finding one improved strategy and then spending
the remaining budget failing to refine it. Two further diversity sources:
merge combines lineages that improved different modules, and each candidate
records its parent so the search tree is inspectable (Appendix J).

## 2026, Agrawal: optimize_anything: A Universal API for Optimizing any Text Parameter

Saved as [2605.19633v1.md](../2605.19633v1.md).

Generalizes the front to three settings: per-example scores (generalization
mode), per-metric sub-scores for a single task, and a front shared across
independent problems in multi-task mode. Their trajectory analysis credits the
front with retaining candidates from several algorithmic families at once
(greedy, LP, SLP, bilevel, CMA-ES) on different quality dimensions, so a
candidate that dominates on raw score does not evict structurally different
ones that lead on stability metrics, and those later seed hybrids. Even broken
mutations scoring zero were recovered through the refiner and kept. In
multi-task mode, gains grew from 10 to 20 problems and frontier size did not
bottleneck scaling because parents are sampled by frontier frequency; the same
sharing hurt when problems had no transferable structure.

## 2023, Guo: EvoPrompt: Connecting LLMs with Evolutionary Algorithms Yields Powerful Prompt Optimizers

Saved as [2309.08532v3.md](../2309.08532v3.md).

Two retention rules. The genetic variant generates N offspring, merges them
with the N incumbents, and keeps the top N by score, which is pure
exploitation. The differential variant pits one challenger against each
incumbent and keeps the better, so population size is constant and every
member is protected until something beats it in its own slot. The
differential variant's population stayed more diverse (longer prompts, higher
length variance, more new words mutated late in the run), which the authors
credit for its better escape from local optima and its advantage when seeds
are poor. Population size from 4 to 12 helped on harder tasks and saturated
early on an easy one. Parent selection by roulette wheel slightly beat
tournament and random.

Era caveat: the task executor is Alpaca-7B (a 2023 LLaMA-1 fine-tune) for
classification and generation, with GPT-3.5 as the optimizer and for BBH;
tasks are SST-2/SST-5, Subj, AGNews-style classification, SAMSum, ASSET,
and 3-shot BBH. The initialization and operator ablations are on SST-5
only. Modern small models are near ceiling on these classification sets
and far more robust to prompt wording, so the absolute effect sizes (1 to 6
points) and the population-size curve should not be assumed to carry over.
The relative findings (bad-only seeds hurt, differing-parts mutation beats
whole-prompt mutation) are the part worth keeping.

## 2026, Liu: Reflection in the Dark: Exposing and Escaping the Black Box in Reflective Prompt Optimization

Saved as [2603.18388v2.md](../2603.18388v2.md).

Keeps GEPA's Pareto pool with frequency-weighted parent sampling and adds two
explicit exploration mechanisms. Within a round, each of K = 3 hypothesis
slots is unconstrained with probability epsilon = 0.1, otherwise drawn from
the taxonomy. Across rounds, with probability 0.2 the candidate is built by
restart from model behavior instead of by mutation. The labeled trace lets the
hypothesis agent avoid directions already tried. Ablations: epsilon = 0 cost
about two points, epsilon = 1 collapsed the method, restart alone added two
points over GEPA, and K = 5 underperformed K = 3.

## 2022, Zhou: Large Language Models are Human-Level Prompt Engineers

Saved as [2211.01910v2.md](../2211.01910v2.md).

There is no maintained population beyond the initial sample; iterative APE
filters to the top candidates and asks the model for variations of them.
Survival curves shift upward each round and plateau after about three, but the
best candidate is usually unchanged, and the gain over one-shot sampling was
marginal except on tasks where the initial set was poor. The authors chose
non-iterative sampling as the default. This is the clearest evidence in the
set that local resampling around good candidates narrows rather than widens
the search.

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

## 2026, Gao: p1: Better Prompt Optimization with Fewer Prompts

Saved as [2604.08801v2.md](../2604.08801v2.md).

No population is retained. Each step samples N = 16 fresh system prompts from
the current generator policy at temperature 1, scores them, and updates the
policy; diversity is a property of the policy's sampling distribution and
shrinks as the policy sharpens. The paper's explanation for why filtered
training works is that a cleaner reward lets the policy move further from its
initialization, whereas noisy full-dataset reward keeps it near the start,
which is a statement about exploitation being blocked by noise rather than
about diversity being lost.

## 2025, Zhao: PMPO: Probabilistic Metric Prompt Optimization for Small and Large Language Models

Saved as [2505.16307v2.md](../2505.16307v2.md). Single-incumbent hill climbing,
the baseline design the other papers depart from.

One prompt is carried between iterations. Each iteration proposes 12 variants,
scores them with the incumbent, keeps the best only if its loss is lower, and
discards the rest. There is no archive, no front, and no explicit exploration;
the only diversity is the temperature sampling of the 4 variants per hard
example and the rotation of which 3 examples are hardest as the prompt
changes. The paper does not report the acceptance rate or how often runs
stall, but the case-study loss curve flattens after about 8 of 20 iterations.

## 2024, Voronov: Mind Your Format: Towards Consistent Evaluation of In-Context Learning Improvements

Saved as [2401.06766v3.md](../2401.06766v3.md). Offers an
alternative to selecting one winner from a population: keep several and
ensemble them.

Template Ensembles average label probabilities across N randomly chosen
formats at test time. Accuracy rose for every model and prediction method
tested and template-induced variance fell, with gains saturating at N of 4
or 5 and small ensembles occasionally dropping when a bad template was drawn.
Majority voting was worse than probability averaging on many-class tasks. The
cost is N forward passes per prediction. For an optimizer this suggests a
different terminal step: rather than returning the single best candidate from
the final pool, return the top few and ensemble, trading inference cost for
robustness to the noise in whichever candidate happened to rank first.

Era caveat: 19 of the 21 models are pre-2024 base models (GPT-J, GPT-NeoX,
BLOOM, OPT, Pythia, LLaMA-1, Llama 2, Falcon), with Llama 3 8B Instruct and
Mistral 7B Instruct v0.3 added in an appendix, and the tasks are SST-2,
DBPedia, AGNews, and TREC with 2 to 4 demonstrations. These are easy
classification sets that current models solve near ceiling, so the absolute
spreads (up to 35 percent of the mean) are almost certainly larger than a
modern model would show on the same tasks. The appendix on the two 2024
instruct models found the same non-transfer and lack of variance reduction,
which is the best evidence in the paper that the qualitative findings
persist; whether they persist on reasoning benchmarks is untested.

