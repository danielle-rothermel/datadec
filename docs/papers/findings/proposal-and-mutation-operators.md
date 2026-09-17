# Proposal and mutation operators

How a new candidate is produced from existing ones, and what the proposer is
shown. Companion to the other files in this directory; sections use the year,
first author, title format and link to the saved full text under
`docs/papers/`.

Summary:

- Five distinct operator families appear: reflective rewriting from traces
  and feedback (Agrawal 2025, Agrawal 2026), hypothesis-targeted rewriting
  from a failure taxonomy (Liu 2026), LLM-implemented crossover and
  differential mutation (Guo 2023), demonstration-conditioned generation and
  paraphrase resampling (Zhou 2022), and an RL-trained generator policy
  (Gao 2026).
- Where ablated, what the proposer is shown mattered more than how it edits:
  side information in Agrawal 2026 and the hypothesis taxonomy in Liu 2026
  each account for most of their paper's gain.
- The one operator-level ablation with a large effect is Guo 2023's: mutating
  only the parts that differ between two parents beat mutating everything by
  about six points on one task.
- Two papers observe that unconstrained reflection can go wrong in specific
  ways: memorizing training questions (Gao 2026's reading of GEPA's AIME
  prompt) and never questioning seed structure (Liu 2026).
- Zhao 2025 adds an explicit attribution step: mask each prompt segment,
  measure the loss change, and hand the table to the rewriter as a hint. It
  is the only operator in the set that measures rather than infers which part
  of the prompt to edit.
- Voronov 2024 warns that component effects interact and flip across
  executors, so per-component attribution and greedy single-part edits are
  conditional on context.
- Sclar 2023 shows the format space is non-monotonic under atomic edits
  (chance-level monotonicity along edit chains), so edit operators should
  act on instruction content and treat format as a sampled variable.

## 2025, Agrawal: GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning

Saved as [2507.19457v2.md](../2507.19457v2.md).

Reflective prompt mutation: run the selected candidate on a minibatch,
capture the execution trace (each module's inputs, outputs, reasoning) and the
evaluation trace (text the metric produced before scoring), pick one module
round-robin, and show the reflection model the current prompt, trajectory,
score, and feedback with instructions to attribute success or failure to
prompt elements and propose a revised instruction. The meta-prompt is in
Appendix C. The child inherits the parent's other modules unchanged and records
its parent, so lessons accumulate along the tree; Figure 5 shows a trajectory
where each step adds a task-specific nuance.

System-aware merge (Appendix D.1): a crossover that combines two descendants
of a common ancestor, taking each module's prompt from whichever descendant
changed it. It is only attempted when the two descendants modified different
modules, checked per module against the ancestor, and at most 5 times per run.

## 2026, Liu: Reflection in the Dark: Exposing and Escaping the Black Box in Reflective Prompt Optimization

Saved as [2603.18388v2.md](../2603.18388v2.md).

Decouples diagnosis from rewriting. A hypothesis agent reads the failure cases
and proposes K = 3 labeled root-cause hypotheses, each drawn either from a
hand-curated taxonomy of failure modes (category, description, suggested fix
direction) with probability 1 minus epsilon, or unconstrained with
probability epsilon (0.1). A reflection agent then rewrites the prompt once per
hypothesis, and the rewrites are verified in parallel on the minibatch. The
winning hypothesis label is attached to the accepted candidate and the
labeled trace is fed back to the hypothesis agent so it can avoid repeated
directions or propose a joint fix when two labels alternate.

Random restart: with probability 0.2 per round, build a prompt from scratch
by running one instance under the current prompt and handing the raw output
and parse error to the rewriter, iterating until the output parses.

Ablation: the taxonomy is the dominant component, contributing nearly 60 of
74 points on the defective-seed task; with epsilon = 1 (no taxonomy) accuracy
collapsed to 23 percent. Restart and parallel sampling added a few points
each. Their own prompts for both agents are in Appendix E.

## 2025, Zhao: PMPO: Probabilistic Metric Prompt Optimization for Small and Large Language Models

Saved as [2505.16307v2.md](../2505.16307v2.md). Adds a credit-assignment step
before rewriting.

Mask-guided importance: the model is asked to split the current prompt into
up to 5 independent, removable segments; each segment is masked in turn and
the change in mean training loss is measured. Positive change means the
segment helps, negative means it hurts or is redundant, near zero means
inert. This per-segment table is passed to the rewriter as a soft hint
alongside the task description, the current prompt, one hard example, and a
fixed list of edit types (rephrase rigid wording, tighten constraints, remove
redundancy, simplify, fix flow, expand underspecified parts, merge
overlapping rules). The rewriter produces 4 variants per hard example by
temperature and top-p sampling and is not restricted to the masked regions.
Separate rewriting templates exist for large models (diagnose then rewrite)
and small models (a fill-in form with the same fields), which is how the
method runs on 0.5B and 1.5B executors.

Ablation on BBH with Qwen2.5-14B: removing the mask analysis cost about 1.6
points, the largest single component in their table. The case study shows
the prompt evolving from "Let's think step by step" into a structured
multi-step instruction over 15 iterations, with most of the loss reduction
in the first few.

## 2023, Guo: EvoPrompt: Connecting LLMs with Evolutionary Algorithms Yields Powerful Prompt Optimizers

Saved as [2309.08532v3.md](../2309.08532v3.md).

Two operators, each implemented as a natural-language instruction to the LLM
with a one-shot worked example prepended.

Genetic algorithm: select two parents by roulette wheel on dev-set score,
ask the LLM to cross them into one offspring, then mutate it. Roulette wheel
beat tournament and random selection by a small margin.

Differential evolution: take two random population members, ask the LLM to
identify the parts where they differ and mutate only those parts, combine the
mutated parts with the current best prompt, then cross the result with the
base prompt being challenged. The rationale is that shared parts of two good
prompts are probably load-bearing and should be preserved. Ablations on Subj:
mutating only the differing parts scored 75.6 versus 69.9 for mutating
everything; using the current best as the combination target scored 75.6
versus 69.8 for a random member and 69.1 for dropping the step. The
differential variant also produced longer, higher-variance prompts and kept
mutating new words late in the run.

Era caveat: the task executor is Alpaca-7B (a 2023 LLaMA-1 fine-tune) for
classification and generation, with GPT-3.5 as the optimizer and for BBH;
tasks are SST-2/SST-5, Subj, AGNews-style classification, SAMSum, ASSET,
and 3-shot BBH. The initialization and operator ablations are on SST-5
only. Modern small models are near ceiling on these classification sets
and far more robust to prompt wording, so the absolute effect sizes (1 to 6
points) and the population-size curve should not be assumed to carry over.
The relative findings (bad-only seeds hurt, differing-parts mutation beats
whole-prompt mutation) are the part worth keeping.

## 2022, Zhou: Large Language Models are Human-Level Prompt Engineers

Saved as [2211.01910v2.md](../2211.01910v2.md).

Generation rather than mutation: the proposer is conditioned on input/output
demonstrations and asked to infer the instruction, either by a forward
meta-prompt or by infilling. The only mutation operator is the resampling
template, which asks for a variation of an instruction with the same meaning;
it raised the whole candidate distribution but rarely changed the best
candidate. The meta-prompt substantially reshaped the proposal distribution,
helping some tasks and hurting others, and one template reached instructions
the other never produced. Proposers other than the executor worked nearly as
well in forward mode, but a proposer whose instruction style did not match the
executor (an infilling model trained for short text) failed.

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

## 2026, Agrawal: optimize_anything: A Universal API for Optimizing any Text Parameter

Saved as [2605.19633v1.md](../2605.19633v1.md).

GEPA's reflective mutation with side information as the feedback channel:
text, structured sub-scores, or images shown to the proposer in a structured
reflection prompt. Adds a refiner step that repairs common generation
artifacts before evaluation. Two observations about the operator in practice:
side information turns mutation from undirected to targeted (their circle
packing trajectory shows the proposer switching algorithm family in response
to specific diagnostics), and tracking a refiner prompt alongside the main
artifact produced a leapfrogging dynamic where each module's advance became
the other's starting point. Proposer choice matters: GPT-5.1 reached 60
percent on AIME from the same seed where GPT-5-nano reached 50.

## 2026, Gao: p1: Better Prompt Optimization with Fewer Prompts

Saved as [2604.08801v2.md](../2604.08801v2.md).

The proposer is a policy trained by RL rather than a fixed reflector: the
generator model samples N = 16 system prompts per step from a one-sentence
meta-prompt at temperature 1, each is scored by mean reward on the training
inputs, and the generator is updated with an on-policy GRPO variant (no KL
term, no standard-deviation normalization, mean-reward baseline). The
proposer sees no traces or feedback; it learns only from the scalar reward of
its own samples. Qualitatively, the prompt it learned on AIME was a general
reasoning-style instruction, whereas GEPA's on the same data contained
question-specific guidance that the authors read as memorization of the
training set.

## 2023, Fernando: Promptbreeder: Self-Referential Self-Improvement via Prompt Evolution

Saved as [2309.16797v1.md](../2309.16797v1.md). Nine operators, one drawn
uniformly per replication, all implemented as an LLM continuation of
(mutation-prompt + parent): zero-order generation from the description ("A
list of 100 hints:"), first-order mutation-prompt-guided rewrite, EDA (list
the population, diversity-filtered at BERT cosine 0.95, unordered, and ask
for more; fitness values were withheld because the LLM copied entries when
shown them), EDA rank-and-index (population listed in ascending fitness but
labelled descending, which they report improves diversity via a recency
effect), lineage (the chronological elite history as context), zero- and
first-order hyper-mutation (rewrite the mutation-prompt itself), Lamarckian
(induce a task-prompt from a correct worked solution, i.e. APE's reverse
engineering used mid-run), plus 10% crossover and few-shot context shuffling.
Success rates on GSM8K (fraction of applications that beat the parent):
zero-order hyper-mutation 42%, lineage 26%, first-order hyper-mutation 23%,
EDA variants 11-13%, direct mutation 12%, Lamarckian 6%. Removing any
self-referential operator hurt on nearly every dataset; with an
under-specified task description the Lamarckian operator became the most
important (81.6% to 64.6% on ETHOS without it).

Two transferable points: the proposer should not be shown raw fitness values
(it copies the top entries), and description-only regeneration and
demonstration-based induction are complementary operators whose relative
value depends on how informative the task description is.

Era caveat: as in [initial-population-generation-and-impact.md](initial-population-generation-and-impact.md).

## 2025, Yang: What Prompts Don't Say: Understanding and Managing Underspecification in LLM Prompts

Saved as [2505.13360v3.md](../2505.13360v3.md). Two operators over an
explicit requirement list rather than over free text. (1) Bayesian selection
of which requirements to state: each of n requirements is a binary
hyperparameter and a Tree-structured Parzen Estimator searches the 2^n
inclusion patterns on the training split; gains of +3.8% average accuracy
with 41-45% fewer prompt tokens, dropping mostly global, format and
developer-written requirements (the ones models follow by default).
Selections were stable across training subsets (Jaccard 0.75). (2) COPRO
with requirement-specific validators as the metric instead of a generic
1-10 LLM judge (+5.8%); the rewrites reorder and merge requirements. Off-the-
shelf optimizers with a generic judge gave inconsistent results (+2.8% on two
tasks, -1.1% on one). Budget: 9 candidate prompts, 30 training examples.

For us: the inclusion-pattern search is a cheap, structured proposal
operator for an instruction that has been decomposed into parts, and it is
the natural way to test whether our small models are hurt by longer
instructions. The validator result is the requirements-level version of the
feedback finding elsewhere in this file: a metric that says which part
failed beats a scalar.

Era caveat: as in [initial-population-generation-and-impact.md](initial-population-generation-and-impact.md).

## 2024, Schnabel: Symbolic Prompt Program Search (SAMMO): A Structure-Aware Approach to Efficient Compile-Time Prompt Optimization

Saved as [2404.02319v2.md](../2404.02319v2.md). Widest operator set in this
collection because the prompt is a symbolic program. Text operators:
Paraphrase, InduceInstructions (regenerate instructions from examples),
ShortenText, TextToBulletPoints, RemoveStopwords. Attribute operators:
ChangeSectionFormat (markdown vs XML), ChangeDataFormat (JSON, XML,
plaintext), DecreaseInContextExamples. Structural operators: DropSection,
RepeatSection. Search is beam search with mutators drawn uniformly at random
per candidate; budget 24 to 48 evaluations on 100 training examples. In the
compression experiment, the operators that most often improved the objective
were rewriting and dropping in-context examples, and the useful set differed
by backend (Figure 7); GPT-4 tolerated fewer examples and a dropped
introduction better than the others. Reframes APE as InitCandidates +
Paraphrase under beam search and GrIPS as constituent-level
Add/Delete/Swap/Paraphrase.

Era caveat: as in [template-selection.md](template-selection.md).

## 2024, Voronov: Mind Your Format: Towards Consistent Evaluation of In-Context Learning Improvements

Saved as [2401.06766v3.md](../2401.06766v3.md). Not an operator paper, but
its component analysis constrains how edit operators should be evaluated.

Templates were decomposed into four parts and each part's score distribution
measured with the others varying. Every part had high variance, the best
choice for a part flipped between models (an output verbalizer that ranked
first for Llama 2 70B was among the worst for Falcon 40B), and optimal
templates contained individually suboptimal parts. Two consequences for
operators that edit one component at a time, including differential mutation
(Guo 2023) and mask-guided rewriting (Zhao 2025): a component's measured
contribution is conditional on the rest of the prompt and on the executor,
so an edit accepted in one context can be wrong after other edits land, and
greedy per-component improvement can miss combinations that are only good
together.

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

## 2023, Sclar: Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design or: How I learned to start worrying about prompt formatting

Saved as [2310.11324v2.md](../2310.11324v2.md). A negative
result for edit-based operators over format.

Along 300 chains per task of three formats each one atomic edit apart
(change a separator, a space, casing, or numbering), accuracy was monotonic
in 32 to 34 percent of chains, the rate expected from random ordering. The
authors conclude that local search methods such as simulated annealing are
not effective in this space. Individual atomic features also rarely predict
accuracy on their own; only the descriptor separator and enumeration
numbering had strong marginal effects on more than a few tasks, yet single
edits could shift accuracy by tens of points. Together with Voronov 2024's
interaction result, this says the format dimension of a prompt should be
sampled and selected, not mutated; the reflective and differential operators
in this file are operators over instruction content and should leave format
fixed or draw it from a grammar.

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

