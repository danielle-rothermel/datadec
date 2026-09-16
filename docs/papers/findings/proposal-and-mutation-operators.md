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

