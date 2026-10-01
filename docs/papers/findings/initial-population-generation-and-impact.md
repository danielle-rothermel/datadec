# Initial population generation and its impact on prompt optimization

What the papers under `docs/papers/` say about where the starting prompts for a
prompt optimizer come from, how to sample a diverse or strong set of them, and
how much the choice matters. Sections are ordered from most to least relevant.
Each heading gives year, first author, and title; the file name links to the
saved full text.

Summary across all six papers:

- Three papers start from a population rather than a single seed: Zhou 2022
  samples it from a proposer, Guo 2023 mixes human prompts with paraphrases,
  and Voronov 2024 samples from a factored grid of components. Only Guo 2023
  ablates the population's composition. The others
  either start from one hand-written seed (Agrawal 2025, Agrawal 2026,
  Liu 2026) or sample a population purely as a measurement instrument
  (Gao 2026).
- The cheapest evidence-backed recipe is Zhou 2022's: condition the proposer on
  a few input/output demonstrations, draw roughly 50 to 64 candidates, score
  with adaptive filtering, and skip local refinement unless the set is poor.
  Guo 2023 adds that padding a few human prompts with LLM paraphrases is enough
  and that ranking seeds is unnecessary, while an all-bad population does hurt.
- Two levers matter more than seed curation in every paper that measured them:
  proposer model strength (Zhou 2022, Agrawal 2026) and the meta-prompt used
  to elicit candidates (Zhou 2022, untested in Gao 2026).
- No paper measures the diversity of a candidate population directly, samples
  from several meta-prompts on purpose, or tests structurally different
  strategies as seeds. Single-seed methods obtain diversity during the run
  through Pareto-front retention, not at initialization.
- Zhao 2025's segment masking is an untested but mechanical route from one
  seed to a population: the seed plus its single-segment ablations, each with
  a measured loss delta. Ablative rather than lexical variation, bounded by
  the seed's content.
- Voronov 2024 builds its population by a third route: factor the prompt
  into slots, curate a small option pool per slot, and sample from the
  Cartesian product. That gives enumerable, structured diversity and
  component-level analysis for free. Its rankings are setting-specific
  (top-10 formats rarely overlap even within a model family) and components
  have no stable individual effect, so the space cannot be pruned by parts.
- Sclar 2023 generalizes the grid to a validated grammar with semantic
  equivalence classes, and shows the format space is non-monotonic under
  single edits, so for format the sampled population is the search rather
  than a starting point for local mutation.

## 2024, Lin: Prompt Optimization with Human Feedback (APOHF)

Saved as [2405.17346v1.md](../2405.17346v1.md); full summary in
[candidate-evaluation-and-budget-allocation.md](candidate-evaluation-and-budget-allocation.md).
The candidate pool is fixed at the start and never extended: 200 instructions,
each produced by APE-style induction from 5 exemplars drawn at random from a
100-exemplar set (a different draw per candidate), so pool diversity comes
entirely from which demonstrations the proposer saw. Everything after that is
selection. This is the same one-shot pool design as our APE stage, and the
paper's dependence on the pool being wide enough is why its exploration bonus
matters: with a fixed pool, search quality is bounded by what the initial
draw contained.

Era caveat: as in [candidate-evaluation-and-budget-allocation.md](candidate-evaluation-and-budget-allocation.md).

## 2024, Opsahl-Ong: Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs (MIPRO)

Saved as [2406.11695v2.md](../2406.11695v2.md). Instruction candidates are
proposed once, up front, by a "grounded" proposer: the meta-prompt can include
an LM-written summary of the dataset, a summary of the program, a set of
bootstrapped demonstrations, previously scored instructions, and one of six
short "tips" (none / creative / simple / descriptive / high-stakes / persona),
with the proposer temperature as a further knob. MIPRO++ then learns, with a
Bayesian model over trials, which of these to use for a given task; the
learned importances put the choice of demonstrations shown to the proposer and
the tip at the top across tasks, the dataset summary high for one task and
near the bottom for two others (Lesson 4). Grounding helped on two of three
tasks and hurt on the third.

Two points for our seed generation: the tip is the same device as
Promptbreeder's thinking-style draw (a framing factor recorded per candidate),
here with evidence that it is one of the two most important proposal knobs;
and which demonstrations the proposer sees matters more than the data summary,
which supports rendering the demonstrations exactly as scored (our per-format
rule) over describing the task in prose.

Era caveat: as in [candidate-evaluation-and-budget-allocation.md](candidate-evaluation-and-budget-allocation.md).

## 2022, Zhou: Large Language Models are Human-Level Prompt Engineers

Saved as [2211.01910v2.md](../2211.01910v2.md). APE is the canonical
sample-then-select method, so the whole paper is about the seed space.

How candidates are generated. The proposer is conditioned on a handful of
input/output demonstrations and asked to infer the instruction that produced
them, either in forward mode (a meta-prompt such as "I gave a friend an
instruction and they produced these outputs; the instruction was") or in
reverse mode (an infilling model fills the instruction slot). About 50
candidates are drawn per task and scored by execution accuracy on held-out
demonstrations, with adaptive filtering so that only promising candidates
receive the full evaluation budget. An optional iterative variant paraphrases
the best candidates and re-scores; the resampling template it uses is the one
Guo 2023 later adopted for population initialization.

Findings about the proposal space:

- Sample count is the dominant lever with diminishing returns. Best-candidate
  accuracy rose monotonically from 4 to 128 samples and reached human level
  around 64. Selection over many samples rescues weak proposers: small models
  rarely produce a good instruction, but with enough draws they produce some.
- Proposer quality still matters and hard tasks expose it. Larger and
  instruction-tuned proposers gave better distributions. On an easy task every
  proposal from the best model was usable; on a harder one, half were
  off-topic even from the best model.
- Iterating around the best barely helps. Resampling shifted the whole
  distribution upward and plateaued after about three rounds, but the top
  instruction usually stayed the same. It helped only on tasks where the
  initial set was poor. One-shot sampling with no iteration is their default.
- The meta-prompt substantially reshapes the distribution. Swapping templates
  helped some tasks and hurt others, and one template could reach instructions
  the other never proposed. Meta-prompt engineering for proposal distributions
  is explicitly left as future work.
- Match proposer to executor. Instructions selected for one model dropped
  sharply when run on another. A weaker model generating and selecting its own
  prompts beat both a stronger model's prompts and human prompts on itself.
- Larger aligned proposers are cheaper overall because they write more concise
  instructions, which lowers scoring cost.
- Execution accuracy beats log-probability as a selection score, and selection
  can be gamed: on a rhyming task the selected instructions told the model to
  echo the input.

Caveat: 2022-era models on short single-step instruction-induction tasks with
a few demonstrations. Whether 50 samples saturate the space for a multi-rule
reasoning system prompt is untested here.

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

## 2023, Guo: EvoPrompt: Connecting LLMs with Evolutionary Algorithms Yields Powerful Prompt Optimizers

Saved as [2309.08532v3.md](../2309.08532v3.md). The only paper here that runs
an evolutionary population and ablates how it was initialized.

How the population is built. About 10 prompts mixing existing human-written
instructions from prior work and prompt collections with LLM paraphrases
generated by APE's resampling template. The stated rationale is to combine
human prior knowledge with the randomness evolutionary algorithms normally get
from random initialization. The same recipe is used for classification and
generation tasks.

Initialization ablation (Table 6, SST-5, population of 10, genetic-algorithm
and differential-evolution variants):

- Careful seed selection is not essential. A random selection of 10 prompts
  finished within about half a point of the top-10 selection, inside the
  reported standard deviations.
- Bad seeds do hurt. Starting from the 10 worst prompts cost roughly 1 to 1.5
  points relative to random, the largest effect in the table.
- Paraphrase variations help a little on top of strong seeds and not at all on
  random ones. Replacing half the top prompts with LLM variations gave a small
  gain; doing the same to random prompts made no difference for differential
  evolution.
- The operator matters more than the seeds when seeds are weak. Differential
  evolution, which mutates only the parts that differ between two prompts,
  recovered better from bottom-10 seeds than the genetic-algorithm variant.

Diversity beyond initialization:

- Population size helps on harder tasks. Scaling from 4 to 12 improved
  classification results, more so for differential evolution because its
  mutation exploits differences between members. On an easy simplification
  task a population of 6 matched 10 at less than half the cost.
- Diversity is maintained by the operator during the run. Differential
  evolution produced longer, higher-variance prompts and mutated more new words
  in later iterations than the genetic algorithm.

Caveat: the ablation is on a five-class sentiment task with a 2023-era model,
effect sizes are small relative to variance, and diversity means paraphrases
of a fixed set of human prompts rather than structurally different strategies.

Era caveat: the task executor is Alpaca-7B (a 2023 LLaMA-1 fine-tune) for
classification and generation, with GPT-3.5 as the optimizer and for BBH;
tasks are SST-2/SST-5, Subj, AGNews-style classification, SAMSum, ASSET,
and 3-shot BBH. The initialization and operator ablations are on SST-5
only. Modern small models are near ceiling on these classification sets
and far more robust to prompt wording, so the absolute effect sizes (1 to 6
points) and the population-size curve should not be assumed to carry over.
The relative findings (bad-only seeds hurt, differing-parts mutation beats
whole-prompt mutation) are the part worth keeping.

## 2025, Yang: What Prompts Don't Say: Understanding and Managing Underspecification in LLM Prompts

Saved as [2505.13360v3.md](../2505.13360v3.md). Treats a prompt as a set of
atomic requirements (20 per task, curated from existing prompts, brainstorming
and error analysis) and builds the prompt population combinatorially: each
prompt states N of the 20 requirements, chosen by a cyclic design so every
prompt has the same N and every requirement is specified equally often. That
is a balanced, structured seed population, and it is the same idea as
building seeds by masking parts of one long prompt (our earlier note), with
the mask chosen for balance rather than at random.

Findings that matter for seeding: models follow many requirements without
being told (41% of unspecified requirements at above 98% accuracy; format
requirements 71%), but which ones is model-specific and unstable across
prompts (2x the std of specified ones) and across model versions (2x the
regression rate). Specifying everything is an anti-pattern: accuracy on
specified requirements falls from 98.7% in isolation to 85% (GPT-4o) and 80%
(Llama-3.3-70B) at 19 requirements. So the seed population should span how
much is specified, not only what, and smaller models should get shorter
prompts.

Era caveat: Llama-3.3-70B-Instruct, three GPT-4o versions and o3-mini on
open-ended generation tasks (code explanation, trip advice, product
descriptions) with LLM-judge validators; nothing is likelihood-scored, and
the models are far stronger instruction followers than ours. The
combinatorial construction transfers; the guess rates do not.

## 2023, Lu: Strings from the Library of Babel: Random Sampling as a Strong Baseline for Prompt Optimisation

Saved as [2311.09569v2.md](../2311.09569v2.md); full summary in
[template-selection.md](template-selection.md). Bears on initialization as a
control: a seed population of random strings (vocabulary tokens or LM-prior
phrases) scored on the training set matched LLM-proposed populations to
within 1% after best-of-160 selection. Any claim that a generated seed pool is
better than random should be tested against this population, drawn with the
same budget and selected the same way. Their "language space is rich with
good separators" result (a >40% chance per draw of beating the human default on
base models) is also the reason a wide, cheap initial population can beat a
narrow, expensive one.

Era caveat: as in [template-selection.md](template-selection.md).

## 2023, Fernando: Promptbreeder: Self-Referential Self-Improvement via Prompt Evolution

Saved as [2309.16797v1.md](../2309.16797v1.md). Initial population of 50
units, each unit = task-prompt(s) + its own mutation-prompt. Every initial
task-prompt is generated by concatenating a randomly drawn mutation-prompt
(from a hand-written list of ~50, e.g. "Make a variant of the prompt"), a
randomly drawn thinking-style (from ~40, e.g. "Let's think step by step"), and
the task description, and letting the LLM continue after "INSTRUCTION MUTANT:".
The two random draws are the diversity mechanism: the same description is
re-described 50 ways through different framings. The ablation (Appendix L,
population 10, 200 evaluations) found this thinking-style-guided
initialization the single most valuable component across all math datasets;
replacing it with the bare task description hurt more than removing any
mutation operator. Random draw of the initial mutation-prompt was the one
component that hurt on one task (GSM8K).

For our seeding: this is an APE-style pool generated from a description
rather than from demonstrations, with diversity injected by crossing two
small hand-written option lists. It is the cheapest way we have not tried to
widen a description-only pool (our `p1_description` style used one fixed
framing), and the ablation says the widening is where the value is.

Era caveat: PaLM 2-L (2023), two-stage generated answers on GSM8K-family
arithmetic, CommonsenseQA/StrategyQA, ETHOS and the APE instruction-induction
tasks; fitness is accuracy on a 100-item batch. No likelihood scoring, and
modern instruct models are far less sensitive to thinking-style framings than
PaLM 2 was, so the size of the initialization effect is era-specific; the
mechanism is not.

## 2025, Ferreira: Diverse Prompts: Illuminating the Prompt Space of Large Language Models with MAP-Elites

Saved as [2504.14367v1.md](../2504.14367v1.md); full summary in
[population-maintenance-and-search-diversity.md](population-maintenance-and-search-diversity.md).
Initial population is 50 random expansions of a structural grammar
(role-context yes/no, 0/few/many shots, reasoning-depth instruction yes/no
with depth 1-10, plus task-specific request and answer instruction). The
MAP-Elites archive then converts that into a population stratified by
structure (shots x length x reasoning depth), which is the paper's actual
contribution to initialization: a seed set is judged by how many structural
cells it fills with a competent prompt, not by its best member. Random
sampling from the same grammar filled far fewer cells with competent prompts.

Era caveat: as in the population-maintenance file; 2024-2025 small instruct
models, binary BigBench Lite tasks, 50-instance fitness.

## 2024, Schnabel: Symbolic Prompt Program Search (SAMMO): A Structure-Aware Approach to Efficient Compile-Time Prompt Optimization

Saved as [2404.02319v2.md](../2404.02319v2.md). The initial population is
not the focus, but two things bear on it. First, InitCandidates is an
explicit step in the search skeleton, and APE's few-shot induction is one
instantiation of it (InduceInstructions is a mutator, so an initial
population can be regenerated from examples at any point in the search, not
only at the start). Second, the enumerative mode is itself a way to build a
seed population over format: a small explicit grid of structural choices,
evaluated exhaustively with 24 calls, before any text mutation. Because
candidate scores correlate weakly across backends (Figure 5), a seed
population built for one model is a poor prior for another.

Era caveat: as in [template-selection.md](template-selection.md); 2023-era
black-box backends, generation-scored tasks.

## 2024, Voronov: Mind Your Format: Towards Consistent Evaluation of In-Context Learning Improvements

Saved as [2401.06766v3.md](../2401.06766v3.md). Not an optimizer, but its
template construction is a population-generation method in its own right,
and a different kind from the others here.

How the population is built. A template is factored into four slots: input
verbalizer, output verbalizer, intra-separator (between input and label), and
inter-separator (between demonstrations). Each slot has a small pool of
human-written options gathered from prior work (the LM-BFF verbalizers of
Gao et al. 2021, minimal "{}" templates, and universal "input/output" style
labels), listed in Table 2. Every combination is a valid template, giving 216
formats for SST-2 and 168 for the other datasets. Experiments then draw 10
templates uniformly at random per demonstration seed, and 30 for the transfer
analysis. So the recipe is: curate option pools per component, take the
Cartesian product, sample uniformly. Label words themselves were held fixed
and are noted as an unexplored slot.

What this recipe offers that sampling and paraphrase do not:

- Structured, enumerable diversity. Members differ along known axes, the
  whole space can be listed, and coverage is uniform by construction rather
  than dependent on a proposer's temperature or a meta-prompt.
- Component-level analysis for free. Because every member is a known
  combination, the effect of each option can be measured marginally over the
  others (Appendix D). That analysis is what produced the findings below.
- Cheap extension. New options in any slot multiply the space without new
  design work.

What it costs: the axes and options are hand-chosen, so the population can
only contain what the designer put in the pools, and the space grows
multiplicatively, which is why they sample rather than enumerate. It also
only covers format; content variation would need slots of its own.

Findings about the resulting population:

- The best members do not transfer. Intersection-over-union of the top-10
  templates between models exceeded 0.5 for only a few pairs, including
  models in the same family trained on the same data. Transfer between
  prediction methods (Direct, Channel, Calibration) was similarly low, and
  changing the demonstration set, even adding demonstrations chosen by the
  same method, reordered the top templates. A population ranked for one
  executor or scoring rule has to be re-ranked for another; Zhou 2022 reached
  the same conclusion for instructions.
- The top of the ranking is flat. The tenth-best of 30 templates averaged
  about 90 percent of the best template's score, so a modest random draw
  from the grid reliably contains a near-best member, the same
  diminishing-return shape Zhou 2022 saw for instruction samples.
- Components have no stable individual effect. No verbalizer or separator was
  consistently bad, part rankings flipped between models, and good templates
  were assembled from parts that were individually mediocre. The space
  cannot be pruned by dropping options. This is also a caution for the
  masking note under Zhao 2025 below: a single-segment loss delta is specific
  to the executor and to the other segments present, so a masked
  population's attribution labels are local to the setting they were
  measured in, not properties of the segments.

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

Saved as [2310.11324v2.md](../2310.11324v2.md). Extends
Voronov 2024's grid to a recursive grammar, and adds two results about
searching the resulting population.

How the population is built. A Backus-Naur grammar composes basic fields
(descriptor, separator, text slot) with joiners, enumerations, casing
functions, and item-numbering functions, each drawing from a small
user-defined constant set. It generates the formats of 100+ human-written
tasks, so it is a validated model of the plausible-format space rather than a
hand-picked list. An equivalence relation restricts sampling to
meaning-preserving variants of a given starting format, and contextual
restrictions rule out unnatural combinations. This is the grid recipe with
structure: nested rather than flat slots, and an explicit notion of which
members are semantically the same as the seed. Descriptor paraphrase is
mentioned as a possible extra function but not used.

What it says about populations of formats:

- Sample rather than mutate. Accuracy along chains of single atomic edits
  was monotonic at chance rate, so a local search from a seed format has no
  gradient to follow; the paper's own tool samples formats and allocates
  evaluation budget among them instead of mutating. For the format
  component of a prompt, an initial population drawn from the grammar is
  the search, not a warm start for it.
- The population's ranking is model-specific. If one format beats another
  on one model, the same order holds on a second model with probability
  below 0.62; there are no inherently good formats to seed with.
- Ten members is a lower bound. Spread kept growing from 10 to 20 sampled
  formats on about a sixth of tasks, so population size for formats should
  be set by budget, not by an assumed saturation.

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

## 2026, Gao: p1: Better Prompt Optimization with Fewer Prompts

Saved as [2604.08801v2.md](../2604.08801v2.md). Samples a population of system
prompts, but only as an instrument for filtering the training set, not as a
starting point for optimization.

How candidates are sampled. N system prompts (N = 16) are drawn from the
untrained generator policy, the same model as the response model, conditioned
on a one-sentence meta-prompt given verbatim in Appendix A. Training-time
sampling is temperature 1 with top-p 1; the paper does not state which setting
the filtering stage used. Generation length is 4,096 tokens. There is no
diversity mechanism, dedup, or seeding; diversity comes only from independent
draws.

What the candidates are used for. For every candidate and every user prompt,
M responses are sampled to estimate a success-probability matrix. Every
subset of user prompts of size K_top (default 2) is scored by the variance of
candidate mean rewards over the subset minus an estimate of response-sampling
noise, and the argmax subset becomes the training set. Training on two AIME 24
questions chosen this way beat full-dataset RL and GEPA on held-out reasoning
benchmarks; on the more homogeneous IFBench the filter underperformed the full
set.

What this says about initial populations:

- The ranking of user prompts is a function of the specific 16 candidates, so a
  different draw changes the selection. The paper acknowledges this indirectly
  by reporting two top-ranked subsets per setting.
- No ablation of N, the meta-prompt, or sampling temperature for the
  filtering stage.
- The filtered subset did not measurably help or hurt GEPA (all GEPA rows sit
  at the base-model level on AIME), and the filtering stage itself costs
  N x K x M rollouts, which at the analysis settings is more compute than the
  optimization it is meant to shorten.

## 2026, Liu: Reflection in the Dark: Exposing and Escaping the Black Box in Reflective Prompt Optimization

Saved as [2603.18388v2.md](../2603.18388v2.md). Single-seed like GEPA, but the
paper's first claimed limitation is about the seed, and its proposed method
adds a restart mechanism.

The seed-trap finding. Reflective optimizers inherit structural defects in the
seed because every child descends from it and the reflector never proposes
seed-level root causes. GEPA's official GSM8K seed has its output fields in the
wrong order, disabling chain-of-thought; GEPA never fixed this across any
round or reflector and dropped accuracy below the unoptimized seed. With a
repaired seed, all methods landed within a point of each other.

What VISTA does about it. With probability p per round (0.2 in experiments) it
builds a fresh prompt by running one training instance under the current
prompt, handing the raw output and any parse error to the rewriting model, and
iterating until the output parses. The restart is conditioned on the model's
natural behavior rather than the seed. It produces one candidate at a time and
is kept only if it beats the current prompt on the minibatch. Within a round,
K = 3 rewrites target different hypothesized root causes drawn epsilon-greedily
from a hand-curated failure taxonomy.

What the ablation says. The heuristic taxonomy accounts for nearly 60 of the
74-point gain; restart and parallel sampling add a few points each.
Performance dropped at K = 5, attributed to noise from extra candidates.

What this says about initial populations: the seed is not neutral even when
the algorithm treats it as a plain input, and a restart from model behavior is
one way to escape it. Nothing here builds or evaluates a population of seeds.

## 2025, Agrawal: GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning

Saved as [2507.19457v2.md](../2507.19457v2.md). Single seed, set by the
experiment designer.

Initialization. Algorithm 1 takes the system as input and initializes the
candidate pool as a single-element list containing its prompts as given.
Everything after that is mutation of existing candidates, either reflective
rewriting of one module's prompt on a minibatch or a merge of two candidates.
There is no sampling of an initial population, no meta-prompt, and no warm-up.
The seed's per-instance scores on the Pareto split are computed once before the
loop, so a weak seed anchors the Pareto front until a child beats it on at
least one instance.

Seeds used. Appendix L lists deliberately minimal base prompts per module:
"Respond to the query" for the IFBench responder, and DSPy auto-generated
signature text such as "Given the fields question, summary_1, produce the
fields query" for retrieval modules. All task-specific content came from
reflection.

Where diversity comes from. Pareto-based candidate selection keeps any
candidate that is best on at least one instance and samples parents in
proportion to how often they appear on the front. Diversity is therefore a
property of the run, not of initialization.

## 2026, Agrawal: optimize_anything: A Universal API for Optimizing any Text Parameter

Saved as [2605.19633v1.md](../2605.19633v1.md). GEPA generalized to arbitrary
text artifacts; initialization is unchanged.

Initialization. The core loop initializes the pool with one artifact and only
ever mutates pool members. Every main experiment started from a deliberately
weak seed: a single generic sentence for the AIME system prompt, a ten-line
single-call agent for ARC-AGI, a greedy heuristic for circle packing, plain
Dijkstra for cloud routing. These are presented as evidence the optimizer does
not need a good start, not as a study of what a good start would be.

Seedless mode. Passing no seed makes the proposer write one bootstrap
candidate from a natural-language objective. Demonstrated once (a 3D unicorn)
with no ablation and no comparison against a hand-written seed. It produces one
candidate, not a population.

Relevant findings. No experiment varies the seed. Diversity is attributed to
Pareto-front retention of candidates from several algorithmic families and to
an auxiliary refiner prompt tracked on the same front. Proposer strength
matters more than the seed in their data: on AIME the same seed reached 60
percent with GPT-5.1 as proposer and 50 percent with GPT-5-nano, from a 46.67
percent baseline.

## 2025, Zhao: PMPO: Probabilistic Metric Prompt Optimization for Small and Large Language Models

Saved as [2505.16307v2.md](../2505.16307v2.md). Single seed, fixed across methods.

Every optimizer in their comparison starts from the same chain-of-thought
baseline prompt, "Let's think step by step" (Appendix A.2.3), so the paper
says nothing about seed choice beyond holding it constant for fairness. The
first iteration's rewrite already replaces it with a task-specific
instruction, so the seed's content has little influence on the result;
what it does fix is the segmentation the mask analysis starts from.

Note from Danielle: the masking step is itself a way to turn one seed into an
initial population automatically. Segmenting the seed into up to 5 removable
units and masking each one yields the seed plus up to 5 variants, each
differing from the seed by exactly one identifiable component, and the same
forward passes that score them produce the per-segment loss deltas. Masking
subsets of segments extends this to up to 2^m variants with known structure.
Unlike paraphrase resampling (Zhou 2022, Guo 2023) the variation is
ablative rather than lexical, it is deterministic given the segmentation, and
every member comes with an attribution of what it lacks. The limits are the
mirror image: all members are subsets of the seed, so the population cannot
contain anything the seed did not, and the segmenter's choice of units bounds
the diversity. A natural combination is to use masked variants as the
structured core of a population and sampled or paraphrased candidates for
content the seed lacks.
