# In-context example selection

Which demonstrations go inside the prompt, how many, in what order, and
whether they are optimized at all. This is about the few-shot examples the
executor sees, not the training inputs the optimizer scores candidates on;
that is [user-input-selection.md](user-input-selection.md). Sections use the
year, first author, title format and link to the saved full text under
`docs/papers/`.

Summary:

- Under template variation, learned demonstration-selection methods did not
  reliably beat random selection, and one was consistently worse
  (Voronov 2024). More demonstrations raise the mean without reducing
  format variance, and changing them reorders which templates are best.
- Optimizing the instruction alone now beats jointly optimizing instructions
  and demonstrations on every benchmark GEPA tested, with a smaller
  generalization gap and much shorter prompts (Agrawal 2025), reversing the
  earlier finding that exemplars generalize better.
- Instruction gains and demonstration gains are additive: an optimized
  instruction stayed about 5 points ahead of the base instruction at every
  shot count from 0 to 9, and demonstration gains saturated around 5 shots
  (Zhao 2025).
- A candidate instruction should be selected in the shot setting it will be
  used in; zero-shot-selected instructions sometimes hurt with
  demonstrations added (Zhou 2022).
- Most optimizers in this set hold demonstrations fixed or use none:
  Gao 2026 is zero-shot, Guo 2023 fixes 0, 1, or 3 shots per task family,
  and Voronov 2024 randomizes them as a nuisance variable.
- Local: OLMES uses a fixed, curated 5-shot set per task, the same five
  ARC-Easy questions for every model and both formulations, pinned in the
  fork's `fewshot_sources.py`. Demonstrations are therefore a controlled
  variable in our evals, and any optimization we build has to decide up
  front whether they are part of the optimized artifact or part of the
  fixed frame.

## TODO

- Pull the OLMES paper (Gu et al. 2024) for how the curated few-shot sets
  were chosen and why they are fixed rather than sampled.
- Pull Wan et al. 2024 ("Teach Better or Show Smarter? On Instructions and
  Exemplars in Automatic Prompt Optimization"), the paper GEPA's
  instruction-versus-exemplar claim is arguing against, and Opsahl-Ong et
  al. 2024 (MIPROv2) for the joint bootstrapping procedure.
- Candidates from Voronov 2024's related work on demonstration effects:
  Lu et al. 2022 (order sensitivity), Min et al. 2022 (what demonstrations
  actually convey), Liu et al. 2021 (KATE, similarity-based retrieval).

## 2024, Opsahl-Ong: Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs (MIPRO)

Saved as [2406.11695v2.md](../2406.11695v2.md). Demonstrations are
bootstrapped: training inputs are run through the program and any trace whose
final output passes the metric becomes a candidate few-shot example; N sets of
K such examples are then searched over (random search, or the TPE surrogate).
Lesson 1 of the paper: optimising bootstrapped demonstrations alone beat the
best instruction-only optimiser on every task but one (Wilcoxon p < .05), and
different demonstration sets vary a lot in outcome, so which examples are shown
carries information about successful reasoning rather than just format. Lesson
3: instruction optimisation only wins when the task has conditional rules the
model does not know and that a few examples cannot convey; on those tasks a
seed instruction stating the rules is necessary because the optimiser cannot
infer them.

For us: OLMES fixes five curated demonstrations; this paper is the strongest
argument in the collection that demonstration choice is a larger lever than
instruction text, and that a demonstration-count-and-selection axis belongs in
the format grid (see [template-selection.md](template-selection.md)).

Era caveat: as in [candidate-evaluation-and-budget-allocation.md](candidate-evaluation-and-budget-allocation.md); demonstrations here are full reasoning traces, not question-answer pairs.

## 2024, Voronov: Mind Your Format: Towards Consistent Evaluation of In-Context Learning Improvements

Saved as [2401.06766v3.md](../2401.06766v3.md). The only paper here that
evaluates demonstration-selection methods under controlled format variation.

Methods compared, 4-shot, Direct scoring, 10 templates times 3 seeds: random
selection; ITM, which learns a latent task-concept model and picks examples
that best predict the concept (concept model trained once on GPT-2 Large
and reused across executors, since the authors report its picks transfer);
and z-ICL, which retrieves the most similar unlabeled sentences to the test
input and assigns random labels.

Findings:

- ITM raised the mean over random in most model and dataset cells but with
  standard deviations across templates as large as random's, so its gain
  was inside the format spread. z-ICL was more consistent and worse. For
  instruction-tuned models ITM was slightly more template-robust than
  random, and z-ICL was again worst.
- Going from 0 to 2 to 4 demonstrations raised the mean for both base and
  instruction-tuned models but did not reduce variance across templates.
- Demonstrations and templates interact: adding demonstrations chosen by
  the same method reordered the top-10 templates (Section 5.4), so a
  demonstration set cannot be chosen independently of the format it will be
  shown in.
- Order matters and does not transfer between models, citing Lu et al.
  2022; this was not re-measured here.
- The papers proposing ITM and z-ICL used different templates and label
  words from each other (Appendix A), so their published comparisons are
  not on equal footing.

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

## 2025, Agrawal: GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning

Saved as [2507.19457v2.md](../2507.19457v2.md). GEPA optimizes instructions
only; MIPROv2, its main baseline, jointly optimizes instructions and
bootstrapped demonstration sets (18 instruction candidates and 18 few-shot
sets in the heavy setting, searched by a Bayesian optimizer). GEPA beat
MIPROv2 on all six tasks with both executors, by up to 11 points, with more
than double the aggregate gain over baseline. The authors acknowledge this
reverses Opsahl-Ong et al. 2024 and Wan et al. 2024, who found exemplar
optimization stronger, and attribute the change to better instruction
following and self-reflection in current models. They re-ran Wan et al.'s
generalization-gap analysis and found evolved instructions now have the
smaller gap. Evolved instructions were also much shorter than demonstration
prompts, which matters when one demonstration for a complex task is already
long. The evolved prompts contained declarative task rules rather than
quasi-exemplars.

## 2025, Zhao: PMPO: Probabilistic Metric Prompt Optimization for Small and Large Language Models

Saved as [2505.16307v2.md](../2505.16307v2.md). Appendix A.4.3 varies shot
count from 0 to 9 on BBH with a base and a PMPO-optimized instruction. The
optimized instruction led by roughly 5 points at every shot count (79.2
versus 73.8 at zero shots, 82.4 versus 78.0 at nine). Demonstration gains
were concentrated in the first few shots and flat from about 5 onward for
both instructions. Their reading is that instructions and demonstrations
address different things, generality versus task-specific grounding, and
stack. Demonstrations were randomly drawn, not selected.

## 2022, Zhou: Large Language Models are Human-Level Prompt Engineers

Saved as [2211.01910v2.md](../2211.01910v2.md). Demonstrations play two
roles. They condition the proposer: candidates are inferred from a few
input-output pairs. And they are part of the deployment context: inserting
a selected instruction before demonstrations matched or beat plain few-shot
on 21 of 24 tasks, but hurt on three, which the authors attribute to
instructions selected by zero-shot accuracy overfitting that setting.
Selecting by few-shot accuracy instead recovered all but one task. The
number of demonstrations shown to the proposer and the number of proposals
per demonstration set were tuned hyperparameters that moved five tasks to
human level.

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

Saved as [2309.08532v3.md](../2309.08532v3.md). Demonstrations are fixed per
task family and not optimized: 1-shot for classification with Alpaca, 0-shot
for generation, and 3-shot chain-of-thought for BBH. The evolved instruction
is inserted into a fixed task template around them.

Era caveat: the task executor is Alpaca-7B (a 2023 LLaMA-1 fine-tune) for
classification and generation, with GPT-3.5 as the optimizer and for BBH;
tasks are SST-2/SST-5, Subj, AGNews-style classification, SAMSum, ASSET,
and 3-shot BBH. The initialization and operator ablations are on SST-5
only. Modern small models are near ceiling on these classification sets
and far more robust to prompt wording, so the absolute effect sizes (1 to 6
points) and the population-size curve should not be assumed to carry over.
The relative findings (bad-only seeds hurt, differing-parts mutation beats
whole-prompt mutation) are the part worth keeping.

## 2026, Gao: p1: Better Prompt Optimization with Fewer Prompts

Saved as [2604.08801v2.md](../2604.08801v2.md). Zero-shot: the prompt is the
system prompt followed by the user prompt with no demonstrations, so the
paper has no bearing on example selection except as a design point where
the whole task-specific burden is carried by the instruction.
