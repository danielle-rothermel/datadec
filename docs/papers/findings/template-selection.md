# Template selection

The template is the fixed format wrapper around a prompt: how the
instruction, demonstrations, and query are laid out, the verbalizers and
separators, the answer format (free text, choice text, or a letter), and the
field order in structured outputs. It is distinct from the instruction's
content, which the other files cover, and from demonstration choice, covered
in [icl-example-selection.md](icl-example-selection.md). Sections use the
year, first author, title format and link to the saved full text under
`docs/papers/`.

Summary:

- Format alone moves scores by a large fraction of their value, and the best
  format is specific to the executor, the scoring rule, and the
  demonstrations (Voronov 2024). Any comparison between candidate prompts,
  or between optimizers, made on a single template is partly a comparison
  of template luck.
- Templates are a factored space and can be enumerated and sampled; no
  component is safely prunable because components interact (Voronov 2024).
  SAMMO (Schnabel 2024) searches the structural end of that space
  (serialization, grouping, section markup, example count) as an explicit
  grid or through mutators, and finds format winners transfer weakly
  between models.
- A template defect can silently cap an optimizer: a wrong output-field
  order disabled chain-of-thought and reflective rewriting never diagnosed
  it (Liu 2026). Optimizers that only edit instruction text inherit the
  template they are given (Agrawal 2025, Zhao 2025).
- The meta-prompt used to elicit candidates is itself a template with large
  effects on what gets proposed (Zhou 2022).
- Ensembling predictions over 4 or 5 templates is a cheap way to remove
  format variance at inference time (Voronov 2024); Bozhenko 2025 finds
  probability-averaged ensembles cost accuracy while majority voting does
  not, and that Batch Calibration removes more format spread than either
  at zero cost. Likelihood ranking is more format-robust than generation.
- Local: the OLMES RC and MC formulations are two templates for the same
  items. For the 150M DataDecide model MC is at chance (0.26) while RC is
  0.51, so at small scale the template decides whether the score carries any
  signal. Both formats and the five fixed demonstrations are pinned in the
  OLMES fork, so template is a controlled variable in our local evals.
- Sclar 2023 measured the same effect at larger scale with a
  meaning-preserving format grammar: median spread 7.5 points from 10
  formats, maxima above 70, model rankings that reverse with significance
  in both directions, and a format space that is non-monotonic under atomic
  edits. FormatSpread estimates the spread interval with a Thompson-sampling
  bandit at a few percent of exhaustive cost.

## TODO

- Pull the OLMES paper (Gu et al. 2024, "OLMES: A Standard for Language
  Model Evaluations") for its rationale on the RC and MC formats, the
  curated few-shot sets, and its per-format normalization choices.
- Pull Zhao et al. 2021 ("Calibrate Before Use") for the calibration
  scoring rule.

## 2025, Bozhenko: When Punctuation Matters: A Large-Scale Comparison of Prompt Robustness Methods for LLMs

Saved as [2508.11383v1.md](../2508.11383v1.md). Benchmarks five format-robustness
methods on 8 open models (Llama 3.x 1B/3B/8B, Qwen 2.5 1.5B/3B/7B, Gemma 2
2B/9B) across 52 Natural Instructions classification and multiple-choice
tasks (1000 items each), 2-shot with demonstrations rendered in the same
format as the query, 10 formats per task drawn from a Sclar-style component
grammar (descriptor case, descriptor separator, inter-field space,
text-to-option separator, option item style, option item wrapper; 4 to 16
values per component, Appendix G), plus GPT-4.1 and DeepSeek V3 on 10 tasks.
Robustness is spread (max minus min accuracy over formats) and the std over
formats; method wins are decided by a paired t-test over tasks.

Findings that bear on format selection:

- Probability ranking over the answer options is always more format-robust
  than greedy generation of the answer (Section 4.3); Gemma is the extreme
  case. Our likelihood scoring is already the robust choice.
- Batch Calibration, which subtracts the batch-mean log-probability of each
  option before the argmax (Zhou et al. 2024), raises accuracy on all 8 models
  and significantly reduces spread on 6 of 8, with no training and no extra
  forward passes. It is the single most effective intervention they test.
- Template Ensembles (probability averaging over 5 formats) reduce spread
  on 4 of 8 models but lower accuracy, because one bad format in the
  ensemble drags the mean; majority voting over formats does not have that
  failure and slightly improves frontier-model accuracy.
- LoRA with format augmentations raises accuracy but improves spread on
  only 1 of 8 models; robustness is not learned from exposure alone.
- Spread grows with format complexity (number of components in the format,
  Appendix E), and frontier models are far more robust (spread 0.03 to
  0.05 vs 0.16 to 0.19 for 7B-8B open models) but still show 8 to 10 point
  spreads on individual tasks.

Relation to our grid: the component grammar is the same family as ours and
Sclar's, with two components we lack (text-to-option separator, option
wrapper as a separate axis from item style) and larger value pools. The
paper does not select formats; it samples 10 at random per task and treats
the spread across them as the quantity to minimize.

Era caveat: models are 2024-2025 open instruction-tuned models from 1B to 9B
plus two frontier APIs, so the executors are current; tasks are Natural
Instructions classification and MC with 2 to 4 options, and the accuracy
reported is over generated or ranked labels, not likelihood. All methods are
inference- or training-time robustness fixes rather than optimizers.

## 2023, Lu: Strings from the Library of Babel: Random Sampling as a Strong Baseline for Prompt Optimisation

Saved as [2311.09569v2.md](../2311.09569v2.md). The "separator" is the string
between the input and the scored label, i.e. exactly our answer-descriptor slot
("Answer:"). Three ways of drawing it at random: tokens sampled uniformly from
the vocabulary, phrases sampled from the model's own prior with no context, and
phrases sampled with a few training examples in the meta-prompt. Each draw is
scored on 64 training items by label-probability argmax (likelihood scoring,
as in our RC/MC), up to 160 draws, best-on-train kept and reported on a
held-out test set. Nine classification tasks, eight models from GPT-2 Large to
ChatGPT.

Results: random vocabulary strings beat the human "Answer:" by 10% relative on
average and sit within 1% of OPRO and APE; natural-language random phrases add
0.5%, task-conditioned ones a further 0.3%. On AGNews a single random draw
beats "Answer:" 37-71% of the time for base models and 14-21% for chat models.
Best random separators transfer poorly across tasks (Table 10) but reasonably
across demonstration sets within a task (73% vs 51% for human ones). On GSM8K
the best of the random draws beat the human chain-of-thought trigger by 15-30%
relative, though the average random draw did not.

For our grid: the descriptor-pair axis is a three-option slice of this space,
and the paper says the slice is where a large part of prompt-optimisation gains
actually live. Two uses: (1) random separator draws are the null baseline any
optimiser must beat, and belong in every seed population as controls; (2)
selection of a best-of-k random string on a small training set is itself a
prompt optimiser, so its train-test gap is the winner's-curse floor for our
own searches.

Era caveat: 2023 models (GPT-2, Llama-2 7B, Mistral 7B, Alpaca, ChatGPT 0613)
on short-text classification with one-shot demonstrations; selection on 64
items has a binomial sd of about 0.06, and "chance of beating the human
baseline" is measured on those same 64 items. Current instruct models are less
separator-sensitive (the chat-model rows already show it), so the size of the
random-string effect is era-specific; the baseline argument is not.

## 2025, Ferreira: Diverse Prompts: Illuminating the Prompt Space of Large Language Models with MAP-Elites

Saved as [2504.14367v1.md](../2504.14367v1.md); full summary in
[population-maintenance-and-search-diversity.md](population-maintenance-and-search-diversity.md).
Relevant here for the axes it treats as template structure: number of
demonstrations (0, 1-2, 3+), presence of a role-context preamble, and a
reasoning-depth instruction, all above a fixed task request and answer
instruction. Number of shots was the one axis with a task-dependent effect
(zero-shot best on three tasks, few/many-shot best on three others); the
preamble and reasoning slots, filled from generic templates, did nothing. It
is a structural counterpart to our surface-level grid, with the same lesson
as SAMMO: demonstration count is a live axis and generic preambles are not.

Era caveat: as in the population-maintenance file.

## 2024, Schnabel: Symbolic Prompt Program Search (SAMMO): A Structure-Aware Approach to Efficient Compile-Time Prompt Optimization

Saved as [2404.02319v2.md](../2404.02319v2.md). Treats the whole prompt as a
symbolic program (a graph of sections, data renderers, and example lists)
so that format is a first-class search variable alongside instruction text.

Format choices are searched two ways. Enumerative search when the choices
are known a priori: on retrieval-augmented semantic parsing the grid was
in-context example format (JSON, plaintext, XML) x grouping (by item vs by
input/output) x number of examples (5, 10) x DSL spec (full vs signatures),
24 evaluations total, and it gave 30 to 133 percent relative gains
depending on the backend. Iterative (beam) search when the space is implicit:
structural mutators such as ChangeSectionFormat (markdown vs XML section
rendering), ChangeDataFormat, DecreaseInContextExamples, DropSection and
RepeatSection sit next to text mutators (Paraphrase, InduceInstructions,
ShortenText, TextToBulletPoints, RemoveStopwords), sampled uniformly at
random per round. Two findings matter for us: which mutators help depends on
the backend model (Figure 7), and the training scores of the same 24 format
candidates correlate only weakly across backends (Figure 5), so a format
found for one model should not be assumed to transfer. Their evaluation and
training sets are 100 examples each, chosen as the practical ceiling for
hand labeling, with a budget of 24 to 48 candidate evaluations.

Relation to our grid: SAMMO's format axes are coarser and more structural
than Sclar's (data serialization, grouping, section markup, example count)
where ours are surface-level (descriptor words, separators, casing, label
style). The two are complementary rather than alternatives; our grid has no
structural axis at all.

Era caveat: backends are GPT-3.5 (0613), GPT-4 (0613), Llama-2-70B-chat and a
Mixtral 8x7B fine-tune, all black-box text generation with no
probabilities; tasks are BigBench zero-shot classification with headroom
below 0.9, semantic parsing to DSLs, and Super-NaturalInstructions
compression. The instruction-tuning gains shrink as the backend gets
stronger (2x for Llama-2, 10 percent for GPT-3.5), which is the same trend
we see between DataDecide and Qwen. Nothing is scored by likelihood.

## 2024, Voronov: Mind Your Format: Towards Consistent Evaluation of In-Context Learning Improvements

Saved as [2401.06766v3.md](../2401.06766v3.md). The primary source for this
topic; the population-construction view of the same paper is in
[initial-population-generation-and-impact.md](initial-population-generation-and-impact.md).

Template space. Four slots (input verbalizer, output verbalizer,
intra-separator, inter-separator) with small human-curated option pools
drawn from prior work, giving 168 to 216 combinations per dataset; label
words were fixed. Ten random templates per demonstration seed, three seeds,
21 models, four classification datasets, scored by label likelihood.

Findings:

- Sensitivity is large at every scale. Standard deviation across templates
  reached 35 percent of the mean for 40B and 70B models, a poor template
  reduced strong models to chance, and instruction-tuned models were not
  more robust. More demonstrations raised the mean but did not shrink the
  variance.
- Reported method gains sit inside the template spread. Advanced prediction
  methods (Channel, Calibration) and example-selection methods (ITM, z-ICL)
  overlapped with the plain baseline's range across templates; Appendix A
  shows the papers proposing these methods used disjoint template sets, so
  their reported orderings may be template artifacts.
- No universal best template. Top-10 overlap between models exceeded 0.5
  for few pairs, including within a family; overlap between scoring rules
  was 0.23 to 0.54; adding or changing demonstrations reordered the top.
  The tenth-best of 30 was about 90 percent of the best, so the top is flat
  but its membership is setting-specific.
- Components interact. Every slot showed high variance, the best option per
  slot flipped between models, and optimal templates contained individually
  poor parts, so the space cannot be reduced by dropping options.
- Template Ensembles. Averaging label probabilities over 5 random templates
  raised accuracy for every model and scoring rule and cut variance; gains
  saturate at 4 or 5; majority vote was worse on many-class tasks; cost is N
  forward passes.

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

Saved as [2310.11324v2.md](../2310.11324v2.md). FormatSpread. The
larger-scale predecessor of Voronov 2024, with the same conclusion and a
search tool.

Format space. A hand-written grammar over descriptors, separators, spaces,
casing, and enumeration styles (letters, numerals, wrappers) that generates
the formats of 100+ Super-NaturalInstructions tasks, with an equivalence
relation so that only meaning-preserving variants of a task's original
format are explored, and contextual restrictions (for example no newline
inside a field when fields are joined without newlines) to keep formats
natural. Instruction and demonstration choice are held fixed so format is
the only variable.

Findings, 53 classification and multiple-choice tasks, LLaMA-2 7B to 70B,
Falcon 7B and 7B-Instruct, GPT-3.5, 1- and 5-shot:

- Spread is large and does not go away. With only 10 sampled formats the
  median spread was 7.5 accuracy points, 20 percent of tasks had at least 15
  points under every LLaMA-2 setting, several tasks exceeded 70 points, and
  the maximum for LLaMA-2-13B was 76. Larger models, more shots, and
  instruction tuning did not remove it; 4-bit LLaMA-2-70B at 1-shot had a
  median spread of 17 points across 320 formats, GPT-3.5 a median of 6.4
  with a maximum of 56.
- Ten formats is a lower bound. About 17 percent of tasks gain at least 5
  points of spread going from 10 to 20 sampled formats.
- Model comparisons reverse under format change. Given model M beats M' by
  at least 2 points on one format, M' beats M by at least 2 points on
  another with probability about 0.14 for both 13B-versus-70B and
  7B-versus-Falcon-7B, and in 76 and 47 percent of those reversals both
  directions were statistically significant on 1,000 examples. If format A
  beats B on one model, it beats B on another with probability under 0.62.
  Formats are not inherently good or bad.
- Few atomic features predict performance alone. Over 500 formats on 31
  tasks, only the descriptor-text separator and the enumeration numbering
  style produced strongly different accuracy distributions on more than a
  handful of tasks; spacing, item wrappers, and casing never did, despite
  each having large variance. Single-character changes still moved
  accuracy by up to 78 points (a colon after the descriptor versus none).
- The space is non-monotonic. Along chains of three formats each one atomic
  edit apart, accuracy was monotonic 32 to 34 percent of the time, which is
  chance. Local search over formats has nothing to climb.
- Formats are identifiable in the embedding. The top 100 principal
  components of the last-layer prompt embedding classify which of 10
  formats produced it with at least 0.98 accuracy, and separability in the
  top two components correlates moderately (0.42 to 0.56) with spread.
- Recommendation: report a performance interval over sampled plausible
  formats rather than one number, especially when comparing models; a
  single format remains a valid engineering choice for building a system.

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

Saved as [2603.18388v2.md](../2603.18388v2.md). The seed-trap result is a
template result. GEPA's official GSM8K seed specifies a JSON output with the
answer field before the reasoning field, so the model commits to an answer
before reasoning. Across every round and reflector, GEPA's rewrites targeted
reasoning quality and instruction clarity and never touched field order;
accuracy fell from 24 to 14 percent. With the field order repaired, all
methods reached about 86 percent. Their fix was a restart that rebuilds the
prompt from the model's raw output and parse errors, which is a way of
letting the format be re-derived rather than inherited. The general point:
an instruction optimizer treats the template as fixed, so template defects
are invisible to it and must be checked separately.

## 2025, Agrawal: GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning

Saved as [2507.19457v2.md](../2507.19457v2.md). Templates are supplied by
the DSPy framework: each module has a signature that fixes input and output
field names and their layout, and the seed instruction is the auto-generated
"Given the fields ..., produce the fields ..." text. GEPA evolves the
instruction inside that frame and never the frame itself. The paper argues
that evolved instructions are shorter and cheaper than demonstration-heavy
prompts, which is an argument about what occupies the template's slots, not
about the template. Baseline comparisons ported the same signature and
parsing prompts to the Trace framework so that template was held constant
across optimizers, the control Voronov 2024 says prior work lacked.

## 2022, Zhou: Large Language Models are Human-Level Prompt Engineers

Saved as [2211.01910v2.md](../2211.01910v2.md). Two template-level results.
The meta-prompt used to elicit candidate instructions is a template
(forward mode places the instruction at the end after demonstrations,
reverse mode infills it anywhere), and switching it changed which
instructions could be proposed at all, helping some tasks and hurting
others. And where the instruction sits relative to demonstrations matters
for selection: instructions chosen by zero-shot accuracy sometimes hurt
when demonstrations were added, and selecting by few-shot accuracy fixed
most cases. The template the candidate will be deployed in should be the
one it is scored in.

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

Saved as [2505.16307v2.md](../2505.16307v2.md). Treats the instruction as
segmentable into up to five removable units, which is a learned rather than
designed decomposition of the prompt. The rewriter templates differ for
large and small executors (diagnose-then-rewrite versus a fill-in form), and
optimized instructions stacked additively with 0 to 9 demonstrations, which
suggests instruction content and demonstration slots contribute separately
within a fixed frame. No result on the frame itself.
