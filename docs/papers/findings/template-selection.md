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
- A template defect can silently cap an optimizer: a wrong output-field
  order disabled chain-of-thought and reflective rewriting never diagnosed
  it (Liu 2026). Optimizers that only edit instruction text inherit the
  template they are given (Agrawal 2025, Zhao 2025).
- The meta-prompt used to elicit candidates is itself a template with large
  effects on what gets proposed (Zhou 2022).
- Ensembling predictions over 4 or 5 templates is a cheap way to remove
  format variance at inference time (Voronov 2024).
- Local: the OLMES RC and MC formulations are two templates for the same
  items. For the 150M DataDecide model MC is at chance (0.26) while RC is
  0.51, so at small scale the template decides whether the score carries any
  signal. Both formats and the five fixed demonstrations are pinned in the
  OLMES fork, so template is a controlled variable in our local evals.

## TODO

- Pull the OLMES paper (Gu et al. 2024, "OLMES: A Standard for Language
  Model Evaluations") for its rationale on the RC and MC formats, the
  curated few-shot sets, and its per-format normalization choices.
- Pull Sclar et al. 2023 ("Quantifying Language Models' Sensitivity to
  Spurious Features in Prompt Design", FormatSpread), the format-sensitivity
  study Voronov 2024 builds on, and Zhao et al. 2021 ("Calibrate Before
  Use") for the calibration scoring rule.

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

## 2025, Zhao: PMPO: Probabilistic Metric Prompt Optimization for Small and Large Language Models

Saved as [2505.16307v2.md](../2505.16307v2.md). Treats the instruction as
segmentable into up to five removable units, which is a learned rather than
designed decomposition of the prompt. The rewriter templates differ for
large and small executors (diagnose-then-rewrite versus a fill-in form), and
optimized instructions stacked additively with 0 to 9 demonstrations, which
suggests instruction content and demonstration slots contribute separately
within a fixed frame. No result on the frame itself.
