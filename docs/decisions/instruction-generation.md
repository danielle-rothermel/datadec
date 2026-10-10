# Instruction generation (APE / seed system prompts)

How candidate instructions are generated, what the proposer sees, and how pools are
organized. Log, most recent first.

## 2026-09-17 14:10 — Decision: proposer is a recorded factor; Terra pools added

The same 36 cells were regenerated with `openai/gpt-5.6-terra` (reasoning high). Terra returns
the framing almost verbatim (5-13 words; half the RC grid collapses to three near-duplicate
groups; aware cells often identical to non-aware), GPT-5.1 elaborates (14-34 words with
answer-form and no-explanation clauses). Keep both pools and treat proposer as a factor
crossed with the framings, since the two proposers span the instruction-length axis. The
near-duplicate filter is required for Terra pools even at 18 cells. Results:
`../results/2026-09-17-1345-ape-framed-pools-dd300m.md` (Terra section).

## 2026-09-17 13:45 — Executed: first framed round (3 x 3 x aware, canonical format, RC and MC)

Run as decided: GPT-5.1 reasoning high, output limit unset; demonstrations = the five OLMES
items rendered exactly as scored; framings in `configs/po/framings.json` (operators plain /
rule / short; stances eliminate / commonsense / none); aware cells describe DataDecide-300M
and likelihood scoring. 18 calls per formulation, all accepted. Findings and grouping in
`../results/2026-09-17-1345-ape-framed-pools-dd300m.md`. Two observations for the next round:
the aware slot barely changed the proposals (no candidate adapts to likelihood scoring), so a
stronger phrasing of the reader description is worth one more cell; and the eliminate stance
is MC-flavoured, so RC needs its own stance list. Sameness filter: difflib ratio >= 0.8 for
now (`scripts/po_pool_report.py`), one RC pair caught.

## 2026-09-17 13:10 — Decision: random-string controls in every instruction pool

Every per-format pool includes a small set of random-string instructions (random
vocabulary tokens and LM-prior phrases of comparable length, seeded and recorded as their
own generation style) alongside the generated candidates. They are the null hypothesis
for the pool: a generated pool that does not beat the random controls under the same
selection (choose on train, report on dev) has not shown anything. Lu 2023 found
best-of-160 random strings within 1% of OPRO/APE on 2023 base models; the size of the
effect on our models is an empirical question the controls answer.

## 2026-09-17 11:00 — Decision: framing-pair generation and near-duplicate filtering

- Generate description-style seeds by crossing two short hand-written lists
  (Promptbreeder initialization): a rewrite operator (state plainly / as a rule / as
  expert advice / as short as possible / explicit about answer form / silent about
  answer form / ...) and a stance suited to likelihood-scored small models (prefer the
  common-sense option / eliminate wrong options first / answer as a science teacher /
  treat as recall / ...), about ten each. Each candidate records its (operator, stance)
  pair. Sample 16-24 pairs per pool. This is a third factor beside the per-format
  demonstrations and the model-aware slot; all three are recorded per candidate.
- Filter near-duplicates before a pool is used for selection or as GEPA seeds. Now: an
  embedding cosine threshold (Promptbreeder used 0.95 with BERT; pick the threshold by
  inspecting the pool). Later: an LLM-as-judge definition of "same instruction" (same
  requirements stated, same answer-form guidance), which catches paraphrase collapse
  that embeddings miss and separates instructions that embed close but say different
  things. Record which filter produced a pool.
- Keep both generation styles (demonstration-induced and description-based) in every
  per-format pool; their relative value depends on how informative the task description
  is (Promptbreeder's under-specified-description ablation).

## 2026-09-17 10:50 — Decision: one APE pool per format, demonstrations matched to the target setting

- The demonstrations shown in the APE meta-prompt must be rendered exactly as they
  will appear in the scored prompt: same format (all axes, including demonstration
  count and grouping) and same formulation (RC answer text vs MC label). The
  2026-09-16 pools violated this (canonical-format RC demonstrations for every format,
  and the RC pool applied to MC), which is why all of those sweeps are disregarded.
- Therefore run one APE pool per (formulation, format) we intend to evaluate, even
  though that multiplies proposer cost. Afterwards decide whether to select shared seed
  instructions across formats or keep every pool separate; that choice is downstream of
  seeing the pools, not a reason to skip generating them.
- Experiment early with telling the meta-prompt what kind of model the instruction is
  for (size, base vs instruction-tuned, family, scored by likelihood over choices rather
  than generating text). This is expected to change what gets proposed and should be
  measured as its own factor (model-aware vs model-agnostic meta-prompt, same seeds and
  demonstrations) before it is folded into the default.
- Existing tooling: `datadec.po.ape` meta-prompts (`ape_forward`, `p1_description`,
  `ape_forward_mc`, `p1_description_mc`) render demonstrations in the canonical format
  only; the per-format rendering and the model-description slot are to be added.
