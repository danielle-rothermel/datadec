# Instruction generation (APE / seed system prompts)

How candidate instructions are generated, what the proposer sees, and how pools are
organized. Log, most recent first.

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
