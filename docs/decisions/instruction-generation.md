# Instruction generation (APE / seed system prompts)

How candidate instructions are generated, what the proposer sees, and how pools are
organized. Log, most recent first.

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
