# Format selection

The prompt-format grid (seed formats) and how formats are chosen and scored. Log, most recent first.

## 2026-09-17 10:35 — Decision: wrapper axis scoring (see evaluation-setup 10:35)

If the option wrapper becomes its own axis, the scored continuation must not change
with the wrapper. Adopted: score bare label plus displayed wrapper form as a set. Case
is an item-style property (upper/lower letters are already separate styles) and is not
normalized away at scoring time.

## 2026-09-17 10:25 — Direction: axes and selection rules suggested by SAMMO and Bozhenko

- Add a structural axis (demonstration grouping: interleaved vs inputs-then-outputs;
  demonstration count) — SAMMO's gains came from structural choices, and our grid has
  none.
- Treat the format grid as an enumerative first pass at a fixed instruction, then
  search instructions at the chosen format, instead of sampling format x instruction
  pairs (our 51 pairs confound the two axes).
- Select the seed format per model or family: SAMMO's 24 format candidates correlate
  only weakly across backends; Bozhenko's spread is 0.16-0.19 for 7-8B open models vs
  0.03-0.05 for frontier models.
- Split `choice_label_style` into item style (A/1/a/I) and wrapper (`A.`, `A)`, `(A)`,
  `[A]`); add a text-to-option separator axis. Bozhenko's Appendix E: spread grows with
  the number of format components, so weight sampling toward component-heavy formats to
  stress a model, sparse ones for a stable baseline.
- Regenerate the instruction pool per format (APE's InduceInstructions is a mutator in
  SAMMO, not a one-time step), addressing the audit's canonical-format conditioning
  mismatch.

## 2026-09-17 10:20 — Record: provenance of the current grid

The seven axes (descriptor pair, descriptor separator, descriptor case, answer break,
example separator, choice label style, choices header) come from Sclar 2023's
meaning-preserving grammar; the shape (few named options per axis, scored by
likelihood) from Voronov 2024. The option pools (`rule`, `dash`, ...) are ours. Sclar
found only the descriptor separator and enumeration style moved most tasks; several of
our axes are coverage, not expected effects. Canonical format reproduces OLMES byte for
byte (`daa93775`).
