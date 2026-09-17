# Format selection

The prompt-format grid (seed formats) and how formats are chosen and scored. Log, most recent first.

## 2026-09-17 13:25 — Decision: first-pass format run (300M, one factor at a time, axes 1/2/5/6)

Six axes chosen for the first pass out of the existing seven plus proposed additions: descriptor
separator, descriptor pair (with two random-string answer descriptors), option item style,
option wrapper, text-to-option separator, demonstration count. Of those, the first run uses
axes 1 (separator), 2 (descriptor pair), 5 (text-to-option separator, MC only) and 6 (shots),
one factor at a time from canonical, instruction none: 9 RC/MC formats plus 2 MC-only, 20
tasks per subset, on the SNR train and dev subsets (dev is the replicate for the winner's-curse
check), DataDecide 300M only. Pairs file `configs/po/pairs/arc-fmt-ofat-1256.json`; sweeps under
`~/drotherm/data/runs/po/clean-20260917/sweeps/fmt-ofat-1256-dd300m-{train,dev}`.

- Random descriptors drawn once and fixed in the fork: `random_vocab` = "ceil Ai aggress"
  (3 uniform tokens from the OLMo vocabulary, seed 0), `random_phrase` = "View Mobile Site"
  (4 tokens sampled from DataDecide-300M with no context, seed 0). Question descriptor stays
  "Question"; the RC unconditional context becomes the random string plus colon.
- Shot count is an OLMES task parameter (first k of the curated five); it travels in the pair
  and the task alias (`|k0`, `|k1`), not in the format id.
- The new `choice_text_separator` axis enters a format id only when non-canonical, so earlier
  ids (canonical `daa93775`) are unchanged.
- Full grids costed: RC 45 formats ~1.5 h on 300M; MC over axes 1/2/5/6 135 formats ~2 h on
  300M, all single-token; adding item style and wrapper makes the MC grid 1620 and infeasible.
  Item style, wrapper and the surface-form scoring change are deferred to a second pass.
- Whether DD models stay at chance on MC under other formats is an open question this run
  starts to answer; MC is scored the current way (bare label for dot styles).

## 2026-09-17 13:10 — Decision: widen the answer-descriptor axis and add random-string descriptors

Lu 2023: the string between the input and the scored label (our answer-descriptor slot)
is where a large share of prompt-optimisation gains live; random vocabulary strings there
matched LLM-driven optimisers. Amend the stratified grid (10:50): the descriptor axis gets
a few more conventional pairs and a few random-string descriptors (vocabulary tokens and
LM-prior phrases, fixed by seed and recorded), so the axis is a probe rather than a fixed
choice. Random-descriptor formats also serve as the null control for any format search:
a best-of-k random selection with the same budget and the same train/dev protocol.

## 2026-09-17 10:50 — Decision: stratified seed formats and a demonstration-count axis

- Add a demonstration-count axis to the format grid, including zero shots (the fixed
  five is OLMES's choice, not ours). Ferreira 2025 and SAMMO both found shot count the
  structural axis that moves results; keep the curated OLMES demonstrations as the
  source and take the first k.
- Build the seed format set by stratification rather than uniform sampling: define
  structural cells (demonstration count x demonstration grouping x a coarse surface
  class) and keep the best-scoring format per cell (MAP-Elites archive rule), so the
  seed set spans structure by construction. Cells and the surface classes are still to
  be defined; the enumerative first pass at a fixed instruction (10:25 direction) is
  how each cell's best member is found.

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
