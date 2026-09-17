# Framed instruction pools at the canonical format, RC and MC (2026-09-17 13:45)

First framed APE round: GPT-5.1 (reasoning high, temperature 1, output limit unset) asked to
write the instruction behind the five OLMES demonstrations rendered exactly as the scored
prompt renders them (RC: question and answer text; MC: lettered list and letter), crossed
with 3 rewrite operators (plain / rule / short) x 3 stances (rule out wrong options /
prefer common sense / none) x model-aware (reader described as DataDecide-300M, scored by
likelihood) vs not. 18 calls per formulation, 36 total; every call accepted with a normal
stop. Pools: `~/drotherm/data/runs/po/ape/20260917T173637Z-framed-rc-dd300m` and
`...173745Z-framed-mc-dd300m`; candidates, meta-prompts, the framing lists and the pool
report are copied under `assets/2026-09-17-ape-framed-pools-dd300m/`. Nothing has been
evaluated yet; this note is about what the proposer produced.

## Summary

- **The framings work as factors.** Every candidate carries its operator and stance:
  "short" gives 14-15 words on average against 26-34 for plain and rule; "rule" produces
  the only modal phrasings (You must / Always, 5 of 6 RC rule candidates); the stances
  appear verbatim as content (rule-out language in exactly the 6 eliminate candidates,
  common-sense in exactly the 6 common-sense candidates, per formulation).
- **Model awareness changed little that a reader would notice.** Aware candidates are
  slightly longer (27 vs 21-23 words) and are the only ones that name the "Answer:"
  descriptor (2 RC) or list the letters A-D more often (6 vs 4 MC). No candidate
  mentions the model's size or family, and none adapts to likelihood scoring; all still
  describe an output behaviour ("respond with only the letter"). The aware slot is an
  input the proposer largely ignored at this reasoning level.
- **Near-duplicates are rare at this pool size.** One RC pair above the 0.8 similarity
  threshold (the two plain/short no-stance variants); median pairwise similarity 0.42-0.44.
  Nine cells per condition are not enough to collapse; the filter matters at 16+ per pool.
- **Every candidate spends words on answer form.** 15 of 18 RC candidates say to write a
  word or phrase, 18 of 18 MC say to write the letter, and 13-16 say "no explanation". The
  demonstrations already show the form, so by Yang 2025's finding this is the redundant
  content models follow unasked; it is also the content most likely to be inert under
  likelihood scoring.
- **The eliminate stance leaks MC framing into RC.** All 6 RC eliminate candidates talk
  about ruling out options, and 2 call the task multiple-choice, though the RC prompt shows
  no options. Kept as a recorded factor rather than regenerated: the sweep will show
  whether it costs anything.

## Auto-grouping (rule-based tags; regexes in `scripts/po_pool_report.py`)

| tag | RC (of 18) | MC (of 18) |
|---|---|---|
| answer form: text (word/phrase) | 15 | 0 |
| answer form: letter | 1 | 18 |
| elimination language | 4 (+2 phrased as "options") | 6 |
| common-sense language | 6 | 6 |
| no-explanation clause | 13 | 16 |
| mentions science | 8 | 6 |
| names the "Answer:" descriptor | 2 | 0 |
| lists the letters A-D | 0 | 10 |
| modal / rule phrasing | 5 | 2 |
| calls the task multiple-choice or mentions options | 6 | 14 |
| length: short / medium / long (<15 / 15-30 / >30 words) | 4 / 8 / 6 | 3 / 12 / 3 |

Proposed grouping for later analysis, from these tags: (a) answer-form-only instructions
(no stance content, no science mention: the "short/none" cells and similar), (b)
stance-carrying instructions (elimination or common-sense), (c) rule-phrased instructions
(modal opening), (d) descriptor-anchored instructions (mention "Answer:"). Groups (a) vs
(b) is the test of whether stance content does anything for a likelihood-scored model;
(a) vs the empty instruction and the random-string controls is the test of whether any
instruction does.

## Cost and settings

36 calls, 36.5k total tokens (reasoning included), under a minute of wall time per grid at
concurrency 6. Seeds 5000-5017 (RC) and 6000-6017 (MC).

## Next

Evaluate both pools plus controls (empty instruction, the two random-string instructions)
at the canonical format on the SNR train subset for 300M, RC pool on RC and MC pool on MC,
then the dev replicate for the selected candidates; add 1B and Qwen after the format
one-factor runs finish. The model-aware factor is measured as the paired difference of
aware vs not-aware candidates in the same cell.
