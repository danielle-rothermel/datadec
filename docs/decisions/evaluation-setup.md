# Evaluation setup

Metrics, scoring rules, formulations, and what is comparable to what. Log, most recent first.

## 2026-09-17 10:39 — Decision: which likelihood metrics are comparable across tokenizers

String log-probabilities are tokenizer-independent, so `norm_correct_prob`, the per-char
variants, the unconditional variants, and the MC label scores are comparable between
DataDecide (OLMo tokenizer) and Qwen. Anything normalized per token
(`acc_per_token`, `norm_correct_prob_per_token`, `logits_per_token_corr`) is not, and is
never a headline metric. Residual caveat: canonical-tokenization-only scoring leaves a
little mass unaccounted for, differently per tokenizer; second order.

## 2026-09-17 16:50 — Implemented: generation formulation (gen_rc, gen_mc); baseline runs launched

The 10:38 spec is implemented as fork tasks `arc_easy:gen:fmt` / `arc_easy:mc:gen:fmt` (and the
ARC-Challenge pair) with metric `PrefixMatch` (`oe_eval/metrics/prefix_match.py`), and in datadec
as formulations `gen_rc` / `gen_mc` (primary `text_match` / `label_match`, no likelihood twin,
`primary_likelihood` None). Same prompt bytes as RC / MC, greedy, cap 96, stop at newline; the raw
generation is kept in the predictions file and on the items table (`continuation`).

- DataDecide checkpoints generate with the KV cache after all (17:00 revision): HF generate
  hands hf_olmo a `DynamicCache` where it expects a list of (k, v) tuples, so the fork's generate
  override runs a short greedy loop with the model's own cache format for hf_olmo models; its
  output is byte-identical to the no-cache path on the smoke items. `po_sweep.py --no-cache`
  remains as an escape hatch. The override also pins the mps autocast dtype to float32 because
  hf_olmo casts its attention bias to that dtype whenever an attention mask is passed (generate
  always passes one; the likelihood path never does), which otherwise fails with a Half/float
  mismatch.
- Five-item smoke test on 300M: RC generations are answer text (`the earth's gravity`), MC
  generations a bare letter; 300M answers `B` on every MC item. Runtime without cache is small at
  this answer length (under a second for five RC items).
- Caveat: OLMES's `num_tokens` for a generation is the batch's generated length, not the row's,
  so `max_tokens_reached` is an upper bound.
- Label match under RC prompts reads a leading article "a" as label A; it is secondary there.
- Runs: `gen-base-<model>-<train|dev>` for 300M, 1B, Qwen3-1.7B-Base, Qwen3-1.7B at canonical
  format, no instruction, 5 shots (`driver-gen-base.sh`).

## 2026-09-17 10:38 — Decision: generation formulation spec

Third formulation beside RC and MC: same prompt, greedy decoding, cap 96 tokens for both
RC and MC (longest ARC gold answer is 46 tokens; p99 is 17), stop on newline and the
format's example separator, raw output and stop reason stored. Scoring on the normalized
string prefix (strip, lowercase, drop punctuation, word boundaries): label match (first
word equals a label) and text match (output begins with an option's full text, longest
wins), both computed on every generation; primary = label match under MC prompts, text
match under RC prompts; no-answer counted wrong and reported. An LLM judge over the
stored text is an optional later column, not part of the primary score.

## 2026-09-17 10:35 — Decision: MC scoring over a surface-form set (not yet implemented)

For each MC option score the log-sum-exp of two continuations: the bare label (` A`) and
the displayed wrapper form (` (A)`, ` A.`, ...). Doubles MC requests (half multi-token).
Case variants deferred. OLMES's canonical uppercase bare-label `acc_raw` is kept alongside
under its own name for DataDecide comparability. Rationale: a model that copies the
list's wrapper into its answer should not lose credit for our choice of scored string;
for ranking metrics the shared wrapper tokens mostly cancel, so existing paren-style
results are slow rather than biased.

## 2026-09-17 10:30 — Result-backed decision: Batch Calibration as an MC metric (post hoc)

Computable from `predictions.jsonl` (subtract each label position's batch-mean
log-probability, then argmax). On the full set it removes the label prior (300M answers
B 84% of the time raw) but leaves accuracy at chance: 150M 0.262 -> 0.250, 300M
0.247 -> 0.252; our 100-item 300M value 0.16 -> 0.26. Worth reporting as the honest MC
number; it does not make MC informative for DataDecide models. If added, it is a
task-level derived metric in the results loader; inside GEPA the position means must
come from a fixed reference run, not the minibatch.

## 2026-09-17 09:25 — Decision: DataDecide metric names, `primary` and `primary_likelihood`

Verified against DataDecide's published 150M row on the full ARC-Easy run (15 metrics
match to 5-6 digits): what we had called `correct_prob` is DataDecide's
`norm_correct_prob`. `datadec.po.metrics` now owns `norm_correct_prob{,_per_char,
_per_token,_uncond}`, `norm_margin_*`, `PRIMARY_METRIC` and the accuracy-to-likelihood
pairing. Every item and task row carries `primary` (the OLMES primary accuracy:
acc_per_char for ARC-Easy RC, acc_uncond for ARC-Challenge RC, acc_raw for MC) and
`primary_likelihood` (its softmax twin). Reports, seed ranking and GEPA jobs use these;
GEPA's `score_metric` job field is `primary` or `primary_likelihood`, default
`primary_likelihood` (`GEPA_SCORE_METRIC`). Open: which of the two GEPA should optimize.
Caveat: the per-char share is compressed toward chance (DD ARC-Easy RC 0.28-0.31), so
absolute values are not "probability on the right answer"; paired differences are fine.

## 2026-09-17 09:20 — Note: checkpoint naming

Our 300M runs use step 45000 (where DataDecide's published results stop); step 45787 on
HF is the true end of training (30.0B tokens). 0.5B tokens on a plateau; state it in
reports, no rerun.

## 2026-09-17 09:15 — Decision: likelihood ranking stays the scoring rule

Bozhenko 2025: probability ranking over options is uniformly more format-robust than
greedy generation, on current open models at our sizes. Generation is added only as a
reference formulation (10:38), not as the comparison metric.
