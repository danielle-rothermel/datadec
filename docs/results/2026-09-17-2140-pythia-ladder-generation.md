# Pythia ladder (160M–2.8B) and generation baselines for eight models (2026-09-17 21:40)

Two additions to the clean ARC-Easy program. (1) The Pythia ladder (EleutherAI, Pile, GPT-NeoX
tokenizer, `step143000`, non-deduped, seed 1234) at 160M, 410M, 1B and 2.8B: full-test-set
baselines, the one-factor format run on the SNR train and dev subsets, and generation baselines.
(2) The generation formulation (`gen_rc`, `gen_mc`: same prompt, greedy, cap 96, stop at newline,
scored by normalized prefix match) at canonical format for all eight models. Assets in
`assets/2026-09-17-pythia-ladder-generation/`; runs under `~/drotherm/data/runs/olmes/EleutherAI--pythia-*`
and `~/drotherm/data/runs/po/clean-20260917/sweeps/{fmt-ofat-1256,gen-base}-<model>-{train,dev}`.

## Summary

- **Pythia sits below DataDecide at matched size on RC and at chance on MC at every size.**
  Full test (2376): RC per-char 0.433 / 0.555 / 0.602 / 0.683 for 160M / 410M / 1B / 2.8B; MC
  0.254 / 0.248 / 0.234 / 0.242. Pythia-2.8B lands just under DataDecide-1B (0.707).
- **Canonical is optimal or within 1 se of it for every Pythia size on RC.** The only resolved
  gain across the ladder is colon-newline on 410M (+0.0011 ± 0.0004 share), which reverses at 1B
  and 2.8B. Zero shots is the largest effect and grows with size (−0.002, −0.010, −0.013, −0.013
  share). Both random-string descriptors hurt at 1B and 2.8B, the vocabulary string more than the
  LM-prior phrase, the opposite of DataDecide-300M.
- **MC generation reproduces MC likelihood on every model** (within a point; a single dominant
  letter for the small models, evenly spread letters for Qwen). Greedy first-token generation and
  single-token likelihood ranking are the same decision. `gen_mc` adds nothing over `mc` and is
  dropped from future sweeps unless a format makes the answer multi-token.
- **RC generation is a different task from RC likelihood.** Text match rises with capability
  (0.01 at Pythia-160M to 0.28 at Qwen) but stays far below RC likelihood accuracy (0.43 to 0.84),
  with 64–90 % of generations matching no option. The stored answers are mostly on-topic and
  often correct free answers ("centripetal force", "the force of gravity"), so prefix match
  measures option adherence, not knowledge. An LLM-judge column over the stored text is the way
  to separate those; not run.
- **Qwen instruct and base generate alike under this plain few-shot prompt**, within two points
  on both formulations, as in the likelihood runs.

## Pythia full-set baselines

| model | RC acc_per_char | RC acc_raw | RC acc_uncond | MC acc_raw |
|---|---|---|---|---|
| Pythia-160M | 0.433 | 0.455 | 0.410 | 0.254 |
| Pythia-410M | 0.555 | 0.561 | 0.505 | 0.248 |
| Pythia-1B | 0.602 | 0.601 | 0.553 | 0.234 |
| Pythia-2.8B | 0.683 | 0.678 | 0.610 | 0.242 |

Subset-size bootstrap plots per model are in the packet
`~/drotherm/data/.claude/datadec/2026-09-17/0949-arc-easy-subset-size/pythia*/`.

## Format effects, RC, pooled train+dev (n=600), Δ share ± se vs canonical

Canonical RC accuracy / share on these items: 160M 0.405 / 0.269; 410M 0.528 / 0.285;
1B 0.597 / 0.290; 2.8B 0.688 / 0.304. Full tables incl. accuracy and MC in `effects-pythia.md`.

| format | 160M | 410M | 1B | 2.8B |
|---|---|---|---|---|
| sep=colon_newline | +0.0002 | **+0.0011** | −0.0021 | −0.0014 |
| sep=dash | −0.0003 | −0.0006 | −0.0012 | −0.0005 |
| desc=q_a | −0.0001 | +0.0002 | −0.0005 | −0.0006 |
| desc=input_output | −0.0006 | −0.0032 | −0.0024 | −0.0019 |
| desc=random_vocab | −0.0015 | −0.0008 | −0.0052 | −0.0034 |
| desc=random_phrase | −0.0005 | 0.0000 | −0.0041 | −0.0022 |
| shots=1 | −0.0010 | −0.0026 | −0.0048 | −0.0023 |
| shots=0 | −0.0021 | −0.0104 | −0.0129 | −0.0132 |

Paired se on the share is 0.0002–0.0015. MC: at chance under every format at every size; every
MC share effect within 1.5 se except 410M, where all formats nudge a slightly-below-chance
canonical (0.222) back toward 0.25 (label prior moving, not signal).

Charts: `format-effects-rc-pythia.png` (all formats), `format-effects-group-shots-pythia.png`.

## Generation baselines, canonical format, 5 shots (train / dev)

| model | gen_mc label match | gen_rc text match | gen_rc no answer | MC lik. acc | RC lik. acc |
|---|---|---|---|---|---|
| Pythia-160M | 0.250 / 0.260 | 0.007 / 0.010 | 0.90 / 0.88 | 0.255 | 0.405 |
| Pythia-410M | 0.237 / 0.207 | 0.073 / 0.070 | 0.81 / 0.84 | 0.222 | 0.528 |
| DD-300M | 0.250 / 0.210 | 0.100 / 0.087 | 0.77 / 0.81 | 0.230 | 0.567 |
| Pythia-1B | 0.243 / 0.253 | 0.080 / 0.100 | 0.78 / 0.77 | 0.248 | 0.597 |
| DD-1B | 0.230 / 0.233 | 0.163 / 0.200 | 0.72 / 0.68 | 0.232 | 0.715 |
| Pythia-2.8B | 0.257 / 0.250 | 0.150 / 0.167 | 0.70 / 0.71 | 0.242 | 0.688 |
| Qwen3-1.7B-Base | 0.907 / 0.923 | 0.213 / 0.280 | 0.71 / 0.64 | 0.915 | 0.843 |
| Qwen3-1.7B | 0.893 / 0.903 | 0.207 / 0.283 | 0.71 / 0.66 | 0.898 | 0.790 |

Likelihood columns are the canonical cells of the format run on the same pooled items. Full
per-run rows (label/text/any match, no-answer, mean tokens, dominant output) in
`generation-baselines.csv`. Dominant MC letters: 300M B (83 %), DD-1B D (65 %), Pythia-160M C
(64 %), 410M A (51 %), 1B D (54 %), 2.8B D (80 %); Qwen spread evenly.

## Caveats

- OLMES's `num_tokens` for a generation is the batch's generated length, so mean tokens and
  `max_tokens_reached` are upper bounds.
- Label match under RC prompts reads a leading article "a" as label A; it is secondary there.
- The generation path needed three fixes today (KV cache format for hf_olmo, sdpa NaN on
  left-padded rows on mps, Qwen's generation_config overriding the cap); see
  `docs/decisions/evaluation-setup.md` 16:50 and 17:35. All eight runs post-date the fixes.
- Pythia is a 2023 suite on the Pile; per the model-survey rule it is a historical anchor, not a
  recommended ladder member (survey: `~/drotherm/data/.claude/datadec/2026-09-17/1836-model-ladders/`).
