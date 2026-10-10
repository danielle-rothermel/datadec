# OLMES-default baselines on the 100-item subsets (2026-09-17 09:30)

Canonical OLMES prompt (format `daa93775`, no instruction), 5 curated demonstrations,
100-item seed-0 test subsets (ids in `assets/2026-09-17-arc-easy-baselines/`). Files:
`~/drotherm/data/runs/po/pipeline/arc-20260916/baselines/<sweep>/<rc|mc>/` (manifest
copied to `assets/2026-09-17-arc-easy-baselines/baselines-manifest.jsonl`).

## Summary

- RC carries signal for every model; MC is at chance for all DataDecide sizes and only
  works for Qwen (0.91-0.95).
- The 100-item subset misleads on MC: 300M reads 0.16 here but 0.247 on the full test
  set (see `2026-09-17-1040-arc-easy-subset-size.md`).
- On ARC-Challenge RC the raw likelihood share sits below chance (150M 0.173) while the
  OLMES primary `acc_uncond` is 0.32; the unconditional share (0.307) agrees with the
  accuracy story. This is why `primary_likelihood` follows the OLMES primary rule.

| sweep | form | primary_metric | primary | primary_likelihood_metric | primary_likelihood |
|---|---|---|---|---|---|
| arc_easy-dd150m | RC | acc_per_char | 0.57 | norm_correct_prob_per_char | 0.282 |
| arc_easy-dd300m | RC | acc_per_char | 0.61 | norm_correct_prob_per_char | 0.291 |
| arc_easy-dd530m | RC | acc_per_char | 0.65 | norm_correct_prob_per_char | 0.299 |
| arc_easy-qwen1.7b-base | RC | acc_per_char | 0.80 | norm_correct_prob_per_char | 0.339 |
| arc_easy-qwen1.7b | RC | acc_per_char | 0.80 | norm_correct_prob_per_char | 0.413 |
| arc_easy-dd150m | MC | acc_raw | 0.28 | norm_correct_prob | 0.250 |
| arc_easy-dd300m | MC | acc_raw | 0.16 | norm_correct_prob | 0.244 |
| arc_easy-dd530m | MC | acc_raw | 0.21 | norm_correct_prob | 0.243 |
| arc_easy-qwen1.7b-base | MC | acc_raw | 0.95 | norm_correct_prob | 0.900 |
| arc_easy-qwen1.7b | MC | acc_raw | 0.91 | norm_correct_prob | 0.909 |
| arc_challenge-dd150m | RC | acc_uncond | 0.32 | norm_correct_prob_uncond | 0.307 |
| arc_challenge-dd300m | RC | acc_uncond | 0.37 | norm_correct_prob_uncond | 0.338 |
| arc_challenge-dd150m | MC | acc_raw | 0.32 | norm_correct_prob | 0.257 |
| arc_challenge-dd300m | MC | acc_raw | 0.24 | norm_correct_prob | 0.252 |

Missing: ARC-Challenge for 530M and both Qwen models (never run).

## Details: all metrics

`norm_cp` = `norm_correct_prob`. A dash means undefined (no unconditional request for MC).

### ARC-Easy RC

| model | acc_raw | acc_per_char | acc_per_token | acc_uncond | norm_cp | norm_cp_per_char | norm_cp_per_token | norm_cp_uncond | norm_margin | norm_margin_per_char | norm_margin_per_token | norm_margin_uncond | correct_logprob |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DD 150M | 0.580 | 0.570 | 0.540 | 0.470 | 0.527 | 0.282 | 0.398 | 0.433 | 0.151 | 0.005 | 0.049 | -0.016 | -12.99 |
| DD 300M | 0.600 | 0.610 | 0.620 | 0.540 | 0.582 | 0.291 | 0.442 | 0.492 | 0.227 | 0.016 | 0.120 | 0.089 | -11.40 |
| DD 530M | 0.670 | 0.650 | 0.660 | 0.680 | 0.625 | 0.299 | 0.477 | 0.585 | 0.302 | 0.026 | 0.172 | 0.255 | -10.09 |
| Qwen3-1.7B-Base | 0.850 | 0.800 | 0.820 | 0.780 | 0.811 | 0.339 | 0.611 | 0.729 | 0.647 | 0.075 | 0.377 | 0.501 | -7.30 |
| Qwen3-1.7B | 0.790 | 0.800 | 0.780 | 0.860 | 0.791 | 0.413 | 0.687 | 0.830 | 0.593 | 0.151 | 0.467 | 0.674 | -12.65 |

### ARC-Easy MC

| model | acc_raw | acc_per_char | acc_per_token | acc_uncond | norm_cp | norm_cp_per_char | norm_cp_per_token | norm_cp_uncond | norm_margin | norm_margin_per_char | norm_margin_per_token | norm_margin_uncond | correct_logprob |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DD 150M | 0.280 | 0.280 | 0.280 | – | 0.250 | 0.251 | 0.250 | – | -0.056 | -0.028 | -0.056 | – | -1.436 |
| DD 300M | 0.160 | 0.160 | 0.160 | – | 0.244 | 0.248 | 0.244 | – | -0.063 | -0.031 | -0.063 | – | -1.439 |
| DD 530M | 0.210 | 0.210 | 0.210 | – | 0.243 | 0.247 | 0.243 | – | -0.045 | -0.022 | -0.045 | – | -1.434 |
| Qwen3-1.7B-Base | 0.950 | 0.950 | 0.950 | – | 0.900 | 0.766 | 0.900 | – | 0.818 | 0.629 | 0.818 | – | -0.208 |
| Qwen3-1.7B | 0.910 | 0.910 | 0.910 | – | 0.909 | 0.901 | 0.909 | – | 0.818 | 0.807 | 0.818 | – | -0.826 |

In MC every label has the same character count, so the per-char share is the raw share
at a flatter temperature, not a different ranking.

### ARC-Challenge (RC and MC)

| model | form | acc_raw | acc_per_char | acc_per_token | acc_uncond | norm_cp | norm_cp_per_char | norm_cp_per_token | norm_cp_uncond | norm_margin | norm_margin_per_char | norm_margin_per_token | norm_margin_uncond | correct_logprob |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DD 150M | RC | 0.180 | 0.270 | 0.200 | 0.320 | 0.173 | 0.253 | 0.228 | 0.307 | -0.517 | -0.033 | -0.204 | -0.236 | -19.23 |
| DD 300M | RC | 0.200 | 0.280 | 0.180 | 0.370 | 0.213 | 0.256 | 0.236 | 0.338 | -0.460 | -0.031 | -0.191 | -0.181 | -17.29 |
| DD 150M | MC | 0.320 | 0.320 | 0.320 | – | 0.257 | 0.253 | 0.257 | – | -0.069 | -0.034 | -0.069 | – | -1.419 |
| DD 300M | MC | 0.240 | 0.240 | 0.240 | – | 0.252 | 0.250 | 0.252 | – | -0.049 | -0.024 | -0.049 | – | -1.411 |

## Details: post-hoc Batch Calibration on MC (full test set and the 100-item subset)

Subtract each label position's batch-mean log-probability, then argmax (Zhou et al.
2024, via Bozhenko 2025). Computed from the full-set runs' `predictions.jsonl`.

| model | items | acc_raw | acc_bc | raw predicted-position share A/B/C/D (%) | after BC |
|---|---|---|---|---|---|
| 150M | full 2376 | 0.262 | 0.250 | 41 / 30 / 18 / 11 | 29 / 23 / 20 / 27 |
| 150M | our 100 | 0.280 | 0.300 | 44 / 27 / 13 / 16 | 29 / 25 / 21 / 25 |
| 300M | full 2376 | 0.247 | 0.252 | 0 / 84 / 11 / 5 | 26 / 20 / 21 / 33 |
| 300M | our 100 | 0.160 | 0.260 | 0 / 82 / 12 / 6 | 27 / 20 / 19 / 34 |

The label prior is large (300M answers B on 84% of items) and BC removes it, but
accuracy stays at chance: there is no signal underneath for these models.

## Provenance and caveats

- Metrics verified against DataDecide's published 150M row (15 metrics match to 5-6
  digits on the full 2376-item run).
- 300M is step 45000 (where DataDecide's published results stop); step 45787 is the true
  final checkpoint.
- Baseline bootstrap 95% intervals on 100 items are about +/-0.09 on accuracy and
  +/-0.01 on the per-char share; see the subset-size note for the full curves.
