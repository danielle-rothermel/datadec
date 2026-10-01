# SNR-filtered ARC-Easy subsets and their OLMES-default baselines (2026-09-17 12:55)

Three disjoint 300-item subsets (train, dev, test; test untouched for now) drawn with seed 0
from the 2292 ARC-Easy test items that pass the signal-to-noise filter (signal = RC
per-char share dispersion across DD 150M/300M/1B finals, noise = 300M checkpoint sd,
SNR >= 2; 84 items excluded). Selection script `scripts/po_select_items.py`; files in
`configs/po/subsets/arc_easy-test-snr-{train,dev,test}-n300-seed0.json`, copies and the
per-item pool ranking in `assets/2026-09-17-arc-easy-snr-subsets/`. Decision:
`../decisions/dataset-sample-selection.md` (12:55).

## Summary

- The baselines below are read from the full-set runs (no new evaluation): filter the
  per-item predictions by subset id.
- All three subsets sit within their expected sampling bands of the full-set values
  (n=300 half-widths: ~0.055 on accuracy, ~0.006 on the RC share). Dev happens to be the
  easier draw for the Qwen models (+0.04 to +0.05 RC accuracy over full) and train the
  harder; paired comparisons within a subset are unaffected by that offset.
- MC is at chance for the DD models on every subset, as on the full set.

| model | subset | RC acc | RC share | MC acc | MC share |
|---|---|---|---|---|---|
| DD 150M | train | 0.483 | 0.280 | 0.267 | 0.251 |
| DD 150M | dev | 0.503 | 0.280 | 0.267 | 0.251 |
| DD 150M | test | 0.530 | 0.280 | 0.237 | 0.251 |
| DD 150M | full | 0.509 | 0.281 | 0.262 | 0.251 |
| DD 300M | train | 0.577 | 0.287 | 0.250 | 0.250 |
| DD 300M | dev | 0.557 | 0.289 | 0.210 | 0.246 |
| DD 300M | test | 0.567 | 0.288 | 0.247 | 0.248 |
| DD 300M | full | 0.588 | 0.290 | 0.247 | 0.250 |
| DD 1B | train | 0.707 | 0.305 | 0.230 | 0.250 |
| DD 1B | dev | 0.723 | 0.310 | 0.233 | 0.250 |
| DD 1B | test | 0.677 | 0.305 | 0.233 | 0.248 |
| DD 1B | full | 0.707 | 0.308 | 0.237 | 0.250 |
| Qwen3-1.7B-Base | train | 0.817 | 0.336 | 0.907 | 0.867 |
| Qwen3-1.7B-Base | dev | 0.870 | 0.346 | 0.923 | 0.891 |
| Qwen3-1.7B-Base | test | 0.837 | 0.341 | 0.953 | 0.897 |
| Qwen3-1.7B-Base | full | 0.831 | 0.341 | 0.926 | 0.883 |
| Qwen3-1.7B | train | 0.763 | 0.403 | 0.893 | 0.894 |
| Qwen3-1.7B | dev | 0.817 | 0.424 | 0.903 | 0.905 |
| Qwen3-1.7B | test | 0.810 | 0.417 | 0.907 | 0.904 |
| Qwen3-1.7B | full | 0.795 | 0.414 | 0.897 | 0.895 |

Also in `assets/2026-09-17-arc-easy-snr-subsets/baselines-on-snr-subsets.csv`.

## Details

- Power at n=300 for a paired prompt comparison on the RC per-char share (paired item
  sd about 0.03): minimum detectable difference about 0.005 at 80% power, two-sided 5%.
- Pool statistics: signal median 0.030, noise median 0.0026, SNR median 11.6; the 84
  excluded items have SNR below 2 (their model dispersion is within twice their checkpoint
  jitter).
- 1B checkpoint noise is not yet in the noise term; when it lands the pool can be
  re-ranked, which would change at most a few items at the margin, and the subsets will be
  kept as drawn unless the re-ranking excludes one of their members.
