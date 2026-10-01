# Pair files

- `arc-pairs-n50-seed0.json`: **RC-conditioned** pairs (formats x RC APE pool + none). Use for RC sweeps only;
  the MC results produced from it on 2026-09-16/17 are quarantined under the pipeline's `broken-mc/`.
- `arc-pairs-mc-n50-seed0.json`: MC-conditioned pairs, matched pair-for-pair to the RC file (same format, same
  candidate index in the MC pool). Use for MC sweeps.
