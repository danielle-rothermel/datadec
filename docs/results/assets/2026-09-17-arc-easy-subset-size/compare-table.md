# Model comparison on the full ARC-Easy test set (n_full=2376): dd150m, dd300m, dd1b

Bands are 2.5–97.5 percentiles of the mean over random n-item subsets. Differences are paired per item.

## RC primary accuracy

| model | full-set value | n=100 band | n=100 half-width | our subset |
|---|---|---|---|---|
| dd150m | 0.509 | [0.410, 0.600] | 0.095 | 0.570 |
| dd300m | 0.588 | [0.490, 0.680] | 0.095 | 0.610 |
| dd1b | 0.707 | [0.620, 0.790] | 0.085 | 0.690 |

| pair | full-set diff | item sd | n=100 band | MDD n=100 | n for 80% power | our subset diff |
|---|---|---|---|---|---|---|
| dd300m − dd150m | +0.079 | 0.442 | [+0.000, +0.170] | 0.124 | 247 | +0.040 |
| dd1b − dd150m | +0.197 | 0.499 | [+0.100, +0.300] | 0.140 | 50 | +0.120 |
| dd1b − dd300m | +0.119 | 0.451 | [+0.030, +0.210] | 0.126 | 113 | +0.080 |

## MC primary accuracy

| model | full-set value | n=100 band | n=100 half-width | our subset |
|---|---|---|---|---|
| dd150m | 0.262 | [0.180, 0.350] | 0.085 | 0.280 |
| dd300m | 0.247 | [0.170, 0.330] | 0.080 | 0.160 |
| dd1b | 0.237 | [0.160, 0.320] | 0.080 | 0.250 |

| pair | full-set diff | item sd | n=100 band | MDD n=100 | n for 80% power | our subset diff |
|---|---|---|---|---|---|---|
| dd300m − dd150m | -0.014 | 0.594 | [-0.120, +0.100] | 0.166 | 13501 | -0.120 |
| dd1b − dd150m | -0.024 | 0.634 | [-0.150, +0.100] | 0.178 | 5288 | -0.030 |
| dd1b − dd300m | -0.010 | 0.606 | [-0.130, +0.110] | 0.170 | 28205 | +0.090 |

## RC primary likelihood

| model | full-set value | n=100 band | n=100 half-width | our subset |
|---|---|---|---|---|
| dd150m | 0.281 | [0.272, 0.290] | 0.009 | 0.282 |
| dd300m | 0.290 | [0.280, 0.300] | 0.010 | 0.291 |
| dd1b | 0.308 | [0.296, 0.320] | 0.012 | 0.307 |

| pair | full-set diff | item sd | n=100 band | MDD n=100 | n for 80% power | our subset diff |
|---|---|---|---|---|---|---|
| dd300m − dd150m | +0.009 | 0.029 | [+0.004, +0.015] | 0.008 | 79 | +0.009 |
| dd1b − dd150m | +0.027 | 0.044 | [+0.020, +0.036] | 0.012 | 20 | +0.025 |
| dd1b − dd300m | +0.018 | 0.035 | [+0.012, +0.025] | 0.010 | 28 | +0.015 |

## MC primary likelihood

| model | full-set value | n=100 band | n=100 half-width | our subset |
|---|---|---|---|---|
| dd150m | 0.251 | [0.240, 0.261] | 0.011 | 0.250 |
| dd300m | 0.250 | [0.240, 0.259] | 0.010 | 0.244 |
| dd1b | 0.250 | [0.245, 0.255] | 0.005 | 0.252 |

| pair | full-set diff | item sd | n=100 band | MDD n=100 | n for 80% power | our subset diff |
|---|---|---|---|---|---|---|
| dd300m − dd150m | -0.001 | 0.072 | [-0.015, +0.013] | 0.020 | 33625 | -0.006 |
| dd1b − dd150m | -0.001 | 0.060 | [-0.013, +0.011] | 0.017 | 48845 | +0.002 |
| dd1b − dd300m | +0.000 | 0.052 | [-0.010, +0.010] | 0.015 | 183976 | +0.008 |

