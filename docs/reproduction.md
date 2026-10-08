# DataDecide single-scale reproduction

The claim inventory in `configs/repro_claims/magnusson2025-datadecide.toml`
links unique claims to passages in the bundled paper source. List those passages
without downloading anything:

```bash
uv run python scripts/repro/claims.py
```

To generate single-scale ranking evidence, download the processed OLMES table
if it is not already present, then run from the repository root:

```bash
uv run datadec download --olmes
uv run python scripts/repro/evaluate_claims.py
# Optional paths and maximum relative compute undershoot for matched comparisons:
uv run python scripts/repro/evaluate_claims.py \
  --data-dir data --output-dir outputs/repro/ranking \
  --matched-compute-tolerance 0.05
```

The runner reads local data and writes ignored run artifacts. It does not fit
scaling laws. Math/code, seed-noise, and scaling-law claims are outside this
experiment; missing recipe identities in other source tables are not guessed.

## Generic evaluation API

`eval.ranking.predict_recipe_ranking` in `src/eval/` accepts an evaluation
DataFrame with `params`, `step`, `task`, `data` (recipe), `seed`, `compute`, and
metric columns. Pass a sorted `(params, step, task)` MultiIndex for repeated
calls. Its result contains checkpoint compute, ordered recipe scores, gold
scores, and every seed's pairwise decisions, mean accuracy, and population
standard deviation.

```python
from pathlib import Path

from eval import predict_recipe_ranking
from repro.ranking import load_evaluations, SMALL_SEEDS, TARGET_SEEDS

rows = load_evaluations(Path("data"))
result = predict_recipe_ranking(
    rows,
    predictor_size="150M",
    predicted_size="1B",
    task="hellaswag",
    predictor_task_metric="correct_prob_per_char",
    predicted_task_metric="primary_metric",
    predictor_step=37500,
    predicted_step=69369,
    predictor_seeds=SMALL_SEEDS,
    predicted_seeds=TARGET_SEEDS,
)
print(result.decision_accuracy, result.compute_ratio)
```

Steps are exact and separate on each side. Metric direction defaults to higher
is better; callers using losses must set the appropriate
`predictor_higher_is_better` or `predicted_higher_is_better` flag to false.
The generic helper infers omitted recipe/seed sets from observed rows. Pass
explicit sets to enforce expected coverage; `repro.ranking.paper_ranking` does
this using the catalog recipes and paper seed labels.

Each target recipe's score is the mean over target seeds. Every predictor seed
makes its own decisions against that gold mean, then decision accuracy is
averaged across predictor seeds. Predictor scores are not averaged before
comparison. Every unordered recipe pair contributes equally. Exact ties have
sign zero and count as correct only against another tie. Duplicate identities
and inconsistent compute are errors. Missing checkpoints, recipe/seed cells,
or finite metric values raise `UnavailableRankingError`.

## Paper policy and evidence

The `src/repro/` wrappers own DataDecide-specific choices:

- Gold scores use curated `primary_metric` at **1B, step 69369**, averaged over
  `default`, `large aux 2`, and `large aux 3`. Small predictors use `default`,
  `small aux 2`, and `small aux 3`; 1B predictors use the gold seed labels.
  Predictions using 1B seeds share runs with the gold target, as in the paper.
- MMLU is the unweighted mean over its 57 subjects. OLMES is the unweighted
  mean over ten benchmarks, with MMLU counting once. Missing component tasks
  fail rather than changing macro weights; missing component metrics remain
  unavailable.
- The sweep includes observed positive-compute checkpoints at or below target
  compute and raw/token/character accuracy and probability proxy metrics.
  All catalog recipes and all three requested seeds must be present. Coverage
  failures are recorded, not silently intersected or dropped.
- Compute is the processed table's exact-parameter FLOP estimate. Ratios are
  fractions, so **0.01% = 0.0001**. The compute budget selector chooses the
  largest observed positive checkpoint at or below the budget. It never
  interpolates or selects a more expensive checkpoint.
- A checkpoint is marked `schedule_complete` only if its observed step reaches
  its configured `total_steps`. Latest observed checkpoints are not assumed to
  be completed runs. For intermediate-versus-completed comparisons, the
  default 5% maximum compute undershoot is an explicit analysis tolerance,
  adjustable with the CLI flag. Unmatched budgets are omitted from that
  diagnostic; `checkpoints.csv` retains the source coverage. The crossover
  diagnostic requires the complete catalog recipe set at every included
  completed scale; missing recipes fail instead of reducing the pair count.

Outputs under `outputs/repro/ranking/`:

| File | Contents |
| --- | --- |
| `claims.json` | Input SHA-256, policy, claim IDs/statements/source locations, statuses, evidence row IDs, remaining judgments |
| `claims.md` | Readable claim evidence and remaining validation todo |
| `rankings.parquet` | One row per checkpoint/task/metric, seed accuracies, compute, recipe/pair counts, coverage failures; index `evidence_id` |
| `checkpoints.csv` | Observed compute and schedule-completion coverage |
| `curves.csv` | Per-size descriptive log-compute slopes, R², adjacent decreases, and endpoints |
| `proxy_comparisons.parquet` | Proxy advantage over curated Accuracy at the identical checkpoint |
| `matched_compute.csv` | Completed/intermediate evidence IDs, actual compute gaps, accuracy differences |
| `recipe_crossovers.csv` | Strict recipe-order reversals between adjacent observed completed scales, using seed means |

The candidate claim scopes are declared in `repro.evaluate`. DD-0014, DD-0015,
and DD-0016 state explicit numerical existence bounds: at least one observed
continuous proxy/checkpoint must exceed 0.80 decision accuracy within 0.0001
of target compute. ARC is evaluated separately on both Easy and Challenge;
both must satisfy the bound. `supported` is observed numerical support,
`not_supported` means the fully available observed grid does not meet the
bound, and `insufficient_data` records absent evidence that prevents a verdict.
These are observed-grid checks, not statistical significance or universal
reproducibility guarantees.

Other candidate claims receive `requires_judgment`, with supporting rows and a
specific remaining decision. Terms such as “strong,” “roughly log-linear,”
“equivalent,” “small,” and “frequently” do not have numerical tolerances in the
paper. Inspect their evidence and settle those criteria before relying on a
reproduction verdict. DD-0011 retains the observed 150M curve rather than
pretending its latest checkpoint is fully trained. DD-0207 also requires
checking metric definitions for its incorrect-answer penalty assertion.
