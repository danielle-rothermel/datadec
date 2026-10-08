# DataDecide single-scale reproduction

The claim inventory in `configs/repro_claims/magnusson2025-datadecide.toml`
links unique claims to passages in the bundled paper source. View all claims
and extracted quotes without downloading anything:

```bash
uv run python scripts/repro/claims.py
```

The Rich viewer adds evidence from the most recent completed run for this
inventory under `outputs/repro/`, searching recursively. Recency is the
modification time of `run.json`, which evaluation writes last; a completed run
must also contain `claim_results.parquet`. Incomplete directories and runs for
other inventories are excluded. If a saved run is malformed, viewing fails
rather than silently substituting older evidence. Claims missing from the
selected run explicitly say no relevant evidence has been extracted.

Select a specific run or a different search directory:

```bash
uv run python scripts/repro/claims.py --run-dir outputs/repro/ranking
uv run python scripts/repro/claims.py --reports-dir outputs/repro
# Optional fixed width, useful for redirected output:
uv run python scripts/repro/claims.py --width 120
```

The viewer is read-only: it loads saved measurements/verdicts and extracts
paper passages. It does not rerun analysis or require the original OLMES input
or evaluation TOML. Terminal output uses colors when supported and plain text
when redirected. The evidence tables show per-task observed maxima, seed
standard deviations, predictor checkpoints/metrics, compute ratios, and coverage.
Full seed scores, FLOPs, and evidence IDs remain in the saved dataset.

To generate evidence, download the processed OLMES table if absent, then run
from the repository root:

```bash
uv run datadec download --olmes
uv run python scripts/repro/evaluate_claims.py
# Optional experiment config, paths, and maximum relative compute undershoot:
uv run python scripts/repro/evaluate_claims.py \
  --config configs/repro_evaluations/magnusson2025-datadecide.toml \
  --data-dir data --output-dir outputs/repro/custom \
  --matched-compute-tolerance 0.05
```

Evaluation writes a local analysis report as structured datasets and a run
manifest, then prints the claims command for that run. Use a fresh output
directory for each new run. Reports stay ignored. Math/code, seed-noise, and
scaling-law claims are outside this experiment; missing recipe identities in
other source tables are not guessed.

## Configuration and module layout

The default experiment lives in
`configs/repro_evaluations/magnusson2025-datadecide.toml`. It owns run paths,
paper source directory, target size/step/metric/seeds, predictor seed sets,
benchmark aggregation, sweep metrics, compute matching tolerance, and claim
analysis scopes. Claim statements and paper locations remain in the separate
`repro_claims` inventory.
The default evaluation config is also included in the wheel.

Named task and metric groups keep repeated selections in one place. Each
`[claims.<id>]` entry selects explicit names and/or named groups, an optional
predictor size and compute range, supporting tables, and the judgment still
required. Numerical claims reference a criterion in `[criteria]`; the runner
applies the criterion without branching on claim IDs. The configured
`accuracy_gt` bound is strict and must be met separately for every selected
task. These paper wrappers use higher-is-better metrics.

Use `--config PATH` to choose another experiment. Paths inside the TOML resolve
from the repository root; explicit CLI paths resolve from the current working
directory. `--data-dir`, `--output-dir`, and `--matched-compute-tolerance`
override the corresponding TOML settings. `run.json` records the input SHA-256
and effective, validated configuration, including overrides. The claims viewer
loads this manifest rather than rereading the evaluation TOML. Invalid references,
unknown fields, and invalid compute ranges fail before the sweep.

```text
configs/
  repro_claims/magnusson2025-datadecide.toml       # Claims and paper locations
  repro_evaluations/magnusson2025-datadecide.toml  # Experiment and analysis policy
src/
  eval/
    checkpoint_scores.py  # Exact selection and recipe/seed coverage
    ranking.py            # Generic predict_recipe_ranking algorithm
    results.py            # Scores, checkpoints, and pairwise result types
  repro/
    claims.py             # Inventory parsing and quote lookup
    config.py             # Validated experiment config and group resolution
    aggregation.py        # Local OLMES reads and macro averages
    checkpoints.py        # Observed schedules and compute-budget selection
    sweep.py              # Paper ranking policy and experiment execution
    observations.py       # Typed observations and sweep DataFrame conversion
    claim_evaluation.py   # Evidence selection and numerical verdicts
    results.py            # Claim status and computed measurement types
    datasets.py           # Claim-results Parquet schema and run manifest I/O
    reporting.py          # Join claims and saved evidence; render with Rich
    diagnostics/
      curves.py           # Compute/accuracy slopes and reversals
      proxies.py          # Proxy advantages at identical checkpoints
      crossovers.py       # Recipe-order reversals across completed scales
      compute_matches.py  # Intermediate/completed compute comparisons
scripts/repro/evaluate_claims.py  # Run analyses, save report, print viewer command
scripts/repro/claims.py           # View paper quotes and latest/specified evidence
```

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
from repro.aggregation import load_evaluations
from repro.config import load_evaluation_config

config = load_evaluation_config()
rows = load_evaluations(Path("data"), config)
result = predict_recipe_ranking(
    rows,
    predictor_size="150M",
    predicted_size=config.target.size,
    task="hellaswag",
    predictor_task_metric="correct_prob_per_char",
    predicted_task_metric=config.target.metric,
    predictor_step=37500,
    predicted_step=config.target.step,
    predictor_seeds=config.predictors.seeds_for("150M"),
    predicted_seeds=config.target.seeds,
)
print(result.decision_accuracy, result.compute_ratio)
```

Steps are exact and separate on each side. Metric direction defaults to higher
is better; callers using losses must set the appropriate
`predictor_higher_is_better` or `predicted_higher_is_better` flag to false.
The generic helper infers omitted recipe/seed sets from observed rows. Pass
explicit sets to enforce expected coverage; `repro.sweep.paper_ranking` does
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
| `claim_results.parquet` | One computed record per claim: verdict, per-task coverage, best observed score and checkpoint/metric/compute/seed details, applied bound, evidence IDs, remaining criterion notes |
| `run.json` | Input SHA-256 and effective configuration, including claim-inventory and paper-source paths |
| `rankings.parquet` | One row per checkpoint/task/metric, seed accuracies, compute, recipe/pair counts, coverage failures; index `evidence_id` |
| `checkpoints.csv` | Observed compute and schedule-completion coverage |
| `curves.csv` | Per-size descriptive log-compute slopes, R², adjacent decreases, and endpoints |
| `proxy_comparisons.parquet` | Proxy advantage over curated Accuracy at the identical checkpoint |
| `matched_compute.csv` | Completed/intermediate evidence IDs, actual compute gaps, accuracy differences |
| `recipe_crossovers.csv` | Strict recipe-order reversals between adjacent observed completed scales, using seed means |

The evaluator writes numerical results without generating prose summaries.
`claim_results.parquet` stores per-task maxima and their exact witnesses, the
three predictor seed accuracies and their population standard deviation,
coverage counts, the applied strict bound (if any), and its outcome. Evidence
IDs link to every selected checkpoint/metric row in `rankings.parquet`; the
separate diagnostic datasets retain curve, proxy, compute-match, and crossover
measurements. Best observations are descriptive maxima over the selected grid,
not aggregate evidence that every qualitative assertion is true.

The claims viewer reads persisted verdicts and measurements, then uses the
claim inventory's source locations to extract quotes from paper text. Numerical
values and verdicts are not recalculated during viewing. Authored `judgment`
text in the config describes the analysis criterion or a remaining research
decision; it is not a computed finding.

Sweep observations have a frozen typed owner in `repro.observations`; the
DataFrame conversion owns their tabular column names and null representation.
Diagnostic functions take only their required metric, task, and tolerance
arguments; the runner supplies these from validated configuration.

The candidate claim scopes are declared in the evaluation TOML and applied by
`repro.claim_evaluation`. DD-0014, DD-0015,
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
