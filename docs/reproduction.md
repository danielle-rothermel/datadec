# DataDecide single-scale reproduction

The claim inventory in `configs/repro_claims/magnusson2025-datadecide.toml`
stores only source coordinates and IDs for passages in the bundled paper.
View the extracted passages without downloading anything:

```bash
uv run python scripts/repro/claims.py
```

The Rich viewer adds evidence from the most recent completed run for this
inventory under `outputs/repro/`, searching recursively. Recency is the
modification time of `run.json`, which evaluation writes last; a completed run
must also contain `claim_evidence.parquet` and `rankings.parquet`. Incomplete
directories and runs for other inventories are excluded. If a saved run is malformed, viewing fails
rather than silently substituting older evidence. Every selection is displayed,
including those with zero linked measurements.

Select a specific run or a different search directory:

```bash
uv run python scripts/repro/claims.py --run-dir outputs/repro/ranking
uv run python scripts/repro/claims.py --reports-dir outputs/repro
# Optional fixed width, useful for redirected output:
uv run python scripts/repro/claims.py --width 120
```

The viewer is read-only: it loads saved measurements and extracts
paper passages. It does not rerun analysis or require the original OLMES input
or evaluation TOML. Terminal output uses Rich formatting and plain text when
redirected. The evidence tables show independent per-task maxima for the per-seed mean
and mean-score prediction approaches, predictor checkpoints/metrics, compute
ratios, and coverage. Seed standard deviation applies only to the per-seed
approach; the aggregate prediction has one accuracy, shown without a standard
deviation.
Seed accuracies, FLOPs, and evidence IDs remain in the saved ranking dataset.
Viewing a saved run requires both the claim links and ranking dataset.

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
row-selection scopes. The separate `repro_claims` inventory contains source
filenames, line/slice coordinates, and IDs. Paper passages are read at display
time; there are no authored claim statements or section labels.
The default evaluation config is also included in the wheel.

Named task and metric groups keep repeated selections in one place. Each
`[claims.<id>]` entry selects explicit names and/or named groups, an optional
predictor size and compute range, and related data tables. These are filters
for collecting measurements. The runner does not assign support labels or
interpret the paper's wording. These paper wrappers use higher-is-better metrics.

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
  datadec/
    recipes.py            # Canonical recipe names and configured alias resolution
  eval/
    approaches.py         # Named predictor aggregation approaches
    checkpoint_scores.py  # Exact selection and recipe/seed coverage
    ranking.py            # Generic predict_recipe_ranking algorithm
    recipe_scores.py      # Immutable scores keyed by DataRecipeName
    results.py            # Checkpoints and single-seed/pairwise result types
  repro/
    claims.py             # Inventory parsing and quote lookup
    config.py             # Validated experiment config and group resolution
    aggregation.py        # Local OLMES reads and macro averages
    approaches.py         # Labelled comparisons for diagnostic tables
    checkpoints.py        # Observed schedules and compute-budget selection
    sweep.py              # Paper ranking policy and experiment execution
    observations.py       # Typed observations and sweep DataFrame conversion
    evidence.py           # Row selection and maximum-row references
    results.py            # Task-specific row links and derived coverage counts
    datasets.py           # Claim-link schema, ranking reads, and run manifest I/O
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
calls. Its result contains checkpoint compute, canonical recipe identities,
recipe-keyed predictor and target scores, and metric directions. Decisions,
both prediction approaches' accuracies and per-seed population standard
deviation are calculated on demand.

```python
from pathlib import Path

from datadec.recipes import DataRecipeName
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
print(result.decision_accuracy(), result.decision_accuracy_std(), result.compute_ratio)
print(result.seed_rankings[0].ranking.predictor_per_recipe_scores[DataRecipeName.DOLMA17])
print(result.target_per_recipe_scores[DataRecipeName.DOLMA17])
seed = result.seed_rankings[0]
print(seed.ranking.decision(DataRecipeName.DOLMA17, DataRecipeName.C4))
print(seed.ranking.decision_accuracy(
    result.target_per_recipe_scores,
    target_higher_is_better=result.predicted_higher_is_better,
))
aggregate = result.aggregate_ranking
print(aggregate.predictor_per_recipe_scores[DataRecipeName.DOLMA17])
print(aggregate.decision(DataRecipeName.DOLMA17, DataRecipeName.C4))
print(result.aggregate_decision_accuracy())
```

`DataRecipeName` defines one official name per DataDecide recipe. The default
`RecipeNameResolver` also recognizes source names from `[recipe_map]` in
`configs/olmes.toml`; matching is exact, and official names resolve to themselves.
Unknown names fail. Each `SingleSeedRanking` contains `seed` and `ranking`,
a `RecipeRanking` holding the immutable `MultiRecipeScores` mapping in
`predictor_per_recipe_scores` and the predictor metric direction. The target seed
mean uses the same type in `target_per_recipe_scores`. Score keys and the
arguments to `decision()` are enum members, so lookup does not depend on recipe
order. Each seed retains its predictor metric direction; the result retains
the target metric direction. `decision()` returns +1 when the first recipe
ranks higher, -1 when lower, and 0 for a tie. `decision_accuracy()` compares all
unordered pairs without storing a decision tuple. `recipes`, `recipe_count`,
and `pair_count` are derived from the target score mapping. `seed_accuracies()`,
`decision_accuracy()`, and `decision_accuracy_std()` calculate the corresponding
statistics across predictor seeds. `RecipeRanking` owns the shared decision
calculations; `SingleSeedRanking` contains a seed identity and a `RecipeRanking`.
`aggregate_ranking`
derives another `RecipeRanking` by averaging predictor scores per recipe, and
`aggregate_decision_accuracy()` compares its decisions with the same target
seed mean. An aggregate prediction is not assigned a synthetic seed label.

For other source spellings, construct a resolver from an alias dictionary or
load a TOML file containing a `[recipe_map]` table:

```python
from datadec.recipes import RecipeNameResolver
from eval import MultiRecipeScores

resolver = RecipeNameResolver.from_mapping({
    "dolma-source-a": "Dolma1.7",
    "dolma-source-b": "Dolma1.7",
})
# Alternatively: RecipeNameResolver.from_toml(Path("recipe_aliases.toml"))
scores = MultiRecipeScores.from_named_scores(
    {"dolma-source-a": 0.75, "C4": 0.62}, resolver=resolver
)
print(scores[DataRecipeName.DOLMA17])
```

Pass `recipe_name_resolver=resolver` to the ranking helper to resolve both
checkpoint rows and an explicit `recipes` selection. Multiple aliases may
identify one official recipe, but two scores for that recipe at the same
checkpoint/seed fail rather than overwrite or average. Official names cannot
be remapped to a different recipe. Custom alias dictionaries are independent of
the OLMES source contract, whose recipe map remains one-to-one.

Steps are exact and separate on each side. Metric direction defaults to higher
is better; callers using losses must set the appropriate
`predictor_higher_is_better` or `predicted_higher_is_better` flag to false.
The generic helper infers omitted recipe/seed sets from observed rows. Pass
explicit sets to enforce expected coverage; `repro.sweep.paper_ranking` does
this using the catalog recipes and paper seed labels.

Each target recipe's score is the mean over target seeds. Both predictor
approaches use this same target:

- **Per-seed mean:** each predictor seed makes its own decisions, then
  `decision_accuracy()` averages those seeds' accuracies. `seed_accuracies()`
  retains each individual accuracy and `decision_accuracy_std()` measures
  their population standard deviation.
- **Mean scores:** average predictor scores across seeds for each recipe,
  then make decisions and calculate `aggregate_decision_accuracy()`. This
  accuracy cannot be reconstructed from the per-seed accuracies alone.

The two approaches can produce different decisions and accuracies. Every
unordered recipe pair contributes equally. Exact ties have
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
| `claim_evidence.parquet` | Per-selection task links to available/unavailable ranking row IDs, separate maximum row IDs for both approaches, and related data tables |
| `run.json` | Input SHA-256 and effective configuration, including claim-inventory and paper-source paths |
| `rankings.parquet` | One row per checkpoint/task/metric, per-seed accuracies and their mean/std, aggregate prediction accuracy, compute, recipe/pair counts, coverage failures; index `evidence_id` |
| `checkpoints.csv` | Observed compute and schedule-completion coverage |
| `curves.csv` | Per-approach/size descriptive log-compute slopes, R², adjacent decreases, and endpoints |
| `proxy_comparisons.parquet` | Per-approach proxy advantage over curated Accuracy at the identical checkpoint |
| `matched_compute.csv` | Per-approach completed/intermediate evidence IDs, actual compute gaps, accuracy differences |
| `recipe_crossovers.csv` | Strict recipe-order reversals between adjacent observed completed scales, using seed means |

The evaluator writes numerical datasets. `rankings.parquet` owns the ranking
measurements. `claim_evidence.parquet` stores only task-specific available and
unavailable row IDs, `best_per_seed_evidence_id`, `best_aggregate_evidence_id`,
and related table names. The two maxima are selected independently, so they
may reference different checkpoints or metrics.
The viewer joins these references to ranking observations and derives counts
from the linked IDs. Missing rows or mismatched task/availability links fail
explicitly. Maximum IDs are selected during analysis; viewing does not rerun
the experiment or select new maxima. Curve, proxy, and compute-match diagnostic
tables label `approach` as `per_seed` or `aggregate` and keep their comparisons
separate. Recipe crossovers continue to compare seed-mean recipe scores across
completed scales; checkpoint coverage is common to both approaches.

The claims viewer extracts each selected passage directly from its source
coordinates and presents saved measurements as Rich tables. Scientific text
comes from the paper; measurements come from code applied to data. IDs,
filenames, table headings, and availability counts are presentation metadata.
There are no authored paraphrases, findings, judgments, or support verdicts.

The summary counts a selection as having measurements when at least one
configured task has an available observation. Partial coverage still counts;
per-task available/unavailable counts remain visible. This is an accounting
of linked data, not an assessment of the paper's statements. The current runner
collects OLMES ranking measurements; selections without configured measurements
remain visible with zero linked rows.

Sweep observations have a frozen typed owner in `repro.observations`.
`RankingStatistics` retains seed accuracies, recipe count, and aggregate
prediction accuracy. Per-seed mean accuracy, population standard deviation,
and pair count are derived. The DataFrame writer
materializes these summaries as columns for analysis. The reader reconstructs
observations from canonical values and computes their summaries. The conversion
owns tabular column names and null representation.
Diagnostic functions take only their required metric, task, and tolerance
arguments; the runner supplies these from validated configuration.
