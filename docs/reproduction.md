# Reproduction commands and APIs

Run commands from the repository root.

## View saved evidence

```bash
uv run python scripts/repro/claims.py
uv run python scripts/repro/claims.py --run-dir outputs/repro/measurement-helpers
uv run python scripts/repro/claims.py --claim DD-0098 --max-rows 0
uv run python scripts/repro/claims.py --reports-dir outputs/repro --width 120
```

`--claim` is repeatable. `--max-rows` limits rows per numerical table; zero
shows every linked row. The displayed row counts identify partial displays.

The viewer reads saved datasets and extracts verbatim passages from the bundled
paper using `configs/repro_claims/magnusson2025-datadecide.toml`. It does not
require the original analysis inputs or rerun calculations. Automatic discovery
selects the newest completed matching run by `run.json` modification time.
Missing or malformed referenced datasets fail explicitly.

## Generate a report

```bash
uv run datadec download --olmes --scaling-law --published-results
uv run python scripts/repro/evaluate_claims.py \
  --output-dir outputs/repro/measurement-helpers
```

Use a fresh output directory for each run. Optional flags are `--config PATH`,
`--data-dir PATH`, and `--matched-compute-tolerance FLOAT`.

The default configuration is
`configs/repro_evaluations/magnusson2025-datadecide.toml`. TOML paths resolve
from the repository root; explicit CLI paths resolve from the working directory.
`run.json` records the effective configuration, input hashes, and table names.
Reports are local and ignored by Git. The evaluator prints the viewer command
when the report has been saved.

## Reusable helpers

| Module | Public helpers |
| --- | --- |
| `eval.score_statistics` | `recipe_scores_at_checkpoint`, `summarize_recipe_scores`, `score_noise_and_spread` |
| `eval.ranking` | `predict_recipe_ranking` |
| `eval.budgets` | `ranking_at_compute_budget` |
| `eval.checkpoints` | `observed_checkpoints`, `select_compute_budget` |
| `eval.scaling_observations` | `scaling_law_observations` |
| `eval.prediction_errors` | `prediction_errors` |

Scores are keyed by canonical `DataRecipeName` values. `RecipeNameResolver`
loads aliases from dictionaries or TOML `[recipe_map]` tables. A
`CheckpointRecipeScores` contains checkpoint metadata and seed-keyed
`MultiRecipeScores`; summaries derive from those values. `ddof` is explicit
for score standard deviations.

`SingleSeedRanking` composes a seed identity and `RecipeRanking`. Use
`seed.ranking.decision(...)` for pairwise decisions. `RankingResult` exposes
`seed_accuracies()`, `decision_accuracy()`, `decision_accuracy_std()`,
`aggregate_ranking`, and `aggregate_decision_accuracy()`. Both approaches
compare against target scores averaged across the selected target seeds.
Ties match only ties; set metric directions explicitly for losses.

Checkpoint steps are exact. Budget selection chooses the latest observed
positive-compute checkpoint at or below the requested budget, without
interpolation or selection on accuracy. Budget compute is per recipe and seed;
`BudgetRanking.total_predictor_compute_per_recipe` includes every selected
predictor seed. Its `relative_undershoot` reports the actual budget gap.

Scaling observations join by recipe, size, seed, and exact checkpoint.
Prediction errors use source score units and the observed score as the relative
error denominator; zero denominators remain missing. Released setup identifiers
are preserved. These helpers do not fit scaling laws.

## Configuration and ownership

General calculations live in `src/eval/`. Paper-specific selection, aggregation,
source adapters, persistence, and presentation live in `src/repro/`.
`measurement_runner.py` coordinates the focused score, budget, and scaling
wrappers. `numerical_tables.py` renders linked numerical rows.

`[measurements]` configures score metrics, additional exact score checkpoints,
standard-deviation convention, budget ratios/metrics, released scaling setups,
and loss metric. Score extraction includes observed completed checkpoints and
explicitly selected additional checkpoints; `schedule_complete` retains their
observed status. `[[measurement_claims.<id>]]` selects table rows by task,
metric, size, step, or released setup. Existing `[claims.<id>]` entries select
ranking rows by task, metric, size, and compute range.

`claim_evidence.parquet` contains row references into `rankings.parquet` and the
manifest-owned numerical tables. Original seed scores, recipe summaries, and
noise/spread measurements are separate datasets. Scaling observations, errors,
and setup/task summaries are separate datasets. The viewer also reads saved
curve, proxy, crossover, and matched-compute diagnostics. Table row IDs remain
available for inspecting the complete Parquet or CSV data.
