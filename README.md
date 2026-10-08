# DataDecide

Library and CLI for downloading, processing, validating, publishing, and
cleaning DataDecide evaluation artifacts.

## Data layout

Artifacts live under `data/` by default:

| Path | Description |
|------|-------------|
| `raw/ppl.parquet` | Raw perplexity export |
| `raw/olmes.parquet` | Raw aggregate OLMES export |
| `raw/olmes-details/models/{recipe}.tar.gz` | Per-recipe OLMES detail archive |
| `raw/scaling-law/*.csv` | Three Google Drive scaling-law sources |
| `reference/published-results/{structured source path}` | Google Drive CSV/JSON sources grouped into publication units |
| `reference/published-results/{figure path}` | 80 download-only PDF/PNG figures |
| `processed/ppl.parquet` | Typed PPL output |
| `processed/olmes.parquet` | Typed aggregate OLMES output |
| `processed/scaling-law/evaluations.parquet` | Typed, precedence-resolved scaling-law task evaluations |
| `processed/scaling-law/checkpoint-losses.parquet` | Typed, reconciled scaling-law checkpoint losses and throughput |
| `processed/olmes-details/tasks.parquet` | Shared detail task summaries for all processed recipes |
| `processed/olmes-details/instances/{recipe}.parquet` | Per-recipe instance shard |
| `processed/olmes-details/choices/{recipe}.parquet` | Per-recipe choice shard |
| `processed/published-results/{schema}.parquet` | One consolidated table for each of the seven published-result schemas |

OLMES table schemas are declared in [`configs/olmes.toml`](configs/olmes.toml).
Scaling-law source precedence, aliases, seed policy, and table schemas are
declared in [`configs/scaling_law.toml`](configs/scaling_law.toml).
Published-result schemas and atomic publication units are declared in
[`configs/published_results.toml`](configs/published_results.toml). Hugging
Face destination paths and commit messages are declared in
[`configs/publishing.toml`](configs/publishing.toml).
Each published-result row begins with required `source_file` and `source_unit`
provenance columns, followed by the original schema columns in their original
order. Rows from every source with the same schema are preserved in the same
table, including duplicates and nulls.
Model definitions and training constants are declared in
[`configs/catalog.toml`](configs/catalog.toml). Each model distinguishes its
nominal parameter count (the size label), training parameter count (the value
used to scale batch size and learning-rate schedules), and exact architectural
parameter count (the value used for FLOP estimates). The configured
`flops_per_token_per_parameter` constant owns the compute multiplier.

Every checkpoint-bearing processed table except the detail instance and choice
tables carries the canonical checkpoint derivations directly, without a later
join: tokens, exact-parameter FLOP compute, model architecture/training details,
`lr_at_step`, and `cumulative_lr`. These fields are present in PPL, aggregate
OLMES, both scaling-law tables, and OLMES detail tasks. Instances and choices
remain evaluation-detail tables keyed to their parent task checkpoint.

## CLI lifecycle

Every command requires an explicit dataset selection. The applicable selectors
are `--ppl`, `--olmes`, repeatable `--olmes-details RECIPE`, `--scaling-law`,
`--published-results`, repeatable `--unit UNIT`, and `--all`. A `--unit`
selection implies published results; `--published-results` without units
selects every structured publication unit. `--data-dir` changes the artifact
root for any command.

`--all` means all processable data: it includes every configured OLMES detail
recipe as well as PPL, aggregate OLMES, scaling-law data, and every structured
published-result unit. Use explicit selectors for an aggregate-only run:

```bash
# Download, process, verify, publish, and then remove selected raw sources
uv run datadec run --ppl --olmes --scaling-law --published-results

# Select individual OLMES detail recipes or structured publication units
uv run datadec run \
  --olmes-details dolma1.7-no-math-no-code \
  --olmes-details c4
uv run datadec run --unit cheap-decisions --unit per-task-arc-challenge

# Run locally and retain raw and processed artifacts
uv run datadec run --ppl --olmes --no-upload
```

`run` downloads missing raw sources, processes the selection, validates its
outputs, publishes final Parquet files to `drotherm/dd_parsed`, verifies every
remote copy at an immutable commit, and only then applies cleanup. Its `default`
cleanup removes selected raw inputs, owned intermediates and OLMES recipe detail
instance and choice shards, retaining aggregate postprocessed results: PPL,
aggregate OLMES, both scaling-law tables, the seven published-result schema
tables, and the shared OLMES detail task table. When everything is selected,
these are 12 retained summary files. Per-recipe instance and choice shards can
be downloaded again when needed.
`--cleanup raw` retains all processed outputs; `--cleanup all` removes them too;
`--cleanup none` retains raw and processed artifacts.
With `--no-upload`, cleanup defaults to `none`; requesting `default`, `raw` or
`all` at the same time is rejected before any work starts. Processing, validation, or
publication failures retain raw inputs and final outputs for diagnosis or retry.
Cleanup itself is not transactional if a filesystem deletion fails.

`download` retrieves verified processed outputs by default. Add `--raw` to
retrieve original sources instead:

```bash
uv run datadec download --ppl --olmes
uv run datadec download --raw --ppl --olmes --scaling-law
uv run datadec download --raw --published-results
```

Published-result `--unit` values select source groups, while processing,
processed download, publication, and cleanup operate on complete schema
families. Selecting one unit therefore expands to every source in each schema
family touched by that unit so rebuilding or deleting a shared table cannot
discard rows from unselected sources. Processed downloads fetch complete files;
selecting any OLMES detail recipe fetches the complete shared task table plus
that recipe's complete instance and choice shards. Filtering rows or columns
happens after download through the local read API.

The 80 published PDF/PNG figures are raw reference artifacts. Select them with
`--published-figures` only on raw download or cleanup operations; they cannot be
processed or published by DataDecide.

```bash
uv run datadec download --raw --published-figures
```

`publish` uploads and verifies existing selected outputs without processing
them again. When a local shared task table contains only some recipes, publication
merges recipes absent locally from a verified immutable remote table before
uploading. Recipes already present locally keep their local rows. `raw-clean` removes selected reproducible raw downloads. `clean`
uses the same `default` retention policy as `run`; choose `--cleanup all` to
remove aggregate outputs too. Before any deletion, it verifies every existing
final output marked for removal against its immutable remote copy. Both
cleanup commands support `--dry-run` and tolerate files that are already
missing.
Standalone cleanup with `--all` also includes downloaded reference figures and
the project-owned raw download caches; automatic run cleanup follows only the
processing selection.

```bash
uv run datadec publish --ppl --olmes --scaling-law
uv run datadec raw-clean --scaling-law --dry-run
uv run datadec clean --unit cheap-decisions --dry-run
uv run datadec clean --all --cleanup default --dry-run
uv run datadec clean --ppl --cleanup all --dry-run
```

Scaling-law preprocessing requires all three local raw CSVs. It validates the
fixed source schema, excludes the recorded blank/6198 legacy-seed rows and
invalid source groups, resolves the historical `baseline` alias and source
overlaps by configured policy, and derives corrected token/compute schedules
from the pinned batch sizes and exact model parameter counts. It writes both
output tables only after both temporary parquet files validate successfully.
After both final tables validate, `run` publishes them in one atomic commit and
removes the three raw CSVs only after immutable remote verification when the
selected cleanup policy permits it.

Select all three Drive-backed source categories to reconstruct all 134 source
files (11.92 GB). Downloads use a pinned inventory from the public Drive folder
rather than crawling its current contents.

The lifecycle CLI operates on canonical artifact paths. The low-level
processors remain directly callable from Python for research workflows with
custom inputs and outputs; no CLI path-override interface is provided. For
example, aggregate OLMES supports explicit input and output paths:

```python
from pathlib import Path

from datadec.data.artifacts import DataArtifacts
from datadec.data.preprocess import preprocess_olmes

artifacts = DataArtifacts(Path("data"))
preprocess_olmes(
    artifacts,
    input_path=Path("path/to/raw.parquet"),
    output_path=Path("path/to/out.parquet"),
)
```

`preprocess_olmes_details` likewise accepts explicit archive and table paths.
Custom inputs supplied to these processors are outside automatic cleanup
ownership.

Use `read_processed_table` to project columns and filter rows from known local
tables with Arrow expressions:

```python
import pyarrow.dataset as ds

from datadec.data.read import read_processed_table

target_pairs = read_processed_table(
    artifacts,
    "published-results/target_pairs",
    filters=ds.field("source_unit") == "outputs2",
    columns=["source_file", "pair_index"],
)
```

Known table names are `ppl`, `olmes`, both `scaling-law/*` tables, each
`published-results/{schema}` table, and `olmes-details/tasks`,
`olmes-details/instances`, and `olmes-details/choices`. The instance and choice
names read their shard directories and accept `recipe` filters. This API reads
local data only; it does not provide partial network downloads. Tables are
written in useful key order with Parquet row-group statistics, so a separate
index is not maintained.

## Library organization

| Module | Responsibility |
| --- | --- |
| `datadec.config` | Validated dataset, schema, source, and publication contracts |
| `datadec.data.selection` | Deterministic dataset, recipe, and publication-unit selection |
| `datadec.data.artifacts` | Canonical paths and explicit artifact ownership |
| `datadec.data.preprocess` | Local-input/local-output dataset processors |
| `datadec.data.verify` | Selection-aware output and cross-source verification |
| `datadec.data.download` | Verified raw-source and processed-output downloads |
| `datadec.data.read` | Local projection and Arrow-filtered reads of known processed tables |
| `datadec.data.publication` / `publish` | Atomic publication units, upload, and immutable remote verification |
| `datadec.data.cleanup` | Scoped raw and full cleanup plans |
| `datadec.data.pipeline` | End-to-end stage coordination and cleanup policy |

## OLMES detail preprocessing

Detail preprocessing streams one checkpoint at a time through the recipe
archive. Each recipe update atomically replaces these contract-typed outputs:

- **tasks** — one shared `tasks.parquet` table of task-level metrics and config
  JSON; an update replaces the selected recipe rows and retains all other
  recipes
- **instances** — one recipe shard per
  `(recipe, params, seed_value, step, task, doc_id)`; heterogeneous native IDs
  normalized to nullable `native_id` + `native_id_kind`
- **choices** — one recipe shard with a row per choice index in each instance's
  `model_output`

Pipeline processing restores the published shared task table when it is absent
locally before updating a recipe, which preserves other recipes. Standalone
processors remain local-only. Nullable byte/unconditional fields remain null
when absent in the source checkpoint.

## Verification

**Default tests** (`uv run pytest`) use small fixtures. They cover schema
mapping, nullability, lifecycle coordination, CLI wiring, and verification
logic, but not full live archives.

**Manual verification** is for representative recipes after download + preprocess. This checks:

- task / instance / choice counts and primary-key uniqueness
- cross-source parity on the 482 overlapping checkpoints between aggregate and detail for a recipe such as `dolma1.7-no-math-no-code`
- reconstruction of task metrics from instance rows

Manual raw-dependent verification requires its source files locally. After raw
cleanup, redownload the applicable sources with `download --raw` before running
these checks; the full pipeline performs them before cleanup.

`bits_per_byte_corr` is declared non-reconstructible from the detail slice in `configs/olmes.toml` and is excluded from reconstruction checks.

Checkpoint derivation verification covers PPL, aggregate OLMES, scaling-law
evaluations, scaling-law checkpoint losses, and OLMES detail task outputs.
These derivation checks exclude the instance and choice tables; selected detail
verification separately checks their counts and metric reconstruction.
Derivation verification checks every available raw token/compute value
against the canonical schedule and reports when a raw source instead encodes
nominal-parameter compute. No current preprocessing source records learning-rate
schedule values, so LR derivations can be checked for internal consistency but
not independently confirmed against raw evidence.

The 2026-08-19 full-data validation produced zero token, exact-compute,
model-detail, or LR contradictions in all five processed outputs: 22,709 PPL
rows; 1,410,750 aggregate OLMES rows; 1,788,996 scaling-law evaluation rows;
27,106 scaling-law checkpoint-loss rows; and 35,772 OLMES detail task rows.
The aggregate OLMES raw source also had zero token or exact-compute
contradictions across 1,410,750 rows. Of 2,245,848 raw Google Drive scaling-law
rows, 489,258 contained token and compute evidence: their token values all
matched, their compute values all matched nominal-parameter compute, and all
therefore differed from the standardized exact-parameter compute. Embedded
OLMES detail model configuration had zero contradictions across 35,772 task
rows. The raw Google Drive distinction is diagnostic evidence: it is reported
without blocking the pipeline. Exact-compute mismatches in generated outputs
remain correctness failures and stop publication and cleanup.

Measured full local preprocessing wall times for that validation were 0.78s
for PPL, 30.58s for aggregate OLMES, 228.94s for scaling-law, and 863.36s for
the 542-checkpoint OLMES detail archive. The detail run wrote 20,423,644
instance rows and 74,384,622 choice rows in addition to its task rows.

Full-recipe detail preprocessing and verification can take a long time and require multi-GB local data; they are intentionally excluded from the default test suite.

## Paper reproduction

From the repository root, print the 64 distinct claim IDs and statements,
followed by all 74 section-labelled source quotes:

```bash
uv run python scripts/repro/claims.py
```

The required arXiv v2 paper text is committed in
[`docs/papers/2504.11393v2/`](docs/papers/2504.11393v2/README.md), so no paper or
evaluation-data download is needed. Use Python 3.12 or newer and `uv`;
`uv run` installs the project and its dependencies on first use. Output preserves
the paper's LaTeX macros. This command lists claims and quotes; it does not run
reproduction experiments.

The inventory lives in
[`configs/repro_claims/magnusson2025-datadecide.toml`](configs/repro_claims/magnusson2025-datadecide.toml).
Each claim has a statement, its original registry entry IDs, and one or more
paper locations. Repeated assertions share one claim; `original_entry_ids`
preserves their provenance. Locations specify a source file, paper section,
1-based line, and 0-based character range with an exclusive end.

`repro.claims` provides `load_claims`, the frozen `Claim` and `QuoteLocation`
dataclasses, and `read_quotes`. The loader validates the TOML structure and
coordinate ranges, rejects unknown fields, and preserves claim and location
order. [`scripts/repro/claims.py`](scripts/repro/claims.py) loads the inventory
and prints the quotes. The helpers can also be called directly:

```python
from pathlib import Path

from repro.claims import load_claims, read_quotes

claims = load_claims(Path("configs/repro_claims/magnusson2025-datadecide.toml"))
quotes = read_quotes(claims, Path("docs/papers/2504.11393v2"))
```

Source provenance, licensing, and hashes are recorded beside the bundled paper
text. Five figure-only targets without textual quotes are documented as omitted
in the inventory's header comments.

## Development

```bash
uv run ruff check src scripts tests
uv run ty check src
uv run pytest
```
