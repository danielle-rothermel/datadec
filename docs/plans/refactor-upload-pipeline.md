# Refactor the data upload pipeline

## Core steps

- [x] Split configuration and schema contracts into focused modules.
- [ ] Centralize dataset selection, artifact paths, and file ownership.
- [ ] Keep processors local-only and separate verification from processing.
- [ ] Separate publication-unit construction from upload and remote verification.
- [ ] Add selective downloads of processed outputs from Hugging Face.
- [ ] Add shared, scoped cleanup with dry-run support.
- [ ] Implement the full pipeline coordinator in `src`.
- [ ] Consolidate the Typer CLI in `src` and retire superseded scripts.
- [ ] Update callers, documentation, packaging, and focused tests together.

## Implementation log

- **Step 1 — Split configuration and schema contracts into focused modules:** Split config.py into focused modules; moved PPL, checkpoint-enrichment, and published-result schemas to canonical configuration owners. All 316 tests and installed-wheel resource checks passed.

## Goal

Provide one library-owned lifecycle for downloading, processing, validating,
publishing, and cleaning selected DataDecide datasets. Raw files remain
recoverable from their original sources; final outputs are archived to
`drotherm/dd_parsed` and can be downloaded selectively.

Automatic cleanup occurs only after all selected processing, validation, and
remote verification succeed. Processing scratch files remain disposable local
implementation artifacts, rather than Hugging Face publication outputs.

## Library organization

Keep the existing dataset processors and research functionality, while giving
each lifecycle responsibility a clear owner:

| Location | Responsibility |
| --- | --- |
| `datadec/config/` | Focused configuration and schema modules for catalog, sources, PPL, OLMES, scaling-law, published results, and publishing; shared TOML/resource loading |
| `datadec/data/selection.py` | Resolve dataset, recipe, and publication-unit selections; expand `all`, validate names, and deduplicate |
| `datadec/data/artifacts.py` | Canonical local paths and ownership of raw, processed, intermediate, and identifiable cache files; replaces `paths.py` |
| `datadec/data/preprocess/` | Local-input/local-output processors, shared Parquet export mechanics, checkpoint identity normalization, and model enrichment |
| `datadec/data/verify/` | Selected-output verification, cross-source parity, reconstruction, and derivation checks |
| `datadec/data/download.py` | Download selected raw sources or published processed outputs |
| `datadec/data/publication.py` | Construct concrete atomic publication units using canonical schema and destination contracts |
| `datadec/data/publish.py` | Upload files and verify their remote sizes and hashes at immutable commits; no local deletion |
| `datadec/data/cleanup.py` | Resolve deletion plans, preview them, and execute scoped cleanup |
| `datadec/data/pipeline.py` | Coordinate stages and enforce upload, verification, and cleanup policy |
| `datadec/cli.py` | Typer argument handling, help, progress, and result formatting |

Use concrete functions and typed results, with frozen dataclasses for internal
requests/results and Pydantic at configuration boundaries. Avoid a generic
workflow framework or plugin registry. Leave `ingest/` and `wandb/` behavior
unchanged except for required import updates.

## Pipeline behavior

`run_pipeline(request)` performs these stages for one resolved selection:

1. Resolve artifacts and verification prerequisites.
2. Download missing sources, reusing valid local files.
3. Process selected datasets.
4. Validate outputs and run applicable cross-source checks.
5. Publish final outputs and verify every remote copy.
6. Apply cleanup only after the complete selected run succeeds.

On failure, retain raw inputs and final outputs for retry. Earlier successful
uploads may remain remote: publication is atomic per publication unit, not
across the entire run. Preserve existing OLMES checkpoint resumption; other
processing stages may recompute on retry without a new persistent run ledger.

Verification uses the requested selection, not unrelated files discovered
locally. Report checks skipped because their cross-source prerequisites are
outside the selection. All applicable raw-dependent checks finish before
cleanup.

## CLI and cleanup

`scripts/data.py` becomes a small launcher for `datadec.cli.app`. Commands are
`run`, `download`, `publish`, `raw-clean`, and `clean`.

Share `--data-dir` and applicable selectors: `--ppl`, `--olmes`, repeatable
`--olmes-details`, `--scaling-law`, `--published-results`, and `--all`.
Preserve published-result unit selection. Require an explicit selection.
Published figures remain download-only and are outside processing/publication.

- `run --cleanup raw` is the default: remove selected raw files after success.
- `run --cleanup all` also removes verified published final outputs.
- `run --cleanup none` retains raw and processed artifacts.
- `run --no-upload` retains artifacts; reject explicitly contradictory cleanup
  requests. Normal processor scratch-file cleanup still applies.
- `download` retrieves selected processed outputs; `download --raw` retrieves
  their original sources.
- `publish` uploads existing outputs without processing again.
- `raw-clean` explicitly removes selected reproducible raw downloads.
- `clean` additionally removes owned intermediates and processed outputs, but
  first verifies that each existing final output has an identical remote copy.
  Refuse to discard unpublished or changed final outputs.
- Both cleanup commands support `--dry-run`, use the same deletion plan for
  preview and execution, and tolerate already-missing files.

Preserve custom-input protection: automatic cleanup does not delete arbitrary
input overrides. Selective cleanup preserves shared cache entries with uncertain
ownership; explicit all-data cleanup can clear the project-owned download cache.

## Cutover and verification

Replace existing processing/publication CLI orchestration with the canonical
CLI in one coherent change. Update imports, examples, package resources, and CLI
dependency declarations; remove superseded entry points without compatibility
wrappers. Keep low-level processors independently callable from Python.

Focus tests on stage ordering, failure retention, remote mismatch protection,
selection isolation, dry-run accuracy, selective downloads, and equivalent
behavior through library and CLI entry points. Reuse existing processor and
publication tests.

The existing `ingest_from_hf` consumer can still redownload raw PPL and OLMES
files after cleanup. Migrating it to processed outputs is outside this refactor.
