# Refactor the data upload pipeline

## Core steps

- [x] Split configuration and schema contracts into focused modules.
- [x] Centralize dataset selection, artifact paths, and file ownership.
- [x] Keep processors local-only and separate verification from processing.
- [x] Separate publication-unit construction from upload and remote verification.
- [x] Add selective downloads of processed outputs from Hugging Face.
- [x] Add shared, scoped cleanup with dry-run support.
- [x] Implement the full pipeline coordinator in `src`.
- [x] Consolidate the Typer CLI in `src` and retire superseded scripts.
- [x] Update callers, documentation, packaging, and focused tests together.

## Implementation log

- **Step 1 — Split configuration and schema contracts into focused modules:** Split config.py into focused modules; moved PPL, checkpoint-enrichment, and published-result schemas to canonical configuration owners. All 316 tests and installed-wheel resource checks passed.

- **Step 2 — Centralize dataset selection, artifact paths, and file ownership:** Added deterministic dataset selection and explicit artifact ownership; migrated all callers and removed paths.py and duplicate selectors/path mappings. Recipe-specific metadata cleanup is isolated; shared caches require --all. All 335 tests passed.

- **Step 3 — Keep processors local-only and separate verification from processing:** Moved verification into data/verify and added typed selection-aware reports. Base-only derivation checks never scan detail files; cross-source prerequisites are reported explicitly. Extracted shared checkpoint identity normalization and staging ownership. All 352 tests passed. Live-data follow-up: the pinned scaling-law inputs encode nominal compute in 489,258 rows; those differences are diagnostic while exact-compute checks on generated outputs remain blocking.

- **Step 4 — Separate publication-unit construction from upload and remote verification:** Separated publication units from CAS upload and immutable size/hash verification. Publishing never deletes local files; added read-only verification for cleanup. Updated tests to prove raw/output retention on success and failure. All 347 integrated tests passed.

- **Step 5 — Add selective downloads of processed outputs from Hugging Face:** Added exact selected processed downloads from an immutable Hugging Face commit, with size/hash/schema verification before replacing any existing outputs. Raw downloads now use the same resolved selection and support published-result units. All 356 integrated tests passed.

- **Step 6 — Add shared, scoped cleanup with dry-run support:** Added shared raw/full cleanup with exact dry-run plans and all remote checks before deletion. Missing files are idempotent; subsets preserve shared caches and unrelated files; raw cleanup preserves processor staging. Symlink escapes and unexpected directories are rejected. All 370 integrated tests passed. Follow-up: full cleanup handles selected DuckDB spill directories explicitly; 23 focused cleanup/artifact tests passed.

- **Step 7 — Implement the full pipeline coordinator in `src`:** Added the typed coordinator: raw download, explicit local processors, all schemas and selected checks, immutable publication, then cleanup. No-upload defaults retain files; contradictory cleanup is rejected before side effects. All 386 integrated tests passed, including failure retention and partial-publication checks.

- **Step 8 — Consolidate the Typer CLI in `src` and retire superseded scripts:** Added the library-owned Typer CLI and tiny scripts/data.py launcher; removed superseded download, processing and publication scripts and tests. Shared selection, dry-run and diagnostics are covered; the installed CLI uses a working-directory-relative data root. All 374 integrated tests passed.

- **Step 9 — Update callers, documentation, packaging, and focused tests together:** Updated supported commands, library ownership and cleanup guarantees; moved Typer to runtime and added the packaged datadec entry point. All 374 tests, full src type check, lockfile check, wheel build and installed-wheel CLI/configuration-resource checks passed.

- **Review — one full round:** Fixed all three findings: automatic `run --all` preserves unselected figures, the manual derivation verifier uses the canonical diagnostic policy, and the detail-verifier message names the supported command. All 378 tests and `src`/`scripts` type checks passed.

- **Live validation:** Reran PPL, aggregate OLMES, scaling-law, and all structured published-result processors. All 55 final files in 18 publication units matched verified Hugging Face copies at immutable commit `a7a576a27d06aaf359e12cdd69850c4a00464fba`; publication was a verified no-op. Output derivation checks passed; historical raw compute differences were diagnostic. Automatic cleanup removed all 56 selected raw source files.

- **Cleanup and recovery validation:** Confirmed PPL-only full cleanup and selective download restored the identical SHA-256. Standalone all-raw cleanup removed the two remaining owned cache trees. No dataset raw files or owned download caches remain; all 75 per-instance output files retain their original sizes and modification times. Retired and archived the historical tracked download note in this branch while preserving the unrelated original checkout's tracked source.

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

On processing, validation, or publication failure, retain raw inputs and final
outputs for retry. Cleanup is not transactional on filesystem deletion failure.
Earlier successful
uploads may remain remote: publication is atomic per publication unit, not
across the entire run. Preserve existing OLMES checkpoint resumption; other
processing stages may recompute on retry without a new persistent run ledger.

Verification uses the requested selection, not unrelated files discovered
locally. Report checks skipped because their cross-source prerequisites are
outside the selection. All applicable raw-dependent checks finish before
cleanup. Historical nominal-compute values in raw scaling-law inputs are reported
as diagnostics; generated tables must satisfy the exact-compute contract.

## CLI and cleanup

`scripts/data.py` becomes a small launcher for `datadec.cli.app`. Commands are
`run`, `download`, `publish`, `raw-clean`, and `clean`.

Share `--data-dir` and applicable selectors: `--ppl`, `--olmes`, repeatable
`--olmes-details`, `--scaling-law`, `--published-results`, and `--all`.
Preserve published-result unit selection. Require an explicit selection.
Published figures remain download-only and are outside processing/publication.

- `run --cleanup default` is the default: remove selected raw and intermediate
  files and recipe detail outputs, retaining PPL, aggregate OLMES, scaling-law
  tables, and structured published results.
- `run --cleanup raw` removes selected raw files after success.
- `run --cleanup all` also removes verified published final outputs.
- `run --cleanup none` retains raw and processed artifacts.
- `run --no-upload` retains artifacts; reject explicitly contradictory cleanup
  requests. Normal processor scratch-file cleanup still applies.
- `download` retrieves selected processed outputs; `download --raw` retrieves
  their original sources.
- `publish` uploads existing outputs without processing again.
- `raw-clean` explicitly removes selected reproducible raw downloads.
- `clean` defaults to retaining aggregate postprocessed results while removing
  selected raw files, intermediates, and recipe details. Use `--cleanup all` to
  also remove aggregate outputs. First verify that every final output marked
  for deletion has an identical remote copy; refuse to discard unpublished or
  changed final outputs.
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
