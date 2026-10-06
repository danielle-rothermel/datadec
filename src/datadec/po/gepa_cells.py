"""GEPA cell execution (gepa-run-contract, po-gepa/2).

A cell is one (model, task): 2 starts (``empty``, ``ape``) x 3 GEPA seeds = 6 runs. This module writes the cell's
job files and ``gepa`` pool items, builds the per-run Test eval sweep and its ``eval`` pool item (the gepa item's
follow-up), and aggregates the paired optimized-minus-canonical effects into the tidy tables with a hierarchical
bootstrap (runs, then items within runs).

The GEPA search device is a job field: ``cuda`` for cluster pool runs, ``mps`` for the Mac lane
(``scripts/po_gepa_local.py``), which writes jobs without pool items, runs them locally and pushes finished runs to
the cluster. The Test eval sweep is always ``cuda`` whatever device searched, because Test evaluation is canonical on
the cluster; the per-run ``gepa_runs`` table records the search device.

Paths follow the contracts: runs at ``ROOT/gepa/<cell_id>/<run_id>/``, jobs at ``ROOT/gepa/<cell_id>/<run_id>.job.json``,
eval sweeps at ``ROOT/sweeps/ge-<run_id>``, pool items as ``<pool>/pending/<NNNN>-<name>.json`` (po-pool-item/1).
Items are written straight into the pool directories (atomic rename into ``pending/``); nothing here depends on
the pool module. Every gepa item has class ``gepa`` (its ``est_minutes`` scales with model size); the follow-up
runs ``uv run --directory <datadec checkout>``, so it does not depend on the worker honouring ``followup_cwd``.

Aggregation includes a run only when its eval sweep covers every Test item in both arms, so a sweep still running
(or one with a failed chunk) never contributes partial deltas; such runs are reported as incomplete.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum, unique
from pathlib import Path

import numpy as np
import pandas as pd

from datadec.po.ape import PROMPT_CAP, SOL_SETTINGS
from datadec.po.formats import CANONICAL, format_id
from datadec.po.model_cards import model_card, nominal_params
from datadec.po.results import load_sweep
from datadec.po.subsets import DATASETS, ROW_INDEX, ItemSubset
from datadec.po.sweep import SweepSpec, write_sweep


@unique
class Start(StrEnum):
    EMPTY = "empty"
    APE = "ape"


@unique
class ReflectionPromptVersion(StrEnum):
    """The fork driver's reflection meta-prompt (job field ``reflection_prompt_version``; values fixed by the fork)."""

    V1 = "v1"
    V2 = "v2"


DEFAULT_REFLECTION_PROMPT_VERSION = ReflectionPromptVersion.V1
GEPA_SEEDS = (0, 1, 2)
MAX_PROPOSALS = 12
METRIC_CALLS_PER_TRAIN_DEV_ITEM = 40  # max_metric_calls = 40 x |Train-Dev|, a high cap; the proposal count stops runs
REFLECTION_MINIBATCH_SIZE = 3
NUM_SHOTS = 5
SCORE_METRIC = "primary_likelihood"
FORMULATION = "rc"
COMPONENT = "system_prompt"  # the fork adapter's candidate component name
NONE_INSTRUCTION = {"id": "none", "text": None}
GEPA_INSTRUCTION_PREFIX = "gepa-"
EVAL_SWEEP_PREFIX = "ge-"
GEPA_ITEM_PREFIX = "gepa-"

SECRETS_ENV_FILE = "/scratch/ddr8143/.config/secrets/openrouter.env"
POOL_ITEM_SCHEMA = "po-pool-item/1"
POOL_STATE_DIRS = ("pending", "claimed", "done", "failed")
SMALL_MAX_PARAMS = 4e9  # strictly below -> small (eval class, shorter gepa estimate)
BIG_MAX_PARAMS = 30e9  # strictly below -> big; at or above -> 32b (eval) and bfloat16 weights
DEFAULT_DEVICE = "cuda"  # GEPA search device of cluster pool runs; the Mac lane writes "mps"
EVAL_DEVICE = "cuda"  # Test eval sweeps run on the cluster regardless of the search device
GEPA_CLASS = "gepa"  # gepa items that share a GPU under MPS (two slots): models under GEPA_SHARED_MAX_PARAMS
GEPA_BIG_CLASS = "gepa-big"  # gepa items needing the whole GPU (one slot): 8B fp32 and up
GEPA_SHARED_MAX_PARAMS = 6e9  # strictly below -> class gepa (two 4B fp32 models fit one 80 GB A100; two 8B do not)
GEPA_EST_MINUTES_SMALL, GEPA_EST_MINUTES_BIG = 180, 360
EVAL_EST_MINUTES = {"small": 30, "big": 90, "32b": 180}
PRECISION = {"float32": "fp32-tf32", "bfloat16": "bf16"}

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 0
METRIC_COLUMNS = {"accuracy": "primary", "likelihood": "primary_likelihood"}  # tidy metric -> items column
CONTRAST_TYPE = "gepa"
CONTRAST_METHOD = "hierarchical-bootstrap"
BOTH_STARTS = "both"


@dataclass(frozen=True, slots=True)
class CellSubsets:
    """Repo-relative subset files of one task: Train (reflection minibatches), Train-Dev (Pareto selection), Test."""

    train: str
    train_dev: str
    test: tuple[str, ...]


def _sub(name: str) -> str:
    return f"configs/po/subsets/{name}.json"


CELL_SUBSETS: dict[str, CellSubsets] = {
    "arc_easy": CellSubsets(_sub("arc_easy-train-part-train-n1495-seed0"), _sub("arc_easy-train-part-traindev-n748-seed0"),
                            (_sub("arc_easy-validation-n570-seed0"), _sub("arc_easy-test-part-test1-n1188-seed0"))),
    "arc_challenge": CellSubsets(_sub("arc_challenge-train-part-train-n739-seed0"), _sub("arc_challenge-train-part-traindev-n369-seed0"),
                                 (_sub("arc_challenge-validation-n299-seed0"), _sub("arc_challenge-test-part-test1-n586-seed0"))),
    "openbookqa": CellSubsets(_sub("openbookqa-train-part-train-n500-seed0"), _sub("openbookqa-train-part-traindev-n250-seed0"),
                              (_sub("openbookqa-validation-n500-seed0"), _sub("openbookqa-test-part-test1-n250-seed0"))),
    "csqa": CellSubsets(_sub("csqa-train-part-train-n1221-seed0"), _sub("csqa-train-part-traindev-n610-seed0"),
                        (_sub("csqa-validation-part-test-n610-seed0"), _sub("csqa-train-part-testaug-n611-seed0"))),
    "socialiqa": CellSubsets(_sub("socialiqa-train-part-train-n1954-seed0"), _sub("socialiqa-train-part-traindev-n977-seed0"),
                             (_sub("socialiqa-validation-part-test-n977-seed0"), _sub("socialiqa-train-part-testaug-n977-seed0"))),
    "piqa": CellSubsets(_sub("piqa-train-part-train-n1838-seed0"), _sub("piqa-train-part-traindev-n919-seed0"),
                        (_sub("piqa-validation-part-test-n919-seed0"), _sub("piqa-train-part-testaug-n919-seed0"))),
    "winogrande": CellSubsets(_sub("winogrande-train-part-train-n1267-seed0"), _sub("winogrande-train-part-traindev-n633-seed0"),
                              (_sub("winogrande-validation-part-test-n633-seed0"), _sub("winogrande-train-part-testaug-n634-seed0"))),
    "hellaswag": CellSubsets(_sub("hellaswag-train-part-train-n2000-seed0"), _sub("hellaswag-train-part-traindev-n1000-seed0"),
                             (_sub("hellaswag-validation-part-test-n2000-seed0"),)),
}


def ape_seed_path(task: str) -> str:
    return f"configs/po/instructions/ape-seed-{task}.json"


# --- ids, classes, defaults -----------------------------------------------------------------------------------

def default_dtype(model: str) -> str:
    """fp32 weights (TF32 matmuls) everywhere except the 32B models, which run in bfloat16."""
    return "bfloat16" if nominal_params(model) >= BIG_MAX_PARAMS else "float32"


def model_key(model: str, revision: str | None, dtype: str) -> str:
    """The tidy model_key `<model>@<revision>@<precision>` (revision `main` when unpinned)."""
    return f"{model}@{revision or 'main'}@{PRECISION[dtype]}"


def model_slug(key: str) -> str:
    return key.replace("/", "-").replace("@", "_")


def cell_id(key: str, task: str) -> str:
    return f"{model_slug(key)}__{task}"


def run_id(cell: str, start: Start, seed: int) -> str:
    return f"{cell}__{start}__s{seed}"


def gepa_class(model: str) -> str:
    """Worker class of a gepa item: shared-GPU "gepa" below GEPA_SHARED_MAX_PARAMS, else "gepa-big"."""
    return GEPA_CLASS if nominal_params(model) < GEPA_SHARED_MAX_PARAMS else GEPA_BIG_CLASS


def gepa_est_minutes(model: str) -> int:
    return GEPA_EST_MINUTES_SMALL if nominal_params(model) < SMALL_MAX_PARAMS else GEPA_EST_MINUTES_BIG


def eval_class(model: str) -> str:
    params = nominal_params(model)
    return "small" if params < SMALL_MAX_PARAMS else ("big" if params < BIG_MAX_PARAMS else "32b")


def default_batch_size(model: str) -> int:
    """Scoring batch size matching the phase-2 task sweeps (Qwen3's large vocabulary needs smaller batches)."""
    params = nominal_params(model)
    if model.startswith("Qwen/"):
        return 16 if params < 8e9 else (8 if params < 14e9 else 4)
    if params < 1.2e9:
        return 32 if model.startswith("allenai/OLMo-2-") else 64
    for limit, size in ((2e9, 32), (4e9, 16), (10e9, 8), (20e9, 4)):
        if params < limit:
            return size
    return 2


# --- job files and gepa items ---------------------------------------------------------------------------------

def _ids(task: str, subset: ItemSubset) -> list:
    """Row-position tasks carry their ids as strings (gepa-run-contract); native ids stay as stored."""
    return [str(i) for i in subset.ids] if DATASETS[task].id_field == ROW_INDEX else list(subset.ids)


def seed_instruction(start: Start, task: str, repo: Path) -> dict:
    if start is Start.EMPTY:
        return dict(NONE_INSTRUCTION)
    record = json.loads((repo / ape_seed_path(task)).read_text())
    return {"id": record["id"], "text": record["text"]}


def build_job(*, model: str, revision: str | None, dtype: str, batch_size: int, task: str, start: Start, seed: int,
              root: Path, repo: Path, device: str = DEFAULT_DEVICE,
              reflection_prompt_version: ReflectionPromptVersion = DEFAULT_REFLECTION_PROMPT_VERSION) -> dict:
    """One GEPA job in the contract's field set; ``device`` is the GEPA search device the fork driver loads onto and
    ``reflection_prompt_version`` selects the driver's reflection meta-prompt."""
    key = model_key(model, revision, dtype)
    cell = cell_id(key, task)
    rid = run_id(cell, start, seed)
    subsets = CELL_SUBSETS[task]
    train = ItemSubset.load(repo / subsets.train)
    train_dev = ItemSubset.load(repo / subsets.train_dev)
    if set(train.ids) & set(train_dev.ids):
        raise ValueError(f"{task}: Train and Train-Dev overlap")
    return {
        "job_id": rid, "cell_id": cell, "start": str(start), "seed": seed,
        "model": model, "revision": revision, "device": device, "dtype": dtype, "batch_size": batch_size,
        "task": task, "formulation": FORMULATION, "prompt_format": dict(CANONICAL), "format_id": format_id(CANONICAL),
        "num_shots": NUM_SHOTS, "score_metric": SCORE_METRIC,
        "seed_instruction": seed_instruction(start, task, repo),
        "split": train.split, "val_split": train_dev.split,
        "train_ids": _ids(task, train), "val_ids": _ids(task, train_dev),
        "subset_paths": {"train": str(repo / subsets.train), "train_dev": str(repo / subsets.train_dev)},
        "max_metric_calls": METRIC_CALLS_PER_TRAIN_DEV_ITEM * train_dev.n, "max_proposals": MAX_PROPOSALS,
        "reflection_minibatch_size": REFLECTION_MINIBATCH_SIZE,
        "reflection_model": SOL_SETTINGS.model, "reflection_reasoning": SOL_SETTINGS.reasoning,
        "reflection_token_limit": SOL_SETTINGS.token_limit,
        "prompt_cap": {"words": PROMPT_CAP.words, "chars": PROMPT_CAP.chars, "shortening_turns": PROMPT_CAP.shortening_turns},
        "reflection_prompt_version": str(ReflectionPromptVersion(reflection_prompt_version)),
        "model_card": model_card(model, revision),
        "run_dir": str(root / "gepa" / cell / rid),
    }


def job_path(root: Path, cell: str, rid: str) -> Path:
    return root / "gepa" / cell / f"{rid}.job.json"


def _write_job(path: Path, job: dict) -> None:
    """Write once; an existing identical job is reused, a different one is an error (runs may already exist)."""
    text = json.dumps(job, indent=1) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError(f"{path} exists with different content; remove it deliberately to rewrite")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def gepa_item(job: dict, *, job_file: Path, pool_dir: Path, repo: Path, olmes_repo: Path) -> dict:
    run_dir = job["run_dir"]
    return {
        "schema": POOL_ITEM_SCHEMA, "name": GEPA_ITEM_PREFIX + job["job_id"], "kind": "gepa", "class": gepa_class(job["model"]),
        "est_minutes": gepa_est_minutes(job["model"]), "cwd": str(olmes_repo),
        "command": ["uv", "run", "local/gepa_arc.py", "--job", str(job_file)],
        "followup": ["uv", "run", "--directory", str(repo), "python", "scripts/po_gepa_cells.py", "eval-item",
                     "--run-dir", run_dir, "--pool", str(pool_dir)],
        "followup_cwd": str(repo),
        "env_files": [SECRETS_ENV_FILE],
        "complete": {"type": "file", "path": f"{run_dir}/result.json"},
        "meta": {"model": job["model"], "revision": job["revision"], "task": job["task"],
                 "subsets": [job["subset_paths"]["train"], job["subset_paths"]["train_dev"]],
                 "batch_size": job["batch_size"], "dtype": job["dtype"], "quant": None,
                 "cell_id": job["cell_id"], "run_id": job["job_id"], "start": job["start"], "seed": job["seed"]},
        "created_utc": _now(), "attempts": [],
    }


def write_jobs(*, model: str, revision: str | None, task: str, root: Path, repo: Path, dtype: str | None = None,
               batch_size: int | None = None, device: str = DEFAULT_DEVICE,
               reflection_prompt_version: ReflectionPromptVersion = DEFAULT_REFLECTION_PROMPT_VERSION,
               ) -> list[tuple[Path, dict]]:
    """The cell's 6 job files (empty starts first, then ape; seeds in order) as (path, job) in that order. An existing
    identical job file is reused; a different one is an error."""
    if task not in CELL_SUBSETS:
        raise ValueError(f"task must be one of {sorted(CELL_SUBSETS)}")
    dtype = dtype or default_dtype(model)
    if dtype not in PRECISION:
        raise ValueError(f"dtype must be one of {sorted(PRECISION)}")
    batch_size = batch_size or default_batch_size(model)
    jobs = []
    for start in Start:
        for seed in GEPA_SEEDS:
            job = build_job(model=model, revision=revision, dtype=dtype, batch_size=batch_size, task=task, start=start,
                            seed=seed, root=root, repo=repo, device=device,
                            reflection_prompt_version=reflection_prompt_version)
            path = job_path(root, job["cell_id"], job["job_id"])
            _write_job(path, job)
            jobs.append((path, job))
    return jobs


def write_cell(*, model: str, revision: str | None, task: str, pool_dir: Path | None, root: Path, repo: Path,
               olmes_repo: Path | None = None, dtype: str | None = None, batch_size: int | None = None,
               device: str = DEFAULT_DEVICE,
               reflection_prompt_version: ReflectionPromptVersion = DEFAULT_REFLECTION_PROMPT_VERSION) -> list[Path]:
    """The cell's 6 job files and, when ``pool_dir`` is given, its 6 gepa items in the same order. Returns the item
    paths written; items already present in the pool (any state) are not written again. With ``pool_dir=None``
    (the Mac lane) only the job files are written and the result is empty."""
    jobs = write_jobs(model=model, revision=revision, task=task, root=root, repo=repo, dtype=dtype,
                      batch_size=batch_size, device=device, reflection_prompt_version=reflection_prompt_version)
    if pool_dir is None:
        return []
    if olmes_repo is None:
        raise ValueError("olmes_repo is required when writing pool items")
    return append_items(pool_dir, [gepa_item(job, job_file=path, pool_dir=pool_dir, repo=repo, olmes_repo=olmes_repo)
                                   for path, job in jobs])


# --- pool item files ------------------------------------------------------------------------------------------

_SEQ = re.compile(r"^(\d+)-(.+)\.json$")


def _pool_files(pool_dir: Path) -> list[Path]:
    return [p for d in POOL_STATE_DIRS if (pool_dir / d).is_dir() for p in (pool_dir / d).iterdir() if _SEQ.match(p.name)]


def append_items(pool_dir: Path, items: list[dict]) -> list[Path]:
    """Write items into <pool>/pending/ as <NNNN>-<name>.json, continuing the highest sequence number present across
    pending/claimed/done/failed. Each file is written beside the state dirs and renamed in, so a worker never sees
    a partial item. Items whose name already exists in any state dir are skipped."""
    files = _pool_files(pool_dir)
    present = {_SEQ.match(p.name).group(2) for p in files}
    seq = max((int(_SEQ.match(p.name).group(1)) for p in files), default=0)
    pending = pool_dir / "pending"
    pending.mkdir(parents=True, exist_ok=True)
    written = []
    for item in items:
        if item["name"] in present:
            continue
        seq += 1
        final = pending / f"{seq:04d}-{item['name']}.json"
        tmp = pool_dir / f".tmp-{os.getpid()}-{final.name}"
        tmp.write_text(json.dumps(item, indent=1) + "\n")
        os.rename(tmp, final)
        present.add(item["name"])
        written.append(final)
    return written


# --- eval follow-up -------------------------------------------------------------------------------------------

def load_run(run_dir: Path) -> tuple[dict, dict]:
    """(job, result) of a finished run; the job comes from the contract path beside the run dir."""
    run_dir = Path(run_dir)
    job = json.loads((run_dir.parent / f"{run_dir.name}.job.json").read_text())
    result = json.loads((run_dir / "result.json").read_text())
    return job, result


def best_text(result: dict) -> str | None:
    text = (result.get("best_candidate") or {}).get(COMPONENT)
    return text if text else None


def eval_sweep_spec(job: dict, best: str | None, repo: Path) -> SweepSpec:
    """Test subsets of the task, RC, canonical format x [none, gepa-<run_id>] (both ids always present). The sweep runs
    on ``cuda`` whatever device the GEPA search used: Test evaluation is canonical on the cluster."""
    fmt = dict(CANONICAL)
    pairs = tuple({"format": fmt, "format_id": format_id(fmt), "instruction": instr}
                  for instr in (dict(NONE_INSTRUCTION), {"id": GEPA_INSTRUCTION_PREFIX + job["job_id"], "text": best}))
    return SweepSpec(
        name=EVAL_SWEEP_PREFIX + job["job_id"], subset_paths=tuple(repo / p for p in CELL_SUBSETS[job["task"]].test),
        model=job["model"], revision=job["revision"], task=job["task"], formulations=(FORMULATION,), pairs=pairs,
        num_shots=NUM_SHOTS, device=EVAL_DEVICE, batch_size=int(job["batch_size"]), dtype=job["dtype"],
    )


def eval_item(spec: SweepSpec, sweep_dir: Path, job: dict, repo: Path) -> dict:
    item_class = eval_class(spec.model)
    return {
        "schema": POOL_ITEM_SCHEMA, "name": spec.name, "kind": "eval", "class": item_class,
        "est_minutes": EVAL_EST_MINUTES[item_class], "cwd": str(repo),
        "command": ["uv", "run", "python", "scripts/po_run_sweep.py", "--sweep-dir", str(sweep_dir)],
        "followup": None, "env_files": [],
        "complete": {"type": "sweep", "sweep_dir": str(sweep_dir)},
        "meta": {"model": spec.model, "revision": spec.revision, "task": spec.task,
                 "subsets": [str(p) for p in spec.subset_paths], "batch_size": spec.batch_size, "dtype": spec.dtype,
                 "quant": None, "cell_id": job["cell_id"], "run_id": job["job_id"]},
        "created_utc": _now(), "attempts": [],
    }


def write_eval_item(run_dir: Path, pool_dir: Path, repo: Path) -> tuple[Path, list[Path]]:
    """The gepa item's follow-up: write ROOT/sweeps/ge-<run_id> (reused when present) and append its eval item.
    Returns (sweep_dir, item paths written)."""
    run_dir = Path(run_dir)
    if run_dir.parent.parent.name != "gepa":
        raise ValueError(f"{run_dir} is not ROOT/gepa/<cell_id>/<run_id>")
    job, result = load_run(run_dir)
    spec = eval_sweep_spec(job, best_text(result), Path(repo))
    sweeps_root = run_dir.parents[2] / "sweeps"
    sweep_dir = sweeps_root / spec.name
    if not (sweep_dir / "sweep.json").exists():
        sweep_dir = write_sweep(spec, root=sweeps_root)
    return sweep_dir, append_items(Path(pool_dir), [eval_item(spec, sweep_dir, job, Path(repo))])


# --- aggregation ----------------------------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class RunDiffs:
    """Paired per-item differences optimized - canonical on Test for one run, keyed by tidy metric, and the number of
    paired items per Test subset label."""

    diffs: dict[str, np.ndarray]
    same_text: bool
    subset_counts: dict[str, int]

    @property
    def n_items(self) -> int:
        return len(next(iter(self.diffs.values())))


def run_diffs(items: pd.DataFrame, rid: str, texts: dict[str, str | None]) -> RunDiffs | None:
    """Pair items by (subset, native_id). When the optimized text equals the canonical one (empty best on the empty
    start) OLMES runs one task, the optimized rows are missing, and the effect is exactly zero (same_text)."""
    gid = GEPA_INSTRUCTION_PREFIX + rid
    base = items[items["instruction_id"] == NONE_INSTRUCTION["id"]].set_index(["subset", "native_id"])
    opt = items[items["instruction_id"] == gid].set_index(["subset", "native_id"])
    same_text = False
    if opt.empty and (texts.get(gid) or "") == (texts.get(NONE_INSTRUCTION["id"]) or ""):
        opt, same_text = base, True
    keys = base.index.intersection(opt.index)
    if len(keys) == 0:
        return None
    counts = {str(k): int(v) for k, v in pd.Series(keys.get_level_values("subset")).value_counts().items()}
    return RunDiffs({m: (opt.loc[keys, col] - base.loc[keys, col]).to_numpy(float) for m, col in METRIC_COLUMNS.items()},
                    same_text, counts)


def item_bootstrap(diff: np.ndarray, *, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    boot = diff[rng.integers(0, len(diff), size=(resamples, len(diff)))].mean(axis=1)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def hierarchical_bootstrap(diffs: list[np.ndarray], *, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """Resample runs with replacement, then items with replacement within each drawn run; the statistic is the mean
    of run means. Returns the 2.5 and 97.5 percentiles."""
    rng = np.random.default_rng(seed)
    n_runs = len(diffs)
    stats = np.empty(resamples)
    for b in range(resamples):
        drawn = rng.integers(0, n_runs, size=n_runs)
        stats[b] = np.mean([diffs[r][rng.integers(0, len(diffs[r]), size=len(diffs[r]))].mean() for r in drawn])
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def _instruction_texts(manifest: dict) -> dict[str, str | None]:
    return {i["id"]: i.get("text") for i in manifest["instructions"]}


def _test_counts(manifest: dict) -> dict[str, int]:
    """Items per Test subset label the eval sweep was written for."""
    return {s["label"]: int(s["n"]) for s in manifest["subsets"]}


def _load_items(sweep_dir: Path) -> pd.DataFrame:
    return load_sweep(sweep_dir)["items"]


def proposal_count(run_dir: Path, result: dict | None = None) -> int | None:
    """Proposal rows (kind "proposal") in the run's proposals.jsonl; without the file, result.json's reflection_calls."""
    path = run_dir / "proposals.jsonl"
    if path.exists():
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return sum(r.get("kind") == "proposal" for r in rows)
    return (result or {}).get("reflection_calls")


def collect_runs(gepa_root: Path, sweeps_dir: Path, load_items: Callable[[Path], pd.DataFrame] = _load_items
                 ) -> tuple[pd.DataFrame, dict[str, RunDiffs], list[str]]:
    """Every finished run under gepa_root whose eval sweep pairs every Test item in both arms: one gepa_runs row each
    (including the job's GEPA search ``device``), plus its diffs. Runs with result.json whose eval sweep is not written yet, empty or only partly covered are
    returned as incomplete run ids and contribute nothing."""
    rows, diffs, incomplete = [], {}, []
    for job_file in sorted(Path(gepa_root).glob("*/*.job.json")):
        rid = job_file.name.removesuffix(".job.json")
        run_dir = job_file.parent / rid
        sweep_dir = Path(sweeps_dir) / (EVAL_SWEEP_PREFIX + rid)
        if not (run_dir / "result.json").exists():
            continue
        if not (sweep_dir / "sweep.json").exists():
            incomplete.append(rid)
            continue
        manifest = json.loads((sweep_dir / "sweep.json").read_text())
        items = load_items(sweep_dir)
        rd = None if items.empty else run_diffs(items, rid, _instruction_texts(manifest))
        if rd is None or rd.subset_counts != _test_counts(manifest):
            incomplete.append(rid)
            continue
        job, result = load_run(run_dir)
        best = best_text(result) or ""
        usage = result.get("reflection_usage") or {}
        row = {
            "run_id": rid, "cell_id": job["cell_id"], "model_key": model_key(job["model"], job["revision"], job["dtype"]),
            "task": job["task"], "start": job["start"], "seed": job["seed"], "device": job["device"],
            "seed_val_score": result.get("seed_val_score"), "best_val_score": result.get("best_val_score"),
            "proposals": proposal_count(run_dir, result), "num_candidates": result.get("num_candidates"),
            "total_metric_calls": result.get("total_metric_calls"),
            "reflection_calls": usage.get("calls", result.get("reflection_calls")),
            "reflection_prompt_tokens": usage.get("prompt_tokens"), "reflection_completion_tokens": usage.get("completion_tokens"),
            "reflection_reasoning_tokens": usage.get("reasoning_tokens"),
            "reflection_total_tokens": usage.get("total_tokens", result.get("reflection_total_tokens")),
            "shortening_turns_total": usage.get("shortening_turns_total"), "truncations": usage.get("truncations"),
            "best_candidate_words": result.get("best_candidate_words", len(best.split())),
            "best_candidate_chars": result.get("best_candidate_chars", len(best)),
            "best_text": best, "same_text": rd.same_text, "n_items": rd.n_items, "eval_sweep": sweep_dir.name,
        }
        for metric, diff in rd.diffs.items():
            lo, hi = item_bootstrap(diff)
            row |= {f"{metric}_delta": float(diff.mean()), f"{metric}_lo": lo, f"{metric}_hi": hi}
        rows.append(row)
        diffs[rid] = rd
    return pd.DataFrame(rows), diffs, incomplete


def cell_contrasts(runs: pd.DataFrame, diffs: dict[str, RunDiffs]) -> pd.DataFrame:
    """Per cell and start group (both, empty, ape) and metric: delta = mean of run means, hierarchical CI."""
    rows = []
    for cell, g in runs.groupby("cell_id", sort=True):
        groups = [(BOTH_STARTS, g)] + [(str(s), g[g["start"] == str(s)]) for s in Start]
        for label, sub in groups:
            if sub.empty:
                continue
            run_list = [diffs[r] for r in sub["run_id"]]
            for metric in METRIC_COLUMNS:
                per_run = [rd.diffs[metric] for rd in run_list]
                lo, hi = hierarchical_bootstrap(per_run)
                rows.append({
                    "contrast_type": CONTRAST_TYPE, "a_key": sub["model_key"].iloc[0], "b_key": f"gepa:{cell}:{label}",
                    "task": sub["task"].iloc[0], "role": "test", "metric": metric,
                    "delta": float(np.mean([d.mean() for d in per_run])), "lo": lo, "hi": hi,
                    "n_items": int(min(len(d) for d in per_run)), "n_runs": len(per_run), "method": CONTRAST_METHOD,
                })
    return pd.DataFrame(rows)


def _replace_rows(path: Path, new: pd.DataFrame, stale: Callable[[pd.DataFrame], pd.Series]) -> pd.DataFrame:
    if path.exists():
        old = pd.read_parquet(path)
        new = pd.concat([old[~stale(old)], new], ignore_index=True) if len(old) else new
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_parquet(path, index=False)
    new.to_csv(path.with_suffix(".csv"), index=False)
    return new


def aggregate(tidy_dir: Path, gepa_root: Path, sweeps_dir: Path, load_items: Callable[[Path], pd.DataFrame] = _load_items
              ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Append gepa rows to contrasts.parquet and write gepa_runs.parquet (csv twins beside), replacing any earlier
    rows of the cells aggregated now. Returns (contrast rows, run rows, incomplete run ids) of this aggregation."""
    runs, diffs, incomplete = collect_runs(gepa_root, sweeps_dir, load_items)
    if runs.empty:
        return pd.DataFrame(), runs, incomplete
    contrasts = cell_contrasts(runs, diffs)
    cells = set(runs["cell_id"])
    prefixes = tuple(f"gepa:{c}:" for c in cells)
    tidy_dir = Path(tidy_dir)
    _replace_rows(tidy_dir / "contrasts.parquet", contrasts,
                  lambda df: (df["contrast_type"] == CONTRAST_TYPE) & df["b_key"].astype(str).str.startswith(prefixes))
    _replace_rows(tidy_dir / "gepa_runs.parquet", runs, lambda df: df["cell_id"].isin(cells))
    return contrasts, runs, incomplete
