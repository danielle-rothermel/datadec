"""Tidy results tables for the phase-2 sweeps (contract po-tidy/1).

`build(sweeps_root, out)` reads every selected sweep directory and writes, as parquet plus csv twins:

- `models`: one row per (model, revision, dtype, quant) seen, attributes from `datadec.po.model_registry`.
- `tasks`: one row per sweep x OLMES task (task x subset x formulation x format x instruction).
- `summary`: one row per model_key x task x role x instruction_id, items pooled across the role's subsets.
- `contrasts`: paired per-item differences (b - a) for the natural contrasts on the canonical prompt
  (instruction `none`), one row per contrast x task x role x metric. `gepa` rows are appended later by the
  GEPA aggregation step and never computed here.

Accuracy is the primary metric (acc_per_char); likelihood is `primary_likelihood`, the gold choice's share of the
per-char-normalized choice probabilities. Chance per item is 1 / number of choices (from the choices table).
CIs are 95 % percentile item bootstraps, 2000 resamples, seed 0, drawn afresh for each statistic so a value does
not depend on build order.

When several sweeps cover the same (model_key, task, subset, formulation, format, instruction) - retries and the
canonical pair inside GEPA eval sweeps - the summary and contrasts use one source: non-GEPA-eval first, then the
one with more items, then the first sweep name.
"""

from __future__ import annotations

import itertools
import json
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from enum import StrEnum, UNIQUE, verify
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from datadec.po.model_registry import CheckpointKind, Precision, Variant, describe
from datadec.po.results import load_sweep

INCLUDE_PREFIXES = ("baselines-", "ladder-dd-", "pythia-ladder-", "g4-", "mid-", "peak-", "quant-olmo3", "tasks-",
                    "t-", "ge-")
EXCLUDE_PREFIXES = ("g5-", "quant-smoke", "bscheck")  # g5: Granite dropped from the model set
GEPA_EVAL_PREFIX = "ge-"
TASK_SWEEP_PREFIX = "t-"
RETRY_PREFIX = "retry-"
TASKS = ("arc_easy", "arc_challenge", "hellaswag", "csqa", "socialiqa", "openbookqa", "piqa", "winogrande")
BASELINE_INSTRUCTION = "none"
N_RESAMPLES = 2000
BOOTSTRAP_SEED = 0
TABLES = ("models", "tasks", "summary", "contrasts")
METRICS = {"accuracy": "primary", "likelihood": "primary_likelihood"}
RECIPE_ORDER = ("dclm", "dolma", "c4")


@verify(UNIQUE)
class Role(StrEnum):
    TRAIN = "train"
    TRAIN_DEV = "train-dev"
    TEST = "test"


@verify(UNIQUE)
class ContrastType(StrEnum):
    SIZE_ADJACENT = "size-adjacent"
    RECIPE = "recipe"
    BASE_VS_INSTRUCT = "base-vs-instruct"
    STAGE = "stage"
    PRECISION = "precision"
    CHECKPOINT = "checkpoint"
    SEED = "seed"
    GEPA = "gepa"


CONTRAST_METHOD = "paired-item-bootstrap"
# subset-label fragment -> role; checked in order (traindev before train)
ROLE_FRAGMENTS = (("-part-traindev-", Role.TRAIN_DEV), ("-part-train-", Role.TRAIN), ("-part-testaug-", Role.TEST),
                  ("-part-test1-", Role.TEST), ("-part-test-", Role.TEST), ("-validation-", Role.TEST))
TASK_COLUMNS = ("sweep", "model_key", "task", "subset", "role", "formulation", "format_id", "instruction_id",
                "n_items", "accuracy", "likelihood", "chance_accuracy", "chance_likelihood", "processing_time",
                "run_dir", "complete")
CONTRAST_COLUMNS = ("contrast_type", "a_key", "b_key", "task", "role", "metric", "delta", "lo", "hi", "n_items",
                    "n_runs", "method")


def role_for_subset(label: str) -> Role:
    for fragment, role in ROLE_FRAGMENTS:
        if fragment in label:
            return role
    raise ValueError(f"subset label {label!r} has no role")


def _inner_name(name: str) -> str:
    """The original sweep name behind a cross-task sweep (t-<task>-[retry-]<name>)."""
    if name.startswith(TASK_SWEEP_PREFIX):
        rest = name[len(TASK_SWEEP_PREFIX):]
        for task in TASKS:
            if rest.startswith(task + "-"):
                rest = rest[len(task) + 1:]
                return rest[len(RETRY_PREFIX):] if rest.startswith(RETRY_PREFIX) else rest
    return name


@dataclass(frozen=True, slots=True)
class SweepSelection:
    included: tuple[Path, ...]
    excluded: dict[str, int]  # prefix (or reason) -> count


def select_sweeps(root: Path) -> SweepSelection:
    included, excluded = [], Counter()
    for d in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        inner = _inner_name(d.name)
        if hit := next((p for p in EXCLUDE_PREFIXES if inner.startswith(p)), None):
            excluded[hit] += 1
        elif not d.name.startswith(INCLUDE_PREFIXES):
            excluded[d.name.split("-")[0] + "-"] += 1
        elif not (d / "sweep.json").exists():
            excluded["no sweep.json"] += 1
        else:
            included.append(d)
    return SweepSelection(tuple(included), dict(sorted(excluded.items())))


MODEL_ID_COLUMNS = ("model", "revision", "dtype", "quant")


def _none(v: object) -> object:
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else v


def load_sweep_items(sweep_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(task rows, item rows) of one sweep in tidy form; empty frames when the sweep has no finished task."""
    manifest = json.loads((Path(sweep_dir) / "sweep.json").read_text())
    tables = load_sweep(sweep_dir)
    tasks, items, choices = tables["tasks"], tables["items"], tables["choices"]
    if tasks.empty:
        return pd.DataFrame(), pd.DataFrame()
    key = ["run_dir", "task_idx", "doc_id"]
    n_choices = choices.groupby(key).size().rename("n_choices").reset_index()
    items = items.merge(n_choices, on=key, how="left", suffixes=("_lik", ""))
    if items["n_choices"].isna().any():
        raise ValueError(f"{sweep_dir}: items without choices")
    complete = len(tasks) == manifest["num_tasks"]
    # older task files carry no dtype/quant in their metadata; the manifest records what the runner was given
    ident = tuple(manifest.get(c) for c in MODEL_ID_COLUMNS)
    for col, want in zip(MODEL_ID_COLUMNS, ident):
        seen = {_none(v) for v in items[col]} - {None}
        if seen - {want}:
            raise ValueError(f"{sweep_dir}: task files disagree with sweep.json on {col}: {seen} vs {want!r}")
    info = describe(*ident)
    items = items.assign(
        task=manifest["task"],
        model_key=info.model_key,
        chance=1.0 / items["n_choices"].astype(float),
        native_id=items["native_id"].astype(str),
    )
    keep = ["sweep", "model_key", "model", "revision", "dtype", "quant", "task", "subset", "formulation", "format_id",
            "instruction_id", "num_shots", "run_dir", "task_idx", "native_id", "primary", "primary_likelihood",
            "chance"]
    items = items[keep]
    task_rows = (items.groupby(["run_dir", "task_idx"], sort=False)
                 .agg(sweep=("sweep", "first"), model_key=("model_key", "first"), task=("task", "first"),
                      subset=("subset", "first"), formulation=("formulation", "first"),
                      format_id=("format_id", "first"), instruction_id=("instruction_id", "first"),
                      num_shots=("num_shots", "first"), n_items=("native_id", "size"), accuracy=("primary", "mean"),
                      likelihood=("primary_likelihood", "mean"), chance_accuracy=("chance", "mean"))
                 .reset_index())
    task_rows = task_rows.merge(tasks[["run_dir", "task_idx", "processing_time"]], on=["run_dir", "task_idx"])
    task_rows["chance_likelihood"] = task_rows["chance_accuracy"]
    task_rows["role"] = task_rows["subset"].map(role_for_subset).astype(str)
    task_rows["complete"] = complete
    task_rows["model_info"] = [info.row()] * len(task_rows)
    return task_rows, items


# ---------------------------------------------------------------- bootstrap


@lru_cache(maxsize=2)
def _resample_index(n: int) -> np.ndarray:
    return np.random.default_rng(BOOTSTRAP_SEED).integers(0, n, size=(N_RESAMPLES, n), dtype=np.int32)


def bootstrap_ci(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """95 % percentile bootstrap CI of the column means of values (n x k): resample rows, 2000 draws, seed 0."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    idx = _resample_index(len(values))
    means = np.stack([values[:, j][idx].mean(axis=1) for j in range(values.shape[1])], axis=1)
    return np.percentile(means, 2.5, axis=0), np.percentile(means, 97.5, axis=0)


# ---------------------------------------------------------------- selection of one source per item set


def dedupe_sources(task_rows: pd.DataFrame) -> pd.DataFrame:
    """One task row per (model_key, task, subset, formulation, format_id, instruction_id, num_shots)."""
    key = ["model_key", "task", "subset", "formulation", "format_id", "instruction_id", "num_shots"]
    ranked = task_rows.assign(_ge=task_rows["sweep"].str.startswith(GEPA_EVAL_PREFIX))
    ranked = ranked.sort_values(["_ge", "n_items", "sweep"], ascending=[True, False, True], kind="stable")
    return ranked.drop_duplicates(key, keep="first").drop(columns="_ge")


# ---------------------------------------------------------------- summary


def summarize(items: pd.DataFrame) -> pd.DataFrame:
    """Pool a task's selected items per (model_key, task, role, instruction_id); bootstrap both metrics."""
    rows = []
    groups = items.groupby(["model_key", "task", "role", "instruction_id"], sort=True)
    for (key, task, role, instruction), g in sorted(groups, key=lambda kv: len(kv[1])):
        for col in ("formulation", "format_id", "num_shots"):
            if g[col].nunique() > 1:
                raise ValueError(f"{key} {task} {role} {instruction}: mixed {col} {sorted(g[col].unique())}")
        vals = g[["primary", "primary_likelihood"]].to_numpy(float)
        lo, hi = bootstrap_ci(vals)
        mean = vals.mean(axis=0)
        chance = float(g["chance"].mean())
        rows.append({"model_key": key, "task": task, "role": role, "instruction_id": instruction, "n_items": len(g),
                     "accuracy": mean[0], "accuracy_lo": lo[0], "accuracy_hi": hi[0],
                     "likelihood": mean[1], "likelihood_lo": lo[1], "likelihood_hi": hi[1],
                     "chance_accuracy": chance, "chance_likelihood": chance,
                     "margin_accuracy": mean[0] - chance, "margin_likelihood": mean[1] - chance,
                     "n_sweeps": g["sweep"].nunique()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- natural contrasts


def _groups(models: pd.DataFrame, cols: list[str]):
    return models.groupby(cols, dropna=False, sort=True)


def natural_contrasts(models: pd.DataFrame) -> list[tuple[str, str, str]]:
    """(contrast_type, a_key, b_key) for every natural contrast among the models; delta is b - a."""
    out: list[tuple[str, str, str]] = []
    m = models
    # adjacent sizes within (family, recipe, variant, precision, checkpoint_kind, seed_label)
    for _, g in _groups(m, ["family", "recipe", "variant", "precision", "checkpoint_kind", "seed_label"]):
        g = g.sort_values("params")
        if g["params"].duplicated().any():
            raise ValueError(f"duplicate sizes in a size ladder: {sorted(g['model_key'])}")
        keys = list(g["model_key"])
        out += [(ContrastType.SIZE_ADJACENT, a, b) for a, b in itertools.pairwise(keys)]
    # recipes at matched size (DataDecide), fixed recipe order
    dd = m[m["recipe"].notna()]
    for _, g in _groups(dd, ["family", "size_label", "variant", "precision", "checkpoint_kind", "seed_label"]):
        by_recipe = dict(zip(g["recipe"], g["model_key"]))
        present = [r for r in RECIPE_ORDER if r in by_recipe]
        out += [(ContrastType.RECIPE, by_recipe[a], by_recipe[b]) for a, b in itertools.combinations(present, 2)]
    # base vs instruct, and successive stages, within a family and size
    for _, g in _groups(m, ["family", "size_label", "precision", "checkpoint_kind", "seed_label"]):
        base = g[g["variant"] == Variant.BASE]["model_key"].tolist()
        instruct = g[g["variant"] == Variant.INSTRUCT]["model_key"].tolist()
        out += [(ContrastType.BASE_VS_INSTRUCT, a, b) for a in base for b in instruct]
        stages = sorted(g["stage_index"].unique())
        if len(stages) >= 3:  # a two-stage family (base, instruct) is covered by base-vs-instruct
            for s0, s1 in itertools.pairwise(stages):
                lo_keys = sorted(g[g["stage_index"] == s0]["model_key"])
                hi_keys = sorted(g[g["stage_index"] == s1]["model_key"])
                out += [(ContrastType.STAGE, a, b) for a in lo_keys for b in hi_keys]
    # each precision vs fp32-tf32 for the same weights
    for _, g in _groups(m, ["model", "revision"]):
        ref = g[g["precision"] == Precision.FP32_TF32]["model_key"].tolist()
        others = sorted(g[g["precision"] != Precision.FP32_TF32]["model_key"])
        out += [(ContrastType.PRECISION, a, b) for a in ref for b in others]
    # mid and peak vs final for the same run
    for _, g in _groups(m, ["model", "seed_label", "precision"]):
        final = g[g["checkpoint_kind"] == CheckpointKind.FINAL]["model_key"].tolist()
        inter = sorted(g[g["checkpoint_kind"] != CheckpointKind.FINAL]["model_key"])
        out += [(ContrastType.CHECKPOINT, a, b) for a in final for b in inter]
    # each aux seed vs default
    seeded = m[m["seed_label"].notna()]
    for _, g in _groups(seeded, ["model", "checkpoint_kind", "precision"]):
        default = g[g["seed_label"] == "default"]["model_key"].tolist()
        aux = sorted(g[g["seed_label"] != "default"]["model_key"])
        out += [(ContrastType.SEED, a, b) for a in default for b in aux]
    return [(str(t), a, b) for t, a, b in out]


def paired_contrasts(items: pd.DataFrame, pairs: list[tuple[str, str, str]]) -> pd.DataFrame:
    """Paired per-item b - a on the canonical prompt for each pair and each (task, role) both models cover."""
    base = items[items["instruction_id"] == BASELINE_INSTRUCTION]
    vals = {k: g.set_index(["task", "role", "subset", "format_id", "native_id"])[["primary", "primary_likelihood"]]
            for k, g in base.groupby("model_key")}
    jobs = []
    for ctype, a, b in pairs:
        if a not in vals or b not in vals:
            continue
        joined = vals[a].join(vals[b], how="inner", lsuffix="_a", rsuffix="_b")
        for (task, role), g in joined.groupby(level=["task", "role"]):
            d = g[["primary_b", "primary_likelihood_b"]].to_numpy(float) - g[["primary_a", "primary_likelihood_a"]].to_numpy(float)
            jobs.append((ctype, a, b, task, role, d))
    rows = []
    for ctype, a, b, task, role, d in sorted(jobs, key=lambda j: len(j[5])):
        lo, hi = bootstrap_ci(d)
        mean = d.mean(axis=0)
        for j, metric in enumerate(METRICS):
            rows.append({"contrast_type": ctype, "a_key": a, "b_key": b, "task": task, "role": role, "metric": metric,
                         "delta": mean[j], "lo": lo[j], "hi": hi[j], "n_items": len(d), "n_runs": 1,
                         "method": CONTRAST_METHOD})
    return pd.DataFrame(rows, columns=list(CONTRAST_COLUMNS))


def _task_stats(args: tuple[pd.DataFrame, list[tuple[str, str, str]]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    items, pairs = args
    return summarize(items), paired_contrasts(items, pairs)


# ---------------------------------------------------------------- build


@dataclass(frozen=True, slots=True)
class BuildResult:
    tables: dict[str, pd.DataFrame]
    selection: SweepSelection
    duplicate_task_rows: int
    seconds: float


def build_tables(sweeps_root: Path, *, workers: int | None = None) -> BuildResult:
    start = time.monotonic()
    selection = select_sweeps(sweeps_root)
    workers = workers or os.cpu_count() or 1
    with ProcessPoolExecutor(max_workers=workers) as pool:
        loaded = list(pool.map(load_sweep_items, selection.included, chunksize=4))
    task_rows = pd.concat([t for t, _ in loaded if not t.empty], ignore_index=True)
    items = pd.concat([i for _, i in loaded if not i.empty], ignore_index=True)
    models = (pd.DataFrame(task_rows["model_info"].tolist()).drop_duplicates("model_key")
              .sort_values(["family", "variant", "recipe", "params", "checkpoint_fraction", "precision", "seed_label"],
                           na_position="first").reset_index(drop=True))
    models["checkpoint_step"] = models["checkpoint_step"].astype("Int64")
    task_rows = task_rows.drop(columns="model_info")
    chosen = dedupe_sources(task_rows)
    items = items.merge(chosen[["run_dir", "task_idx", "role"]], on=["run_dir", "task_idx"], how="inner")
    pairs = natural_contrasts(models)
    per_task = [(g, pairs) for _, g in items.groupby("task")]
    with ProcessPoolExecutor(max_workers=min(workers, len(per_task))) as pool:
        stats = list(pool.map(_task_stats, per_task))
    summary = pd.concat([s for s, _ in stats], ignore_index=True).sort_values(
        ["model_key", "task", "role", "instruction_id"]).reset_index(drop=True)
    contrasts = pd.concat([c for _, c in stats], ignore_index=True).sort_values(
        ["contrast_type", "a_key", "b_key", "task", "role", "metric"]).reset_index(drop=True)
    tasks = task_rows[list(TASK_COLUMNS)].sort_values(["sweep", "task", "subset", "instruction_id"]).reset_index(drop=True)
    return BuildResult({"models": models, "tasks": tasks, "summary": summary, "contrasts": contrasts}, selection,
                       duplicate_task_rows=len(task_rows) - len(chosen), seconds=time.monotonic() - start)


def write_tables(result: BuildResult, out: Path) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in result.tables.items():
        df.to_parquet(out / f"{name}.parquet", index=False)
        df.to_csv(out / f"{name}.csv", index=False)
    complete = result.tables["tasks"].groupby("sweep")["complete"].first()
    (out / "build.json").write_text(json.dumps({
        "contract": "po-tidy/1",
        "sweeps_included": len(result.selection.included),
        "sweeps_with_tasks": int(len(complete)),
        "sweeps_incomplete": sorted(complete[~complete].index),
        "sweeps_excluded": result.selection.excluded,
        "duplicate_task_rows_not_pooled": result.duplicate_task_rows,
        "rows": {k: len(v) for k, v in result.tables.items()},
        "seconds": round(result.seconds, 1),
    }, indent=1) + "\n")


def read_tables(tidy: Path) -> dict[str, pd.DataFrame]:
    return {name: pd.read_parquet(Path(tidy) / f"{name}.parquet") for name in TABLES}
