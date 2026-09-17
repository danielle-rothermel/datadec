"""Select best/worst seeds from a sweep and write GEPA job files."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from datadec.po.model_cards import model_card
from datadec.po.subsets import ItemSubset

DEFAULT_ROOT = Path.home() / "drotherm" / "data" / "runs" / "po" / "gepa"


def rank_seeds(items: pd.DataFrame, *, metric: str = "correct_prob") -> pd.DataFrame:
    """Mean metric per (model, revision, formulation, format_id, instruction_id), ranked within (model, formulation)."""
    keys = ["model", "revision", "formulation", "format_id", "instruction_id"]
    items = items.assign(revision=items["revision"].fillna("main"))  # pandas drops NaN group keys by default
    g = items.groupby(keys, dropna=False).agg(score=(metric, "mean"), acc_raw=("acc_raw", "mean"), n=("native_id", "size")).reset_index()
    g["rank"] = g.groupby(["model", "revision", "formulation"])["score"].rank(ascending=False, method="first")
    g["n_seeds"] = g.groupby(["model", "revision", "formulation"])["score"].transform("size")
    return g.sort_values(["model", "revision", "formulation", "rank"]).reset_index(drop=True)


def select_seeds(ranked: pd.DataFrame, n: int) -> pd.DataFrame:
    top = ranked[ranked["rank"] <= n].assign(group="best")
    bottom = ranked[ranked["rank"] > ranked["n_seeds"] - n].assign(group="worst")
    return pd.concat([top, bottom]).sort_values(["model", "revision", "formulation", "rank"]).reset_index(drop=True)


def write_jobs(
    *,
    name: str,
    sweep_dir: Path,
    selected: pd.DataFrame,
    train_subset: ItemSubset,
    val_subset: ItemSubset,
    max_metric_calls: int,
    reflection_minibatch_size: int = 3,
    reflection_model: str = "openai/gpt-5.1",
    reflection_reasoning: str = "medium",
    device: str = "mps",
    batch_size: int = 4,
    formulations: tuple[str, ...] | None = None,
    max_group_score: float | None = None,
    root: Path = DEFAULT_ROOT,
) -> list[Path]:
    """Write one job per selected seed. max_group_score skips (model, formulation) groups whose best seed
    already scores at or above it (no headroom)."""
    sweep = json.loads((Path(sweep_dir) / "sweep.json").read_text())
    formats = {_fid(f): f for f in sweep["formats"]} | {p["format_id"]: p["format"] for p in sweep.get("pairs", [])}
    instructions = {i["id"]: i["text"] for i in sweep["instructions"]} | {"none": None}
    if set(train_subset.ids) & set(val_subset.ids):
        raise ValueError("train and val subsets overlap")
    out_root = root / name
    jobs_dir = out_root / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    skipped_groups = []
    for _, row in selected.iterrows():
        if formulations and row["formulation"] not in formulations:
            continue
        if max_group_score is not None:
            group_best = selected[(selected["model"] == row["model"]) & (selected["formulation"] == row["formulation"])]["score"].max()
            if group_best >= max_group_score:
                skipped_groups.append((row["model"], row["formulation"], round(float(group_best), 3)))
                continue
        text = instructions[row["instruction_id"]] or ""  # no-instruction seeds start GEPA from an empty prompt
        job_id = f"{_slug(row['model'])}--{row['formulation']}--{row['group']}-r{int(row['rank']):02d}--{row['format_id']}--{_slug(row['instruction_id'])}"
        job = {
            "job_id": job_id, "group": row["group"], "seed_rank": int(row["rank"]), "seed_score": float(row["score"]),
            "model": row["model"], "revision": None if row["revision"] in (None, "main") else row["revision"],
            "device": device, "batch_size": batch_size, "task": sweep.get("task", "arc_easy"), "formulation": row["formulation"],
            "prompt_format": formats[row["format_id"]], "format_id": row["format_id"],
            "seed_instruction": {"id": row["instruction_id"], "text": text},
            "split": train_subset.split, "val_split": val_subset.split,
            "train_ids": list(train_subset.ids), "val_ids": list(val_subset.ids),
            "max_metric_calls": max_metric_calls, "reflection_minibatch_size": reflection_minibatch_size,
            "reflection_model": reflection_model,
            "reflection_reasoning": reflection_reasoning,
            "model_card": model_card(row["model"], None if row["revision"] in (None, "main") else row["revision"]),
            "source_sweep": str(sweep_dir), "run_dir": str(out_root / "runs" / job_id), "seed": 0,
        }
        p = jobs_dir / f"{job_id}.json"
        p.write_text(json.dumps(job, indent=1) + "\n")
        paths.append(p)
    (out_root / "selection.csv").write_text(selected.to_csv(index=False))
    if skipped_groups:
        (out_root / "skipped_groups.json").write_text(json.dumps(sorted(set(skipped_groups)), indent=1) + "\n")
    return paths


def _fid(fmt: dict) -> str:
    from datadec.po.formats import AXES, format_id
    return format_id({a: fmt[a] for a in AXES})


def _slug(value: str) -> str:
    return value.replace("/", "--")
