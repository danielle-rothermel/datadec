"""Aggregate a pipeline root into per-sweep summaries and GEPA seed-vs-optimized deltas."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from datadec.po.results import load_sweep

METRICS = ("primary", "primary_likelihood", "acc_raw", "acc_per_char", "acc_uncond",
           "norm_correct_prob", "norm_correct_prob_per_char", "norm_correct_prob_uncond", "norm_margin")
HEADLINE = "primary_likelihood"  # spread attribution and best/worst pairs are ranked on this


def sweep_summary(sweep_dir: Path) -> dict:
    """Per-formulation spread across (format, instruction) pairs, baseline pair (read from the baselines dir
    beside the sweeps, or from the sweep itself if it was never moved), and format/instruction attribution."""
    tables = load_sweep(sweep_dir)
    items = tables["items"]
    if items.empty:
        return {"sweep": sweep_dir.name, "empty": True}
    spec = json.loads((sweep_dir / "sweep.json").read_text())
    out = {"sweep": sweep_dir.name, "task": spec.get("task"), "model": spec["model"], "revision": spec.get("revision"),
           "n_items": int(items["native_id"].nunique()), "formulations": {}}
    per_pair = items.groupby(["formulation", "format_id", "instruction_id"])[list(METRICS)].mean().reset_index()
    for form, g in per_pair.groupby("formulation"):
        form_items = items[items["formulation"] == form]
        canon = form_items[(form_items["instruction_id"] == "none") & (form_items["format_id"] == "daa93775")]
        entry = {"n_pairs": int(len(g)),
                 "primary_metric": form_items["primary_metric"].iloc[0],
                 "primary_likelihood_metric": form_items["primary_likelihood_metric"].iloc[0]}
        for m in METRICS:
            entry[m] = {"mean": float(g[m].mean()), "std": float(g[m].std()), "min": float(g[m].min()), "max": float(g[m].max())}
            if len(canon):
                entry[m]["baseline"] = float(canon[m].mean())
        # crude attribution: variance of pair means explained by format vs instruction main effects
        for key, label in (("format_id", "format"), ("instruction_id", "instruction")):
            means = g.groupby(key)[HEADLINE].mean()
            entry[f"{label}_effect_std"] = float(means.std()) if len(means) > 1 else None
        top = g.nlargest(3, HEADLINE)[["format_id", "instruction_id", "primary_likelihood", "primary"]]
        bot = g.nsmallest(3, HEADLINE)[["format_id", "instruction_id", "primary_likelihood", "primary"]]
        entry["best_pairs"] = top.to_dict("records")
        entry["worst_pairs"] = bot.to_dict("records")
        out["formulations"][form] = entry
    return out


def gepa_summary(gepa_dir: Path, eval_sweep_dir: Path | None) -> list[dict]:
    """One row per GEPA job: seed vs best val score, and test-subset scores of seed vs optimized if evaluated."""
    rows = []
    test = load_sweep(eval_sweep_dir)["items"] if eval_sweep_dir and (eval_sweep_dir / "analysis").exists() else pd.DataFrame()
    for job_path in sorted((gepa_dir / "jobs").glob("*.json")):
        job = json.loads(job_path.read_text())
        res_path = Path(job["run_dir"]) / "result.json"
        row = {"job_id": job["job_id"], "group": job["group"], "seed_rank": job["seed_rank"], "formulation": job["formulation"],
               "format_id": job["format_id"], "seed_instruction_id": job["seed_instruction"]["id"], "seed_round_score": job["seed_score"],
               "completed": res_path.exists()}
        if res_path.exists():
            r = json.loads(res_path.read_text())
            row |= {"val_seed": r["seed_val_score"], "val_best": r["best_val_score"], "num_candidates": r["num_candidates"],
                    "reflection_calls": r["reflection_calls"], "reflection_tokens": r["reflection_total_tokens"],
                    "optimized_text": r["best_candidate"]["system_prompt"]}
            if not test.empty:
                for tag in ("seed", "gepa"):
                    sub = test[(test["instruction_id"] == f"{tag}-{job['job_id']}") & (test["formulation"] == job["formulation"])]
                    if len(sub):
                        for m in METRICS:
                            row[f"test_{tag}_{m}"] = float(sub[m].mean())
        rows.append(row)
    return rows


def pipeline_report(root: Path) -> dict:
    sweeps_dir = root / "sweeps"
    report = {"root": str(root), "sweeps": [], "gepa": {}}
    for d in sorted(sweeps_dir.iterdir()) if sweeps_dir.exists() else []:
        if (d / "analysis" / "tasks.parquet").exists() and not d.name.endswith("-gepa-eval"):
            report["sweeps"].append(sweep_summary(d))
    gepa_root = root / "gepa"
    for d in sorted(gepa_root.iterdir()) if gepa_root.exists() else []:
        if (d / "jobs").exists():
            report["gepa"][d.name] = gepa_summary(d, sweeps_dir / f"{d.name}-gepa-eval")
    return report
