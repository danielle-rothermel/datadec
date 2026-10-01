"""Paired per-item effects of GEPA-optimized instructions against their own seed instructions.

Reads evaluation sweeps built from a GEPA ``optimized_pairs.json`` (instruction ids ``gepa-<job_id>`` and
``seed-<job_id>``), pairs items by native_id within (model, formulation, job), and bootstraps the mean
difference for the primary metric (accuracy) and primary_likelihood (per-char share). One row per
(sweep, model, formulation, job, metric) goes to effects.csv plus a markdown table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer

from datadec.po.results import load_sweep

app = typer.Typer()
GEPA_PREFIX = "gepa-"
SEED_PREFIX = "seed-"
METRICS = ("primary", "primary_likelihood")


def _subset_label(sweep_dir: Path) -> str:
    manifest = json.loads((sweep_dir / "sweep.json").read_text())
    return Path(manifest["subset_path"]).stem


def paired_effects(items: pd.DataFrame, *, rng: np.random.Generator, resamples: int) -> list[dict]:
    rows = []
    for (model, form), g in items.groupby(["model", "formulation"]):
        jobs = sorted({i[len(GEPA_PREFIX):] for i in g["instruction_id"].unique() if i.startswith(GEPA_PREFIX)})
        for job in jobs:
            opt = g[g["instruction_id"] == GEPA_PREFIX + job].set_index("native_id")
            seed = g[g["instruction_id"] == SEED_PREFIX + job].set_index("native_id")
            ids = opt.index.intersection(seed.index)
            if len(ids) == 0:
                continue
            opt, seed = opt.loc[ids], seed.loc[ids]
            for metric in METRICS:
                diff = (opt[metric] - seed[metric]).to_numpy(float)
                boot = diff[rng.integers(0, len(diff), size=(resamples, len(diff)))].mean(axis=1)
                rows.append({
                    "model": model, "formulation": form, "job": job, "metric": metric, "n": int(len(diff)),
                    "seed_mean": float(seed[metric].mean()), "optimized_mean": float(opt[metric].mean()),
                    "diff": float(diff.mean()), "se": float(diff.std(ddof=1) / np.sqrt(len(diff))),
                    "ci_lo": float(np.percentile(boot, 2.5)), "ci_hi": float(np.percentile(boot, 97.5)),
                })
    return rows


@app.command()
def main(
    sweep_dirs: Annotated[list[Path], typer.Argument(help="evaluation sweep dirs (e.g. one for dev, one for test)")],
    out: Annotated[Path, typer.Option("--out")],
    resamples: Annotated[int, typer.Option("--resamples")] = 2000,
    seed: Annotated[int, typer.Option("--seed")] = 0,
) -> None:
    rng = np.random.default_rng(seed)
    rows = []
    for d in sweep_dirs:
        items = load_sweep(d)["items"]
        if items.empty:
            typer.echo(f"no items in {d}")
            continue
        label = _subset_label(d)
        rows += [r | {"sweep": d.name, "subset": label} for r in paired_effects(items, rng=rng, resamples=resamples)]
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "effects.csv", index=False)
    md = ["# GEPA optimized vs seed (paired per-item bootstrap, 95% CI)", ""]
    for (sweep, subset), sub in df.groupby(["sweep", "subset"]):
        md += [f"## {sweep} ({subset})", "", "| model | form | job | metric | n | seed | optimized | diff | 95% CI |", "|---|---|---|---|---|---|---|---|---|"]
        for _, r in sub.sort_values(["model", "formulation", "job", "metric"]).iterrows():
            md.append(f"| {r.model} | {r.formulation} | {r.job} | {r.metric} | {r.n} | {r.seed_mean:.4f} | {r.optimized_mean:.4f} | {r['diff']:+.4f} | [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] |")
        md.append("")
    (out / "effects.md").write_text("\n".join(md))
    typer.echo(f"{len(df)} rows -> {out / 'effects.csv'}")


if __name__ == "__main__":
    app()
