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

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

matplotlib.use("Agg")

from datadec.po.results import load_sweep

app = typer.Typer()
GEPA_PREFIX = "gepa-"
SEED_PREFIX = "seed-"
METRICS = {"primary": "accuracy", "primary_likelihood": "share"}


def _subset_label(sweep_dir: Path) -> str | None:
    """Single-subset sweeps: the subset stem from the manifest; multi-subset sweeps carry it per item instead."""
    manifest = json.loads((sweep_dir / "sweep.json").read_text())
    paths = manifest.get("subset_paths") or [manifest.get("subset_path")]
    return Path(paths[0]).stem if len(paths) == 1 and paths[0] else None


def _instruction_texts(sweep_dir: Path) -> dict[str, str]:
    manifest = json.loads((sweep_dir / "sweep.json").read_text())
    return {i["id"]: (i.get("text") or "") for i in manifest["instructions"]}


def paired_effects(items: pd.DataFrame, *, rng: np.random.Generator, resamples: int, texts: dict[str, str] | None = None) -> list[dict]:
    """texts: instruction id -> text from the sweep manifest. When GEPA returns the seed itself as its best
    candidate the two prompts are identical and OLMES runs them as one task, so the seed rows are missing; the
    effect is then exactly zero and is reported with same_text=True."""
    rows = []
    texts = texts or {}
    for (model, form), g in items.groupby(["model", "formulation"]):
        jobs = sorted({i[len(GEPA_PREFIX):] for i in g["instruction_id"].unique() if i.startswith(GEPA_PREFIX)})
        for job in jobs:
            opt = g[g["instruction_id"] == GEPA_PREFIX + job].set_index("native_id")
            seed = g[g["instruction_id"] == SEED_PREFIX + job].set_index("native_id")
            same_text = False
            if seed.empty and texts.get(GEPA_PREFIX + job) is not None and texts.get(GEPA_PREFIX + job) == texts.get(SEED_PREFIX + job):
                seed, same_text = opt, True
            if seed.empty:
                # Several sets can share one seed text (de-duplicated to a single seed-* id when the pairs files were
                # merged): fall back to any seed row for the same model whose id ends in the same seed instruction id.
                tail = job.rsplit("--", 1)[-1]
                alt = [i for i in g["instruction_id"].unique() if i.startswith(SEED_PREFIX) and i.rsplit("--", 1)[-1] == tail]
                if alt:
                    seed = g[g["instruction_id"] == alt[0]].set_index("native_id")
            ids = opt.index.intersection(seed.index)
            if len(ids) == 0:
                continue
            opt, seed = opt.loc[ids], seed.loc[ids]
            for metric in METRICS:
                diff = (opt[metric] - seed[metric]).to_numpy(float)
                boot = diff[rng.integers(0, len(diff), size=(resamples, len(diff)))].mean(axis=1)
                rows.append({
                    "model": model, "formulation": form, "job": job, "metric": metric, "n": int(len(diff)), "same_text": same_text,
                    "seed_mean": float(seed[metric].mean()), "optimized_mean": float(opt[metric].mean()),
                    "diff": float(diff.mean()), "se": float(diff.std(ddof=1) / np.sqrt(len(diff))),
                    "ci_lo": float(np.percentile(boot, 2.5)), "ci_hi": float(np.percentile(boot, 97.5)),
                })
    return rows


def plot_effects(df: pd.DataFrame, out: Path, *, labels: dict[str, str]) -> None:
    """Per model: optimized minus seed with 95% CI, one point per GEPA job, dev and test side by side."""
    subsets = sorted(df["subset"].unique())
    models = list(dict.fromkeys(df["model"]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    offsets = {s: (i - (len(subsets) - 1) / 2) * 0.22 for i, s in enumerate(subsets)}
    colors = {s: c for s, c in zip(subsets, ("#1f77b4", "#d62728", "#2ca02c"))}
    for ax, (metric, name) in zip(axes, METRICS.items()):
        for i, m in enumerate(models):
            for sub in subsets:
                g = df[(df.model == m) & (df.metric == metric) & (df.subset == sub)]
                if g.empty:
                    continue
                x = i + offsets[sub] + np.linspace(-0.06, 0.06, len(g))
                ax.errorbar(x, g["diff"], yerr=[g["diff"] - g["ci_lo"], g["ci_hi"] - g["diff"]], fmt="o", ms=4, capsize=2,
                            color=colors[sub], label=sub if i == 0 else None)
        ax.axhline(0, color="0.2", linewidth=0.9)
        ax.set_xticks(range(len(models)), [labels.get(m, m.split("/")[-1]) for m in models], rotation=20, ha="right")
        ax.set_title(f"{name}: optimized − seed (one point per GEPA job, 95% CI)")
        ax.set_ylabel("Δ " + name)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color="0.9")
        ax.set_axisbelow(True)
    axes[0].legend(title="subset", frameon=False, fontsize=9, title_fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


@app.command()
def main(
    sweep_dirs: Annotated[list[Path], typer.Argument(help="evaluation sweep dirs (e.g. one for dev, one for test)")],
    out: Annotated[Path, typer.Option("--out")],
    resamples: Annotated[int, typer.Option("--resamples")] = 2000,
    seed: Annotated[int, typer.Option("--seed")] = 0,
    model_labels: Annotated[str | None, typer.Option("--model-labels", help="comma-separated hf_id=label")] = None,
    subset_labels: Annotated[str | None, typer.Option("--subset-labels", help="comma-separated subset_stem=label, e.g. ...-dev-n300-seed0=dev")] = None,
) -> None:
    rng = np.random.default_rng(seed)
    rows = []
    for d in sweep_dirs:
        items = load_sweep(d)["items"]
        if items.empty:
            typer.echo(f"no items in {d}")
            continue
        texts = _instruction_texts(d)
        if "subset" in items and items["subset"].notna().any():
            for label, sub in items.groupby("subset"):
                rows += [r | {"sweep": d.name, "subset": label} for r in paired_effects(sub, rng=rng, resamples=resamples, texts=texts)]
        else:
            rows += [r | {"sweep": d.name, "subset": _subset_label(d)} for r in paired_effects(items, rng=rng, resamples=resamples, texts=texts)]
    df = pd.DataFrame(rows)
    if subset_labels:
        df["subset"] = df["subset"].replace(dict(kv.split("=") for kv in subset_labels.split(",")))
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "effects.csv", index=False)
    plot_effects(df, out / "effects.png", labels=dict(kv.split("=") for kv in model_labels.split(",")) if model_labels else {})
    md = ["# GEPA optimized vs seed (paired per-item bootstrap, 95% CI)", ""]
    for (sweep, subset), sub in df.groupby(["sweep", "subset"]):
        md += [f"## {sweep} ({subset})", "", "| model | form | job | metric | n | seed | optimized | diff | 95% CI |", "|---|---|---|---|---|---|---|---|---|"]
        for _, r in sub.sort_values(["model", "formulation", "job", "metric"]).iterrows():
            note = " (best = seed)" if r.get("same_text") else ""
            md.append(f"| {r.model} | {r.formulation} | {r.job}{note} | {r.metric} | {r.n} | {r.seed_mean:.4f} | {r.optimized_mean:.4f} | {r['diff']:+.4f} | [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] |")
        md.append("")
    (out / "effects.md").write_text("\n".join(md))
    typer.echo(f"{len(df)} rows -> {out / 'effects.csv'}")


if __name__ == "__main__":
    app()
