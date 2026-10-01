"""Seed-prompt effects: every seed instruction vs the canonical no-instruction prompt, per model.

Inputs are (seeds sweep, canonical sweep) pairs on the same item subset and prompt format. For each
(model, formulation, seed) the per-item difference against the canonical pair is bootstrapped for the
primary metric (accuracy) and primary_likelihood (share). Outputs: per_seed.csv, summary.md, a violin
figure of per-seed mean effects, and a per-item violin figure for the worst / median / best seed.
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

from datadec.po.results import load_sweep

matplotlib.use("Agg")
app = typer.Typer()
METRICS = {"primary": "accuracy", "primary_likelihood": "share"}
CANONICAL_ID = "none"


def seed_effects(seeds: pd.DataFrame, canon: pd.DataFrame, *, rng: np.random.Generator, resamples: int) -> tuple[list[dict], dict[tuple, np.ndarray]]:
    rows, per_item = [], {}
    for form, g in seeds.groupby("formulation"):
        base = canon[(canon["formulation"] == form) & (canon["instruction_id"] == CANONICAL_ID)].set_index("native_id")
        for seed_id, h in g.groupby("instruction_id"):
            h = h.set_index("native_id").reindex(base.index)
            for metric in METRICS:
                diff = (h[metric] - base[metric]).to_numpy(float)
                boot = diff[rng.integers(0, len(diff), size=(resamples, len(diff)))].mean(axis=1)
                per_item[(form, seed_id, metric)] = diff
                rows.append({"formulation": form, "instruction_id": seed_id, "metric": metric, "n": int(len(diff)),
                             "canon_mean": float(base[metric].mean()), "seed_mean": float(h[metric].mean()), "diff": float(diff.mean()),
                             "se": float(diff.std(ddof=1) / np.sqrt(len(diff))),
                             "ci_lo": float(np.percentile(boot, 2.5)), "ci_hi": float(np.percentile(boot, 97.5))})
    return rows, per_item


def _proposer_label(name: str) -> str:
    """Short legend label for a proposer model id (e.g. openai/gpt-5.6-terra -> terra)."""
    if "terra" in name:
        return "terra"
    if "gpt-5.1" in name:
        return "gpt-5.1"
    return name


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_mean_effects(df: pd.DataFrame, models: list[str], labels: dict[str, str], proposer: dict[str, str], out: Path, *, task_label: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {"gpt-5.1": "#1f77b4", "terra": "#d62728"}
    for ax, (metric, name) in zip(axes, METRICS.items()):
        data = [df[(df.model == m) & (df.metric == metric)]["diff"].to_numpy() for m in models]
        parts = ax.violinplot(data, positions=range(len(models)), showmeans=False, showextrema=False, widths=0.75)
        for body in parts["bodies"]:
            body.set_facecolor("0.85")
            body.set_edgecolor("0.6")
            body.set_alpha(1.0)
        for i, m in enumerate(models):
            sub = df[(df.model == m) & (df.metric == metric)]
            jitter = rng_jitter(len(sub), i)
            cols = [colors.get(proposer.get(s, ""), "0.3") for s in sub["instruction_id"]]
            ax.scatter(jitter, sub["diff"], s=16, c=cols, zorder=3, linewidths=0)
        ax.axhline(0, color="0.2", linewidth=0.9)
        ax.set_xticks(range(len(models)), [labels.get(m, m) for m in models], rotation=20, ha="right")
        ax.set_title(f"{name}: seed − canonical (mean over items)")
        ax.set_ylabel("Δ " + name)
        _style(ax)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, label=p) for p, c in colors.items()]
    axes[0].legend(handles=handles, title="proposer", frameon=False, fontsize=9, title_fontsize=9)
    n_seeds = df.groupby("model")["instruction_id"].nunique().max()
    fig.suptitle(f"{n_seeds} seed prompts, {task_label}, RC, 300 train items: effect vs no-instruction canonical prompt", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def rng_jitter(n: int, center: int, width: float = 0.18) -> np.ndarray:
    return center + np.random.default_rng(center).uniform(-width, width, size=n)


def plot_per_item(per_item: dict, df: pd.DataFrame, models: list[str], labels: dict[str, str], out: Path, *, task_label: str) -> None:
    metric = "primary_likelihood"
    fig, axes = plt.subplots(1, len(models), figsize=(3.0 * len(models), 4.0), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), models):
        sub = df[(df.model == m) & (df.metric == metric)].sort_values("diff").reset_index(drop=True)
        picks = [("worst", sub.iloc[0]), ("median", sub.iloc[len(sub) // 2]), ("best", sub.iloc[-1])]
        data = [per_item[m][("rc", r["instruction_id"], metric)] for _, r in picks]
        parts = ax.violinplot(data, positions=range(3), showmeans=True, showextrema=False, widths=0.8)
        for body in parts["bodies"]:
            body.set_facecolor("#9ecae1")
            body.set_edgecolor("#3182bd")
            body.set_alpha(1.0)
        parts["cmeans"].set_color("0.1")
        ax.axhline(0, color="0.2", linewidth=0.9)
        ax.set_xticks(range(3), [f"{k}\n{r['instruction_id']}" for k, r in picks], fontsize=8)
        ax.set_title(labels.get(m, m))
        _style(ax)
    np.atleast_1d(axes)[0].set_ylabel("Δ share per item (seed − canonical)")
    fig.suptitle(f"{task_label}: per-item share effects for the worst, median and best seed of each model", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


@app.command()
def main(
    pair: Annotated[list[str], typer.Option("--pair", help="model_key=seeds_sweep_dir,canonical_sweep_dir (repeatable)")],
    out: Annotated[Path, typer.Option("--out")],
    instructions: Annotated[Path | None, typer.Option("--instructions", help="instruction set JSON with proposer per id")] = None,
    model_labels: Annotated[str | None, typer.Option("--model-labels", help="comma-separated key=label")] = None,
    task_label: Annotated[str, typer.Option("--task-label", help="task name shown in figure titles")] = "ARC-Easy",
    resamples: Annotated[int, typer.Option("--resamples")] = 2000,
    seed: Annotated[int, typer.Option("--seed")] = 0,
) -> None:
    rng = np.random.default_rng(seed)
    proposer = {}
    if instructions:
        raw = json.loads(instructions.read_text())
        proposer = {i["id"]: _proposer_label(i.get("proposer", "")) for i in (raw["instructions"] if isinstance(raw, dict) else raw)}
    labels = dict(kv.split("=") for kv in model_labels.split(",")) if model_labels else {}
    models, rows, per_item = [], [], {}
    for spec in pair:
        key, dirs = spec.split("=")
        seeds_dir, canon_dir = (Path(d) for d in dirs.split(","))
        seeds, canon = load_sweep(seeds_dir)["items"], load_sweep(canon_dir)["items"]
        r, p = seed_effects(seeds, canon, rng=rng, resamples=resamples)
        rows += [x | {"model": key, "proposer": proposer.get(x["instruction_id"], "")} for x in r]
        per_item[key] = p
        models.append(key)
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_seed.csv", index=False)
    plot_mean_effects(df, models, labels, proposer, out / "violin_mean_effects.png", task_label=task_label)
    plot_per_item(per_item, df, models, labels, out / "violin_per_item.png", task_label=task_label)

    md = ["# Seed-prompt effects vs canonical (paired per-item bootstrap, 95% CI)", ""]
    md += ["| model | metric | canonical | seed mean (min / median / max) | spread | seeds above 0 (CI excludes 0) | seeds below 0 (CI excludes 0) |", "|---|---|---|---|---|---|---|"]
    for m in models:
        for metric, name in METRICS.items():
            s = df[(df.model == m) & (df.metric == metric)]
            up, down = int((s.ci_lo > 0).sum()), int((s.ci_hi < 0).sum())
            md.append(f"| {labels.get(m, m)} | {name} | {s.canon_mean.iloc[0]:.4f} | {s.seed_mean.min():.4f} / {s.seed_mean.median():.4f} / {s.seed_mean.max():.4f} | {s.seed_mean.max() - s.seed_mean.min():.4f} | {up} / {len(s)} | {down} / {len(s)} |")
    md.append("")
    for metric, name in METRICS.items():
        wide = df[df.metric == metric].pivot(index="instruction_id", columns="model", values="diff")[models]
        corr = wide.corr(method="spearman")
        md += [f"## Rank correlation of seed effects across models ({name}, Spearman over 27 seeds)", "", "| | " + " | ".join(labels.get(m, m) for m in models) + " |", "|---|" + "---|" * len(models)]
        for m in models:
            md.append(f"| {labels.get(m, m)} | " + " | ".join(f"{corr.loc[m, n]:+.2f}" for n in models) + " |")
        md.append("")
    for m in models:
        md += [f"## {labels.get(m, m)}: per-seed effects", "", "| seed | proposer | Δ accuracy [95% CI] | Δ share [95% CI] |", "|---|---|---|---|"]
        acc = df[(df.model == m) & (df.metric == "primary")].set_index("instruction_id")
        sh = df[(df.model == m) & (df.metric == "primary_likelihood")].set_index("instruction_id")
        for sid in sh.sort_values("diff", ascending=False).index:
            a, b = acc.loc[sid], sh.loc[sid]
            md.append(f"| {sid} | {a.proposer} | {a['diff']:+.4f} [{a.ci_lo:+.4f}, {a.ci_hi:+.4f}] | {b['diff']:+.4f} [{b.ci_lo:+.4f}, {b.ci_hi:+.4f}] |")
        md.append("")
    (out / "summary.md").write_text("\n".join(md))
    typer.echo(f"{len(df)} rows -> {out}")


if __name__ == "__main__":
    app()
