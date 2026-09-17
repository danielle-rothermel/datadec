"""Per-format paired effects against the canonical pair for one-factor format sweeps.

Reads sweeps named <prefix>-<model>-<subset>, pairs items by native_id against the canonical
5-shot no-instruction pair, and writes effects.csv plus a markdown table per model.
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


@app.command()
def main(
    root: Annotated[Path, typer.Option("--root")],
    prefix: Annotated[str, typer.Option("--prefix")],
    models: Annotated[str, typer.Option("--models", help="comma-separated model keys used in sweep names")],
    pairs: Annotated[Path, typer.Option("--pairs")],
    out: Annotated[Path, typer.Option("--out")],
    subsets: Annotated[str, typer.Option("--subsets")] = "train,dev",
    plot_subset: Annotated[str, typer.Option("--plot-subset", help="subset drawn in bars.png")] = "train",
    resamples: Annotated[int, typer.Option("--resamples")] = 2000,
    seed: Annotated[int, typer.Option("--seed")] = 0,
    model_labels: Annotated[str | None, typer.Option("--model-labels", help="comma-separated display names, same order as --models")] = None,
) -> None:
    rng = np.random.default_rng(seed)
    labels = {p["format_id"] + f"|k{p['num_shots']}": p["label"] for p in json.loads(pairs.read_text())["pairs"]}
    order = [p["label"] for p in json.loads(pairs.read_text())["pairs"]]
    rows = []
    for model in models.split(","):
        for sub in subsets.split(","):
            d = root / f"{prefix}-{model}-{sub}"
            if not (d / "tasks.jsonl").exists():
                continue
            it = load_sweep(d)["items"]
            if it.empty:
                continue
            it["label"] = (it.format_id + "|k" + it.num_shots.astype(str)).map(labels)
            for form, g in it.groupby("formulation"):
                base = g[g.label == "canonical"].set_index("native_id")
                for label, h in g.groupby("label"):
                    h = h.set_index("native_id").reindex(base.index)
                    for metric in ("primary", "primary_likelihood"):
                        diff = (h[metric] - base[metric]).to_numpy(float)
                        boot = diff[rng.integers(0, len(diff), size=(resamples, len(diff)))].mean(axis=1)
                        rows.append({"model": model, "subset": sub, "formulation": form, "format": label, "metric": metric,
                                     "mean": h[metric].mean(), "diff": diff.mean(), "se": diff.std(ddof=1) / np.sqrt(len(diff)),
                                     "ci_lo": float(np.percentile(boot, 2.5)), "ci_hi": float(np.percentile(boot, 97.5)), "n": int(len(diff))})
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "effects.csv", index=False)
    md = [f"# Format effects vs canonical ({prefix})", ""]
    for model in models.split(","):
        for form in ("rc", "mc"):
            sub_df = df[(df.model == model) & (df.formulation == form)]
            if sub_df.empty:
                continue
            subs = [s for s in subsets.split(",") if s in set(sub_df.subset)]
            md += [f"## {model} {form.upper()}", "", "| format | " + " | ".join(f"{s} acc Δ | {s} share Δ ± se" for s in subs) + " |", "|---|" + "---|---|" * len(subs)]
            for label in order:
                cells = []
                for s in subs:
                    a = sub_df[(sub_df.subset == s) & (sub_df.format == label) & (sub_df.metric == "primary")]
                    lk = sub_df[(sub_df.subset == s) & (sub_df.format == label) & (sub_df.metric == "primary_likelihood")]
                    if a.empty:
                        cells += ["", ""]
                        continue
                    cells += [f"{a['diff'].iloc[0]:+.3f} ({a['mean'].iloc[0]:.3f})", f"{lk['diff'].iloc[0]:+.4f} ± {lk['se'].iloc[0]:.4f} ({lk['mean'].iloc[0]:.4f})"]
                if any(cells):
                    md.append(f"| {label} | " + " | ".join(cells) + " |")
            md.append("")
    (out / "effects.md").write_text("\n".join(md) + "\n")
    typer.echo("\n".join(md))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    model_keys = models.split(",")
    names = model_labels.split(",") if model_labels else model_keys
    panels = [("rc", "primary", "RC accuracy Δ vs canonical"), ("rc", "primary_likelihood", "RC likelihood Δ vs canonical"),
              ("mc", "primary", "MC accuracy Δ vs canonical"), ("mc", "primary_likelihood", "MC likelihood Δ vs canonical")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), constrained_layout=True)
    width = 0.8 / len(model_keys)
    for ax, (form, metric, title) in zip(axes.flat, panels, strict=True):
        fmts = [f for f in order if f != "canonical" and not df[(df.formulation == form) & (df.format == f) & (df.subset == plot_subset)].empty]
        x = np.arange(len(fmts))
        for mi, (mk, nm) in enumerate(zip(model_keys, names, strict=True)):
            t = df[(df.model == mk) & (df.subset == plot_subset) & (df.formulation == form) & (df.metric == metric)].set_index("format").reindex(fmts)
            if t["diff"].isna().all():
                continue
            ax.bar(x + (mi - (len(model_keys) - 1) / 2) * width, t["diff"], width, color=f"C{mi}", label=nm,
                   yerr=[t["diff"] - t["ci_lo"], t["ci_hi"] - t["diff"]], capsize=2, error_kw={"lw": 0.9})
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(fmts, rotation=35, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.savefig(out / "bars.png", dpi=150)
    typer.echo(f"wrote {out / 'bars.png'}")


if __name__ == "__main__":
    app()
