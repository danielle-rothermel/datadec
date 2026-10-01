"""Per-item signal and noise on a full test-set run, after Heineman et al. 2025, at item granularity.

Noise per item: standard deviation of the item's score across trailing checkpoints of one model
(--ckpt-run-dir, repeatable) and across seeds at the final checkpoint (--seed-run-dir, repeatable,
include the default seed's final run). Signal per item: dispersion (max minus min) of the item's score
across the models in --model-items (items.csv files from po_subset_bootstrap, with --model-labels).
Writes items.csv (one row per item per metric family), summary.json, table.md, plot.png under --out.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer

from datadec.po.metrics import likelihood_metrics, primary_likelihood_metric

app = typer.Typer()
FAMILIES = ("rc_primary", "rc_primary_likelihood", "mc_primary", "mc_primary_likelihood")


def load_run(run_dir: Path) -> pd.DataFrame:
    """Per-item rc/mc primary and primary_likelihood from a run dir holding both tasks."""
    frames = []
    for mp in sorted(run_dir.glob("task-*-metrics.json")):
        m = json.loads(mp.read_text())
        form = "mc" if ":mc" in m["task_config"]["task_name"] else "rc"
        primary = m["task_config"]["primary_metric"]
        lik = primary_likelihood_metric(primary)
        rows = []
        for line in mp.with_name(mp.name.replace("-metrics.json", "-predictions.jsonl")).read_text().splitlines():
            if line.strip():
                p = json.loads(line)
                rows.append({"native_id": p["native_id"], f"{form}_primary": p["metrics"][primary],
                             f"{form}_primary_likelihood": likelihood_metrics(p["model_output"], p["label"])[lik]})
        frames.append(pd.DataFrame(rows).set_index("native_id"))
    df = frames[0]
    for f in frames[1:]:
        df = df.join(f, how="inner")
    return df


def _stack(run_dirs: list[Path]) -> dict[str, pd.DataFrame]:
    """family -> items x runs matrix."""
    runs = [load_run(d) for d in run_dirs]
    ids = runs[0].index
    for r in runs[1:]:
        ids = ids.intersection(r.index)
    return {fam: pd.concat([r.loc[ids, fam].rename(str(i)) for i, r in enumerate(runs)], axis=1) for fam in FAMILIES if fam in runs[0]}


@app.command()
def main(
    out: Annotated[Path, typer.Option("--out")],
    ckpt_run_dir: Annotated[list[Path], typer.Option("--ckpt-run-dir", help="trailing checkpoints of one model")],
    seed_run_dir: Annotated[list[Path] | None, typer.Option("--seed-run-dir", help="final checkpoints of different seeds")] = None,
    model_items: Annotated[list[Path] | None, typer.Option("--model-items", help="items.csv per model for cross-model signal")] = None,
    model_labels: Annotated[str | None, typer.Option("--model-labels")] = None,
    subset: Annotated[Path | None, typer.Option("--subset", help="ItemSubset JSON marking 'our' subset")] = None,
) -> None:
    ck = _stack(ckpt_run_dir)
    sd_seed = _stack(seed_run_dir) if seed_run_dir else {}
    models = {}
    if model_items:
        names = (model_labels or ",".join(p.parent.name for p in model_items)).split(",")
        frames = [pd.read_csv(p).set_index("native_id") for p in model_items]
        ids = frames[0].index
        for f in frames[1:]:
            ids = ids.intersection(f.index)
        models = {fam: pd.concat([f.loc[ids, fam].rename(nm) for f, nm in zip(frames, names, strict=True)], axis=1) for fam in FAMILIES}
    sub = set(json.loads(subset.read_text())["ids"]) if subset else set()

    rows = []
    for fam, M in ck.items():
        t = pd.DataFrame({"native_id": M.index, "family": fam, "ckpt_mean": M.mean(axis=1).values,
                          "ckpt_sd": M.std(axis=1, ddof=1).values, "n_ckpts": M.shape[1]})
        if fam.endswith("primary"):
            flips = (M.diff(axis=1).abs().iloc[:, 1:] > 0).sum(axis=1)
            t["ckpt_flips"] = flips.values  # number of adjacent-checkpoint changes of the 0/1 outcome
        if fam in sd_seed:
            S = sd_seed[fam].reindex(M.index)
            t["seed_sd"] = S.std(axis=1, ddof=1).values
            t["n_seeds"] = S.shape[1]
        if fam in models:
            X = models[fam].reindex(M.index)
            t["model_dispersion"] = (X.max(axis=1) - X.min(axis=1)).values
            t["model_sd"] = X.std(axis=1, ddof=1).values
            t["snr_ckpt"] = t["model_dispersion"] / t["ckpt_sd"].replace(0, np.nan)
        t["in_subset"] = t["native_id"].isin(sub)
        rows.append(t)
    items = pd.concat(rows, ignore_index=True)
    out.mkdir(parents=True, exist_ok=True)
    items.to_csv(out / "items.csv", index=False)

    summary, md = {}, [f"# Per-item noise: {len(ckpt_run_dir)} trailing checkpoints" + (f", {len(seed_run_dir)} seeds" if seed_run_dir else "") + (f", models {','.join(models[FAMILIES[0]].columns)}" if models else ""), ""]
    for fam, M in ck.items():
        t = items[items.family == fam]
        agg = M.mean(axis=0)
        s = {"aggregate_ckpt_mean": float(agg.mean()), "aggregate_ckpt_sd": float(agg.std(ddof=1)),
             "item_ckpt_sd_median": float(t["ckpt_sd"].median()), "item_ckpt_sd_p90": float(t["ckpt_sd"].quantile(0.9)),
             "items_constant_across_ckpts": float((t["ckpt_sd"] == 0).mean())}
        if "ckpt_flips" in t:
            s["items_flipping_at_least_once"] = float((t["ckpt_flips"] > 0).mean())
        if "seed_sd" in t:
            S = sd_seed[fam]
            s |= {"aggregate_seed_sd": float(S.mean(axis=0).std(ddof=1)), "item_seed_sd_median": float(t["seed_sd"].median()),
                  "item_seed_sd_p90": float(t["seed_sd"].quantile(0.9))}
        if "model_dispersion" in t:
            s |= {"item_model_dispersion_median": float(t["model_dispersion"].median()),
                  "items_with_snr_below_1": float((t["snr_ckpt"] < 1).mean()), "items_with_snr_above_3": float((t["snr_ckpt"] > 3).mean())}
            if sub:
                s |= {"subset_items_with_snr_below_1": float((t[t.in_subset]["snr_ckpt"] < 1).mean())}
        summary[fam] = s
        md += [f"## {fam}", ""] + [f"- {k}: {v:.4f}" for k, v in s.items()] + [""]
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (out / "table.md").write_text("\n".join(md) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fams = [f for f in FAMILIES if f in ck]
    ncol = 2 if models else 1
    fig, axes = plt.subplots(len(fams), ncol, figsize=(6.5 * ncol, 3.6 * len(fams)), squeeze=False)
    for r, fam in enumerate(fams):
        t = items[items.family == fam]
        ax = axes[r][0]
        ax.hist(t["ckpt_sd"], bins=40, alpha=0.7, label="all items")
        if sub:
            ax.hist(t[t.in_subset]["ckpt_sd"], bins=40, alpha=0.7, label="our subset")
        ax.set_title(f"{fam}: per-item sd across {M.shape[1]} checkpoints")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        if models:
            ax = axes[r][1]
            ax.scatter(t["ckpt_sd"], t["model_dispersion"], s=6, alpha=0.4, label="all items")
            if sub:
                ax.scatter(t[t.in_subset]["ckpt_sd"], t[t.in_subset]["model_dispersion"], s=14, color="C3", label="our subset")
            lim = max(t["ckpt_sd"].max(), t["model_dispersion"].max()) or 1
            ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="signal = noise")
            ax.set_xlabel("checkpoint sd (noise)")
            ax.set_ylabel("cross-model dispersion (signal)")
            ax.set_title(f"{fam}: signal vs noise per item")
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "plot.png", dpi=130)
    typer.echo((out / "table.md").read_text())
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    app()
