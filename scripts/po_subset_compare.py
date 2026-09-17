"""Compare full-test-set runs across models: per-model subsampling bands and paired between-model differences vs n.

Inputs are the items.csv files written by po_subset_bootstrap.py (one per model; same item ids). Writes
overlay.png (each model's subsampling band per metric), differences.png (paired model-minus-reference
bands per metric), table.md, summary.json under --out.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer

app = typer.Typer()
Z80 = 2.80
SERIES = ("rc_primary", "mc_primary", "rc_primary_likelihood", "mc_primary_likelihood")
TITLES = {"rc_primary": "RC primary accuracy", "mc_primary": "MC primary accuracy",
          "rc_primary_likelihood": "RC primary likelihood", "mc_primary_likelihood": "MC primary likelihood"}


def subsample(values: np.ndarray, n: int, draws: int, rng: np.random.Generator) -> np.ndarray:
    if n >= len(values):
        return np.full(draws, values.mean())
    idx = np.stack([rng.choice(len(values), n, replace=False) for _ in range(draws)])
    return values[idx].mean(axis=1)


@app.command()
def main(
    items: Annotated[list[Path], typer.Option("--items", help="repeatable: items.csv from po_subset_bootstrap")],
    labels: Annotated[str, typer.Option("--labels", help="comma-separated names, same order as --items")],
    out: Annotated[Path, typer.Option("--out")],
    subset: Annotated[Path | None, typer.Option("--subset", help="ItemSubset JSON marking 'our' subset")] = None,
    reference: Annotated[int, typer.Option("--reference", help="index into --items used as the difference baseline")] = 0,
    sizes: Annotated[str, typer.Option("--sizes")] = "25,50,100,200,400,800,1600,full",
    draws: Annotated[int, typer.Option("--draws")] = 2000,
    seed: Annotated[int, typer.Option("--seed")] = 0,
) -> None:
    names = labels.split(",")
    if len(names) != len(items):
        raise typer.BadParameter("--labels must match --items")
    frames = [pd.read_csv(p).set_index("native_id") for p in items]
    ids = frames[0].index
    for f in frames[1:]:
        ids = ids.intersection(f.index)
    frames = [f.loc[ids] for f in frames]
    n_full = len(ids)
    size_list = [n_full if s == "full" else int(s) for s in sizes.split(",")]
    rng = np.random.default_rng(seed)
    sub_pos = None
    if subset:
        sub_ids = [i for i in json.loads(subset.read_text())["ids"] if i in set(ids)]
        sub_pos = np.array([ids.get_loc(i) for i in sub_ids])

    rows, drows = [], []
    for key in SERIES:
        vals = {nm: f[key].to_numpy(float) for nm, f in zip(names, frames, strict=True)}
        for nm, v in vals.items():
            for n in size_list:
                lo, hi = np.percentile(subsample(v, n, draws, rng), [2.5, 97.5])
                rows.append({"series": key, "model": nm, "n": n, "full_mean": float(v.mean()), "lo": float(lo), "hi": float(hi),
                             "ours": float(v[sub_pos].mean()) if sub_pos is not None else None})
        pairs = [(names[reference], nm) for nm in names if nm != names[reference]] + [
            (a, b) for a, b in combinations(names, 2) if names[reference] not in (a, b)]
        for a, b in pairs:
            d = vals[b] - vals[a]
            sd = float(d.std(ddof=1))
            for n in size_list:
                lo, hi = np.percentile(subsample(d, n, draws, rng), [2.5, 97.5])
                drows.append({"series": key, "pair": f"{b} − {a}", "n": n, "full_diff": float(d.mean()), "lo": float(lo), "hi": float(hi),
                              "item_sd": sd, "mdd_80": float(Z80 * sd / np.sqrt(n)),
                              "ours": float(d[sub_pos].mean()) if sub_pos is not None else None,
                              "n_for_80pct_power": float((Z80 * sd / d.mean()) ** 2) if d.mean() else None})
    bands, diffs = pd.DataFrame(rows), pd.DataFrame(drows)
    out.mkdir(parents=True, exist_ok=True)
    bands.to_csv(out / "bands.csv", index=False)
    diffs.to_csv(out / "differences.csv", index=False)

    md = [f"# Model comparison on the full ARC-Easy test set (n_full={n_full}): {', '.join(names)}", "",
          "Bands are 2.5–97.5 percentiles of the mean over random n-item subsets. Differences are paired per item.", ""]
    for key in SERIES:
        md += [f"## {TITLES[key]}", "", "| model | full-set value | n=100 band | n=100 half-width | our subset |", "|---|---|---|---|---|"]
        for nm in names:
            r = bands[(bands.series == key) & (bands.model == nm) & (bands.n == 100)].iloc[0]
            ours = "–" if r["ours"] is None or pd.isna(r["ours"]) else f"{r['ours']:.3f}"
            md.append(f"| {nm} | {r['full_mean']:.3f} | [{r['lo']:.3f}, {r['hi']:.3f}] | {(r['hi'] - r['lo']) / 2:.3f} | {ours} |")
        md += ["", "| pair | full-set diff | item sd | n=100 band | MDD n=100 | n for 80% power | our subset diff |", "|---|---|---|---|---|---|---|"]
        for pair in diffs[diffs.series == key]["pair"].unique():
            r = diffs[(diffs.series == key) & (diffs.pair == pair) & (diffs.n == 100)].iloc[0]
            nreq = "n/a" if r["n_for_80pct_power"] is None or pd.isna(r["n_for_80pct_power"]) else f"{r['n_for_80pct_power']:.0f}"
            ours = "–" if r["ours"] is None or pd.isna(r["ours"]) else f"{r['ours']:+.3f}"
            md.append(f"| {pair} | {r['full_diff']:+.3f} | {r['item_sd']:.3f} | [{r['lo']:+.3f}, {r['hi']:+.3f}] | {r['mdd_80']:.3f} | {nreq} | {ours} |")
        md.append("")
    (out / "table.md").write_text("\n".join(md) + "\n")
    (out / "summary.json").write_text(json.dumps({"items": [str(p) for p in items], "labels": names, "reference": names[reference],
                                                  "n_full": n_full, "sizes": size_list, "draws": draws}, indent=1) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for ax, key in zip(axes.flat, SERIES, strict=True):
        for i, nm in enumerate(names):
            t = bands[(bands.series == key) & (bands.model == nm)]
            ax.fill_between(t["n"], t["lo"], t["hi"], alpha=0.25, color=f"C{i}")
            ax.plot(t["n"], t["full_mean"], color=f"C{i}", lw=1.2, label=f"{nm} ({t['full_mean'].iloc[0]:.3f})")
        if key == "mc_primary":
            ax.axhline(0.25, color="grey", lw=0.8, ls="-.")
        ax.set_xscale("log")
        ax.set_title(TITLES[key])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("subset size n (log)")
    fig.suptitle(f"Per-model subsampling bands on ARC-Easy test (n_full={n_full})")
    fig.tight_layout()
    fig.savefig(out / "overlay.png", dpi=130)

    pairs = list(diffs["pair"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for ax, key in zip(axes.flat, SERIES, strict=True):
        for i, pair in enumerate(pairs):
            t = diffs[(diffs.series == key) & (diffs.pair == pair)]
            ax.fill_between(t["n"], t["lo"], t["hi"], alpha=0.2, color=f"C{i}")
            ax.plot(t["n"], t["full_diff"], color=f"C{i}", lw=1.2, label=f"{pair} ({t['full_diff'].iloc[0]:+.3f})")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xscale("log")
        ax.set_title(f"{TITLES[key]}: paired difference")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("subset size n (log)")
    fig.suptitle(f"Paired between-model differences vs subset size (n_full={n_full})")
    fig.tight_layout()
    fig.savefig(out / "differences.png", dpi=130)
    typer.echo((out / "table.md").read_text())
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    app()
