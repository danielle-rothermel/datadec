"""Presentation plots from full-test-set runs: per-model bars and paired-difference heatmaps.

Inputs are items.csv files from po_subset_bootstrap.py (one per model, same items). Writes
bars.png (full-set value per model per metric with a bootstrap 95% interval), diff_heatmap.png
(paired column-minus-row differences), power_heatmap.png (items needed for 80% power on each
paired difference), and summary.csv under --out.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer

app = typer.Typer()
Z80 = 2.80
SERIES = ("rc_primary", "mc_primary", "rc_primary_likelihood", "mc_primary_likelihood")
TITLES = {"rc_primary": "RC accuracy", "mc_primary": "MC accuracy",
          "rc_primary_likelihood": "RC likelihood", "mc_primary_likelihood": "MC likelihood"}


@app.command()
def main(
    items: Annotated[list[Path], typer.Option("--items")],
    labels: Annotated[str, typer.Option("--labels")],
    out: Annotated[Path, typer.Option("--out")],
    resamples: Annotated[int, typer.Option("--resamples")] = 2000,
    seed: Annotated[int, typer.Option("--seed")] = 0,
    chance: Annotated[float, typer.Option("--chance", help="reference line on MC panels")] = 0.25,
) -> None:
    names = labels.split(",")
    frames = [pd.read_csv(p).set_index("native_id") for p in items]
    ids = frames[0].index
    for f in frames[1:]:
        ids = ids.intersection(f.index)
    frames = [f.loc[ids] for f in frames]
    rng = np.random.default_rng(seed)
    n = len(ids)
    rows = []
    for key in SERIES:
        for nm, f in zip(names, frames, strict=True):
            v = f[key].to_numpy(float)
            boot = v[rng.integers(0, n, size=(resamples, n))].mean(axis=1)
            rows.append({"series": key, "model": nm, "mean": v.mean(), "lo": np.percentile(boot, 2.5), "hi": np.percentile(boot, 97.5)})
    summary = pd.DataFrame(rows)
    diff = np.full((len(SERIES), len(names), len(names)), np.nan)
    power = np.full_like(diff, np.nan)
    for si, key in enumerate(SERIES):
        for i, fi in enumerate(frames):
            for j, fj in enumerate(frames):
                if j >= i:
                    continue
                d = fi[key].to_numpy(float) - fj[key].to_numpy(float)
                diff[si, i, j] = d.mean()
                sd = d.std(ddof=1)
                power[si, i, j] = (Z80 * sd / abs(d.mean())) ** 2 if d.mean() != 0 else np.inf
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "summary.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = [f"C{i}" for i in range(len(names))]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, key in zip(axes, SERIES, strict=True):
        t = summary[summary.series == key].set_index("model").loc[names]
        ax.bar(range(len(names)), t["mean"], color=colors, yerr=[t["mean"] - t["lo"], t["hi"] - t["mean"]], capsize=3)
        if key.startswith("mc_"):
            ax.axhline(chance, color="grey", lw=0.8, ls="--")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_title(TITLES[key])
        ax.grid(axis="y", alpha=0.3)
        lo = min(t["lo"].min(), chance if key.startswith("mc_") else t["lo"].min())
        ax.set_ylim(max(0, lo - 0.05), min(1, t["hi"].max() + 0.05))
    fig.tight_layout()
    fig.savefig(out / "bars.png", dpi=150)

    def heat(mat: np.ndarray, fname: str, fmt, cmap: str, center: bool, log: bool) -> None:
        fig, axes = plt.subplots(1, 4, figsize=(17, 4.0), constrained_layout=True)
        for si, (ax, key) in enumerate(zip(axes, SERIES, strict=True)):
            m = mat[si]
            plot = np.log10(m) if log else m
            vmax = np.nanmax(np.abs(plot)) if center else np.nanmax(plot)
            vmin = -vmax if center else np.nanmin(plot)
            im = ax.imshow(np.where(np.isnan(plot), np.nan, plot), cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xlim(-0.5, len(names) - 1.5)
            ax.set_ylim(len(names) - 0.5, 0.5)
            for i in range(len(names)):
                for j in range(len(names)):
                    if not np.isnan(m[i, j]):
                        ax.text(j, i, fmt(m[i, j]), ha="center", va="center", fontsize=8)
            ax.set_xticks(range(len(names) - 1))
            ax.set_xticklabels(names[:-1], rotation=30, ha="right")
            ax.set_yticks(range(1, len(names)))
            ax.set_yticklabels(names[1:])
            ax.set_title(TITLES[key])
            ax.set_xticks(np.arange(-0.5, len(names), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(names), 1), minor=True)
            ax.grid(which="minor", color="white", lw=1)
            ax.tick_params(which="minor", length=0)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            if log:
                ticks = [t for t in range(int(np.floor(vmin)), int(np.ceil(vmax)) + 1)]
                cb.set_ticks(ticks)
                cb.set_ticklabels([f"{10 ** t:,.0f}" for t in ticks])
        fig.savefig(out / fname, dpi=150)

    heat(diff, "diff_heatmap.png", lambda x: f"{x:+.3f}", "RdBu_r", center=True, log=False)
    heat(power, "power_heatmap.png", lambda x: "∞" if not np.isfinite(x) else (f"{x:.0f}" if x < 1e5 else f"{x:.0e}"), "viridis_r", center=False, log=True)
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    app()
