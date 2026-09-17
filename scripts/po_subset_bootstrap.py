"""How much does the item subset size matter? Subsample a full-test-set OLMES run at several sizes.

Given OLMES run dirs holding an RC task and an MC task over the full test split (one dir with both, or
one dir each; tasks are found by name), draw random subsets of each size many times and record the spread of RC primary, MC primary, their paired
difference, and the same for primary_likelihood. Also computes the percentile-bootstrap CI a single subset
would have reported (averaged over subsets, and exactly for the given subset ids), and paired power figures.
Writes plot.png, table.md, table.csv, summary.json, and items.csv (per-item values, consumed by
po_subset_compare.py) under --out.
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
Z80 = 2.80  # (z_{0.975} + z_{0.80}) for a two-sided 5% test at 80% power


def find_task(run_dirs: list[Path], formulation: str) -> Path:
    """Metrics file of the RC or MC task across the given run dirs (MC task names contain ':mc')."""
    hits = [p for d in run_dirs for p in sorted(d.glob("task-*-metrics.json"))
            if (":mc" in json.loads(p.read_text())["task_config"]["task_name"]) == (formulation == "mc")]
    if len(hits) != 1:
        raise typer.BadParameter(f"expected exactly one {formulation} task across {run_dirs}, found {len(hits)}")
    return hits[0]


def load_task(metrics_path: Path) -> pd.DataFrame:
    m = json.loads(metrics_path.read_text())
    primary = m["task_config"]["primary_metric"]
    lik = primary_likelihood_metric(primary)
    rows = []
    pred_path = metrics_path.with_name(metrics_path.name.replace("-metrics.json", "-predictions.jsonl"))
    for line in pred_path.read_text().splitlines():
        if not line.strip():
            continue
        p = json.loads(line)
        rows.append({"native_id": p["native_id"], "primary": p["metrics"][primary],
                     "primary_likelihood": likelihood_metrics(p["model_output"], p["label"])[lik]})
    df = pd.DataFrame(rows).set_index("native_id")
    df.attrs.update({"task": m["task_config"]["task_name"], "primary_metric": primary, "primary_likelihood_metric": lik,
                     "model": m["model_config"]["model"], "revision": m["model_config"].get("revision")})
    return df


def subsample(values: np.ndarray, n: int, draws: int, rng: np.random.Generator) -> np.ndarray:
    """Means of `draws` random n-subsets drawn without replacement (the sampling distribution of a subset mean)."""
    if n >= len(values):
        return np.full(draws, values.mean())
    idx = np.stack([rng.choice(len(values), n, replace=False) for _ in range(draws)])
    return values[idx].mean(axis=1)


def bootstrap_ci(values: np.ndarray, resamples: int, rng: np.random.Generator) -> tuple[float, float]:
    idx = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


@app.command()
def main(
    run_dir: Annotated[list[Path], typer.Option("--run-dir", help="repeatable; dirs holding the RC and MC tasks")],
    subset: Annotated[Path, typer.Option("--subset", help="ItemSubset JSON whose ids mark 'our' subset")],
    out: Annotated[Path, typer.Option("--out")],
    sizes: Annotated[str, typer.Option("--sizes")] = "25,50,100,200,400,800,1600,full",
    draws: Annotated[int, typer.Option("--draws")] = 2000,
    inner_subsets: Annotated[int, typer.Option("--inner-subsets", help="subsets per size for the average within-subset bootstrap CI")] = 100,
    resamples: Annotated[int, typer.Option("--resamples")] = 1000,
    seed: Annotated[int, typer.Option("--seed")] = 0,
) -> None:
    rng = np.random.default_rng(seed)
    rc, mc = load_task(find_task(run_dir, "rc")), load_task(find_task(run_dir, "mc"))
    ids = rc.index.intersection(mc.index)
    rc, mc = rc.loc[ids], mc.loc[ids]
    n_full = len(ids)
    sub_ids = [i for i in json.loads(subset.read_text())["ids"] if i in set(ids)]
    size_list = [n_full if s == "full" else int(s) for s in sizes.split(",")]
    series = {}
    for kind in ("primary", "primary_likelihood"):
        r, m = rc[kind].to_numpy(float), mc[kind].to_numpy(float)
        series[f"rc_{kind}"], series[f"mc_{kind}"], series[f"diff_{kind}"] = r, m, r - m
    labels = {"rc_primary": f"RC {rc.attrs['primary_metric']}", "mc_primary": f"MC {mc.attrs['primary_metric']}",
              "diff_primary": "RC − MC (primary)", "rc_primary_likelihood": f"RC {rc.attrs['primary_likelihood_metric']}",
              "mc_primary_likelihood": f"MC {mc.attrs['primary_likelihood_metric']}", "diff_primary_likelihood": "RC − MC (primary_likelihood)"}
    sub_pos = np.array([ids.get_loc(i) for i in sub_ids])
    rows = []
    for n in size_list:
        for key, v in series.items():
            dist = subsample(v, n, draws, rng)
            lo, hi = np.percentile(dist, [2.5, 97.5])
            widths = []
            for _ in range(inner_subsets if n < n_full else 1):
                pick = v if n >= n_full else v[rng.choice(n_full, n, replace=False)]
                a, b = bootstrap_ci(pick, resamples, rng)
                widths.append((b - a) / 2)
            rows.append({"n": n, "series": key, "full_mean": float(v.mean()), "sub_lo": float(lo), "sub_hi": float(hi),
                         "sub_halfwidth": float((hi - lo) / 2), "boot_halfwidth_avg": float(np.mean(widths)),
                         "item_sd": float(v.std(ddof=1)), "mdd_80": float(Z80 * v.std(ddof=1) / np.sqrt(n))})
    table = pd.DataFrame(rows)
    ours = {}
    for key, v in series.items():
        pick = v[sub_pos]
        a, b = bootstrap_ci(pick, resamples, rng)
        ours[key] = {"n": len(sub_pos), "mean": float(pick.mean()), "boot_lo": a, "boot_hi": b}
    power = {}
    for key in ("diff_primary", "diff_primary_likelihood", "mc_primary", "mc_primary_likelihood"):
        v = series[key]
        effect = v.mean() - (0.25 if key.startswith("mc_") else 0.0)  # MC: distance from 4-way chance
        sd = v.std(ddof=1)
        power[key] = {"observed_effect": float(effect), "item_sd": float(sd),
                      "n_for_80pct_power": float((Z80 * sd / effect) ** 2) if effect else None,
                      "mdd_at_n100": float(Z80 * sd / 10), "mdd_at_full": float(Z80 * sd / np.sqrt(n_full))}

    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "table.csv", index=False)
    pd.DataFrame({"native_id": ids, **{k: v for k, v in series.items() if not k.startswith("diff_")}}).to_csv(out / "items.csv", index=False)
    (out / "summary.json").write_text(json.dumps({
        "run_dirs": [str(d) for d in run_dir], "model": rc.attrs["model"], "revision": rc.attrs["revision"], "n_full": n_full,
        "subset": str(subset), "sizes": size_list, "draws": draws, "inner_subsets": inner_subsets, "resamples": resamples,
        "labels": labels, "our_subset": ours, "power": power}, indent=1) + "\n")
    md = [f"# Subset size vs findings: {rc.attrs['model']} ({rc.attrs['revision']}), ARC-Easy test, n_full={n_full}", "",
          "Subsampling interval = 2.5–97.5 percentile of the mean over random n-subsets (the true spread of what a subset shows).",
          "Bootstrap half-width = what a single n-subset's own percentile bootstrap CI would report (averaged over subsets).",
          "MDD = minimum detectable paired difference at 80% power, two-sided 5%, for that n.", ""]
    for key in labels:
        t = table[table["series"] == key]
        md += [f"## {labels[key]} (full-set value {t['full_mean'].iloc[0]:.3f})", "",
               "| n | subsampling 95% interval | half-width | bootstrap half-width | item sd | MDD |", "|---|---|---|---|---|---|"]
        for _, r in t.iterrows():
            md.append(f"| {r['n']} | [{r['sub_lo']:.3f}, {r['sub_hi']:.3f}] | {r['sub_halfwidth']:.3f} | {r['boot_halfwidth_avg']:.3f} | {r['item_sd']:.3f} | {r['mdd_80']:.3f} |")
        o = ours[key]
        md += ["", f"Our subset (n={o['n']}): {o['mean']:.3f}, its own bootstrap 95% CI [{o['boot_lo']:.3f}, {o['boot_hi']:.3f}].", ""]
    md += ["## Power", "", "| comparison | observed effect | item sd | n for 80% power | MDD at n=100 | MDD at full n |", "|---|---|---|---|---|---|"]
    for key, p in power.items():
        nreq = "n/a" if p["n_for_80pct_power"] is None else f"{p['n_for_80pct_power']:.0f}"
        md.append(f"| {labels[key]} vs {'chance' if key.startswith('mc_') else '0'} | {p['observed_effect']:+.3f} | {p['item_sd']:.3f} | {nreq} | {p['mdd_at_n100']:.3f} | {p['mdd_at_full']:.3f} |")
    (out / "table.md").write_text("\n".join(md) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    order = [["rc_primary", "mc_primary", "diff_primary"], ["rc_primary_likelihood", "mc_primary_likelihood", "diff_primary_likelihood"]]
    for row, keys in zip(axes, order, strict=True):
        for ax, key in zip(row, keys, strict=True):
            t = table[table["series"] == key]
            ax.fill_between(t["n"], t["sub_lo"], t["sub_hi"], alpha=0.25, label="95% subsampling interval")
            ax.plot(t["n"], t["full_mean"], "k--", lw=1, label=f"full set ({t['full_mean'].iloc[0]:.3f})")
            ax.plot(t["n"], t["full_mean"] - t["boot_halfwidth_avg"], "C1:", lw=1, label="avg bootstrap CI of one subset")
            ax.plot(t["n"], t["full_mean"] + t["boot_halfwidth_avg"], "C1:", lw=1)
            if key.startswith("mc_") and key.endswith("primary"):
                ax.axhline(0.25, color="grey", lw=0.8, ls="-.", label="chance")
            ax.set_xscale("log")
            ax.set_title(labels[key])
            ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("subset size n (log)")
    axes[0][0].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"{rc.attrs['model']} ({rc.attrs['revision']}) — ARC-Easy test (n_full={n_full}): what a random n-item subset shows")
    # the given subset's own values are reported in table.md / summary.json, not drawn
    fig.tight_layout()
    fig.savefig(out / "plot.png", dpi=130)
    typer.echo((out / "table.md").read_text())
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    app()
