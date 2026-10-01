"""Select SNR-filtered item pools and disjoint train/dev/test subsets from full-test-set runs.

Signal per item: dispersion of the RC primary likelihood across the models in --model-items (items.csv
from po_subset_bootstrap). Noise per item: checkpoint sd of the RC primary likelihood from one or more
po_item_noise items.csv files (--noise-items; averaged when several). Items with signal/noise below
--min-snr, and the OLMES few-shot ids, are excluded from the pool. The remaining pool is shuffled with
--seed and split into named subsets of --n items each, written as ItemSubset JSON files under --out-dir
(<task>-<split>-snr-<name>-n<N>-seed<S>.json), plus pool.csv with every item's signal, noise and status.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer

from datadec.po.subsets import OLMES_ARC_CHALLENGE_FEWSHOT_IDS, OLMES_ARC_EASY_FEWSHOT_IDS, DATASET_NAMES, ItemSubset

app = typer.Typer()
FEWSHOT = {"arc_easy": OLMES_ARC_EASY_FEWSHOT_IDS, "arc_challenge": OLMES_ARC_CHALLENGE_FEWSHOT_IDS}
FAMILY = "rc_primary_likelihood"


@app.command()
def main(
    task: Annotated[str, typer.Option("--task")],
    model_items: Annotated[list[Path], typer.Option("--model-items")],
    noise_items: Annotated[list[Path], typer.Option("--noise-items")],
    out_dir: Annotated[Path, typer.Option("--out-dir")],
    names: Annotated[str, typer.Option("--names", help="comma-separated subset names, drawn in order")] = "train,dev,test",
    n: Annotated[int, typer.Option("--n")] = 300,
    min_snr: Annotated[float, typer.Option("--min-snr")] = 2.0,
    seed: Annotated[int, typer.Option("--seed")] = 0,
    split: Annotated[str, typer.Option("--split")] = "test",
) -> None:
    frames = [pd.read_csv(p).set_index("native_id")[FAMILY] for p in model_items]
    ids = frames[0].index
    for f in frames[1:]:
        ids = ids.intersection(f.index)
    signal = pd.concat([f.loc[ids] for f in frames], axis=1)
    signal = signal.max(axis=1) - signal.min(axis=1)
    noise_frames = []
    for p in noise_items:
        t = pd.read_csv(p)
        t = t[t.family == FAMILY].set_index("native_id")["ckpt_sd"]
        noise_frames.append(t)
    noise = pd.concat([t.reindex(ids) for t in noise_frames], axis=1).mean(axis=1)
    pool = pd.DataFrame({"signal": signal, "noise": noise})
    pool["snr"] = pool["signal"] / pool["noise"].replace(0, np.nan)
    pool["fewshot"] = pool.index.isin(FEWSHOT[task])
    pool["excluded"] = pool["fewshot"] | pool["snr"].isna() | (pool["snr"] < min_snr)
    eligible = pool.index[~pool["excluded"]].to_numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(eligible)
    subset_names = names.split(",")
    if len(subset_names) * n > len(eligible):
        raise typer.BadParameter(f"need {len(subset_names) * n} eligible items, have {len(eligible)}")
    pool["subset"] = ""
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(subset_names):
        chosen = eligible[i * n:(i + 1) * n]
        pool.loc[chosen, "subset"] = name
        sub = ItemSubset(dataset_path="allenai/ai2_arc", dataset_name=DATASET_NAMES[task], split=split, seed=seed, ids=tuple(sorted(chosen)))
        path = out_dir / f"{task}-{split}-snr-{name}-n{n}-seed{seed}.json"
        path.write_text(sub.to_json())
        typer.echo(f"wrote {path}")
    pool.to_csv(out_dir / f"{task}-{split}-snr-pool-seed{seed}.csv")
    summary = {"task": task, "n_items": int(len(pool)), "excluded_fewshot": int(pool["fewshot"].sum()),
               "excluded_low_snr": int((~pool["fewshot"] & pool["excluded"]).sum()), "min_snr": min_snr,
               "eligible": int(len(eligible)), "subsets": {nm: n for nm in subset_names},
               "signal_median": float(pool["signal"].median()), "noise_median": float(pool["noise"].median()),
               "snr_median": float(pool["snr"].median()), "model_items": [str(p) for p in model_items], "noise_items": [str(p) for p in noise_items]}
    (out_dir / f"{task}-{split}-snr-pool-seed{seed}.summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    typer.echo(json.dumps(summary, indent=1))


if __name__ == "__main__":
    app()
