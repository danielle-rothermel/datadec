from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from datadec.po.gepa_jobs import DEFAULT_ROOT, DEFAULT_SCORE_METRIC, add_empty_seed, rank_seeds, select_seeds, write_jobs
from datadec.po.results import load_sweep
from datadec.po.subsets import ItemSubset

app = typer.Typer()


@app.command()
def main(
    name: Annotated[str, typer.Option("--name")],
    sweep_dir: Annotated[Path, typer.Option("--sweep-dir")],
    train_subset: Annotated[Path, typer.Option("--train-subset")],
    val_subset: Annotated[Path, typer.Option("--val-subset")],
    n: Annotated[int, typer.Option("--n")] = 3,
    metric: Annotated[str, typer.Option("--metric", help="items column used to rank seeds")] = DEFAULT_SCORE_METRIC,
    score_metric: Annotated[str, typer.Option("--score-metric", help="what GEPA optimizes: primary or primary_likelihood")] = DEFAULT_SCORE_METRIC,
    max_metric_calls: Annotated[int, typer.Option("--max-metric-calls")] = 600,
    reflection_minibatch_size: Annotated[int, typer.Option("--reflection-minibatch-size")] = 3,
    formulations: Annotated[str | None, typer.Option("--formulations", help="restrict, e.g. rc")] = None,
    reflection_model: Annotated[str, typer.Option("--reflection-model")] = "openai/gpt-5.1",
    reflection_reasoning: Annotated[str, typer.Option("--reflection-reasoning")] = "medium",
    device: Annotated[str, typer.Option("--device")] = "mps",
    batch_size: Annotated[int, typer.Option("--batch-size")] = 4,
    batch_sizes: Annotated[str | None, typer.Option("--batch-sizes", help="per-model overrides, e.g. 'allenai/DataDecide-dclm-baseline-300M=64,Qwen/Qwen3-1.7B=16'")] = None,
    dtype: Annotated[str | None, typer.Option("--dtype", help="model dtype for GEPA scoring (float32 on the cluster)")] = None,
    include_empty: Annotated[bool, typer.Option("--include-empty/--no-include-empty", help="add an empty-instruction seed per group")] = False,
    max_group_score: Annotated[float | None, typer.Option("--max-group-score", help="skip groups whose best seed scores >= this (no headroom)")] = None,
    root: Annotated[Path, typer.Option("--root")] = DEFAULT_ROOT,
) -> None:
    """Rank a sweep's seeds per (model, formulation), pick the n best and n worst, write GEPA jobs."""
    items = load_sweep(sweep_dir)["items"]
    ranked = rank_seeds(items, metric=metric)
    selected = select_seeds(ranked, n)
    if include_empty:
        selected = add_empty_seed(selected, ranked)
    bs_map = {k: int(v) for k, v in (kv.split("=") for kv in batch_sizes.split(","))} if batch_sizes else None
    paths = write_jobs(
        name=name, sweep_dir=sweep_dir, selected=selected,
        train_subset=ItemSubset.load(train_subset), val_subset=ItemSubset.load(val_subset),
        max_metric_calls=max_metric_calls, reflection_minibatch_size=reflection_minibatch_size, reflection_model=reflection_model, reflection_reasoning=reflection_reasoning,
        device=device, batch_size=batch_size, batch_sizes=bs_map, dtype=dtype, formulations=tuple(formulations.split(",")) if formulations else None,
        max_group_score=max_group_score, score_metric=score_metric, root=root,
    )
    typer.echo(selected[["model", "formulation", "group", "rank", "format_id", "instruction_id", "score_metric", "score", "primary"]].to_string(index=False))
    typer.echo(f"wrote {len(paths)} jobs under {root / name / 'jobs'}")


if __name__ == "__main__":
    app()
