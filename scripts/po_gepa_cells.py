"""GEPA cells (gepa-run-contract): write a cell's jobs and pool items, the per-run eval follow-up, and aggregation.

  write      6 job files under ROOT/gepa/<cell_id>/ and 6 gepa items in the pool (empty and ape starts x seeds 0-2)
  eval-item  follow-up of a finished gepa item: the ge-<run_id> Test sweep plus its eval item in the same pool
  aggregate  per-run and per-cell paired effects (hierarchical bootstrap) into the tidy contrasts and gepa_runs tables
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from datadec.po.gepa_cells import aggregate as aggregate_cells
from datadec.po.gepa_cells import write_cell, write_eval_item

app = typer.Typer(no_args_is_help=True)


@app.command()
def write(
    model: Annotated[str, typer.Option("--model")],
    task: Annotated[str, typer.Option("--task")],
    pool_dir: Annotated[Path, typer.Option("--pool-dir", "--pool", help="ROOT/pool/<pool-name>")],
    root: Annotated[Path, typer.Option("--root", help="run root (ROOT in the pool contract)")],
    repo: Annotated[Path, typer.Option("--repo", help="datadec checkout the items run from")],
    olmes_repo: Annotated[Path, typer.Option("--olmes-repo", help="OLMES fork checkout (gepa item cwd)")],
    revision: Annotated[str | None, typer.Option("--revision")] = None,
    dtype: Annotated[str | None, typer.Option("--dtype", help="float32 (default) or bfloat16 (default for 32B)")] = None,
    batch_size: Annotated[int | None, typer.Option("--batch-size", help="default by model family and size")] = None,
) -> None:
    paths = write_cell(model=model, revision=revision, task=task, pool_dir=pool_dir, root=root, repo=repo,
                       olmes_repo=olmes_repo, dtype=dtype, batch_size=batch_size)
    typer.echo(f"{len(paths)} gepa items written to {pool_dir / 'pending'}")
    for p in paths:
        typer.echo(f"  {p.name}")


@app.command("eval-item")
def eval_item(
    run_dir: Annotated[Path, typer.Option("--run-dir", help="ROOT/gepa/<cell_id>/<run_id>")],
    pool_dir: Annotated[Path, typer.Option("--pool-dir", "--pool")],
    repo: Annotated[Path, typer.Option("--repo", help="datadec checkout (default: the current directory)")] = Path("."),
) -> None:
    sweep_dir, paths = write_eval_item(run_dir, pool_dir, repo.resolve())
    typer.echo(f"sweep {sweep_dir}; " + (f"eval item {paths[0].name}" if paths else "eval item already in the pool"))


@app.command()
def aggregate(
    tidy: Annotated[Path, typer.Option("--tidy", help="tidy table dir (contrasts.parquet, gepa_runs.parquet)")],
    gepa_root: Annotated[Path, typer.Option("--gepa-root", help="ROOT/gepa")],
    sweeps: Annotated[Path, typer.Option("--sweeps", help="dir holding the ge-<run_id> sweeps")],
) -> None:
    contrasts, runs = aggregate_cells(tidy, gepa_root, sweeps)
    if runs.empty:
        typer.echo("no finished runs with eval results")
        return
    typer.echo(f"{len(runs)} runs over {runs['cell_id'].nunique()} cells; {len(contrasts)} gepa contrast rows -> {tidy}")


if __name__ == "__main__":
    app()
