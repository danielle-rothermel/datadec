from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from datadec.po.results import load_sweep, write_analysis

app = typer.Typer()


@app.command()
def main(sweep_dir: Annotated[Path, typer.Argument()]) -> None:
    """Flatten a finished sweep into analysis/{tasks,items,choices}.parquet and print a summary."""
    if not sweep_dir.is_dir() or not (sweep_dir / "sweep.json").exists():
        raise typer.BadParameter(f"{sweep_dir} is not a sweep directory")
    out = write_analysis(sweep_dir)
    tables = load_sweep(sweep_dir)
    t = tables["tasks"]
    typer.echo(f"wrote {out}: {len(t)} tasks, {len(tables['items'])} item rows, {len(tables['choices'])} choice rows")
    if len(t):
        items = tables["items"]
        lik = items.groupby("task_idx")["primary_likelihood"].mean().rename("primary_likelihood")
        t = t.merge(lik, left_on="task_idx", right_index=True, how="left")
        cols = [c for c in ("formulation", "format_id", "instruction_id", "num_instances", "primary_metric", "primary",
                            "primary_likelihood_metric", "primary_likelihood") if c in t]
        typer.echo(t[cols].to_string(index=False))


if __name__ == "__main__":
    app()
