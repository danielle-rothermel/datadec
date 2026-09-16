from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from datadec.po.results import load_sweep, write_analysis

app = typer.Typer()


@app.command()
def main(sweep_dir: Annotated[Path, typer.Argument()]) -> None:
    """Flatten a finished sweep into analysis/{tasks,items,choices}.parquet and print a summary."""
    out = write_analysis(sweep_dir)
    tables = load_sweep(sweep_dir)
    t = tables["tasks"]
    typer.echo(f"wrote {out}: {len(t)} tasks, {len(tables['items'])} item rows, {len(tables['choices'])} choice rows")
    if len(t):
        cols = [c for c in ("formulation", "format_id", "instruction_id", "num_instances", "primary_score") if c in t]
        typer.echo(t[cols].to_string(index=False))


if __name__ == "__main__":
    app()
