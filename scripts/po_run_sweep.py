"""Run one written sweep chunk-wise (resumes: complete chunks are skipped). The command of every pool sweep item."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from datadec.po.sweep import run_sweep

app = typer.Typer()


@app.command()
def main(
    sweep_dir: Annotated[Path, typer.Option("--sweep-dir", help="sweep directory holding sweep.json and tasks.jsonl")],
    chunk_size: Annotated[int, typer.Option("--chunk-size", help="tasks per OLMES process; 0 = all in one")] = 14,
) -> None:
    """Run every task of a sweep through the OLMES fork; exits with the worst chunk return code."""
    raise typer.Exit(run_sweep(sweep_dir, chunk_size=chunk_size or None))


if __name__ == "__main__":
    app()
