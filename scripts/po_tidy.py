"""Build the phase-2 tidy results tables (models, tasks, summary, contrasts) from the copied sweep directories.

    uv run python scripts/po_tidy.py build --sweeps ~/drotherm/data/runs/po/cluster-20261001/sweeps \\
        --out ~/drotherm/data/runs/po/cluster-20261001/tidy
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from datadec.po.tidy import build_tables, write_tables

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


@app.callback()
def _root() -> None:
    """Tidy results tables for the phase-2 sweeps (contract po-tidy/1)."""


@app.command()
def build(
    sweeps: Annotated[Path, typer.Option(help="directory holding the sweep directories")],
    out: Annotated[Path, typer.Option(help="output directory for the parquet/csv tables")],
    workers: Annotated[int | None, typer.Option(help="loader processes (default: CPU count)")] = None,
) -> None:
    """Read every selected sweep, write models/tasks/summary/contrasts (parquet + csv) and build.json."""
    if not sweeps.is_dir():
        raise typer.BadParameter(f"{sweeps} is not a directory")
    result = build_tables(sweeps, workers=workers)
    write_tables(result, out)
    typer.echo(f"sweeps included {len(result.selection.included)}; excluded {result.selection.excluded}")
    typer.echo(f"duplicate task rows not pooled: {result.duplicate_task_rows}")
    for name, df in result.tables.items():
        typer.echo(f"{name}: {len(df)} rows")
    typer.echo(f"wrote {out} in {result.seconds:.0f} s")


if __name__ == "__main__":
    app()
