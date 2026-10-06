"""Render one figure spec over the tidy results tables: a PNG plus its CSV twin of the plotted numbers.

    uv run python scripts/po_figures.py --spec configs/po/figures/figA-model-axes.json \\
        --tidy ~/drotherm/data/runs/po/cluster-20261001/tidy --out <figures dir>
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from datadec.po.figures import load_spec, render
from datadec.po.tidy import read_tables

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(
    spec: Annotated[list[Path], typer.Option(help="figure spec JSON (repeatable)")],
    tidy: Annotated[Path, typer.Option(help="tidy tables directory (po_tidy.py build --out)")],
    out: Annotated[Path, typer.Option(help="output directory; files are named after the spec")],
) -> None:
    """Write <out>/<spec stem>.png and .csv for every spec."""
    tables = read_tables(tidy)
    for path in spec:
        png, csv = render(load_spec(path), tables, out / f"{path.stem}.png")
        typer.echo(f"wrote {png} and {csv}")


if __name__ == "__main__":
    app()
