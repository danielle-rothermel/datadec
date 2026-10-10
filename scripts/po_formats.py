from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from datadec.po.formats import sample_formats, write_formats

app = typer.Typer()
DEFAULT_DIR = Path(__file__).resolve().parents[1] / "configs" / "po" / "formats"


@app.command()
def main(
    n: Annotated[int, typer.Option("--n")] = 10,
    seed: Annotated[int, typer.Option("--seed")] = 0,
    out_dir: Annotated[Path, typer.Option("--out-dir")] = DEFAULT_DIR,
) -> None:
    """Sample n prompt formats (canonical first) and dump them under configs/po/formats."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"arc_easy-fmt-n{n}-seed{seed}.json"
    if path.exists():
        raise typer.BadParameter(f"{path} already exists; format samples are immutable once written")
    write_formats(path, sample_formats(n, seed), seed=seed)
    typer.echo(f"wrote {path}")


if __name__ == "__main__":
    app()
