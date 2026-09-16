from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from datadec.po.ape import load_instructions
from datadec.po.formats import load_formats
from datadec.po.sweep import sample_pairs, write_pairs

app = typer.Typer()
DEFAULT_DIR = Path(__file__).resolve().parents[1] / "configs" / "po" / "pairs"


@app.command()
def main(
    name: Annotated[str, typer.Option("--name")],
    formats: Annotated[Path, typer.Option("--formats")],
    instructions: Annotated[list[Path], typer.Option("--instructions", help="one or more APE candidates.jsonl")],
    n: Annotated[int, typer.Option("--n")] = 50,
    seed: Annotated[int, typer.Option("--seed")] = 0,
    include_none: Annotated[bool, typer.Option("--include-none/--no-include-none")] = True,
    out_dir: Annotated[Path, typer.Option("--out-dir")] = DEFAULT_DIR,
) -> None:
    """Sample n (format, instruction) pairs from a format pool and an instruction pool; baseline pair first."""
    fmts = load_formats(formats)
    pool = []
    for path in instructions:
        for row in load_instructions(path):
            pool.append({"id": f"{path.parent.name}/{row['id']}", "text": row["text"]})
    if include_none:
        pool.append({"id": "none", "text": None})
    pairs = sample_pairs(fmts, pool, n, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}-n{n}-seed{seed}.json"
    if path.exists():
        raise typer.BadParameter(f"{path} exists; pair samples are immutable once written")
    write_pairs(path, pairs, seed=seed, sources={"formats": str(formats), "instructions": [str(p) for p in instructions],
                                                  "n_formats": len(fmts), "n_instructions": len(pool)})
    typer.echo(f"wrote {path}: {len(pairs)} pairs from {len(fmts)} formats x {len(pool)} instructions")
    ids = [p["instruction"]["id"] for p in pairs]
    typer.echo(f"instruction 'none' appears {ids.count('none')}x; distinct formats {len({p['format_id'] for p in pairs})}")


if __name__ == "__main__":
    app()
