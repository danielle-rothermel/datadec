from __future__ import annotations

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
    match: Annotated[Path | None, typer.Option("--match", help="existing pairs file to mirror: same formats, same candidate index from the given pools")] = None,
) -> None:
    """Sample n (format, instruction) pairs from a format pool and an instruction pool; baseline pair first.

    With --match, do not sample: reproduce the given pairs file pair-for-pair, replacing each instruction
    by the candidate with the same index (cNNN) from the corresponding new pool (matched by pool family:
    'ape-forward' or 'p1-description' in the pool directory name)."""
    fmts = load_formats(formats)
    pool = []
    for path in instructions:
        for row in load_instructions(path):
            pool.append({"id": f"{path.parent.name}/{row['id']}", "text": row["text"]})
    if include_none:
        pool.append({"id": "none", "text": None})
    if match:
        from datadec.po.sweep import load_pairs
        by_key = {}
        for entry in pool:
            if entry["id"] == "none":
                by_key[("none", "none")] = entry
            else:
                d, c = entry["id"].split("/")
                fam = "ape-forward" if "ape-forward" in d else "p1-description"
                by_key[(fam, c)] = entry
        pairs = []
        for src in load_pairs(match):
            iid = src["instruction"]["id"]
            key = ("none", "none") if iid == "none" else (("ape-forward" if "ape-forward" in iid else "p1-description"), iid.split("/")[-1])
            if key not in by_key:
                raise typer.BadParameter(f"no matching candidate for {iid}")
            pairs.append({"format": src["format"], "format_id": src["format_id"], "instruction": by_key[key]})
        seed_note = f"matched:{match.name}"
    else:
        pairs = sample_pairs(fmts, pool, n, seed)
        seed_note = None
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}-n{n}-seed{seed}.json"
    if path.exists():
        raise typer.BadParameter(f"{path} exists; pair samples are immutable once written")
    write_pairs(path, pairs, seed=seed, sources={"formats": str(formats), "instructions": [str(p) for p in instructions],
                                                  "n_formats": len(fmts), "n_instructions": len(pool), "matched_to": seed_note})
    typer.echo(f"wrote {path}: {len(pairs)} pairs from {len(fmts)} formats x {len(pool)} instructions")
    ids = [p["instruction"]["id"] for p in pairs]
    typer.echo(f"instruction 'none' appears {ids.count('none')}x; distinct formats {len({p['format_id'] for p in pairs})}")


if __name__ == "__main__":
    app()
