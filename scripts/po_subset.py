from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from datadec.po.subsets import OLMES_ARC_EASY_FEWSHOT_IDS, ItemSubset, sample_subset

app = typer.Typer()
DEFAULT_DIR = Path(__file__).resolve().parents[1] / "configs" / "po" / "subsets"


@app.command()
def main(
    n: Annotated[int, typer.Option("--n")] = 100,
    seed: Annotated[int, typer.Option("--seed")] = 0,
    split: Annotated[str, typer.Option("--split")] = "test",
    out_dir: Annotated[Path, typer.Option("--out-dir")] = DEFAULT_DIR,
    exclude: Annotated[list[Path] | None, typer.Option("--exclude", help="subset files whose ids to exclude")] = None,
    exclude_fewshot: Annotated[bool, typer.Option("--exclude-fewshot/--no-exclude-fewshot")] = True,
) -> None:
    """Dump a seeded ARC-Easy item subset as JSON under configs/po/subsets."""
    excluded: set[str] = set(OLMES_ARC_EASY_FEWSHOT_IDS) if exclude_fewshot else set()
    for path in exclude or []:
        excluded |= set(ItemSubset.load(path).ids)
    subset = sample_subset(n=n, seed=seed, split=split, exclude=frozenset(excluded))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"arc_easy-{split}-n{n}-seed{seed}.json"
    if path.exists():
        raise typer.BadParameter(f"{path} already exists; subsets are immutable once written")
    path.write_text(subset.to_json())
    typer.echo(f"wrote {path} ({subset.n} ids, first {subset.ids[:2]})")


if __name__ == "__main__":
    app()
