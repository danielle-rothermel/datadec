from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from datadec.po.subsets import DATASETS, OLMES_ARC_EASY_FEWSHOT_IDS, OLMES_ARC_CHALLENGE_FEWSHOT_IDS, ItemSubset, sample_subset

app = typer.Typer()
DEFAULT_DIR = Path(__file__).resolve().parents[1] / "configs" / "po" / "subsets"


@app.command()
def main(
    dataset: Annotated[str, typer.Option("--dataset", help="arc_easy, arc_challenge or hellaswag")] = "arc_easy",
    n: Annotated[int, typer.Option("--n")] = 100,
    seed: Annotated[int, typer.Option("--seed")] = 0,
    split: Annotated[str | None, typer.Option("--split", help="default: the dataset's OLMES eval split")] = None,
    out_dir: Annotated[Path, typer.Option("--out-dir")] = DEFAULT_DIR,
    exclude: Annotated[list[Path] | None, typer.Option("--exclude", help="subset files whose ids to exclude")] = None,
    exclude_fewshot: Annotated[bool, typer.Option("--exclude-fewshot/--no-exclude-fewshot")] = True,
) -> None:
    """Dump a seeded item subset as JSON under configs/po/subsets."""
    if dataset not in DATASETS:
        raise typer.BadParameter(f"dataset must be one of {sorted(DATASETS)}")
    path_, name, id_field, default_split = DATASETS[dataset]
    split = split or default_split
    # HellaSwag demos come from the train split, so nothing in its validation split needs excluding.
    fewshot = {"arc_easy": OLMES_ARC_EASY_FEWSHOT_IDS, "arc_challenge": OLMES_ARC_CHALLENGE_FEWSHOT_IDS, "hellaswag": ()}[dataset]
    excluded: set[str | int] = set(fewshot) if exclude_fewshot else set()
    for p in exclude or []:
        excluded |= set(ItemSubset.load(p).ids)
    subset = sample_subset(n=n, seed=seed, split=split, dataset_path=path_, dataset_name=name, exclude=frozenset(excluded), id_field=id_field)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{dataset}-{split}-n{n}-seed{seed}.json"
    if path.exists():
        raise typer.BadParameter(f"{path} already exists; subsets are immutable once written")
    path.write_text(subset.to_json())
    typer.echo(f"wrote {path} ({subset.n} ids, first {subset.ids[:2]})")


if __name__ == "__main__":
    app()
