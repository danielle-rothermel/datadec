from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from datadec.po.subsets import DATASETS, ItemSubset, demo_exclusions, load_split, sample_subset

app = typer.Typer()
DEFAULT_DIR = Path(__file__).resolve().parents[1] / "configs" / "po" / "subsets"


@app.command()
def main(
    dataset: Annotated[str, typer.Option("--dataset", help="a key of datadec.po.subsets.DATASETS")] = "arc_easy",
    n: Annotated[int, typer.Option("--n")] = 100,
    seed: Annotated[int, typer.Option("--seed")] = 0,
    split: Annotated[str | None, typer.Option("--split", help="default: the dataset's OLMES eval split")] = None,
    out_dir: Annotated[Path, typer.Option("--out-dir")] = DEFAULT_DIR,
    exclude: Annotated[list[Path] | None, typer.Option("--exclude", help="subset files whose ids to exclude")] = None,
    exclude_fewshot: Annotated[bool, typer.Option("--exclude-fewshot/--no-exclude-fewshot")] = True,
    tag: Annotated[str | None, typer.Option("--tag", help="role label inserted in the file name, e.g. train1 for a partition of a split")] = None,
) -> None:
    """Dump a seeded item subset as JSON under configs/po/subsets."""
    if dataset not in DATASETS:
        raise typer.BadParameter(f"dataset must be one of {sorted(DATASETS)}")
    spec = DATASETS[dataset]
    split = split or spec.eval_split
    ds = load_split(spec, split)
    # OLMES demonstrations come from the train split; nothing in other splits needs excluding.
    excluded: set[str | int] = set(demo_exclusions(dataset, split, ds)) if exclude_fewshot else set()
    for p in exclude or []:
        excluded |= set(ItemSubset.load(p).ids)
    subset = sample_subset(n=n, seed=seed, split=split, dataset_path=spec.path, dataset_name=spec.name, exclude=frozenset(excluded), id_field=spec.id_field, revision=spec.revision, ds=ds)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{dataset}-{split}-{tag + '-' if tag else ''}n{n}-seed{seed}.json"
    if path.exists():
        raise typer.BadParameter(f"{path} already exists; subsets are immutable once written")
    path.write_text(subset.to_json())
    typer.echo(f"wrote {path} ({subset.n} ids, first {subset.ids[:2]})")


if __name__ == "__main__":
    app()
