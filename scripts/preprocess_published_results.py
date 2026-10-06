from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from datadec.config import (
    load_published_results_manifest,
    load_publishing_contract,
)
from datadec.data.artifacts import DataArtifacts
from datadec.data.selection import resolve_published_result_units
from datadec.data.preprocess.published_results import (
    preprocess_published_results,
)
from datadec.data.publish import (
    publish_unit,
    published_results_publication_units,
)

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"

app = typer.Typer()


@app.command()
def main(
    unit: Annotated[list[str] | None, typer.Option("--unit")] = None,
    upload: Annotated[bool, typer.Option("--upload/--no-upload")] = True,
    keep_sources: Annotated[bool, typer.Option("--keep-sources")] = False,
    data_dir: Annotated[Path, typer.Option("--data-dir")] = DEFAULT_DATA_DIR,
) -> None:
    """Convert structured results to Parquet and publish them by default."""
    manifest = load_published_results_manifest()
    try:
        units = resolve_published_result_units(unit or (), manifest)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--unit") from exc
    paths = DataArtifacts(data_dir)
    results = preprocess_published_results(
        paths, units=units, manifest=manifest, verbose=True
    )
    if not upload:
        return

    publishing = load_publishing_contract()
    publication_units = published_results_publication_units(
        paths,
        units=tuple(result.publication_unit for result in results),
        contract=publishing,
        manifest=manifest,
    )
    for publication_unit in publication_units:
        result = publish_unit(
            publication_unit,
            target=publishing.target,
            keep_sources=keep_sources,
        )
        status = "created" if result.created else "verified no-op"
        typer.echo(f"{result.unit_name}: {status} at {result.commit_oid}")


if __name__ == "__main__":
    app()
