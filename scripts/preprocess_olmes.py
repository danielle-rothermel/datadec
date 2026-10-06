from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from datadec.data.artifacts import DataArtifacts
from datadec.data.preprocess.olmes import preprocess_olmes
from datadec.data.publish import olmes_publication_unit, publish_unit

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"

app = typer.Typer()


@app.command()
def main(
    data_dir: Annotated[Path, typer.Option("--data-dir")] = DEFAULT_DATA_DIR,
    input_path: Annotated[
        Path | None, typer.Option("--input", help="Override raw OLMES parquet input")
    ] = None,
    output_path: Annotated[
        Path | None,
        typer.Option("--output", help="Override processed OLMES parquet output"),
    ] = None,
    upload: Annotated[bool, typer.Option("--upload/--no-upload")] = True,
    keep_sources: Annotated[bool, typer.Option("--keep-sources")] = False,
) -> None:
    """Preprocess the local raw OLMES parquet artifact."""
    paths = DataArtifacts(data_dir)
    result = preprocess_olmes(
        paths,
        input_path=input_path,
        output_path=output_path,
        verbose=True,
    )
    if upload:
        publish_unit(
            olmes_publication_unit(paths, output_path=result.output_path),
            keep_sources=keep_sources,
        )


if __name__ == "__main__":
    app()
