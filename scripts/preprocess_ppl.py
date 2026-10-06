from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from datadec.data.artifacts import DataArtifacts
from datadec.data.preprocess import preprocess_ppl
from datadec.data.publish import ppl_publication_unit, publish_unit

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"

app = typer.Typer()


@app.command()
def main(
    data_dir: Annotated[Path, typer.Option("--data-dir")] = DEFAULT_DATA_DIR,
    upload: Annotated[bool, typer.Option("--upload/--no-upload")] = True,
    keep_sources: Annotated[bool, typer.Option("--keep-sources")] = False,
) -> None:
    """Preprocess the local raw PPL parquet artifact."""
    paths = DataArtifacts(data_dir)
    result = preprocess_ppl(paths, verbose=True)
    if upload:
        publish_unit(
            ppl_publication_unit(paths, output_path=result.output_path),
            keep_sources=keep_sources,
        )


if __name__ == "__main__":
    app()
