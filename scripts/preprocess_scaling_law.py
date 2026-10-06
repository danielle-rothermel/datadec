from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from datadec.data.artifacts import DataArtifacts
from datadec.data.preprocess import preprocess_scaling_law
from datadec.data.publish import publish_unit, scaling_law_publication_unit

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"

app = typer.Typer()


@app.command()
def main(
    data_dir: Annotated[Path, typer.Option("--data-dir")] = DEFAULT_DATA_DIR,
    upload: Annotated[bool, typer.Option("--upload/--no-upload")] = True,
    keep_sources: Annotated[bool, typer.Option("--keep-sources")] = False,
) -> None:
    """Preprocess the three local raw scaling-law CSV artifacts."""
    paths = DataArtifacts(data_dir)
    result = preprocess_scaling_law(paths, verbose=True)
    if upload:
        publish_unit(
            scaling_law_publication_unit(
                paths,
                evaluations_output_path=result.evaluations_output_path,
                checkpoint_losses_output_path=result.checkpoint_losses_output_path,
            ),
            keep_sources=keep_sources,
        )


if __name__ == "__main__":
    app()
