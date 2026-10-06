from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from datadec.config import load_olmes_contract
from datadec.data.artifacts import DataArtifacts
from datadec.data.preprocess.olmes_verify import verify_olmes_details

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_RECIPE = "dolma1.7-no-math-no-code"

app = typer.Typer()


@app.command()
def main(
    recipe: Annotated[str, typer.Option("--recipe")] = DEFAULT_RECIPE,
    data_dir: Annotated[Path, typer.Option("--data-dir")] = DEFAULT_DATA_DIR,
    detail_archive: Annotated[
        Path | None,
        typer.Option(
            "--detail-archive",
            help="Override local recipe detail archive tar.gz",
        ),
    ] = None,
) -> None:
    """Run manual OLMES detail verification against local downloaded artifacts.

    This performs full cross-source parity and metric reconstruction checks on
    the representative recipe, including the 482 overlapping checkpoints. It is
    intentionally excluded from the default pytest suite because it requires large
    local downloads and can take a long time once instances and choices are parsed.

    Note: bits_per_byte_corr is declared non-reconstructible from detailed outputs
    in configs/olmes.toml and is not validated during reconstruction.
    """
    contract = load_olmes_contract()
    paths = DataArtifacts(data_dir)
    archive = detail_archive or (
        data_dir / "raw/olmes-details/models" / f"{recipe}.tar.gz"
    )
    if not archive.is_file():
        raise typer.BadParameter(
            f"detail archive not found: {archive}",
            param_hint="--detail-archive",
        )
    if not paths.get_path("olmes_processed").is_file():
        raise typer.BadParameter(
            f"aggregate parquet not found: {paths.get_path('olmes_processed')}",
            param_hint="--data-dir",
        )
    for table in ("tasks", "instances", "choices"):
        path = getattr(paths, f"olmes_details_{table}_path")(recipe)
        if not path.is_file():
            raise typer.BadParameter(
                f"missing preprocessed detail output: {path}; "
                "run scripts/preprocess_olmes_details.py first",
            )

    result = verify_olmes_details(
        recipe=recipe,
        paths=paths,
        detail_archive=archive,
        contract=contract,
    )
    typer.echo(f"recipe: {recipe}")
    typer.echo(f"overlapping checkpoints: {result.overlapping_checkpoint_count}")
    typer.echo(f"parity rows checked: {result.parity_row_count}")
    typer.echo(f"reconstructed task groups checked: {result.reconstructed_task_count}")
    typer.echo(
        "non-reconstructible from details: "
        f"{', '.join(contract.metrics.not_reproducible_from_details)}"
    )


if __name__ == "__main__":
    app()
