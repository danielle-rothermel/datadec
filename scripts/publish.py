from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from datadec.data.artifacts import DataArtifacts
from datadec.data.publish import publish_existing_outputs

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"

app = typer.Typer()


@app.command()
def main(
    ppl: Annotated[bool, typer.Option("--ppl")] = False,
    olmes: Annotated[bool, typer.Option("--olmes")] = False,
    olmes_details: Annotated[list[str] | None, typer.Option("--olmes-details")] = None,
    scaling_law: Annotated[bool, typer.Option("--scaling-law")] = False,
    published_results: Annotated[bool, typer.Option("--published-results")] = False,
    all_outputs: Annotated[bool, typer.Option("--all")] = False,
    keep_sources: Annotated[bool, typer.Option("--keep-sources")] = False,
    data_dir: Annotated[Path, typer.Option("--data-dir")] = DEFAULT_DATA_DIR,
) -> None:
    """Publish selected existing final DataDecide outputs to Hugging Face."""
    details = olmes_details or []
    if (
        not ppl
        and not olmes
        and not details
        and not scaling_law
        and not published_results
        and not all_outputs
    ):
        raise typer.BadParameter(
            "select --ppl, --olmes, --olmes-details, --scaling-law, "
            "--published-results, or --all"
        )
    if all_outputs:
        ppl = True
        olmes = True
        scaling_law = True
        published_results = True
        details = ["all"]

    try:
        results = publish_existing_outputs(
            DataArtifacts(data_dir),
            ppl=ppl,
            olmes=olmes,
            olmes_details=details,
            scaling_law=scaling_law,
            published_results=published_results,
            keep_sources=keep_sources,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error

    for result in results:
        status = "created" if result.created else "verified no-op"
        typer.echo(f"{result.unit_name}: {status} at {result.commit_oid}")


if __name__ == "__main__":
    app()
