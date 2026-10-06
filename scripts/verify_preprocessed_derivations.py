from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from datadec.data.artifacts import DataArtifacts
from datadec.data.selection import resolve_selection
from datadec.data.verify import verify_selected_outputs

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"

app = typer.Typer()


@app.command()
def main(
    data_dir: Annotated[Path, typer.Option("--data-dir")] = DEFAULT_DATA_DIR,
) -> None:
    """Check canonical schedule derivations against processed and raw values."""
    try:
        report = verify_selected_outputs(
            DataArtifacts(data_dir),
            resolve_selection(ppl=True, olmes=True, scaling_law=True),
        )
    except (AssertionError, ValueError, FileNotFoundError) as error:
        typer.echo(f"verification failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    for check in report.checks:
        typer.echo(f"{check.outcome.value}: {check.name}: {check.detail}")
    typer.echo(
        "raw LR schedule evidence: "
        f"{report.derivations.lr_raw_evidence_count} "
        "(no preprocessing source contains LR schedule values)"
    )


if __name__ == "__main__":
    app()
