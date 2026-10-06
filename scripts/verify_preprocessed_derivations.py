from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from datadec.data.artifacts import DataArtifacts
from datadec.data.preprocess.derivations_verify import (
    verify_preprocessed_derivations,
)

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"

app = typer.Typer()


@app.command()
def main(
    data_dir: Annotated[Path, typer.Option("--data-dir")] = DEFAULT_DATA_DIR,
) -> None:
    """Check proposed schedule derivations against processed and raw values."""
    result = verify_preprocessed_derivations(DataArtifacts(data_dir))
    for output in result.processed_outputs:
        typer.echo(
            f"{output.name}: rows={output.row_count}, "
            f"token evidence={output.token_evidence_count}, "
            f"token contradictions={output.token_mismatch_count}, "
            f"exact-compute evidence={output.compute_evidence_count}, "
            f"exact-compute contradictions={output.exact_compute_mismatch_count}, "
            f"model-detail contradictions={output.model_detail_mismatch_count}, "
            f"LR contradictions={output.lr_mismatch_count}"
        )
    raw_olmes = result.raw_olmes
    typer.echo(
        f"{raw_olmes.name}: rows={raw_olmes.row_count}, "
        f"token contradictions={raw_olmes.token_mismatch_count}, "
        "exact-compute contradictions="
        f"{raw_olmes.exact_compute_mismatch_count}"
    )
    raw_scaling = result.raw_scaling_law
    typer.echo(
        f"raw scaling-law: rows={raw_scaling.row_count}, "
        f"token evidence={raw_scaling.token_evidence_count}, "
        f"token contradictions={raw_scaling.token_mismatch_count}, "
        f"compute evidence={raw_scaling.compute_evidence_count}, "
        "exact-compute contradictions="
        f"{raw_scaling.exact_compute_mismatch_count}, "
        "nominal-compute contradictions="
        f"{raw_scaling.nominal_compute_mismatch_count}"
    )
    for detail in result.detail_tasks:
        typer.echo(
            f"OLMES detail tasks {detail.path.parent.name}: "
            f"rows={detail.row_count}, "
            f"model-config contradictions={detail.max_length_mismatch_count + detail.model_identity_mismatch_count + detail.revision_step_mismatch_count}"
        )
    typer.echo(
        "raw LR schedule evidence: "
        f"{result.lr_raw_evidence_count} "
        "(no preprocessing source contains LR schedule values)"
    )
    typer.echo(f"total contradictions: {result.contradiction_count}")
    if result.contradiction_count:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
