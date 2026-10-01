from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from datadec.po.slurm import SlurmSettings, submit, write_array

app = typer.Typer()


@app.command()
def main(
    jobs: Annotated[Path, typer.Option("--jobs", help="JSON {jobs: [...]}; see datadec.po.slurm")],
    root: Annotated[Path, typer.Option("--root", help="run root; sweeps under <root>/sweeps, array under <root>/slurm/<name>")],
    name: Annotated[str, typer.Option("--name", help="array name (unique under <root>/slurm)")],
    repo: Annotated[Path, typer.Option("--repo", help="datadec checkout the job paths are relative to")] = Path("."),
    partition: Annotated[str, typer.Option("--partition")] = "a100_cds",
    account: Annotated[str, typer.Option("--account")] = "torch_pr_375_cds",
    time: Annotated[str, typer.Option("--time", help="per-task limit, HH:MM:SS")] = "00:30:00",
    max_concurrent: Annotated[int, typer.Option("--max-concurrent")] = 23,
    chunk_size: Annotated[int, typer.Option("--chunk-size", help="tasks per OLMES process inside a sweep")] = 14,
    mem: Annotated[str, typer.Option("--mem")] = "60G",
    cpus: Annotated[int, typer.Option("--cpus")] = 4,
    do_submit: Annotated[bool, typer.Option("--submit/--no-submit")] = True,
) -> None:
    """Write sweeps for a job list and submit them as one Slurm job array (one GPU per sweep)."""
    settings = SlurmSettings(account=account, partition=partition, cpus=cpus, mem=mem, time=time,
                             max_concurrent=max_concurrent, chunk_size=chunk_size)
    array_dir = write_array(jobs, root, name, repo.resolve(), settings)
    n = len((array_dir / "sweeps.txt").read_text().splitlines())
    typer.echo(f"wrote {array_dir} ({n} tasks)")
    if do_submit:
        typer.echo(submit(array_dir))


if __name__ == "__main__":
    app()
