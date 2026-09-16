from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from datadec.po.ape import load_instructions
from datadec.po.formats import CANONICAL, load_formats
from datadec.po.sweep import DEFAULT_ROOT, SweepSpec, load_pairs, run_sweep, write_sweep

app = typer.Typer()


@app.command()
def main(
    name: Annotated[str, typer.Option("--name")],
    subset: Annotated[Path, typer.Option("--subset")],
    model: Annotated[str, typer.Option("--model")],
    revision: Annotated[str | None, typer.Option("--revision")] = None,
    task: Annotated[str, typer.Option("--task", help="arc_easy or arc_challenge")] = "arc_easy",
    pairs: Annotated[Path | None, typer.Option("--pairs", help="sampled (format, instruction) pairs JSON; overrides formats x instructions")] = None,
    formulations: Annotated[str, typer.Option("--formulations")] = "rc,mc",
    formats: Annotated[Path | None, typer.Option("--formats", help="formats JSON; omit for canonical only")] = None,
    instructions: Annotated[Path | None, typer.Option("--instructions", help="APE candidates.jsonl or JSON list")] = None,
    num_shots: Annotated[int, typer.Option("--num-shots")] = 5,
    device: Annotated[str, typer.Option("--device")] = "mps",
    batch_size: Annotated[int, typer.Option("--batch-size")] = 16,
    root: Annotated[Path, typer.Option("--root")] = DEFAULT_ROOT,
    run: Annotated[bool, typer.Option("--run/--no-run")] = True,
) -> None:
    """Write a matched sweep (formulations x formats x instructions on one subset) and run it."""
    spec = SweepSpec(
        name=name, subset_path=subset, model=model, revision=revision, task=task,
        pairs=tuple(load_pairs(pairs)) if pairs else (),
        formulations=tuple(f.strip() for f in formulations.split(",") if f.strip()),
        formats=tuple(load_formats(formats)) if formats else (dict(CANONICAL),),
        instructions=tuple(load_instructions(instructions)) if instructions else (),
        num_shots=num_shots, device=device, batch_size=batch_size,
    )
    sweep_dir = write_sweep(spec, root=root)
    typer.echo(f"wrote {sweep_dir} ({sum(1 for _ in open(sweep_dir / 'tasks.jsonl'))} tasks)")
    if run:
        rc = run_sweep(sweep_dir)
        typer.echo(f"runner exit code {rc}; see {sweep_dir / 'runner.log'}")
        raise typer.Exit(rc)


if __name__ == "__main__":
    app()
