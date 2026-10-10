"""Mac (mps) lane for GEPA cells: write jobs, run them locally, report status, push finished runs to the cluster.

  write   the cell's 6 jobs with device mps under LOCALROOT/gepa/<cell_id>/ (no pool items)
  run     every job without result.json via `mise exec -- uv run local/gepa_arc.py --job J` in the OLMES checkout,
          up to --parallel at once (empty starts, then ape; seeds in order); logs and attempt records beside the jobs;
          exits 1 when any run failed
  status  per cell: jobs / finished / failed / running / stale / pending, and each run's proposals and best val score
  push    rsync one cell's finished runs to <remote-root>/gepa/<cell_id>/ without overwriting anything there, then
          print the eval-item command to run on the cluster for each pushed run

The GEPA search runs on mps; Test evaluation stays on the cluster (the eval-item follow-up writes a cuda sweep).
See datadec.po.gepa_local.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from datadec.po.gepa_cells import DEFAULT_REFLECTION_PROMPT_VERSION, ReflectionPromptVersion, write_jobs
from datadec.po.gepa_local import (
    DEFAULT_LOCAL_ROOT,
    DEFAULT_OLMES_REPO,
    DEFAULT_PARALLEL,
    DEFAULT_REMOTE_ROOT,
    LOCAL_DEVICE,
    Remote,
    RunState,
    cell_dirs,
    cell_status,
    eval_item_command,
    plan_runs,
    push_cell,
    run_jobs,
)

app = typer.Typer(no_args_is_help=True)

RootOption = Annotated[Path, typer.Option("--root", help="local run root (LOCALROOT)")]


@app.command()
def write(
    model: Annotated[str, typer.Option("--model")],
    task: Annotated[str, typer.Option("--task")],
    repo: Annotated[Path, typer.Option("--repo", help="datadec checkout (subset and APE seed files)")] = Path("."),
    root: RootOption = DEFAULT_LOCAL_ROOT,
    revision: Annotated[str | None, typer.Option("--revision")] = None,
    dtype: Annotated[str | None, typer.Option("--dtype", help="float32 (default) or bfloat16 (default for 32B)")] = None,
    batch_size: Annotated[int | None, typer.Option("--batch-size", help="default by model family and size")] = None,
    reflection_prompt_version: Annotated[
        ReflectionPromptVersion, typer.Option("--reflection-prompt-version", help="fork driver reflection meta-prompt")
    ] = DEFAULT_REFLECTION_PROMPT_VERSION,
) -> None:
    jobs = write_jobs(model=model, revision=revision, task=task, root=root.resolve(), repo=repo.resolve(), dtype=dtype,
                      batch_size=batch_size, device=LOCAL_DEVICE, reflection_prompt_version=reflection_prompt_version)
    typer.echo(f"{len(jobs)} {LOCAL_DEVICE} jobs in {jobs[0][0].parent}")
    for _, job in jobs:
        typer.echo(f"  {job['job_id']}")


@app.command()
def run(
    root: RootOption = DEFAULT_LOCAL_ROOT,
    cells: Annotated[list[str] | None, typer.Option("--cell", help="cell id (repeatable; default: every cell)")] = None,
    parallel: Annotated[int, typer.Option("--parallel", min=1)] = DEFAULT_PARALLEL,
    olmes_repo: Annotated[Path, typer.Option("--olmes-repo", help="OLMES fork checkout (driver cwd)")] = DEFAULT_OLMES_REPO,
) -> None:
    plan = plan_runs(cell_dirs(root, cells or []))
    typer.echo(f"{len(plan.todo)} to run, {len(plan.finished)} finished (skipped), {len(plan.running)} running elsewhere (skipped)")
    outcomes = run_jobs(plan.todo, olmes_repo.resolve(), parallel)
    failed = [o for o in outcomes if not o.ok]
    for o in outcomes:
        typer.echo(f"  {'ok    ' if o.ok else 'FAILED'} {o.run_id} (exit {o.exit_code}, result.json {'present' if o.finished else 'missing'})")
    if failed:
        raise typer.Exit(1)


@app.command()
def status(root: RootOption = DEFAULT_LOCAL_ROOT) -> None:
    for cell in cell_dirs(root):
        rows = cell_status(cell)
        counts = {s: sum(r.state is s for r in rows) for s in RunState}
        typer.echo(f"{cell.name}: {len(rows)} jobs, " + ", ".join(f"{counts[s]} {s}" for s in RunState))
        for r in rows:
            best = "-" if r.best_val_score is None else f"{r.best_val_score:.4f}"
            proposals = "-" if r.proposals is None else str(r.proposals)
            typer.echo(f"  {r.run_id}  {r.state:<8} proposals {proposals:>3}  best_val {best}")


@app.command()
def push(
    cell: Annotated[str, typer.Option("--cell")],
    root: RootOption = DEFAULT_LOCAL_ROOT,
    remote_root: Annotated[str, typer.Option("--remote-root", help="host:/cluster/run/root")] = DEFAULT_REMOTE_ROOT,
    pool: Annotated[str | None, typer.Option("--pool", help="cluster pool dir for the printed eval-item commands")] = None,
) -> None:
    remote = Remote.parse(remote_root)
    report = push_cell(root, cell, remote)
    if report.not_finished:
        typer.echo(f"{len(report.not_finished)} runs not finished, not pushed: " + ", ".join(report.not_finished))
    if not report.pushed:
        typer.echo("nothing to push")
        return
    typer.echo(f"pushed {len(report.pushed)} finished runs to {remote.host}:{remote.cell_path(cell)}/")
    if report.skipped_files:
        typer.echo(f"{len(report.skipped_files)} files already on the remote were left untouched:")
        for f in report.skipped_files:
            typer.echo(f"  {f}")
    pool_dir = pool or f"{remote.path}/pool/<pool-name>"
    typer.echo("on the cluster, from the datadec checkout:")
    for rid in report.pushed:
        typer.echo(f"  {eval_item_command(remote, cell, rid, pool_dir)}")


if __name__ == "__main__":
    app()
