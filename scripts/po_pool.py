"""Pull-based worker pool (po-pool-item/1, see datadec.po.pool): create, submit, status, requeue, add-items.

--pool is the pool directory, ROOT/pool/<name>, in every command.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from datadec.po.pool import (
    DEFAULT_CHUNK_SIZE,
    AttemptStatus,
    Pool,
    PlaceCounts,
    SubmitSettings,
    WorkerClass,
    add_items,
    create,
    requeue_failed,
    requeue_stale,
    status_text,
    submit,
    write_worker_script,
)

app = typer.Typer(no_args_is_help=True)

PoolOpt = Annotated[Path, typer.Option("--pool", help="pool directory, ROOT/pool/<name>")]


def _echo_counts(counts: PlaceCounts) -> None:
    typer.echo(f"pending {counts.pending} done {counts.done} existing {counts.existing}")


@app.command("create")
def create_cmd(
    pool: PoolOpt,
    root: Annotated[Path, typer.Option("--root", help="run root; sweeps under <root>/sweeps")],
    repo: Annotated[Path, typer.Option("--repo", help="datadec checkout: job paths are relative to it; items run in it")],
    worker_class: Annotated[WorkerClass, typer.Option("--class", help="worker class the items need")],
    est_minutes: Annotated[int, typer.Option("--est-minutes", help="per-item estimate for the deadline check")],
    jobs: Annotated[list[Path], typer.Option("--jobs", help="job list JSON {jobs: [...]}; repeatable")],
    chunk_size: Annotated[int, typer.Option("--chunk-size", help="tasks per OLMES process inside a sweep")] = DEFAULT_CHUNK_SIZE,
) -> None:
    """Write (or reuse) the sweeps of one or more job lists and place one item per job; completed sweeps go
    straight to done/ as skipped. Also (re)writes <pool>/status.sh."""
    _echo_counts(create(pool.resolve(), root=root.resolve(), repo=repo.resolve(), worker_class=worker_class,
                        est_minutes=est_minutes, job_lists=jobs, chunk_size=chunk_size))


@app.command("add-items")
def add_items_cmd(
    pool: PoolOpt,
    items: Annotated[Path, typer.Option("--items", help="JSONL of po-pool-item/1 items")],
) -> None:
    """Append already-formed items (e.g. from the GEPA cell writer)."""
    _echo_counts(add_items(pool, items))


@app.command("submit")
def submit_cmd(
    pool: PoolOpt,
    worker_class: Annotated[WorkerClass, typer.Option("--class")],
    workers: Annotated[int, typer.Option("--workers", help="array size: one GPU worker per task")],
    time: Annotated[str, typer.Option("--time", help="walltime per worker, under 23:00:00")] = "22:00:00",
    slots: Annotated[int, typer.Option("--slots", help="items run at once per worker GPU (MPS when > 1)")] = 1,
    mem: Annotated[str, typer.Option("--mem")] = "60G",
    cpus: Annotated[int, typer.Option("--cpus")] = 4,
    do_submit: Annotated[bool, typer.Option("--submit/--no-submit", help="--no-submit only writes the script")] = True,
) -> None:
    """Write sbatch/<class>-<n>.sbatch (a worker array; no throttle, no dependency) and sbatch it."""
    settings = SubmitSettings(cpus=cpus, mem=mem, time=time, slots=slots)
    try:
        script = write_worker_script(Pool(pool.resolve()), worker_class, workers, settings)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    typer.echo(f"wrote {script}")
    if do_submit:
        typer.echo(submit(script))


@app.command("status")
def status_cmd(pool: PoolOpt) -> None:
    """Counts per directory and failed items with their status (same output as <pool>/status.sh)."""
    typer.echo(status_text(Pool(pool)), nl=False)


@app.command("requeue")
def requeue_cmd(
    pool: PoolOpt,
    stale: Annotated[bool, typer.Option("--stale", help="claimed items whose Slurm job is gone (one squeue call)")] = False,
    failed: Annotated[bool, typer.Option("--failed", help="every failed item")] = False,
    status: Annotated[list[AttemptStatus] | None, typer.Option("--status", help="failed items whose last attempt has this status; repeatable")] = None,
) -> None:
    """Move items back to pending."""
    if not (stale or failed or status):
        raise typer.BadParameter("give --stale, --failed and/or --status")
    p = Pool(pool)
    moved: list[str] = []
    if stale:
        moved += requeue_stale(p)
    if failed or status:
        moved += requeue_failed(p, None if failed else status)
    typer.echo(f"requeued {len(moved)}")
    for name in moved:
        typer.echo(f"  {name}")


if __name__ == "__main__":
    app()
