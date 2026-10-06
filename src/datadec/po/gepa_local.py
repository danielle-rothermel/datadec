"""Mac (mps) execution lane for GEPA cells (gepa-run-contract, po-gepa/2).

Small-model GEPA searches can run on the Mac while Test evaluation stays on the cluster. The lane uses the cluster
layout under a local root (``LOCALROOT/gepa/<cell_id>/<run_id>.job.json`` and ``.../<run_id>/``) so a finished run
directory moves to the cluster unchanged:

- write: the cell's 6 job files with device ``mps`` and no pool items (``gepa_cells.write_cell(pool_dir=None)``).
- run: every job without ``result.json`` through the fork driver (``mise exec -- uv run local/gepa_arc.py --job J``
  from the OLMES checkout; mise supplies the OpenRouter key), up to ``parallel`` at once, in job order (empty
  starts, then ape; seeds in order). Each attempt appends its output to ``<cell dir>/<run_id>.local.log`` and
  records start, end, exit code (null plus ``error`` when the attempt raised), duration and the launching pid in
  ``<cell dir>/<run_id>.local.json``. A job
  whose record has no end and whose launching process is still alive counts as running and is not started again.
- status: per cell counts of jobs, finished (``result.json``), failed (ended without ``result.json``), running,
  stale (no end, launching process gone) and pending, plus each run's proposal count and best val score.
- push: rsync the finished runs of one cell (job file and run dir; never the ``.local.*`` files or unfinished runs)
  to ``<remote-root>/gepa/<cell_id>/`` with ``--ignore-existing``, so nothing on the cluster is overwritten; files
  already on the remote are listed first and reported as skipped. The operator then runs ``po_gepa_cells.py
  eval-item`` on the cluster for each pushed run, which writes the cuda Test eval sweep and its pool item.

Job files keep the Mac paths in ``run_dir`` and ``subset_paths``; the cluster eval follow-up and aggregation read
neither (they take the run directory they are given and the repo subset paths).
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shlex
import subprocess
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum, unique
from pathlib import Path

from datadec.po.gepa_cells import Start, proposal_count

LOCAL_DEVICE = "mps"
DEFAULT_LOCAL_ROOT = Path.home() / "drotherm" / "data" / "runs" / "po" / "mac-gepa"
DEFAULT_OLMES_REPO = Path.home() / "drotherm" / "repos" / "olmes"
DEFAULT_REMOTE_ROOT = "torch:/scratch/ddr8143/drotherm/data/runs/po/cluster-20261001"
DEFAULT_PARALLEL = 2
JOB_SUFFIX = ".job.json"
LOG_SUFFIX = ".local.log"
RECORD_SUFFIX = ".local.json"
RESULT_FILE = "result.json"
DRIVER = "local/gepa_arc.py"

START_ORDER = {str(s): i for i, s in enumerate(Start)}


@unique
class RunState(StrEnum):
    FINISHED = "finished"
    FAILED = "failed"
    RUNNING = "running"
    STALE = "stale"
    PENDING = "pending"


@dataclass(frozen=True, slots=True)
class LocalJob:
    """One job file of the lane and the paths derived from it."""

    job_file: Path
    run_id: str
    start: str
    seed: int

    @property
    def cell_dir(self) -> Path:
        return self.job_file.parent

    @property
    def run_dir(self) -> Path:
        return self.cell_dir / self.run_id

    @property
    def log(self) -> Path:
        return self.cell_dir / f"{self.run_id}{LOG_SUFFIX}"

    @property
    def record(self) -> Path:
        return self.cell_dir / f"{self.run_id}{RECORD_SUFFIX}"

    @property
    def result(self) -> Path:
        return self.run_dir / RESULT_FILE


def cell_jobs(cell_dir: Path) -> list[LocalJob]:
    """The cell's jobs in writer order: empty starts first, then ape; seeds ascending."""
    jobs = []
    for path in cell_dir.glob(f"*{JOB_SUFFIX}"):
        job = json.loads(path.read_text())
        jobs.append(LocalJob(path, path.name.removesuffix(JOB_SUFFIX), job["start"], int(job["seed"])))
    return sorted(jobs, key=lambda j: (START_ORDER[j.start], j.seed))


def cell_dirs(root: Path, cells: Sequence[str] = ()) -> list[Path]:
    """The named cells under root/gepa (every cell, sorted, when none are named)."""
    gepa = Path(root) / "gepa"
    if not cells:
        return sorted(p for p in gepa.iterdir() if p.is_dir()) if gepa.is_dir() else []
    missing = [c for c in cells if not (gepa / c).is_dir()]
    if missing:
        raise FileNotFoundError(f"no such cell under {gepa}: {', '.join(missing)}")
    return [gepa / c for c in cells]


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_record(job: LocalJob) -> dict | None:
    return json.loads(job.record.read_text()) if job.record.exists() else None


def run_state(job: LocalJob, pid_alive: Callable[[int], bool] = _pid_alive) -> RunState:
    if job.result.exists():
        return RunState.FINISHED
    record = read_record(job)
    if record is None:
        return RunState.PENDING
    if record.get("ended_utc") is None:
        return RunState.RUNNING if pid_alive(int(record["pid"])) else RunState.STALE
    return RunState.FAILED


# --- run ------------------------------------------------------------------------------------------------------

Runner = Callable[[list[str], Path, Path], int]  # (argv, cwd, log path) -> exit code


def driver_argv(job: LocalJob) -> list[str]:
    return ["mise", "exec", "--", "uv", "run", DRIVER, "--job", str(job.job_file)]


def subprocess_runner(argv: list[str], cwd: Path, log: Path) -> int:
    """Run argv without a shell, stdout and stderr appended to log. VIRTUAL_ENV is dropped so the driver's ``uv run``
    uses the OLMES project environment rather than the caller's."""
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    with log.open("a") as out:
        return subprocess.run(argv, cwd=cwd, stdout=out, stderr=subprocess.STDOUT, env=env, check=False).returncode


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _write_record(job: LocalJob, record: dict) -> None:
    tmp = job.record.with_name(f".{job.record.name}.tmp")
    tmp.write_text(json.dumps(record, indent=1) + "\n")
    os.replace(tmp, job.record)


@dataclass(frozen=True, slots=True)
class RunOutcome:
    run_id: str
    exit_code: int | None
    finished: bool  # result.json present after the attempt

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and self.finished


def run_one(job: LocalJob, olmes_repo: Path, runner: Runner = subprocess_runner) -> RunOutcome:
    argv = driver_argv(job)
    record = {"run_id": job.run_id, "job": str(job.job_file), "command": argv, "cwd": str(olmes_repo),
              "log": str(job.log), "pid": os.getpid(), "started_utc": _now(), "ended_utc": None, "exit_code": None,
              "duration_s": None}
    _write_record(job, record)
    t0 = time.monotonic()
    exit_code: int | None = None
    try:
        exit_code = runner(argv, olmes_repo, job.log)
    except Exception as exc:  # the attempt failed to launch or crashed; recorded, and the other jobs go on
        record["error"] = f"{type(exc).__name__}: {exc}"
    record |= {"ended_utc": _now(), "exit_code": exit_code, "duration_s": round(time.monotonic() - t0, 1),
               "finished": job.result.exists()}
    _write_record(job, record)
    return RunOutcome(job.run_id, exit_code, job.result.exists())


@dataclass(frozen=True, slots=True)
class RunPlan:
    todo: list[LocalJob]
    finished: list[LocalJob]
    running: list[LocalJob]


def plan_runs(cells: Sequence[Path], pid_alive: Callable[[int], bool] = _pid_alive) -> RunPlan:
    """Jobs to start in order (pending, failed and stale ones), plus those skipped as finished or running."""
    todo, finished, running = [], [], []
    for cell in cells:
        for job in cell_jobs(cell):
            state = run_state(job, pid_alive)
            if state is RunState.FINISHED:
                finished.append(job)
            elif state is RunState.RUNNING:
                running.append(job)
            else:
                todo.append(job)
    return RunPlan(todo, finished, running)


def run_jobs(jobs: Sequence[LocalJob], olmes_repo: Path, parallel: int = DEFAULT_PARALLEL,
             runner: Runner = subprocess_runner) -> list[RunOutcome]:
    """Run the jobs, at most ``parallel`` at once, starting them in the given order; outcomes in that order."""
    if parallel < 1:
        raise ValueError("parallel must be at least 1")
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = [pool.submit(run_one, job, olmes_repo, runner) for job in jobs]
        return [f.result() for f in futures]


# --- status ---------------------------------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class RunStatus:
    run_id: str
    state: RunState
    proposals: int | None
    best_val_score: float | None


def _proposals_so_far(job: LocalJob) -> int | None:
    try:
        return proposal_count(job.run_dir)
    except json.JSONDecodeError:  # a row being appended right now
        return None


def cell_status(cell: Path, pid_alive: Callable[[int], bool] = _pid_alive) -> list[RunStatus]:
    rows = []
    for job in cell_jobs(cell):
        state = run_state(job, pid_alive)
        result = json.loads(job.result.read_text()) if state is RunState.FINISHED else {}
        rows.append(RunStatus(job.run_id, state, _proposals_so_far(job), result.get("best_val_score")))
    return rows


# --- push -----------------------------------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Remote:
    host: str
    path: str

    @classmethod
    def parse(cls, spec: str) -> Remote:
        host, sep, path = spec.partition(":")
        if not sep or not host or not path.startswith("/"):
            raise ValueError(f"remote root must be host:/absolute/path, got {spec!r}")
        return cls(host, path.rstrip("/"))

    def cell_path(self, cell: str) -> str:
        return f"{self.path}/gepa/{cell}"


CommandRunner = Callable[[list[str]], subprocess.CompletedProcess]


def run_command(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, check=False)


def _files_to_push(job: LocalJob) -> list[str]:
    files = [job.job_file.name]
    files += [str(p.relative_to(job.cell_dir)) for p in sorted(job.run_dir.rglob("*")) if p.is_file()]
    return [f for f in files if not f.endswith((LOG_SUFFIX, RECORD_SUFFIX))]


def remote_list_argv(remote: Remote, cell: str) -> list[str]:
    """ssh argv creating the remote cell dir (rsync here cannot) and listing its files relative to it."""
    path = shlex.quote(remote.cell_path(cell))
    return ["ssh", remote.host, f"mkdir -p {path} && cd {path} && find . -type f"]


def rsync_argv(cell_dir: Path, run_ids: Sequence[str], remote: Remote) -> list[str]:
    includes = [arg for rid in run_ids for arg in (f"--include=/{rid}{JOB_SUFFIX}", f"--include=/{rid}/", f"--include=/{rid}/**")]
    return ["rsync", "-a", "--ignore-existing", f"--exclude=*{LOG_SUFFIX}", f"--exclude=*{RECORD_SUFFIX}", *includes,
            "--exclude=*", f"{cell_dir}/", f"{remote.host}:{remote.cell_path(cell_dir.name)}/"]


def eval_item_command(remote: Remote, cell: str, rid: str, pool: str) -> str:
    return (f"uv run python scripts/po_gepa_cells.py eval-item --run-dir {remote.cell_path(cell)}/{rid} "
            f"--pool {pool}")


@dataclass(frozen=True, slots=True)
class PushReport:
    pushed: list[str]  # finished run ids sent
    not_finished: list[str]  # run ids left on the Mac
    skipped_files: list[str]  # cell-relative files already on the remote, left untouched
    rsync: list[str]  # the rsync argv run (empty when nothing was finished)


def push_cell(root: Path, cell: str, remote: Remote, run_cmd: CommandRunner = run_command) -> PushReport:
    """Send the finished runs of one cell to the remote cell dir without overwriting anything there."""
    cell_dir = cell_dirs(root, [cell])[0]
    jobs = cell_jobs(cell_dir)
    done = [j for j in jobs if j.result.exists()]
    not_finished = [j.run_id for j in jobs if not j.result.exists()]
    if not done:
        return PushReport([], not_finished, [], [])
    listing = run_cmd(remote_list_argv(remote, cell))
    if listing.returncode != 0:
        raise RuntimeError(f"listing {remote.host}:{remote.cell_path(cell)} failed: {listing.stderr.strip()}")
    present = {line.removeprefix("./") for line in listing.stdout.splitlines() if line.strip()}
    skipped = sorted(f for j in done for f in _files_to_push(j) if f in present)
    argv = rsync_argv(cell_dir, [j.run_id for j in done], remote)
    sent = run_cmd(argv)
    if sent.returncode != 0:
        raise RuntimeError(f"rsync failed ({sent.returncode}): {sent.stderr.strip()}")
    return PushReport([j.run_id for j in done], not_finished, skipped, argv)
