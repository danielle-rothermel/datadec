"""Pool worker: claim items of one class from a pool, run them, record each attempt, exit when nothing fits.

Run as `python -m datadec.po.worker --pool DIR --class C --slots K --time HH:MM:SS` (one per Slurm array task; see
`scripts/po_pool.py submit`). Contract: po-pool-item/1, see datadec.po.pool.

- Deadline: from SLURM_JOB_END_TIME, else `squeue -h -j $SLURM_JOB_ID -o %L`, else (under Slurm, when neither gives
  one) worker start plus the submitted --time; none outside Slurm. The worker log records which source was used.
  A slot claims the next item of its class only when est_minutes * 60 < remaining seconds, and otherwise exits.
- Orphan requeue (claimed items whose job is gone go back to pending) runs on start and before every 10th claim.
  An squeue answer that omits this worker's own job is not trusted, and an squeue error is logged; neither stops
  the worker.
- Each of the `slots` threads claims independently and runs one item at a time on the shared GPU. Pending files
  that fail validation are logged and skipped.
- An item's command is skipped when its completion check already passes (a requeued item whose output exists goes
  straight to its followup). Otherwise it runs without a shell, stdout and stderr to the attempt log, with a
  timeout of the deadline minus 60 s (the whole process group is killed on timeout). Env files are sourced by a
  bash wrapper with `set -a`, so their contents never reach the log or the item file.
- Outcome: done (exit 0, completion check passes, followup if any exits 0), failed (non-zero exit, or an exception
  while running the item; exit_code null, traceback in the worker log), no-output (exit 0, check fails),
  followup-failed, timeout.
- Exit code: 1 when any slot hit an exception, else 0.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import threading
import time
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum, unique
from pathlib import Path
from typing import IO

from datadec.po.pool import (
    Attempt,
    AttemptStatus,
    Pool,
    PoolDir,
    PoolItem,
    SqueueRunner,
    WorkerClass,
    claim,
    finish,
    is_complete,
    requeue_stale,
    run_squeue,
    utc_now,
    walltime_seconds,
)

TIMEOUT_MARGIN_S = 60
KILL_GRACE_S = 30
REQUEUE_EVERY = 10
ENV_FILE_FAILED_RC = 97
# Sources each env file with auto-export, then execs the command. `set +x` keeps an inherited xtrace from echoing them.
ENV_WRAPPER = (
    'set +x; n=$1; shift; for ((i = 0; i < n; i++)); do set -a; source "$1" || exit '
    f'{ENV_FILE_FAILED_RC}; set +a; shift; done; exec "$@"'
)

Clock = Callable[[], float]


@unique
class DeadlineSource(StrEnum):
    JOB_END_TIME = "SLURM_JOB_END_TIME"
    SQUEUE = "squeue"
    WALLTIME = "walltime"
    NONE = "none"


def parse_time_left(text: str) -> float | None:
    """squeue %L ([D-][HH:]MM:SS) in seconds; None for UNLIMITED / NOT_SET / INVALID."""
    text = text.strip()
    if not text or not text[0].isdigit():
        return None
    days, _, hms = text.rpartition("-")
    parts = [int(x) for x in hms.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts
    return float(((int(days or 0) * 24 + h) * 60 + m) * 60 + s)


def remaining_seconds(env: Mapping[str, str], now: float, squeue: SqueueRunner | None,
                      walltime_s: float | None = None) -> tuple[float | None, DeadlineSource]:
    """Seconds left in this Slurm job and where that came from; None (no deadline) only outside Slurm, or under
    Slurm when neither SLURM_JOB_END_TIME, squeue %L nor `walltime_s` gives one. An squeue error counts as no
    answer."""
    end = env.get("SLURM_JOB_END_TIME")
    if end:
        return float(end) - now, DeadlineSource.JOB_END_TIME
    job_id = env.get("SLURM_JOB_ID")
    if not job_id:
        return None, DeadlineSource.NONE
    if squeue is not None:
        try:
            left = parse_time_left(squeue(["squeue", "-h", "-j", job_id, "-o", "%L"]))
        except Exception:
            left = None
        if left is not None:
            return left, DeadlineSource.SQUEUE
    if walltime_s is not None:
        return walltime_s, DeadlineSource.WALLTIME
    return None, DeadlineSource.NONE


def wrap_env_files(argv: list[str], env_files: list[str]) -> list[str]:
    if not env_files:
        return list(argv)
    return ["bash", "-c", ENV_WRAPPER, "po-env", str(len(env_files)), *env_files, *argv]


def _kill_group(proc: subprocess.Popen) -> None:
    for sig, grace in ((signal.SIGTERM, KILL_GRACE_S), (signal.SIGKILL, None)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            continue


def run_logged(argv: list[str], *, cwd: str, env_files: list[str], log: IO[str], timeout: float | None) -> int | None:
    """Run argv (env files sourced by the wrapper) with output appended to `log`; None on timeout."""
    log.write(f"# {utc_now()} cwd={cwd} argv={json.dumps(argv)} env_files={len(env_files)}\n")
    log.flush()
    proc = subprocess.Popen(wrap_env_files(argv, env_files), cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)
    try:
        return proc.wait(timeout=None if timeout is None else max(timeout, 0.0))
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        log.write(f"# {utc_now()} killed at the worker deadline\n")
        return None


@dataclass(frozen=True, slots=True)
class Outcome:
    status: AttemptStatus
    exit_code: int | None


def execute(item: PoolItem, log_path: Path, timeout: Callable[[], float | None]) -> Outcome:
    """Run command (skipped when the completion check already passes), completion check and followup;
    `timeout()` gives the seconds allowed from now."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as log:
        if is_complete(item):
            log.write(f"# {utc_now()} command skipped: complete\n")
            log.flush()
        else:
            rc = run_logged(item.command, cwd=item.cwd, env_files=item.env_files, log=log, timeout=timeout())
            if rc is None:
                return Outcome(AttemptStatus.TIMEOUT, None)
            if rc != 0:
                return Outcome(AttemptStatus.FAILED, rc)
            if not is_complete(item):
                return Outcome(AttemptStatus.NO_OUTPUT, 0)
        if item.followup:
            frc = run_logged(item.followup, cwd=item.followup_cwd or item.cwd, env_files=item.env_files, log=log,
                             timeout=timeout())
            if frc is None:
                return Outcome(AttemptStatus.TIMEOUT, None)
            if frc != 0:
                return Outcome(AttemptStatus.FOLLOWUP_FAILED, frc)
        return Outcome(AttemptStatus.DONE, 0)


class Worker:
    def __init__(self, pool: Pool, worker_class: WorkerClass, *, slots: int = 1, env: Mapping[str, str] = os.environ,
                 clock: Clock = time.time, squeue: SqueueRunner | None = run_squeue, node: str | None = None,
                 walltime_s: float | None = None) -> None:
        if slots < 1:
            raise ValueError("slots must be >= 1")
        self.pool, self.worker_class, self.slots = pool, worker_class, slots
        self.env, self.clock, self.squeue = env, clock, squeue
        self.node = node or socket.gethostname()
        self.job_id = env.get("SLURM_JOB_ID")
        self.array_task_id = env.get("SLURM_ARRAY_TASK_ID")
        left, self.deadline_source = remaining_seconds(env, clock(), squeue, walltime_s)
        self.deadline = None if left is None else clock() + left
        self.claims = 0
        self._requeued_at = 0  # claim count at the last periodic orphan requeue
        self.slot_errors = 0
        self._lock = threading.Lock()
        log_id = (f"{env.get('SLURM_ARRAY_JOB_ID', self.job_id)}_{self.array_task_id or 0}" if self.job_id
                  else f"local-{os.getpid()}")
        self.log_path = pool.dir(PoolDir.WORKERS) / f"{log_id}.log"

    def log(self, msg: str) -> None:
        with self._lock, open(self.log_path, "a") as f:
            f.write(f"{utc_now()} {msg}\n")

    def remaining(self) -> float | None:
        return None if self.deadline is None else self.deadline - self.clock()

    def fits(self, item: PoolItem) -> bool:
        left = self.remaining()
        return left is None or item.est_minutes * 60 < left

    def command_timeout(self) -> float | None:
        left = self.remaining()
        return None if left is None else left - TIMEOUT_MARGIN_S

    def requeue_orphans(self) -> None:
        """Requeue orphans; an squeue error or an answer omitting this worker's own job is logged, not raised."""
        if self.squeue is None:
            self.log("orphan requeue skipped: no squeue")
            return
        try:
            moved = requeue_stale(self.pool, self.squeue, must_contain=self.job_id)
        except Exception as e:
            self.log(f"orphan requeue failed: {e!r}")
            return
        if moved:
            self.log(f"requeued orphans: {' '.join(moved)}")

    def claim_next(self) -> tuple[Path, PoolItem] | None:
        with self._lock:
            due = self.claims - self._requeued_at >= REQUEUE_EVERY
            if due:
                self._requeued_at = self.claims
        if due:
            self.requeue_orphans()
        attempt = Attempt(job_id=self.job_id, array_task_id=self.array_task_id, node=self.node, started_utc=utc_now())
        got = claim(self.pool, self.worker_class, attempt, self.fits,
                    on_invalid=lambda f, e: self.log(f"skipped invalid pending file {f.name}: {e!r}"))
        if got is not None:
            with self._lock:
                self.claims += 1
        return got

    def run_item(self, slot: int, path: Path, item: PoolItem) -> None:
        """Execute one claimed item and finish it; an exception while executing finishes it as failed."""
        attempt = item.open_attempt
        assert attempt is not None and attempt.log is not None
        self.log(f"slot {slot} claimed {path.name}")
        t0 = self.clock()
        try:
            outcome = execute(item, Path(attempt.log), self.command_timeout)
        except Exception:
            self.log(f"slot {slot} exception running {path.name}:\n{traceback.format_exc()}")
            with self._lock:
                self.slot_errors += 1
            outcome = Outcome(AttemptStatus.FAILED, None)
        dest = finish(self.pool, path, item, outcome.status, exit_code=outcome.exit_code,
                      duration_s=self.clock() - t0)
        self.log(f"slot {slot} {outcome.status} {path.name} -> {dest.parent.name}")

    def run_slot(self, slot: int) -> None:
        try:
            while (got := self.claim_next()) is not None:
                self.run_item(slot, *got)
        except Exception:
            self.log(f"slot {slot} stopped by an exception:\n{traceback.format_exc()}")
            with self._lock:
                self.slot_errors += 1
            return
        self.log(f"slot {slot} exits (no {self.worker_class} item pending that fits; remaining={self.remaining()})")

    def run(self) -> int:
        self.log(f"start class={self.worker_class} slots={self.slots} node={self.node} job={self.job_id} "
                 f"remaining={self.remaining()} deadline_source={self.deadline_source}")
        self.requeue_orphans()
        threads = [threading.Thread(target=self.run_slot, args=(i,), name=f"slot-{i}") for i in range(self.slots)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.log(f"exit after {self.claims} claims, {self.slot_errors} slot exceptions")
        return 1 if self.slot_errors else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Claim and run pool items of one class until nothing fits.")
    ap.add_argument("--pool", type=Path, required=True, help="pool directory (ROOT/pool/<name>)")
    ap.add_argument("--class", dest="worker_class", required=True, choices=[c.value for c in WorkerClass])
    ap.add_argument("--slots", type=int, default=1, help="items run at once on this worker's GPU")
    ap.add_argument("--time", dest="walltime", default=None,
                    help="submitted walltime [D-]HH:MM:SS; the deadline fallback when Slurm gives none")
    args = ap.parse_args(argv)
    squeue = run_squeue if shutil.which("squeue") else None
    walltime_s = None if args.walltime is None else float(walltime_seconds(args.walltime))
    worker = Worker(Pool(args.pool), WorkerClass(args.worker_class), slots=args.slots, squeue=squeue,
                    walltime_s=walltime_s)
    return worker.run()


if __name__ == "__main__":
    raise SystemExit(main())
