"""Pull-based work pool on a shared filesystem: items move between directories, workers claim by atomic rename.

Contract: po-pool-item/1 (packet ~/drotherm/data/.claude/datadec/2026-10-06/0929-infra-build/pool-item-contract.md).

A pool directory holds pool.json and one subdirectory per state (pending, claimed, done, failed) plus logs/,
workers/ and sbatch/. Each item is one JSON file `<NNNN>-<name>.json`; the zero-padded sequence is creation order,
which is also claim (priority) order. Filenames never change across moves. Every state change appends or closes an
attempt inside the item file. Writes go to a hidden temp file and are renamed into place, and moves that also
rewrite the item first rename the file to a hidden staging name, so concurrent requeuers cannot duplicate an item
and workers never read a half-written file (workers and status.sh only look at `*.json`).

Sweep items come from job lists: {"jobs": [{"name", "model", "revision"?, "subsets": [...], "task"?,
"formulations"?, "pairs"?, "formats"?, "instructions"?, "num_shots"?, "batch_size"?, "dtype"?, "quant"? (int8|nf4,
with dtype bfloat16), "use_cache"?}, ...]}. Each job becomes one sweep directory under <root>/sweeps/<name>
(reused when it exists, since run_sweep resumes chunk-wise, after checking its sweep.json matches the job) and one
item running scripts/po_run_sweep.py. A sweep that an old Slurm array is still running is incomplete and so goes to
pending, where a pool worker would run it a second time in parallel: cancel such arrays before `create`.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import time
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum, unique
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from datadec.po.ape import load_instructions
from datadec.po.formats import CANONICAL, load_formats
from datadec.po.sweep import SweepSpec, load_pairs, sweep_complete, write_sweep

ITEM_SCHEMA = "po-pool-item/1"
POOL_SCHEMA = "po-pool/1"
POOL_FILE = "pool.json"
STATUS_SCRIPT = "status.sh"
SQUEUE_USER = "ddr8143"
DEFAULT_CHUNK_SIZE = 14
MAX_SEQ = 9999
STRANDED_AFTER_S = 600.0  # `requeue --stale` recovers claimed items without an open attempt older than this
STAGING_TAG = "staging"

SqueueRunner = Callable[[list[str]], str]
"""Runs one squeue argv and returns its stdout (injected in tests)."""


@unique
class PoolDir(StrEnum):
    PENDING = "pending"
    CLAIMED = "claimed"
    DONE = "done"
    FAILED = "failed"
    LOGS = "logs"
    WORKERS = "workers"
    SBATCH = "sbatch"


ITEM_DIRS = (PoolDir.PENDING, PoolDir.CLAIMED, PoolDir.DONE, PoolDir.FAILED)


@unique
class ItemKind(StrEnum):
    SWEEP = "sweep"
    GEPA = "gepa"
    EVAL = "eval"


@unique
class WorkerClass(StrEnum):
    SMALL = "small"
    BIG = "big"
    B32 = "32b"
    GEPA = "gepa"


@unique
class AttemptStatus(StrEnum):
    DONE = "done"
    SKIPPED = "skipped"
    FAILED = "failed"
    NO_OUTPUT = "no-output"
    FOLLOWUP_FAILED = "followup-failed"
    TIMEOUT = "timeout"
    ORPHANED = "orphaned"


class SweepCompletion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    type: Literal["sweep"] = "sweep"
    sweep_dir: str


class FileCompletion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    type: Literal["file"] = "file"
    path: str


Completion = Annotated[SweepCompletion | FileCompletion, Field(discriminator="type")]


class Attempt(BaseModel):
    """One try at an item; open (status None) while a worker owns it."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    job_id: str | None = None
    array_task_id: str | None = None
    node: str | None = None
    started_utc: str | None = None
    ended_utc: str | None = None
    exit_code: int | None = None
    duration_s: float | None = None
    log: str | None = None
    status: AttemptStatus | None = None


class PoolItem(BaseModel):
    """The item file (po-pool-item/1). `followup_cwd` defaults to `cwd`."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)
    schema_: Literal["po-pool-item/1"] = Field(default=ITEM_SCHEMA, alias="schema")
    name: str
    kind: ItemKind
    worker_class: WorkerClass = Field(alias="class")
    est_minutes: int
    cwd: str
    command: list[str]
    followup: list[str] | None = None
    followup_cwd: str | None = None
    env_files: list[str] = Field(default_factory=list)
    complete: Completion
    meta: dict = Field(default_factory=dict)
    created_utc: str
    attempts: list[Attempt] = Field(default_factory=list)

    def to_json(self) -> str:
        data = self.model_dump(mode="json", by_alias=True)
        data["attempts"] = [a.model_dump(mode="json", exclude_none=True) for a in self.attempts]
        return json.dumps(data, indent=1) + "\n"

    @classmethod
    def load(cls, path: Path) -> PoolItem:
        return cls.model_validate_json(Path(path).read_text())

    @property
    def open_attempt(self) -> Attempt | None:
        if self.attempts and self.attempts[-1].status is None:
            return self.attempts[-1]
        return None

    @property
    def last_status(self) -> AttemptStatus | None:
        return self.attempts[-1].status if self.attempts else None

    def with_attempt(self, attempt: Attempt) -> PoolItem:
        return self.model_copy(update={"attempts": [*self.attempts, attempt]})

    def close_attempt(self, status: AttemptStatus, *, ended_utc: str | None = None, **fields) -> PoolItem:
        """Close the open attempt with `status`; with no open attempt, append a closed one."""
        if self.open_attempt is None:
            return self.with_attempt(Attempt(status=status, ended_utc=ended_utc, **fields))
        closed = self.attempts[-1].model_copy(update={"status": status, "ended_utc": ended_utc, **fields})
        return self.model_copy(update={"attempts": [*self.attempts[:-1], closed]})


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def is_complete(item: PoolItem) -> bool:
    c = item.complete
    if isinstance(c, SweepCompletion):
        return (Path(c.sweep_dir) / "tasks.jsonl").exists() and sweep_complete(Path(c.sweep_dir))
    p = Path(c.path)
    return p.is_file() and p.stat().st_size > 0


# ---------------------------------------------------------------- layout and file moves


@dataclass(frozen=True, slots=True)
class Pool:
    path: Path

    def dir(self, d: PoolDir) -> Path:
        return self.path / d

    @property
    def meta(self) -> dict:
        return json.loads((self.path / POOL_FILE).read_text())

    def items(self, d: PoolDir) -> list[Path]:
        return sorted(self.dir(d).glob("*.json"))

    def names(self) -> set[str]:
        return {_name_of(p) for d in ITEM_DIRS for p in self.items(d)}

    def next_seq(self) -> int:
        seqs = [int(p.name[:4]) for d in ITEM_DIRS for p in self.items(d) if p.name[:4].isdigit()]
        return max(seqs, default=0) + 1

    def log_path(self, item_file: Path, attempt_no: int) -> Path:
        return self.dir(PoolDir.LOGS) / f"{item_file.stem}.{attempt_no}.log"


def _name_of(item_file: Path) -> str:
    return item_file.stem.split("-", 1)[1]


def _hidden(path: Path, tag: str) -> Path:
    return path.parent / f".{path.name}.{uuid.uuid4().hex}.{tag}"


def write_atomic(path: Path, text: str) -> None:
    tmp = _hidden(path, "tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def move_item(src: Path, dest_dir: Path, item: PoolItem | None = None) -> Path | None:
    """Move src into dest_dir (same filename), optionally rewriting it with `item` on the way.

    The file is first renamed to a hidden staging name (atomic; a FileNotFoundError means another process moved it
    and None is returned), rewritten there, then renamed into dest_dir.
    """
    stage = _hidden(src, STAGING_TAG)
    try:
        os.rename(src, stage)
    except FileNotFoundError:
        return None
    if item is not None:
        stage.write_text(item.to_json())
    dest = dest_dir / src.name
    os.rename(stage, dest)
    return dest


def init_pool(path: Path, *, root: Path, repo: Path) -> Pool:
    """Create the layout and pool.json, or check an existing pool was made for the same root and repo."""
    pool = Pool(Path(path))
    meta_file = pool.path / POOL_FILE
    if meta_file.exists():
        meta = pool.meta
        if (meta["root"], meta["repo"]) != (str(root), str(repo)):
            raise ValueError(f"{pool.path} was created for root={meta['root']} repo={meta['repo']}")
    else:
        pool.path.mkdir(parents=True, exist_ok=True)
        meta = {"schema": POOL_SCHEMA, "name": pool.path.name, "created_utc": utc_now(), "root": str(root),
                "repo": str(repo), "settings": {"squeue_user": SQUEUE_USER}}
        write_atomic(meta_file, json.dumps(meta, indent=1) + "\n")
    for d in PoolDir:
        pool.dir(d).mkdir(exist_ok=True)
    status_sh = pool.path / STATUS_SCRIPT
    status_sh.write_text(status_script(pool.path))
    status_sh.chmod(0o755)
    return pool


# ---------------------------------------------------------------- create / add-items


@dataclass(frozen=True, slots=True)
class PlaceCounts:
    pending: int = 0
    done: int = 0
    existing: int = 0


def place_items(pool: Pool, items: Sequence[PoolItem]) -> PlaceCounts:
    """Write new items in order: to done/ with a "skipped" attempt when their completion check already passes,
    else to pending/. Items whose name is already in the pool are left alone (counted as existing)."""
    seen = pool.names()
    batch = [i.name for i in items]
    dupes = {n for n in batch if batch.count(n) > 1}
    if dupes:
        raise ValueError(f"duplicate item names: {sorted(dupes)}")
    seq = pool.next_seq()
    pending = done = existing = 0
    for item in items:
        if item.name in seen:
            existing += 1
            continue
        if seq > MAX_SEQ:
            raise ValueError(f"pool sequence exceeds {MAX_SEQ}")
        fname = f"{seq:04d}-{item.name}.json"
        seq += 1
        if is_complete(item):
            write_atomic(pool.dir(PoolDir.DONE) / fname,
                         item.with_attempt(Attempt(status=AttemptStatus.SKIPPED)).to_json())
            done += 1
        else:
            write_atomic(pool.dir(PoolDir.PENDING) / fname, item.to_json())
            pending += 1
    return PlaceCounts(pending=pending, done=done, existing=existing)


def build_specs(jobs: list[dict], repo: Path) -> list[SweepSpec]:
    """Sweep specs for a job list (paths in a job are relative to the datadec checkout `repo`)."""
    specs = []
    for j in jobs:
        specs.append(SweepSpec(
            name=j["name"], subset_paths=tuple(repo / s for s in j["subsets"]), model=j["model"], revision=j.get("revision"),
            task=j.get("task", "arc_easy"),
            formulations=tuple(j.get("formulations", ["rc"])),
            formats=tuple(load_formats(repo / j["formats"])) if j.get("formats") else (dict(CANONICAL),),
            instructions=tuple(load_instructions(repo / j["instructions"])) if j.get("instructions") else (),
            pairs=tuple(load_pairs(repo / j["pairs"])) if j.get("pairs") else (),
            num_shots=int(j.get("num_shots", 5)), device="cuda", batch_size=int(j.get("batch_size", 64)),
            dtype=j.get("dtype", "float32"), quant=j.get("quant"), use_cache=bool(j.get("use_cache", True)),
        ))
    return specs


def sweep_item(spec: SweepSpec, sweep_dir: Path, *, repo: Path, worker_class: WorkerClass, est_minutes: int,
               chunk_size: int, job: dict) -> PoolItem:
    return PoolItem(
        name=spec.name, kind=ItemKind.SWEEP, worker_class=worker_class, est_minutes=est_minutes, cwd=str(repo),
        command=["uv", "run", "python", "scripts/po_run_sweep.py", "--sweep-dir", str(sweep_dir),
                 "--chunk-size", str(chunk_size)],
        complete=SweepCompletion(sweep_dir=str(sweep_dir)),
        meta={"model": spec.model, "revision": spec.revision, "task": spec.task, "subsets": list(job["subsets"]),
              "batch_size": spec.batch_size, "dtype": spec.dtype, "quant": spec.quant},
        created_utc=utc_now(),
    )


def sweep_mismatches(spec: SweepSpec, sweep_dir: Path) -> list[str]:
    """Fields where an existing sweep.json differs from `spec` (model, revision, task, batch size, dtype, quant, and
    the subset files by content hash when the sweep recorded them)."""
    have = json.loads((sweep_dir / "sweep.json").read_text())
    diffs = [f"{f}: sweep has {have.get(f)!r}, job has {getattr(spec, f)!r}"
             for f in ("model", "revision", "task", "batch_size", "dtype", "quant") if have.get(f) != getattr(spec, f)]
    if "subsets" in have:
        want = [hashlib.sha1(Path(p).read_bytes()).hexdigest() for p in spec.subset_paths]
        if [s["sha1"] for s in have["subsets"]] != want:
            diffs.append("subsets: subset files differ from the sweep's")
    return diffs


def create(pool_path: Path, *, root: Path, repo: Path, worker_class: WorkerClass, est_minutes: int,
           job_lists: Iterable[Path], chunk_size: int = DEFAULT_CHUNK_SIZE) -> PlaceCounts:
    """Write (or reuse) every job's sweep under <root>/sweeps and place one sweep item per job.

    An existing sweep dir is reused only when its sweep.json matches the job (see sweep_mismatches); otherwise
    nothing is placed and a ValueError names the differences.
    """
    jobs = [j for jl in job_lists for j in json.loads(Path(jl).read_text())["jobs"]]
    specs = build_specs(jobs, repo)
    mismatched = {s.name: d for s in specs if (root / "sweeps" / s.name).exists()
                  and (d := sweep_mismatches(s, root / "sweeps" / s.name))}
    if mismatched:
        raise ValueError("existing sweeps differ from their jobs: " + "; ".join(
            f"{name} ({', '.join(d)})" for name, d in mismatched.items()))
    pool = init_pool(pool_path, root=root, repo=repo)
    items = []
    for spec, job in zip(specs, jobs):
        sweep_dir = root / "sweeps" / spec.name
        if not sweep_dir.exists():
            write_sweep(spec, root=root / "sweeps")
        items.append(sweep_item(spec, sweep_dir, repo=repo, worker_class=worker_class, est_minutes=est_minutes,
                                chunk_size=chunk_size, job=job))
    return place_items(pool, items)


def add_items(pool_path: Path, items_file: Path) -> PlaceCounts:
    """Place already-formed items from a JSONL file into an existing pool."""
    pool = Pool(Path(pool_path))
    if not (pool.path / POOL_FILE).exists():
        raise FileNotFoundError(f"{pool.path / POOL_FILE} missing; create the pool first")
    lines = Path(items_file).read_text().splitlines()
    return place_items(pool, [PoolItem.model_validate_json(line) for line in lines if line.strip()])


# ---------------------------------------------------------------- claim / finish


def claim(pool: Pool, worker_class: WorkerClass, attempt: Attempt,
          fits: Callable[[PoolItem], bool] = lambda _: True,
          on_invalid: Callable[[Path, Exception], None] | None = None) -> tuple[Path, PoolItem] | None:
    """Claim the first pending item of `worker_class` (lexical order) by renaming it into claimed/.

    Returns None when no item of the class is pending, or when the first one does not `fit` (the deadline check):
    claim order is priority order, so a worker never skips ahead. A FileNotFoundError on read or rename means
    another worker won that item; the loop moves on to the next file. A pending file that fails to parse or
    validate is left in place, reported to `on_invalid`, and skipped.
    """
    for src in pool.items(PoolDir.PENDING):
        try:
            item = PoolItem.load(src)
        except FileNotFoundError:
            continue
        except ValueError as e:  # pydantic ValidationError (bad JSON included) and UnicodeDecodeError
            if on_invalid is not None:
                on_invalid(src, e)
            continue
        if item.worker_class != worker_class:
            continue
        if not fits(item):
            return None
        dest = pool.dir(PoolDir.CLAIMED) / src.name
        try:
            os.rename(src, dest)
        except FileNotFoundError:
            continue
        item = PoolItem.load(dest)
        log = pool.log_path(dest, len(item.attempts) + 1)
        item = item.with_attempt(attempt.model_copy(update={"log": str(log)}))
        write_atomic(dest, item.to_json())
        return dest, item
    return None


def finish(pool: Pool, claimed_file: Path, item: PoolItem, status: AttemptStatus, *, exit_code: int | None,
           duration_s: float) -> Path:
    """Close the open attempt and move the item to done/ (status done) or failed/ (anything else)."""
    item = item.close_attempt(status, ended_utc=utc_now(), exit_code=exit_code, duration_s=round(duration_s, 1))
    write_atomic(claimed_file, item.to_json())
    dest_dir = pool.dir(PoolDir.DONE if status == AttemptStatus.DONE else PoolDir.FAILED)
    dest = dest_dir / claimed_file.name
    os.rename(claimed_file, dest)
    return dest


# ---------------------------------------------------------------- requeue


def run_squeue(argv: list[str]) -> str:
    return subprocess.run(argv, capture_output=True, text=True, check=True).stdout


def live_job_ids(user: str, runner: SqueueRunner = run_squeue) -> set[str]:
    """Job ids (one per array element) of every queued or running job of `user`, from one squeue call."""
    out = runner(["squeue", "-h", "-u", user, "-o", "%A", "-r"])
    return {line.strip() for line in out.splitlines() if line.strip()}


class UntrustedSqueueError(RuntimeError):
    """squeue output omitted a job known to be alive, so it cannot decide which jobs are gone."""


def requeue_stale(pool: Pool, runner: SqueueRunner = run_squeue, *, must_contain: str | None = None,
                  stranded_after_s: float | None = None) -> list[str]:
    """Move claimed items whose worker job is gone back to pending, closing the open attempt as orphaned.

    Items with an open attempt are moved when its job id is missing from squeue. With `must_contain` (the calling
    worker's own job id), an squeue answer that omits it raises UntrustedSqueueError and nothing moves. Items with
    no open attempt are normally left alone (a just-claimed file has none until its worker writes it); with
    `stranded_after_s` (operator `requeue --stale`), those whose file is older than that are moved too, with an
    orphaned attempt appended. Returns the moved filenames.
    """
    open_items, to_move = [], []
    now = time.time()
    for f in pool.items(PoolDir.CLAIMED):
        try:
            item = PoolItem.load(f)
            age = now - f.stat().st_mtime
        except FileNotFoundError:
            continue
        if (attempt := item.open_attempt) is not None:
            open_items.append((f, item, attempt.job_id))
        elif stranded_after_s is not None and age > stranded_after_s:
            to_move.append((f, item))
    if open_items:
        user = pool.meta.get("settings", {}).get("squeue_user", SQUEUE_USER)
        live = live_job_ids(user, runner)
        if must_contain is not None and must_contain not in live:
            raise UntrustedSqueueError(f"squeue output omits this worker's own job {must_contain}")
        to_move += [(f, item) for f, item, job_id in open_items if job_id not in live]
    moved = []
    for f, item in sorted(to_move, key=lambda fi: fi[0].name):
        closed = item.close_attempt(AttemptStatus.ORPHANED, ended_utc=utc_now())
        if move_item(f, pool.dir(PoolDir.PENDING), closed) is not None:
            moved.append(f.name)
    return moved


def requeue_failed(pool: Pool, statuses: Iterable[AttemptStatus] | None = None) -> list[str]:
    """Move failed items back to pending; with `statuses`, only those whose last attempt has one of them."""
    wanted = set(statuses) if statuses else None
    moved = []
    for f in pool.items(PoolDir.FAILED):
        try:
            item = PoolItem.load(f)
        except FileNotFoundError:
            continue
        if wanted is not None and item.last_status not in wanted:
            continue
        if move_item(f, pool.dir(PoolDir.PENDING)) is not None:
            moved.append(f.name)
    return moved


# ---------------------------------------------------------------- status


def status_text(pool: Pool) -> str:
    """Counts per item directory, each failed item with its last attempt status, then (only when there are any)
    claimed items without an open attempt and leftover staging files from interrupted moves. Same output as
    status.sh."""
    lines = [f"{d} {len(pool.items(d))}" for d in ITEM_DIRS]
    lines.append("failed items:")
    for f in pool.items(PoolDir.FAILED):
        try:
            status = PoolItem.load(f).last_status
        except FileNotFoundError:
            continue
        lines.append(f"{f.stem} {status or ''}")
    no_attempt = []
    for f in pool.items(PoolDir.CLAIMED):
        try:
            if PoolItem.load(f).open_attempt is None:
                no_attempt.append(f.stem)
        except FileNotFoundError:
            continue
    if no_attempt:
        lines += ["claimed without open attempt:", *no_attempt]
    staging = sorted(f"{d}/{p.name}" for d in ITEM_DIRS for p in pool.dir(d).glob(f".*.{STAGING_TAG}"))
    if staging:
        lines += ["staging leftovers:", *staging]
    return "\n".join(lines) + "\n"


def status_script(pool_path: Path) -> str:
    """Shell-only status for login nodes. The last "status" key in an item file is its last attempt's status,
    because attempts is the item's final key; an attempt is open when its object (the last "{" after "attempts")
    has no "status" key."""
    dirs = " ".join(str(d) for d in ITEM_DIRS)
    return f"""#!/bin/bash
# Pool status without Python: counts per directory, then failed items with their last attempt status.
export LC_ALL=C
P={pool_path}
for d in {dirs}; do
  n=$(find "$P/$d" -maxdepth 1 -name '[!.]*.json' | wc -l | tr -d ' ')
  echo "$d $n"
done
echo "failed items:"
for f in "$P"/failed/*.json; do
  [ -e "$f" ] || continue
  s=$(grep -o '"status": *"[^"]*"' "$f" | tail -n 1 | sed 's/.*"\\([^"]*\\)"$/\\1/')
  echo "$(basename "$f" .json) $s"
done
open_attempt() {{
  awk '/"attempts"/ {{a = 1}} a && /[{{]/ {{o = 1; s = 0}} a && /"status"/ {{s = 1}} END {{exit !(o && !s)}}' "$1"
}}
na=""
for f in "$P"/claimed/*.json; do
  [ -e "$f" ] || continue
  open_attempt "$f" || na="$na$(basename "$f" .json)
"
done
[ -n "$na" ] && printf 'claimed without open attempt:\\n%s' "$na"
st=$(cd "$P" && for d in {dirs}; do find "$d" -maxdepth 1 -name '.*.{STAGING_TAG}'; done | sort)
[ -n "$st" ] && printf 'staging leftovers:\\n%s\\n' "$st"
exit 0
"""


# ---------------------------------------------------------------- worker array sbatch

CLUSTER_ENV = "/scratch/ddr8143/.config/shell/cluster_env.sh"
MAX_WALLTIME_S = 23 * 3600
_WALLTIME = re.compile(r"^(?:(\d+)-)?(\d+):(\d{2}):(\d{2})$")


@dataclass(frozen=True, slots=True)
class SubmitSettings:
    account: str = "torch_pr_375_cilvr"
    partition: str = "a100_cilvr"
    gres: str = "gpu:1"
    cpus: int = 4
    mem: str = "60G"
    time: str = "22:00:00"
    slots: int = 1  # concurrent items per worker on its one GPU (MPS requested when > 1)


def walltime_seconds(value: str) -> int:
    m = _WALLTIME.match(value)
    if not m:
        raise ValueError(f"walltime {value!r} is not [D-]HH:MM:SS")
    d, h, mi, s = (int(x) if x else 0 for x in m.groups())
    return ((d * 24 + h) * 60 + mi) * 60 + s


def worker_script(pool: Pool, worker_class: WorkerClass, workers: int, s: SubmitSettings) -> str:
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if s.slots < 1:
        raise ValueError("slots must be >= 1")
    if walltime_seconds(s.time) >= MAX_WALLTIME_S:
        raise ValueError(f"walltime {s.time} must stay under 23:00:00")
    repo = pool.meta["repo"]
    mps = "#SBATCH --comment=gpu_mps=yes\n" if s.slots > 1 else ""
    return f"""#!/bin/bash
#SBATCH --job-name=pool-{pool.path.name}-{worker_class}
#SBATCH --account={s.account}
#SBATCH --partition={s.partition}
#SBATCH --gres={s.gres}
#SBATCH --cpus-per-task={s.cpus}
#SBATCH --mem={s.mem}
#SBATCH --time={s.time}
#SBATCH --array=0-{workers - 1}
#SBATCH --output={pool.dir(PoolDir.WORKERS)}/%A_%a.out
{mps}export SCRATCH=/scratch/ddr8143
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source {CLUSTER_ENV}
export PATH=/home/ddr8143/.local/bin:$SCRATCH/.local/bin:$PATH
export HOME=/scratch/ddr8143 UV_MANAGED_PYTHON=1 UV_CACHE_DIR=$SCRATCH/.cache/uv UV_LINK_MODE=copy HF_HOME=$SCRATCH/.huggingface
cd {repo}
exec uv run python -m datadec.po.worker --pool {pool.path} --class {worker_class} --slots {s.slots} --time {s.time}
"""


def write_worker_script(pool: Pool, worker_class: WorkerClass, workers: int, s: SubmitSettings) -> Path:
    """Write sbatch/<class>-<n>.sbatch, n = this class's next submission number."""
    text = worker_script(pool, worker_class, workers, s)
    sbatch_dir = pool.dir(PoolDir.SBATCH)
    n = 1 + max((int(p.stem.rsplit("-", 1)[1]) for p in sbatch_dir.glob(f"{worker_class}-*.sbatch")), default=0)
    path = sbatch_dir / f"{worker_class}-{n}.sbatch"
    path.write_text(text)
    return path


def submit(script: Path) -> str:
    """sbatch the worker array (no throttle, no dependency); the sbatch output is kept beside the script."""
    out = subprocess.run(["sbatch", str(script)], capture_output=True, text=True, check=True).stdout.strip()
    script.with_suffix(".submit.txt").write_text(out + "\n")
    return out
