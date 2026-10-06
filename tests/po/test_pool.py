"""Pull-based worker pool: item files, claim race, deadline, orphan requeue, create, worker outcomes, status, sbatch."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from datadec.po.pool import (
    ITEM_SCHEMA,
    Attempt,
    AttemptStatus,
    FileCompletion,
    ItemKind,
    Pool,
    PoolDir,
    PoolItem,
    SubmitSettings,
    SweepCompletion,
    WorkerClass,
    add_items,
    claim,
    create,
    init_pool,
    place_items,
    requeue_failed,
    requeue_stale,
    status_text,
    worker_script,
)
from datadec.po.subsets import ItemSubset
from datadec.po.sweep import sweep_complete
from datadec.po.worker import Worker, parse_time_left, remaining_seconds, run_logged

PY = sys.executable


def _pool(tmp_path: Path) -> Pool:
    return init_pool(tmp_path / "pool" / "p", root=tmp_path, repo=tmp_path / "repo")


def _item(name: str, *, command: list[str] | None = None, out: Path | None = None, cls: WorkerClass = WorkerClass.SMALL,
          est: int = 20, cwd: Path | None = None, **kw) -> PoolItem:
    return PoolItem(name=name, kind=ItemKind.EVAL, worker_class=cls, est_minutes=est, cwd=str(cwd or Path.cwd()),
                    command=command or [PY, "-c", "pass"], complete=FileCompletion(path=str(out or Path("/nonexistent"))),
                    created_utc="2026-10-06T00:00:00+00:00", **kw)


def _write_out(path: Path) -> list[str]:
    return [PY, "-c", f"open({str(path)!r}, 'w').write('ok')"]


def test_persisted_literals_are_pinned() -> None:
    assert ITEM_SCHEMA == "po-pool-item/1"
    assert [s.value for s in AttemptStatus] == ["done", "skipped", "failed", "no-output", "followup-failed", "timeout", "orphaned"]
    assert [d.value for d in PoolDir] == ["pending", "claimed", "done", "failed", "logs", "workers", "sbatch"]
    assert [c.value for c in WorkerClass] == ["small", "big", "32b", "gepa"]
    data = json.loads(_item("x").with_attempt(Attempt(status=AttemptStatus.SKIPPED)).to_json())
    assert data["schema"] == "po-pool-item/1" and data["class"] == "small" and data["followup"] is None
    assert data["attempts"] == [{"status": "skipped"}]
    assert list(data)[-1] == "attempts"  # status.sh relies on attempts being the last key
    with pytest.raises(ValueError):
        PoolItem.model_validate(data | {"unexpected": 1})


def test_claim_race_has_exactly_one_winner(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    place_items(pool, [_item("only")])
    both_read = threading.Barrier(2)

    def fits(_: PoolItem) -> bool:  # runs after each racer has read the pending file and before its rename
        both_read.wait(timeout=10)
        return True

    results: list = [None, None]

    def racer(i: int) -> None:
        results[i] = claim(pool, WorkerClass.SMALL, Attempt(job_id=str(i)), fits)

    threads = [threading.Thread(target=racer, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    winners = [r for r in results if r is not None]
    assert len(winners) == 1
    path, item = winners[0]
    assert path.parent == pool.dir(PoolDir.CLAIMED) and pool.items(PoolDir.PENDING) == []
    assert PoolItem.load(path).open_attempt == item.open_attempt
    assert item.open_attempt is not None and item.open_attempt.log == str(pool.log_path(path, 1))


def test_claim_respects_class_and_order_and_stops_when_first_does_not_fit(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    place_items(pool, [_item("big1", cls=WorkerClass.BIG), _item("long", est=600), _item("short", est=1)])
    assert claim(pool, WorkerClass.SMALL, Attempt(), lambda i: i.est_minutes < 100) is None
    assert len(pool.items(PoolDir.PENDING)) == 3
    path, _ = claim(pool, WorkerClass.SMALL, Attempt()) or (None, None)
    assert path is not None and path.name == "0002-long.json"


def test_deadline_from_env_and_squeue_fallback() -> None:
    assert remaining_seconds({"SLURM_JOB_END_TIME": "2000"}, 1500.0, None) == 500.0
    calls: list[list[str]] = []

    def squeue(argv: list[str]) -> str:
        calls.append(argv)
        return "1-02:03:04\n"

    assert remaining_seconds({"SLURM_JOB_ID": "77"}, 0.0, squeue) == float(((24 + 2) * 60 + 3) * 60 + 4)
    assert calls == [["squeue", "-h", "-j", "77", "-o", "%L"]]
    assert remaining_seconds({}, 0.0, squeue) is None
    assert parse_time_left("05:07") == 307.0 and parse_time_left("UNLIMITED") is None


def test_worker_fits_uses_injected_clock(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    now = [1000.0]
    w = Worker(pool, WorkerClass.SMALL, env={"SLURM_JOB_END_TIME": "2300"}, clock=lambda: now[0], squeue=None)
    item = _item("x", est=20)  # 1200 s
    assert w.fits(item) and w.command_timeout() == 1300.0 - 60
    now[0] = 1100.0
    assert not w.fits(item)


def _claimed(pool: Pool, name: str, seq: int, attempt: Attempt | None) -> Path:
    item = _item(name)
    if attempt is not None:
        item = item.with_attempt(attempt)
    path = pool.dir(PoolDir.CLAIMED) / f"{seq:04d}-{name}.json"
    path.write_text(item.to_json())
    return path


def test_orphan_requeue_with_fake_squeue(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    _claimed(pool, "dead", 1, Attempt(job_id="111", started_utc="t0"))
    _claimed(pool, "live", 2, Attempt(job_id="222", started_utc="t0"))
    _claimed(pool, "fresh", 3, None)  # just renamed by a worker that has not written its attempt yet
    calls: list[list[str]] = []

    def squeue(argv: list[str]) -> str:
        calls.append(argv)
        return "222\n333\n"

    assert requeue_stale(pool, squeue) == ["0001-dead.json"]
    assert calls == [["squeue", "-h", "-u", "ddr8143", "-o", "%A", "-r"]]
    moved = PoolItem.load(pool.dir(PoolDir.PENDING) / "0001-dead.json")
    assert moved.attempts[-1].status == AttemptStatus.ORPHANED and moved.attempts[-1].job_id == "111"
    assert [p.name for p in pool.items(PoolDir.CLAIMED)] == ["0002-live.json", "0003-fresh.json"]
    assert not [p for p in pool.dir(PoolDir.CLAIMED).iterdir() if p.name.startswith(".")]


def _subset(repo: Path, name: str) -> str:
    rel = f"subsets/{name}.json"
    (repo / "subsets").mkdir(parents=True, exist_ok=True)
    (repo / rel).write_text(ItemSubset("allenai/ai2_arc", "ARC-Easy", "test", 0, ("i1", "i2")).to_json())
    return rel


def _complete_sweep(sweep_dir: Path) -> None:
    tasks = (sweep_dir / "tasks.jsonl").read_text()
    (sweep_dir / "chunks").mkdir()
    (sweep_dir / "chunks" / "tasks-chunk000.jsonl").write_text(tasks)
    run = sweep_dir / "olmes" / "20261006-tasks-chunk000"
    run.mkdir(parents=True)
    for i in range(len(tasks.splitlines())):
        (run / f"task-{i:03d}-metrics.json").write_text("{}")


def test_create_reuses_sweeps_and_skips_completed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("datadec.po.sweep.subprocess.run", lambda *a, **k: type("R", (), {"stdout": "deadbeef\n"})())
    repo, root = tmp_path / "repo", tmp_path / "root"
    sub = _subset(repo, "arc_easy-test-n2-seed0")
    jobs_a, jobs_b = tmp_path / "a.json", tmp_path / "b.json"
    jobs_a.write_text(json.dumps({"jobs": [{"name": "s-done", "model": "m", "subsets": [sub]}]}))
    jobs_b.write_text(json.dumps({"jobs": [{"name": "s-todo", "model": "m", "subsets": [sub], "batch_size": 8}]}))
    pool_dir = root / "pool" / "p"
    first = create(pool_dir, root=root, repo=repo, worker_class=WorkerClass.SMALL, est_minutes=20, job_lists=[jobs_a])
    assert (first.pending, first.done) == (1, 0)
    # the sweep completes and its item is lost from the pool: a new pool over the same root skips it
    done_dir = root / "sweeps" / "s-done"
    assert not sweep_complete(done_dir)
    _complete_sweep(done_dir)
    assert sweep_complete(done_dir)
    pool2 = root / "pool" / "q"
    counts = create(pool2, root=root, repo=repo, worker_class=WorkerClass.SMALL, est_minutes=20, job_lists=[jobs_a, jobs_b])
    assert (counts.pending, counts.done, counts.existing) == (1, 1, 0)
    skipped = PoolItem.load(pool2 / "done" / "0001-s-done.json")
    assert [a.status for a in skipped.attempts] == [AttemptStatus.SKIPPED]
    todo = PoolItem.load(pool2 / "pending" / "0002-s-todo.json")
    assert todo.command == ["uv", "run", "python", "scripts/po_run_sweep.py", "--sweep-dir", str(root / "sweeps" / "s-todo"),
                            "--chunk-size", "14"]
    assert todo.complete == SweepCompletion(sweep_dir=str(root / "sweeps" / "s-todo")) and todo.cwd == str(repo)
    assert todo.meta["batch_size"] == 8 and todo.meta["subsets"] == [sub]
    again = create(pool2, root=root, repo=repo, worker_class=WorkerClass.SMALL, est_minutes=20, job_lists=[jobs_b])
    assert (again.pending, again.existing) == (0, 1)
    assert (pool2 / "status.sh").exists()


def test_add_items_appends_in_sequence(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    place_items(pool, [_item("a")])
    done_file = tmp_path / "exists.json"
    done_file.write_text("{}")
    lines = [_item("g1", followup=["true"], followup_cwd="/tmp").to_json().replace("\n", ""),
             _item("g2", out=done_file).to_json().replace("\n", "")]
    (tmp_path / "items.jsonl").write_text("\n".join(lines) + "\n")
    counts = add_items(pool.path, tmp_path / "items.jsonl")
    assert (counts.pending, counts.done) == (1, 1)
    assert [p.name for p in pool.items(PoolDir.PENDING)] == ["0001-a.json", "0002-g1.json"]
    assert [p.name for p in pool.items(PoolDir.DONE)] == ["0003-g2.json"]


def test_env_file_wrapper_never_echoes_contents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / "secrets.env"
    env_file.write_text("PO_TEST_SECRET=hunter2-topsecret\n")
    monkeypatch.setenv("SHELLOPTS", "xtrace")  # an inherited xtrace must not trace the sourcing
    log_path = tmp_path / "attempt.log"
    with open(log_path, "a") as log:
        rc = run_logged([PY, "-c", "import os; print('len', len(os.environ['PO_TEST_SECRET']))"], cwd=str(tmp_path),
                        env_files=[str(env_file)], log=log, timeout=None)
    text = log_path.read_text()
    assert rc == 0 and "len 17" in text
    assert "topsecret" not in text and "PO_TEST_SECRET=" not in text
    with open(log_path, "a") as log:
        assert run_logged([PY, "-c", "pass"], cwd=str(tmp_path), env_files=[str(tmp_path / "missing.env")], log=log,
                          timeout=None) == 97


def test_worker_runs_items_to_their_outcomes(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    ok, ok2, never = tmp_path / "ok.txt", tmp_path / "ok2.txt", tmp_path / "never.txt"
    place_items(pool, [
        _item("ok", command=_write_out(ok), out=ok, followup=[PY, "-c", "import pathlib; pathlib.Path('f.txt').write_text('x')"],
              followup_cwd=str(tmp_path)),
        _item("fails", command=[PY, "-c", "import sys; sys.exit(3)"]),
        _item("silent", out=never),
        _item("followup-bad", command=_write_out(ok2), out=ok2, followup=[PY, "-c", "import sys; sys.exit(5)"]),
        _item("other-class", cls=WorkerClass.BIG),
    ])
    squeue_calls: list[list[str]] = []
    w = Worker(pool, WorkerClass.SMALL, slots=2, env={}, squeue=lambda argv: squeue_calls.append(argv) or "")
    assert w.run() == 0
    assert [p.name for p in pool.items(PoolDir.DONE)] == ["0001-ok.json"]
    assert (tmp_path / "f.txt").read_text() == "x"
    failed = {p.stem: PoolItem.load(p).attempts[-1] for p in pool.items(PoolDir.FAILED)}
    assert {k: (a.status, a.exit_code) for k, a in failed.items()} == {
        "0002-fails": (AttemptStatus.FAILED, 3), "0003-silent": (AttemptStatus.NO_OUTPUT, 0),
        "0004-followup-bad": (AttemptStatus.FOLLOWUP_FAILED, 5)}
    assert [p.name for p in pool.items(PoolDir.PENDING)] == ["0005-other-class.json"]
    done = PoolItem.load(pool.dir(PoolDir.DONE) / "0001-ok.json").attempts[-1]
    assert done.status == AttemptStatus.DONE and done.ended_utc and done.log and Path(done.log).exists()
    assert w.claims == 4 and pool.items(PoolDir.CLAIMED) == []
    assert squeue_calls == []  # nothing claimed at start, so no squeue call was needed


def test_worker_kills_command_at_deadline(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    place_items(pool, [_item("slow", est=0, command=[PY, "-c", "import time; time.sleep(600)"])])
    # fixed clock: 60 s left, so the command timeout (left - 60 s) is zero and the command is killed at once
    w = Worker(pool, WorkerClass.SMALL, env={"SLURM_JOB_END_TIME": "1060"}, clock=lambda: 1000.0, squeue=None)
    w.run()
    attempt = PoolItem.load(pool.dir(PoolDir.FAILED) / "0001-slow.json").attempts[-1]
    assert attempt.status == AttemptStatus.TIMEOUT and attempt.exit_code is None


def test_status_text_and_status_sh_agree(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    place_items(pool, [_item(n) for n in ("a", "b", "c", "d", "e")])
    files = pool.items(PoolDir.PENDING)
    for f, d, status in ((files[0], PoolDir.DONE, AttemptStatus.DONE), (files[1], PoolDir.FAILED, AttemptStatus.TIMEOUT),
                         (files[2], PoolDir.FAILED, AttemptStatus.NO_OUTPUT)):
        item = PoolItem.load(f).with_attempt(Attempt(job_id="1", status=status))
        f.write_text(item.to_json())
        f.rename(pool.dir(d) / f.name)
    (pool.dir(PoolDir.PENDING) / ".0009-x.json.abc.tmp").write_text("{}")  # in-flight temp files are not items
    expected = "pending 2\nclaimed 0\ndone 1\nfailed 2\nfailed items:\n0002-b timeout\n0003-c no-output\n"
    assert status_text(pool) == expected
    sh = subprocess.run(["bash", str(pool.path / "status.sh")], capture_output=True, text=True, check=True).stdout
    assert sh == expected
    assert requeue_failed(pool, [AttemptStatus.TIMEOUT]) == ["0002-b.json"]
    assert status_text(pool).startswith("pending 3\nclaimed 0\ndone 1\nfailed 1\n")


def test_worker_sbatch_content(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    solo = worker_script(pool, WorkerClass.SMALL, 8, SubmitSettings())
    assert "#SBATCH --array=0-7\n" in solo and "%" not in solo.split("--array=")[1].split("\n")[0]
    assert "--dependency" not in solo and "gpu_mps" not in solo
    assert "#SBATCH --account=torch_pr_375_cilvr" in solo and "#SBATCH --partition=a100_cilvr" in solo
    assert "#SBATCH --time=22:00:00" in solo and "#SBATCH --gres=gpu:1" in solo
    assert f"--pool {pool.path} --class small --slots 1" in solo and f"cd {tmp_path / 'repo'}" in solo
    assert "source /scratch/ddr8143/.config/shell/cluster_env.sh" in solo and "HF_HOME=$SCRATCH/.huggingface" in solo
    shared = worker_script(pool, WorkerClass.SMALL, 2, SubmitSettings(slots=3))
    assert "#SBATCH --comment=gpu_mps=yes" in shared and "--slots 3" in shared
    with pytest.raises(ValueError):
        worker_script(pool, WorkerClass.SMALL, 2, SubmitSettings(time="23:00:00"))
    with pytest.raises(ValueError):
        worker_script(pool, WorkerClass.SMALL, 2, SubmitSettings(time="1-00:00:00"))
    assert "--time=22:59:59" in worker_script(pool, WorkerClass.SMALL, 1, SubmitSettings(time="22:59:59"))
