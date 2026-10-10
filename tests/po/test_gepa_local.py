"""Mac GEPA lane: mps job writing, local run ordering and skipping, status counts, and the push plan."""

import json
import os
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from datadec.po.gepa_local import (
    Remote,
    RunState,
    cell_dirs,
    cell_jobs,
    cell_status,
    eval_item_command,
    plan_runs,
    push_cell,
    run_jobs,
)

REPO = Path(__file__).resolve().parents[2]
DD = "allenai/DataDecide-dclm-baseline-150M"
DD_REV = "step37500-seed-default"
CELL = "allenai-DataDecide-dclm-baseline-150M_step37500-seed-default_fp32-tf32__openbookqa"
ORDER = [f"{CELL}__{s}__s{k}" for s in ("empty", "ape") for k in (0, 1, 2)]
REMOTE = Remote.parse("torch:/scratch/x/root")


def _load_script():
    import importlib.util

    spec = importlib.util.spec_from_file_location("po_gepa_local", REPO / "scripts" / "po_gepa_local.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def root(tmp_path):
    root = tmp_path / "mac"
    result = CliRunner().invoke(_load_script().app, ["write", "--model", DD, "--revision", DD_REV, "--task", "openbookqa",
                                                     "--root", str(root), "--repo", str(REPO)])
    assert result.exit_code == 0, result.output
    return root


def _finish(root, rid, best=0.5, proposals=12):
    run_dir = root / "gepa" / CELL / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "proposals.jsonl").write_text("".join(json.dumps({"kind": "proposal"}) + "\n" for _ in range(proposals)))
    (run_dir / "reflection").mkdir(exist_ok=True)
    (run_dir / "reflection" / "reflection-0001.json").write_text("{}")
    (run_dir / "result.json").write_text(json.dumps({"best_val_score": best}))


def _record(root, rid, **fields):
    (root / "gepa" / CELL / f"{rid}.local.json").write_text(json.dumps({"pid": 1, "ended_utc": None, **fields}))


def test_write_makes_six_mps_jobs_and_no_pool_items(root):
    cell = root / "gepa" / CELL
    assert [j.run_id for j in cell_jobs(cell)] == ORDER
    jobs = [json.loads(p.read_text()) for p in cell.glob("*.job.json")]
    assert len(jobs) == 6 and {j["device"] for j in jobs} == {"mps"} and {j["batch_size"] for j in jobs} == {64}
    assert {j["run_dir"] for j in jobs} == {str(cell / rid) for rid in ORDER}
    assert sorted(p.name for p in root.iterdir()) == ["gepa"]  # no pool, no sweeps


def test_run_skips_finished_and_running_and_orders_empty_first(root):
    _finish(root, ORDER[0])
    _record(root, ORDER[4], pid=4242)  # launched by a live process: left alone
    _record(root, ORDER[5], pid=4343)  # launching process gone: retried
    _record(root, ORDER[1], ended_utc="t", exit_code=1)  # failed earlier: retried
    plan = plan_runs(cell_dirs(root), pid_alive=lambda pid: pid == 4242)
    assert [j.run_id for j in plan.finished] == [ORDER[0]] and [j.run_id for j in plan.running] == [ORDER[4]]
    assert [j.run_id for j in plan.todo] == [ORDER[1], ORDER[2], ORDER[3], ORDER[5]]

    calls = []

    def runner(argv, cwd, log):
        job = Path(argv[-1])
        rid = job.name.removesuffix(".job.json")
        calls.append((argv, cwd, log.name))
        assert json.loads((job.parent / f"{rid}.local.json").read_text())["ended_utc"] is None  # recorded before start
        if rid == ORDER[3]:
            return 1
        _finish(root, rid)
        return 0

    outcomes = run_jobs(plan.todo, Path("/x/olmes"), parallel=1, runner=runner)
    assert [o.run_id for o in outcomes] == [c[0][-1].split("/")[-1].removesuffix(".job.json") for c in calls]
    assert [o.run_id for o in outcomes] == [ORDER[1], ORDER[2], ORDER[3], ORDER[5]]
    assert [o.ok for o in outcomes] == [True, True, False, True]
    argv, cwd, log = calls[0]
    assert argv == ["mise", "exec", "--", "uv", "run", "local/gepa_arc.py", "--job", str(root / "gepa" / CELL / f"{ORDER[1]}.job.json")]
    assert cwd == Path("/x/olmes") and log == f"{ORDER[1]}.local.log"
    rec = json.loads((root / "gepa" / CELL / f"{ORDER[3]}.local.json").read_text())
    assert rec["exit_code"] == 1 and rec["ended_utc"] and rec["duration_s"] is not None and rec["pid"] == os.getpid()
    assert not rec["finished"]


def test_run_records_runner_exception_as_failure(root):
    def runner(argv, cwd, log):
        raise OSError("mise not found")

    outcomes = run_jobs(cell_jobs(root / "gepa" / CELL)[:2], Path("/x"), runner=runner)
    assert [(o.exit_code, o.ok) for o in outcomes] == [(None, False), (None, False)]
    rec = json.loads((root / "gepa" / CELL / f"{ORDER[0]}.local.json").read_text())
    assert rec["exit_code"] is None and rec["ended_utc"] and rec["error"] == "OSError: mise not found"
    assert cell_status(root / "gepa" / CELL)[0].state is RunState.FAILED


def test_status_counts(root):
    _finish(root, ORDER[0], best=0.61, proposals=12)
    _record(root, ORDER[1], ended_utc="t", exit_code=1)
    _record(root, ORDER[2], pid=7)
    _record(root, ORDER[3], pid=8)
    (root / "gepa" / CELL / ORDER[2]).mkdir()
    (root / "gepa" / CELL / ORDER[2] / "proposals.jsonl").write_text('{"kind": "proposal"}\n{"kind": "shorten"}\n{"kind": "pro')
    rows = cell_status(root / "gepa" / CELL, pid_alive=lambda pid: pid == 7)
    assert [r.state for r in rows] == [RunState.FINISHED, RunState.FAILED, RunState.RUNNING, RunState.STALE,
                                       RunState.PENDING, RunState.PENDING]
    assert (rows[0].proposals, rows[0].best_val_score) == (12, 0.61)
    assert rows[2].proposals is None and rows[2].best_val_score is None  # partial row being appended
    assert rows[4].proposals is None


def test_push_sends_finished_runs_without_overwriting(root):
    _finish(root, ORDER[0])
    _finish(root, ORDER[3])
    _record(root, ORDER[0], ended_utc="t", exit_code=0)
    (root / "gepa" / CELL / f"{ORDER[0]}.local.log").write_text("log")
    calls = []

    def fake(argv):
        calls.append(argv)
        out = f"./{ORDER[3]}/result.json\n./{ORDER[3]}.job.json\n./other.txt\n" if argv[0] == "ssh" else ""
        return subprocess.CompletedProcess(argv, 0, out, "")

    report = push_cell(root, CELL, REMOTE, run_cmd=fake)
    remote_cell = f"/scratch/x/root/gepa/{CELL}"
    assert calls[0] == ["ssh", "torch", f"mkdir -p {remote_cell} && cd {remote_cell} && find . -type f"]
    assert calls[1] == report.rsync == [
        "rsync", "-a", "--ignore-existing", "--exclude=*.local.log", "--exclude=*.local.json",
        f"--include=/{ORDER[0]}.job.json", f"--include=/{ORDER[0]}/", f"--include=/{ORDER[0]}/**",
        f"--include=/{ORDER[3]}.job.json", f"--include=/{ORDER[3]}/", f"--include=/{ORDER[3]}/**",
        "--exclude=*", f"{root / 'gepa' / CELL}/", f"torch:{remote_cell}/"]
    assert report.pushed == [ORDER[0], ORDER[3]] and report.not_finished == [ORDER[1], ORDER[2], ORDER[4], ORDER[5]]
    assert report.skipped_files == [f"{ORDER[3]}.job.json", f"{ORDER[3]}/result.json"]
    assert eval_item_command(REMOTE, CELL, ORDER[0], "/scratch/x/root/pool/p") == (
        f"uv run python scripts/po_gepa_cells.py eval-item --run-dir {remote_cell}/{ORDER[0]} --pool /scratch/x/root/pool/p")


def test_push_with_nothing_finished_runs_nothing(root):
    def fake(argv):
        raise AssertionError("no remote call expected")

    report = push_cell(root, CELL, REMOTE, run_cmd=fake)
    assert report.pushed == [] and len(report.not_finished) == 6 and report.rsync == []


def test_remote_root_must_be_host_and_absolute_path():
    with pytest.raises(ValueError):
        Remote.parse("/scratch/x")
    assert Remote.parse("torch:/a/b/").cell_path("c") == "/a/b/gepa/c"
