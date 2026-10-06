"""Sweep task construction: several subsets per sweep, RC-only default, and array task grouping."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from datadec.po.formats import CANONICAL
from datadec.po.slurm import SlurmSettings, array_script, task_groups
from datadec.po.subsets import ItemSubset
from datadec.po.sweep import SweepSpec, build_tasks, subset_label, write_sweep


def _subset(tmp_path: Path, name: str, ids: tuple[str, ...]) -> Path:
    p = tmp_path / f"{name}.json"
    p.write_text(ItemSubset("allenai/ai2_arc", "ARC-Easy", "test", 0, ids).to_json())
    return p


def test_default_formulation_is_rc_only() -> None:
    assert SweepSpec(name="x", subset_paths=(Path("a.json"),), model="m", revision=None).formulations == ("rc",)


def test_tasks_cover_every_subset_with_its_label(tmp_path: Path) -> None:
    a = _subset(tmp_path, "arc_easy-test-part-test1-n2-seed0", ("i1", "i2"))
    b = _subset(tmp_path, "arc_easy-validation-n3-seed0", ("v1", "v2", "v3"))
    spec = SweepSpec(name="s", subset_paths=(a, b), model="m", revision=None, formats=(dict(CANONICAL),))
    tasks = build_tasks(spec, [ItemSubset.load(a), ItemSubset.load(b)])
    assert [t["metadata"]["po"]["subset"] for t in tasks] == [subset_label(a), subset_label(b)]
    assert [t["custom_kwargs"]["native_ids"] for t in tasks] == [["i1", "i2"], ["v1", "v2", "v3"]]
    assert [t["limit"] for t in tasks] == [2, 3]
    assert len({t["metadata"]["alias"] for t in tasks}) == 2  # aliases stay unique across subsets


def test_write_sweep_records_every_subset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("datadec.po.sweep.subprocess.run", lambda *a, **k: type("R", (), {"stdout": "deadbeef\n"})())
    a = _subset(tmp_path, "arc_easy-test-part-test1-n1-seed0", ("i1",))
    b = _subset(tmp_path, "arc_easy-validation-n1-seed0", ("v1",))
    spec = SweepSpec(name="s", subset_paths=(a, b), model="m", revision=None, formats=(dict(CANONICAL),))
    sweep_dir = write_sweep(spec, root=tmp_path / "sweeps")
    manifest = json.loads((sweep_dir / "sweep.json").read_text())
    assert [s["label"] for s in manifest["subsets"]] == [subset_label(a), subset_label(b)]
    assert manifest["num_tasks"] == 2
    with pytest.raises(ValueError):
        write_sweep(SweepSpec(name="t", subset_paths=(), model="m", revision=None), root=tmp_path / "sweeps")


def test_task_groups_and_mps_comment() -> None:
    dirs = [Path(f"/s/{i}") for i in range(5)]
    assert task_groups(dirs, 2) == [dirs[0:2], dirs[2:4], dirs[4:5]]
    assert task_groups(dirs, 1) == [[d] for d in dirs]
    with pytest.raises(ValueError):
        task_groups(dirs, 0)
    shared = array_script(Path("/arr"), task_groups(dirs, 2), SlurmSettings(concurrent=2))
    assert "#SBATCH --comment=gpu_mps=yes" in shared and "--array=0-2\n" in shared
    assert "--array=0-2%4" in array_script(Path("/arr"), task_groups(dirs, 2), SlurmSettings(concurrent=2, max_concurrent=4))
    solo = array_script(Path("/arr"), task_groups(dirs, 1), SlurmSettings())
    assert "gpu_mps" not in solo and "--array=0-4\n" in solo and "a100_cilvr" in solo
    assert "steps/task-$SLURM_ARRAY_TASK_ID.log" in solo


def test_quant_and_dtype_recorded_on_tasks(tmp_path: Path) -> None:
    a = _subset(tmp_path, "arc_easy-test-part-test1-n1-seed0", ("i1",))
    spec = SweepSpec(name="q", subset_paths=(a,), model="m", revision=None, formats=(dict(CANONICAL),), dtype="bfloat16", quant="nf4")
    task = build_tasks(spec, [ItemSubset.load(a)])[0]
    assert task["metadata"]["po"]["dtype"] == "bfloat16" and task["metadata"]["po"]["quant"] == "nf4"
