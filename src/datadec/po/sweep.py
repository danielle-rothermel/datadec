"""Build and run matched OLMES sweeps: formulations x formats x instructions on one item subset."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from datadec.po.formats import format_id, validate
from datadec.po.subsets import ItemSubset

DEFAULT_ROOT = Path.home() / "drotherm" / "data" / "runs" / "po" / "sweeps"
OLMES_REPO = Path.home() / "drotherm" / "repos" / "olmes"
FEWSHOT_SOURCE = "OLMES:ARC-Easy"
TASK_NAMES = {"rc": "arc_easy:fmt", "mc": "arc_easy:mc:fmt"}
PRIMARY_METRIC = {"rc": "acc_per_char", "mc": "acc_raw"}


@dataclass(frozen=True, slots=True)
class SweepSpec:
    name: str
    subset_path: Path
    model: str
    revision: str | None
    formulations: tuple[str, ...] = ("rc", "mc")
    formats: tuple[dict, ...] = field(default_factory=tuple)
    instructions: tuple[dict, ...] = field(default_factory=tuple)  # {id, text}; empty -> one "none" entry
    num_shots: int = 5
    device: str = "mps"
    batch_size: int = 16


def _instruction_entries(spec: SweepSpec) -> list[dict]:
    return list(spec.instructions) or [{"id": "none", "text": None}]


def build_tasks(spec: SweepSpec, subset: ItemSubset) -> list[dict]:
    tasks = []
    for formulation in spec.formulations:
        if formulation not in TASK_NAMES:
            raise ValueError(f"unknown formulation {formulation!r}")
        for fmt in spec.formats:
            validate(fmt)
            fid = format_id(fmt)
            for instr in _instruction_entries(spec):
                context_kwargs: dict = {"prompt_format": dict(fmt)}
                if instr["text"]:
                    context_kwargs["description"] = instr["text"]
                tasks.append(
                    {
                        "task_name": TASK_NAMES[formulation],
                        "split": subset.split,
                        "num_shots": spec.num_shots,
                        "fewshot_source": FEWSHOT_SOURCE,
                        "primary_metric": PRIMARY_METRIC[formulation],
                        "limit": len(subset.ids),
                        "context_kwargs": context_kwargs,
                        "custom_kwargs": {"native_ids": list(subset.ids)},
                        "metadata": {
                            "alias": f"{formulation}|{fid}|{instr['id']}",
                            "po": {
                                "sweep": spec.name,
                                "formulation": formulation,
                                "format_id": fid,
                                "format": dict(fmt),
                                "instruction_id": instr["id"],
                                "subset_seed": subset.seed,
                                "subset_n": len(subset.ids),
                            },
                        },
                    }
                )
    return tasks


def write_sweep(spec: SweepSpec, *, root: Path = DEFAULT_ROOT) -> Path:
    subset = ItemSubset.load(spec.subset_path)
    tasks = build_tasks(spec, subset)
    sweep_dir = root / spec.name
    if sweep_dir.exists():
        raise FileExistsError(f"{sweep_dir} exists; sweep names are unique")
    sweep_dir.mkdir(parents=True)
    with open(sweep_dir / "tasks.jsonl", "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    manifest = asdict(spec) | {
        "subset_path": str(spec.subset_path),
        "subset_sha1": hashlib.sha1(Path(spec.subset_path).read_bytes()).hexdigest(),
        "subset_ids": list(subset.ids),
        "num_tasks": len(tasks),
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "olmes_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=OLMES_REPO, capture_output=True, text=True, check=True
        ).stdout.strip(),
    }
    (sweep_dir / "sweep.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return sweep_dir


def run_sweep(sweep_dir: Path, *, olmes_repo: Path = OLMES_REPO) -> int:
    """Invoke the fork's runner once (model loads once) for every task in tasks.jsonl."""
    spec = json.loads((sweep_dir / "sweep.json").read_text())
    cmd = [
        "uv", "run", "local/run_eval.py",
        "--model", spec["model"], "--device", spec["device"], "--batch-size", str(spec["batch_size"]),
        "--task-file", str(sweep_dir / "tasks.jsonl"), "--root", str(sweep_dir / "olmes"),
    ]
    if spec.get("revision"):
        cmd += ["--revision", spec["revision"]]
    (sweep_dir / "run_command.txt").write_text(" ".join(cmd) + "\n")
    with open(sweep_dir / "runner.log", "a") as log:
        return subprocess.run(cmd, cwd=olmes_repo, stdout=log, stderr=subprocess.STDOUT).returncode
