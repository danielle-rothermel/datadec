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
FEWSHOT_SOURCE = {"arc_easy": "OLMES:ARC-Easy", "arc_challenge": "OLMES:ARC-Challenge"}
TASK_NAMES = {
    ("arc_easy", "rc"): "arc_easy:fmt", ("arc_easy", "mc"): "arc_easy:mc:fmt",
    ("arc_challenge", "rc"): "arc_challenge:fmt", ("arc_challenge", "mc"): "arc_challenge:mc:fmt",
}
PRIMARY_METRIC = {("arc_easy", "rc"): "acc_per_char", ("arc_easy", "mc"): "acc_raw",
                  ("arc_challenge", "rc"): "acc_uncond", ("arc_challenge", "mc"): "acc_raw"}


@dataclass(frozen=True, slots=True)
class SweepSpec:
    name: str
    subset_path: Path
    model: str
    revision: str | None
    task: str = "arc_easy"
    formulations: tuple[str, ...] = ("rc", "mc")
    formats: tuple[dict, ...] = field(default_factory=tuple)
    instructions: tuple[dict, ...] = field(default_factory=tuple)  # {id, text}; empty -> one "none" entry
    pairs: tuple[dict, ...] = field(default_factory=tuple)  # {format, instruction{id,text}, formulation?}; overrides the cross
    num_shots: int = 5
    device: str = "mps"
    batch_size: int = 16


def _instruction_entries(spec: SweepSpec) -> list[dict]:
    return list(spec.instructions) or [{"id": "none", "text": None}]


def _combos(spec: SweepSpec) -> list[tuple[str, dict, dict]]:
    """(formulation, format, instruction) triples: sampled pairs if given, else the full cross."""
    out = []
    if spec.pairs:
        for pair in spec.pairs:
            forms = (pair["formulation"],) if pair.get("formulation") else spec.formulations
            for formulation in forms:
                out.append((formulation, pair["format"], pair["instruction"]))
        return out
    for formulation in spec.formulations:
        for fmt in spec.formats:
            for instr in _instruction_entries(spec):
                out.append((formulation, fmt, instr))
    return out


def build_tasks(spec: SweepSpec, subset: ItemSubset) -> list[dict]:
    tasks = []
    for formulation, fmt, instr in _combos(spec):
        if (spec.task, formulation) not in TASK_NAMES:
            raise ValueError(f"unknown task/formulation {(spec.task, formulation)}")
        validate(fmt)
        fid = format_id(fmt)
        context_kwargs: dict = {"prompt_format": dict(fmt)}
        if instr.get("text"):
            context_kwargs["description"] = instr["text"]
        tasks.append(
            {
                "task_name": TASK_NAMES[(spec.task, formulation)],
                "split": subset.split,
                "num_shots": spec.num_shots,
                "fewshot_source": FEWSHOT_SOURCE[spec.task],
                "primary_metric": PRIMARY_METRIC[(spec.task, formulation)],
                "limit": len(subset.ids),
                "context_kwargs": context_kwargs,
                "custom_kwargs": {"native_ids": list(subset.ids)},
                "metadata": {
                    "alias": f"{formulation}|{fid}|{instr['id']}",
                    "po": {
                        "sweep": spec.name,
                        "task": spec.task,
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


def sample_pairs(formats: list[dict], instructions: list[dict], n: int, seed: int, *, include_baseline: bool = True) -> list[dict]:
    """n (format, instruction) pairs drawn without replacement from the cross; the OLMES baseline pair first."""
    import random
    from datadec.po.formats import CANONICAL
    cross = [(f, i) for f in formats for i in instructions]
    baseline = (dict(CANONICAL), {"id": "none", "text": None})
    pool = [c for c in cross if not (c[0] == baseline[0] and c[1]["id"] == "none")]
    picked = random.Random(seed).sample(pool, n)
    chosen = ([baseline] if include_baseline else []) + picked
    return [{"format": f, "format_id": format_id(f), "instruction": i} for f, i in chosen]


def write_pairs(path: Path, pairs: list[dict], *, seed: int | None, sources: dict) -> None:
    path.write_text(json.dumps({"seed": seed, "n": len(pairs), "sources": sources, "pairs": pairs}, indent=1) + "\n")


def load_pairs(path: Path) -> list[dict]:
    raw = json.loads(Path(path).read_text())
    for pair in raw["pairs"]:
        validate(pair["format"])
    return raw["pairs"]


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
        "instructions": [dict(i) for i in spec.instructions] + [
            p["instruction"] for p in spec.pairs if p["instruction"]["id"] not in {i["id"] for i in spec.instructions}
        ],
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
