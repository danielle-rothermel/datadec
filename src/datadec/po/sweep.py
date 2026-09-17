"""Build and run matched OLMES sweeps: formulations x formats x instructions on one item subset."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from datadec.po.formats import format_id, normalize, validate
from datadec.po.metrics import GENERATION_FORMULATIONS, PRIMARY_METRIC
from datadec.po.subsets import ItemSubset

DEFAULT_ROOT = Path.home() / "drotherm" / "data" / "runs" / "po" / "sweeps"
BASELINES_DIRNAME = "baselines"
OLMES_REPO = Path.home() / "drotherm" / "repos" / "olmes"
FEWSHOT_SOURCE = {"arc_easy": "OLMES:ARC-Easy", "arc_challenge": "OLMES:ARC-Challenge"}
TASK_NAMES = {
    ("arc_easy", "rc"): "arc_easy:fmt", ("arc_easy", "mc"): "arc_easy:mc:fmt",
    ("arc_challenge", "rc"): "arc_challenge:fmt", ("arc_challenge", "mc"): "arc_challenge:mc:fmt",
    ("arc_easy", "gen_rc"): "arc_easy:gen:fmt", ("arc_easy", "gen_mc"): "arc_easy:mc:gen:fmt",
    ("arc_challenge", "gen_rc"): "arc_challenge:gen:fmt", ("arc_challenge", "gen_mc"): "arc_challenge:mc:gen:fmt",
}


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
    use_cache: bool = True  # generation formulations only; False for hf_olmo (DataDecide) checkpoints, which cannot generate with a KV cache


def _instruction_entries(spec: SweepSpec) -> list[dict]:
    return list(spec.instructions) or [{"id": "none", "text": None}]


def _combos(spec: SweepSpec) -> list[tuple[str, dict, dict, int]]:
    """(formulation, format, instruction, num_shots) tuples: sampled pairs if given, else the full cross.

    A pair may carry its own num_shots; otherwise the spec's applies.
    """
    out = []
    if spec.pairs:
        for pair in spec.pairs:
            forms = (pair["formulation"],) if pair.get("formulation") else spec.formulations
            for formulation in forms:
                out.append((formulation, pair["format"], pair["instruction"], int(pair.get("num_shots", spec.num_shots))))
        return out
    for formulation in spec.formulations:
        for fmt in spec.formats:
            for instr in _instruction_entries(spec):
                out.append((formulation, fmt, instr, spec.num_shots))
    return out


def build_tasks(spec: SweepSpec, subset: ItemSubset) -> list[dict]:
    tasks = []
    for formulation, fmt, instr, num_shots in _combos(spec):
        if (spec.task, formulation) not in TASK_NAMES:
            raise ValueError(f"unknown task/formulation {(spec.task, formulation)}")
        fmt = normalize(fmt)
        validate(fmt)
        fid = format_id(fmt)
        context_kwargs: dict = {"prompt_format": dict(fmt)}
        if instr.get("text"):
            context_kwargs["description"] = instr["text"]
        extra: dict = {}
        if formulation in GENERATION_FORMULATIONS and not spec.use_cache:
            extra["generation_kwargs"] = {"use_cache": False}
        tasks.append(
            {
                "task_name": TASK_NAMES[(spec.task, formulation)],
                "split": subset.split,
                "num_shots": num_shots,
                "fewshot_source": FEWSHOT_SOURCE[spec.task],
                "primary_metric": PRIMARY_METRIC[(spec.task, formulation)],
                "limit": len(subset.ids),
                "context_kwargs": context_kwargs,
                **extra,
                "custom_kwargs": {"native_ids": list(subset.ids)},
                "metadata": {
                    "alias": f"{formulation}|{fid}|{instr['id']}" + (f"|k{num_shots}" if num_shots != 5 else ""),
                    "po": {
                        "sweep": spec.name,
                        "task": spec.task,
                        "formulation": formulation,
                        "format_id": fid,
                        "format": dict(fmt),
                        "instruction_id": instr["id"],
                        "num_shots": num_shots,
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
        pair["format"] = normalize(pair["format"])
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


def baselines_dir(sweep_dir: Path) -> Path:
    """Where a sweep's OLMES-default (canonical format, no instruction) task files live once moved out.

    Sweeps sit at <root>/sweeps/<name>; their baselines sit beside them at <root>/baselines/<name>/<rc|mc>/,
    each with the task files plus a source.json whose source_run_dir (relative to <root>) names the run dir
    they were taken from.
    """
    sweep_dir = Path(sweep_dir)
    return sweep_dir.parent.parent / BASELINES_DIRNAME / sweep_dir.name


def _moved_baseline_count(sweep_dir: Path, run_dir: Path) -> int:
    """Number of task files moved from run_dir into the sweep's baselines dir."""
    root = Path(sweep_dir).parent.parent
    n = 0
    for src in baselines_dir(sweep_dir).glob("*/source.json"):
        if (root / json.loads(src.read_text())["source_run_dir"]).resolve() == Path(run_dir).resolve():
            n += 1
    return n


def chunk_complete(sweep_dir: Path, run_dir: Path, chunk_len: int) -> bool:
    """A chunk run dir is complete when its metrics files, plus any moved to baselines, cover every task."""
    present = len(list(Path(run_dir).glob("task-*-metrics.json")))
    return present + _moved_baseline_count(sweep_dir, run_dir) == chunk_len


def run_sweep(sweep_dir: Path, *, olmes_repo: Path = OLMES_REPO, chunk_size: int | None = 10) -> int:
    """Run every task in tasks.jsonl through the fork's runner.

    Tasks are split into chunks of chunk_size, one runner process per chunk (each reloads the
    model, ~15 s), because OLMES scoring slows down markedly as many tasks accumulate in a single
    process. Chunk run dirs land side by side under <sweep_dir>/olmes; the results loader globs
    them all. A chunk whose olmes run dir already holds metrics for every task (counting task files
    moved out to the baselines dir) is skipped.
    """
    spec = json.loads((sweep_dir / "sweep.json").read_text())
    tasks = [json.loads(line) for line in (sweep_dir / "tasks.jsonl").read_text().splitlines() if line.strip()]
    size = chunk_size or len(tasks)
    chunks = [tasks[i:i + size] for i in range(0, len(tasks), size)]
    chunk_dir = sweep_dir / "chunks"
    chunk_dir.mkdir(exist_ok=True)
    worst_rc = 0
    for ci, chunk in enumerate(chunks):
        name = f"tasks-chunk{ci:03d}"
        path = chunk_dir / f"{name}.jsonl"
        path.write_text("".join(json.dumps(t) + "\n" for t in chunk))
        done = [d for d in (sweep_dir / "olmes").rglob(f"*-{name}") if chunk_complete(sweep_dir, d, len(chunk))]
        if done:
            continue
        cmd = [
            "uv", "run", "local/run_eval.py",
            "--model", spec["model"], "--device", spec["device"], "--batch-size", str(spec["batch_size"]),
            "--task-file", str(path), "--root", str(sweep_dir / "olmes"),
        ]
        if spec.get("revision"):
            cmd += ["--revision", spec["revision"]]
        with open(sweep_dir / "run_command.txt", "a") as f:
            f.write(" ".join(cmd) + "\n")
        with open(sweep_dir / "runner.log", "a") as log:
            rc = subprocess.run(cmd, cwd=olmes_repo, stdout=log, stderr=subprocess.STDOUT).returncode
        worst_rc = max(worst_rc, rc)
    return worst_rc
