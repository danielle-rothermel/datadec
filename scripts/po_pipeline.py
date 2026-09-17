"""Orchestrate the seed-round sweeps and GEPA runs across tasks and models, in a fixed order.

Idempotent: completed sweeps (analysis/tasks.parquet present) and completed GEPA jobs
(result.json present) are skipped, so the script can be re-run to resume. Progress lines
go to <root>/pipeline.log and stdout.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "configs" / "po"
DEFAULT_ROOT = Path.home() / "drotherm" / "data" / "runs" / "po" / "pipeline"

MODELS: dict[str, tuple[str, str | None]] = {
    "dd150m": ("allenai/DataDecide-dclm-baseline-150M", "step37500-seed-default"),
    "dd300m": ("allenai/DataDecide-dclm-baseline-300M", "step45000-seed-default"),
    "dd530m": ("allenai/DataDecide-dclm-baseline-530M", "step51250-seed-default"),
    "qwen1.7b-base": ("Qwen/Qwen3-1.7B-Base", None),
    "qwen1.7b": ("Qwen/Qwen3-1.7B", None),
    "qwen4b-base": ("Qwen/Qwen3-4B-Base", None),
    "qwen4b": ("Qwen/Qwen3-4B", None),
}
# Qwen's 152k vocabulary makes the per-batch log-softmax tensor several GB at batch 16, which
# pushed the machine into swap; batch 4 keeps it in memory at similar throughput.
BATCH_SIZE = {k: (4 if k.startswith("qwen") else 16) for k in MODELS}
SUBSETS = {
    "arc_easy": {"test": CFG / "subsets/arc_easy-test-n100-seed0.json",
                 "train": CFG / "subsets/arc_easy-train-n60-seed1.json", "val": CFG / "subsets/arc_easy-train-n40-seed2.json"},
    "arc_challenge": {"test": CFG / "subsets/arc_challenge-test-n100-seed0.json",
                      "train": CFG / "subsets/arc_challenge-train-n60-seed1.json", "val": CFG / "subsets/arc_challenge-train-n40-seed2.json"},
}
PAIRS = CFG / "pairs/arc-pairs-n50-seed0.json"
GEPA_N = 3
GEPA_BUDGET = 600
HEADROOM = 0.9  # skip (model, formulation) groups whose best seed scores >= this
DD = ("dd150m", "dd300m", "dd530m")

# (kind, task, model, options) in the order Danielle specified.
STEPS: list[tuple[str, str, str, dict]] = [
    ("sweep", "arc_easy", "dd150m", {}), ("gepa", "arc_easy", "dd150m", {"formulations": "rc"}),
    ("sweep", "arc_easy", "dd300m", {}), ("gepa", "arc_easy", "dd300m", {"formulations": "rc"}),
    ("sweep", "arc_easy", "dd530m", {}),
    ("sweep", "arc_easy", "qwen1.7b-base", {}), ("sweep", "arc_easy", "qwen1.7b", {}),
    *[("sweep", "arc_challenge", m, {}) for m in ("dd150m", "dd300m", "dd530m", "qwen1.7b-base", "qwen1.7b", "qwen4b-base", "qwen4b")],
    ("gepa", "arc_easy", "dd530m", {"formulations": "rc"}),
    ("gepa", "arc_easy", "qwen1.7b-base", {"max_group_score": HEADROOM}), ("gepa", "arc_easy", "qwen1.7b", {"max_group_score": HEADROOM}),
    ("gepa", "arc_challenge", "dd300m", {"formulations": "rc"}), ("gepa", "arc_challenge", "dd530m", {"formulations": "rc"}),
    *[("gepa", "arc_challenge", m, {"max_group_score": HEADROOM}) for m in ("qwen1.7b-base", "qwen1.7b", "qwen4b-base", "qwen4b")],
]


class Pipeline:
    def __init__(self, name: str, root: Path = DEFAULT_ROOT) -> None:
        self.root = root / name
        self.root.mkdir(parents=True, exist_ok=True)
        self.log_path = self.root / "pipeline.log"
        self.state_path = self.root / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {"steps": {}}

    def log(self, msg: str) -> None:
        line = f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
        print(line, flush=True)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    def _save(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=1) + "\n")

    def _run(self, args: list[str], log_name: str) -> int:
        with open(self.root / "commands.log", "a") as f:
            f.write(f"[{dt.datetime.now():%H:%M:%S}] " + " ".join(args) + "\n")
        with open(self.root / "logs" / f"{log_name}.log", "a") as out:
            return subprocess.run(args, cwd=REPO, stdout=out, stderr=subprocess.STDOUT).returncode

    # ---- steps ---------------------------------------------------------

    def sweep(self, task: str, model_key: str, *, pairs: Path, name: str, subset: Path) -> Path | None:
        model, revision = MODELS[model_key]
        sweep_dir = self.root / "sweeps" / name
        if (sweep_dir / "analysis" / "tasks.parquet").exists():
            self.log(f"  sweep {name}: already complete, skipping")
            return sweep_dir
        if sweep_dir.exists():
            failed = sweep_dir.with_name(f"{name}.failed-{dt.datetime.now():%H%M%S}")
            shutil.move(str(sweep_dir), str(failed))
            self.log(f"  sweep {name}: incomplete dir moved to {failed.name}; rerunning")
        args = [sys.executable, "scripts/po_sweep.py", "--name", name, "--subset", str(subset), "--model", model,
                "--task", task, "--pairs", str(pairs), "--root", str(self.root / "sweeps"),
                "--batch-size", str(BATCH_SIZE[model_key])]
        if revision:
            args += ["--revision", revision]
        rc = self._run(args, name)
        if rc != 0:
            self.log(f"  sweep {name}: FAILED (exit {rc})")
            return None
        rc = self._run([sys.executable, "scripts/po_results.py", str(sweep_dir)], name)
        if rc != 0:
            self.log(f"  sweep {name}: results FAILED (exit {rc})")
            return None
        return sweep_dir

    def gepa(self, task: str, model_key: str, opts: dict) -> bool:
        name = f"{task}-{model_key}"
        seed_sweep = self.root / "sweeps" / name
        if not (seed_sweep / "analysis" / "tasks.parquet").exists():
            self.log(f"  gepa {name}: seed sweep missing or incomplete; skipping")
            return False
        gepa_root = self.root / "gepa"
        jobs_dir = gepa_root / name / "jobs"
        if jobs_dir.exists() and not any(jobs_dir.glob("*.json")):
            shutil.rmtree(gepa_root / name)  # a failed selection left an empty jobs dir; redo selection
        if not jobs_dir.exists():
            args = [sys.executable, "scripts/po_select_seeds.py", "--name", name, "--sweep-dir", str(seed_sweep),
                    "--batch-size", str(BATCH_SIZE[model_key]),
                    "--train-subset", str(SUBSETS[task]["train"]), "--val-subset", str(SUBSETS[task]["val"]),
                    "--n", str(GEPA_N), "--max-metric-calls", str(GEPA_BUDGET), "--root", str(gepa_root)]
            if opts.get("formulations"):
                args += ["--formulations", opts["formulations"]]
            if opts.get("max_group_score") is not None:
                args += ["--max-group-score", str(opts["max_group_score"])]
            if self._run(args, f"{name}-gepa") != 0:
                self.log(f"  gepa {name}: seed selection FAILED")
                return False
        n_jobs = len(list(jobs_dir.glob("*.json")))
        if n_jobs == 0:
            self.log(f"  gepa {name}: no jobs (no headroom or nothing selected); skipping")
            return True
        self.log(f"  gepa {name}: {n_jobs} jobs")
        rc = self._run([sys.executable, "scripts/po_gepa.py", str(jobs_dir)], f"{name}-gepa")
        if rc != 0:
            self.log(f"  gepa {name}: one or more jobs FAILED (exit {rc}); continuing with completed ones")
        pairs = gepa_root / name / "optimized_pairs.json"
        if not pairs.exists() or json.loads(pairs.read_text())["n"] == 0:
            self.log(f"  gepa {name}: no optimized instructions to evaluate")
            return False
        out = self.sweep(task, model_key, pairs=pairs, name=f"{name}-gepa-eval", subset=SUBSETS[task]["test"])
        return out is not None

    def run(self) -> None:
        (self.root / "logs").mkdir(exist_ok=True)
        self.log(f"pipeline {self.root.name}: {len(STEPS)} steps")
        for i, (kind, task, model_key, opts) in enumerate(STEPS, 1):
            key = f"{i:02d}-{kind}-{task}-{model_key}"
            if self.state["steps"].get(key) == "done":
                self.log(f"STEP {i}/{len(STEPS)} {key}: done earlier, skipping")
                continue
            self.log(f"STEP {i}/{len(STEPS)} {key}: start")
            t0 = dt.datetime.now()
            if kind == "sweep":
                ok = self.sweep(task, model_key, pairs=PAIRS, name=f"{task}-{model_key}", subset=SUBSETS[task]["test"]) is not None
            else:
                ok = self.gepa(task, model_key, opts)
            elapsed = dt.datetime.now() - t0
            self.state["steps"][key] = "done" if ok else "failed"
            self._save()
            self.log(f"STEP {i}/{len(STEPS)} {key}: {'done' if ok else 'FAILED'} in {elapsed}")
        self.log("pipeline complete")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    a = ap.parse_args()
    Pipeline(a.name, a.root).run()
