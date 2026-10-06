"""Write a set of sweeps and submit them as one Slurm job array (one GPU per sweep).

A job list is a JSON file: {"jobs": [{"name", "model", "revision"?, "subsets": [...], "formulations"?, "pairs"?,
"formats"?, "instructions"?, "num_shots"?, "batch_size"?, "dtype"?, "quant"? (int8|nf4, with dtype bfloat16), "use_cache"?}, ...]}. Each job becomes
one sweep directory under <root>/sweeps/<name> (written up front, so the array script only runs them).
Array tasks take `concurrent` consecutive sweeps each and run them in parallel on the one GPU (MPS is
requested when concurrent > 1), so small models can share a GPU; order the job list accordingly. The array
script appends "STEP <name>: running|done|FAILED" lines to <root>/slurm/<array>/steps/task-<n>.log (authoritative,
one writer each) and to the shared <root>/slurm/<array>/driver.log (convenience; concurrent appends from several
nodes can corrupt it, so read it with `grep -a`).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from datadec.po.ape import load_instructions
from datadec.po.formats import CANONICAL, load_formats
from datadec.po.sweep import SweepSpec, load_pairs, write_sweep

CLUSTER_ENV = "/scratch/ddr8143/.config/shell/cluster_env.sh"
DATADEC_DIR = "/scratch/ddr8143/drotherm/repos/datadec"


@dataclass(frozen=True, slots=True)
class SlurmSettings:
    account: str = "torch_pr_375_cilvr"
    partition: str = "a100_cilvr"
    gres: str = "gpu:1"
    cpus: int = 4
    mem: str = "60G"
    time: str = "01:00:00"
    max_concurrent: int | None = None  # array throttle; None = no cap on our side (decision 2026-10-06: let the cluster's limits apply)
    chunk_size: int = 14
    concurrent: int = 1  # sweeps per array task, run in parallel on the shared GPU


def build_specs(jobs: list[dict], repo: Path) -> list[SweepSpec]:
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


def task_groups(sweep_dirs: list[Path], concurrent: int) -> list[list[Path]]:
    """Consecutive sweeps grouped `concurrent` per array task (the last group may be shorter)."""
    if concurrent < 1:
        raise ValueError("concurrent must be >= 1")
    return [sweep_dirs[i:i + concurrent] for i in range(0, len(sweep_dirs), concurrent)]


def array_script(array_dir: Path, groups: list[list[Path]], s: SlurmSettings) -> str:
    n = len(groups)
    mps = "#SBATCH --comment=gpu_mps=yes\n" if s.concurrent > 1 else ""
    return f"""#!/bin/bash
#SBATCH --job-name={array_dir.name}
#SBATCH --account={s.account}
#SBATCH --partition={s.partition}
#SBATCH --gres={s.gres}
#SBATCH --cpus-per-task={s.cpus}
#SBATCH --mem={s.mem}
#SBATCH --time={s.time}
#SBATCH --array=0-{n - 1}{f"%{s.max_concurrent}" if s.max_concurrent else ""}
#SBATCH --output={array_dir}/task-%a.out
{mps}export SCRATCH=/scratch/ddr8143
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source {CLUSTER_ENV}
export PATH=/home/ddr8143/.local/bin:$SCRATCH/.local/bin:$PATH
export HOME=/scratch/ddr8143 UV_MANAGED_PYTHON=1 UV_CACHE_DIR=$SCRATCH/.cache/uv UV_LINK_MODE=copy HF_HOME=$SCRATCH/.huggingface
LOG={array_dir}/driver.log
mkdir -p {array_dir}/steps
# per-task log is authoritative (concurrent appends from several nodes have clobbered the shared driver.log with NULs)
log() {{ local line="$(date '+%Y-%m-%d %H:%M:%S') $*"; echo "$line" >> "{array_dir}/steps/task-$SLURM_ARRAY_TASK_ID.log"; echo "$line" >> "$LOG"; }}
run_one() {{  # $1 = sweep dir; runs in the background, one per sweep of this task
  local name=$(basename "$1") t0=$(date +%s)
  log "STEP $name: running (job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID on $(hostname))"
  (cd {DATADEC_DIR} && uv run python -c "
from pathlib import Path
from datadec.po.sweep import run_sweep
import sys
sys.exit(run_sweep(Path('$1'), chunk_size={s.chunk_size}))
")
  local rc=$?
  log "STEP $name: $([ $rc -eq 0 ] && echo done || echo FAILED) in $(( $(date +%s) - t0 ))s"
  return $rc
}}
worst=0
for SWEEP in $(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" {array_dir}/sweeps.txt); do
  run_one "$SWEEP" &
  pids="$pids $!"
done
for pid in $pids; do wait $pid || worst=1; done
exit $worst
"""


def write_array(jobs_path: Path, root: Path, array_name: str, repo: Path, settings: SlurmSettings) -> Path:
    """Write every sweep and the array script; return the array dir (contains array.sbatch)."""
    jobs = json.loads(Path(jobs_path).read_text())["jobs"]
    specs = build_specs(jobs, repo)
    sweep_dirs = [write_sweep(spec, root=root / "sweeps") for spec in specs]
    groups = task_groups(sweep_dirs, settings.concurrent)
    array_dir = root / "slurm" / array_name
    array_dir.mkdir(parents=True, exist_ok=False)
    (array_dir / "sweeps.txt").write_text("".join(" ".join(str(d) for d in g) + "\n" for g in groups))
    (array_dir / "jobs.json").write_text(json.dumps({"jobs": jobs, "settings": asdict(settings)}, indent=1) + "\n")
    (array_dir / "array.sbatch").write_text(array_script(array_dir, groups, settings))
    return array_dir


def submit(array_dir: Path, *, dependency: str | None = None) -> str:
    """sbatch the array; `dependency` is passed through (e.g. "afterany:<jobid>") so arrays run one at a time
    under the per-user GPU cap instead of each claiming the full throttle."""
    cmd = ["sbatch"] + ([f"--dependency={dependency}"] if dependency else []) + [str(array_dir / "array.sbatch")]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    (array_dir / "submit.txt").write_text(out + "\n")
    return out
