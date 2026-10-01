"""Write a set of sweeps and submit them as one Slurm job array (one GPU per sweep).

A job list is a JSON file: {"jobs": [{"name", "model", "revision"?, "subset", "formulations"?, "pairs"?,
"formats"?, "instructions"?, "num_shots"?, "batch_size"?, "dtype"?, "use_cache"?}, ...]}. Each job becomes
one sweep directory under <root>/sweeps/<name> (written up front, so the array script only runs them)
and one array task. The array script appends "STEP <name>: running|done|FAILED" lines to
<root>/slurm/<array>/driver.log, the same interface the local drivers used, so a monitor can tail it.
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
    account: str = "torch_pr_375_cds"
    partition: str = "a100_cds"
    gres: str = "gpu:1"
    cpus: int = 4
    mem: str = "60G"
    time: str = "00:30:00"
    max_concurrent: int = 23
    chunk_size: int = 14


def build_specs(jobs: list[dict], repo: Path) -> list[SweepSpec]:
    specs = []
    for j in jobs:
        specs.append(SweepSpec(
            name=j["name"], subset_path=repo / j["subset"], model=j["model"], revision=j.get("revision"),
            task=j.get("task", "arc_easy"),
            formulations=tuple(j.get("formulations", ["rc", "mc"])),
            formats=tuple(load_formats(repo / j["formats"])) if j.get("formats") else (dict(CANONICAL),),
            instructions=tuple(load_instructions(repo / j["instructions"])) if j.get("instructions") else (),
            pairs=tuple(load_pairs(repo / j["pairs"])) if j.get("pairs") else (),
            num_shots=int(j.get("num_shots", 5)), device="cuda", batch_size=int(j.get("batch_size", 64)),
            dtype=j.get("dtype", "float32"), use_cache=bool(j.get("use_cache", True)),
        ))
    return specs


def array_script(array_dir: Path, sweep_dirs: list[Path], s: SlurmSettings) -> str:
    n = len(sweep_dirs)
    return f"""#!/bin/bash
#SBATCH --job-name={array_dir.name}
#SBATCH --account={s.account}
#SBATCH --partition={s.partition}
#SBATCH --gres={s.gres}
#SBATCH --cpus-per-task={s.cpus}
#SBATCH --mem={s.mem}
#SBATCH --time={s.time}
#SBATCH --array=0-{n - 1}%{s.max_concurrent}
#SBATCH --output={array_dir}/task-%a.out
export SCRATCH=/scratch/ddr8143
source {CLUSTER_ENV}
export PATH=/home/ddr8143/.local/bin:$SCRATCH/.local/bin:$PATH
export HOME=/scratch/ddr8143 UV_MANAGED_PYTHON=1 UV_CACHE_DIR=$SCRATCH/.cache/uv UV_LINK_MODE=copy HF_HOME=$SCRATCH/.huggingface
LOG={array_dir}/driver.log
log() {{ echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG"; }}
SWEEP=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" {array_dir}/sweeps.txt)
name=$(basename "$SWEEP")
log "STEP $name: running (job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID on $(hostname))"
t0=$(date +%s)
cd {DATADEC_DIR} && uv run python -c "
from pathlib import Path
from datadec.po.sweep import run_sweep
import sys
sys.exit(run_sweep(Path('$SWEEP'), chunk_size={s.chunk_size}))
"
rc=$?
log "STEP $name: $([ $rc -eq 0 ] && echo done || echo FAILED) in $(( $(date +%s) - t0 ))s"
exit $rc
"""


def write_array(jobs_path: Path, root: Path, array_name: str, repo: Path, settings: SlurmSettings) -> Path:
    """Write every sweep and the array script; return the array dir (contains array.sbatch)."""
    jobs = json.loads(Path(jobs_path).read_text())["jobs"]
    specs = build_specs(jobs, repo)
    sweep_dirs = [write_sweep(spec, root=root / "sweeps") for spec in specs]
    array_dir = root / "slurm" / array_name
    array_dir.mkdir(parents=True, exist_ok=False)
    (array_dir / "sweeps.txt").write_text("".join(f"{d}\n" for d in sweep_dirs))
    (array_dir / "jobs.json").write_text(json.dumps({"jobs": jobs, "settings": asdict(settings)}, indent=1) + "\n")
    (array_dir / "array.sbatch").write_text(array_script(array_dir, sweep_dirs, settings))
    return array_dir


def submit(array_dir: Path) -> str:
    out = subprocess.run(["sbatch", str(array_dir / "array.sbatch")], capture_output=True, text=True, check=True).stdout.strip()
    (array_dir / "submit.txt").write_text(out + "\n")
    return out
