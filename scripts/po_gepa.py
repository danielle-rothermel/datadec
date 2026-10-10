from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Annotated

import typer

from datadec.po.sweep import OLMES_REPO

app = typer.Typer()


@app.command()
def main(
    jobs_dir: Annotated[Path, typer.Argument(help="directory of GEPA job JSON files")],
    olmes_repo: Annotated[Path, typer.Option("--olmes-repo")] = OLMES_REPO,
    only: Annotated[str | None, typer.Option("--only", help="substring filter on job id")] = None,
) -> None:
    """Run GEPA jobs sequentially through the fork's driver; skips jobs with a result.json. Collects optimized instructions."""
    jobs = sorted(jobs_dir.glob("*.json"))
    if only:
        jobs = [j for j in jobs if only in j.name]
    root = jobs_dir.parent
    log = open(root / "runner.log", "a")
    failures = 0
    for job_path in jobs:
        job = json.loads(job_path.read_text())
        typer.echo(f"== {job['job_id']}")
        rc = subprocess.run(["uv", "run", "local/gepa_arc.py", "--job", str(job_path)], cwd=olmes_repo, stdout=log, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            failures += 1
            typer.echo(f"   FAILED (exit {rc}); see {root / 'runner.log'}")
    optimized = []
    for job_path in jobs:
        job = json.loads(job_path.read_text())
        res = Path(job["run_dir"]) / "result.json"
        if res.exists():
            r = json.loads(res.read_text())
            optimized.append({"id": f"gepa-{job['job_id']}", "text": r["best_candidate"]["system_prompt"],
                              "job_id": job["job_id"], "model": job["model"], "formulation": job["formulation"],
                              "format_id": job["format_id"], "seed_instruction_id": job["seed_instruction"]["id"],
                              "seed_val_score": r["seed_val_score"], "best_val_score": r["best_val_score"]})
    (root / "optimized_instructions.json").write_text(json.dumps(optimized, indent=1) + "\n")
    pairs = []
    for job_path in jobs:
        job = json.loads(job_path.read_text())
        res = Path(job["run_dir"]) / "result.json"
        if res.exists():
            r = json.loads(res.read_text())
            pairs.append({"format": job["prompt_format"], "format_id": job["format_id"], "formulation": job["formulation"],
                          "instruction": {"id": f"gepa-{job['job_id']}", "text": r["best_candidate"]["system_prompt"]}})
            pairs.append({"format": job["prompt_format"], "format_id": job["format_id"], "formulation": job["formulation"],
                          "instruction": {"id": f"seed-{job['job_id']}", "text": job["seed_instruction"]["text"] or None}})
    (root / "optimized_pairs.json").write_text(json.dumps({"seed": None, "n": len(pairs), "sources": {"jobs_dir": str(jobs_dir)}, "pairs": pairs}, indent=1) + "\n")
    typer.echo(f"{len(optimized)} optimized instructions written to {root / 'optimized_instructions.json'}; {failures} failures")
    raise typer.Exit(1 if failures else 0)


if __name__ == "__main__":
    app()
