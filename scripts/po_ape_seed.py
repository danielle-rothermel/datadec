"""Generate the APE starting prompt for one task's GEPA cells (gepa-run-contract).

One GPT-5.6 Sol call on the plain instruction-induction meta prompt with the task's five OLMES demonstrations,
then the prompt cap (<= 150 words and <= 1000 characters) enforced by up to two shortening turns and, if still
over, truncation at the last sentence boundary. Evidence lands under ~/drotherm/data/runs/po/ape/; the seed record
is written to --out. Needs OPENROUTER_API_KEY (run through `mise exec -- uv run ...`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from datadec.po.ape import DEFAULT_ROOT, SOL_SETTINGS, build_seed_meta_prompt, generate_seed, olmes_demos
from datadec.po.subsets import DATASETS

app = typer.Typer()


@app.command()
def main(
    task: Annotated[str, typer.Option("--task")],
    out: Annotated[Path, typer.Option("--out", help="e.g. configs/po/instructions/ape-seed-<task>.json")],
    root: Annotated[Path, typer.Option("--root", help="evidence root")] = DEFAULT_ROOT,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="print the meta prompt; no call")] = False,
    force: Annotated[bool, typer.Option("--force", help="overwrite an existing --out")] = False,
) -> None:
    if task not in DATASETS:
        raise typer.BadParameter(f"task must be one of {sorted(DATASETS)}")
    if dry_run:
        typer.echo(build_seed_meta_prompt(task, olmes_demos(5, task)))
        return
    if out.exists() and not force:
        raise typer.BadParameter(f"{out} exists; pass --force to regenerate")
    record = generate_seed(task, settings=SOL_SETTINGS, root=root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1) + "\n")
    typer.echo(f"{task}: raw {record['raw_words']}w/{record['raw_chars']}c -> final {record['final_words']}w/{record['final_chars']}c "
               f"(shortening turns {record['shortening_turns']}, truncated {record['truncated']}); wrote {out}")


if __name__ == "__main__":
    app()
