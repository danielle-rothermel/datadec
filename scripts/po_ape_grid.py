from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from datadec.po.ape import DEFAULT_ROOT, ProposerSettings, build_framed_meta_prompt, olmes_demos, run_ape_grid
from datadec.po.model_cards import model_card

app = typer.Typer()


@app.command()
def main(
    formulation: Annotated[str, typer.Option("--formulation", help="rc or mc")],
    task: Annotated[str, typer.Option("--task", help="arc_easy or hellaswag (demos and framing text)")] = "arc_easy",
    framings: Annotated[Path, typer.Option("--framings")] = Path("configs/po/framings.json"),
    aware_model: Annotated[str | None, typer.Option("--aware-model", help="target model for the model-aware cells")] = None,
    aware_revision: Annotated[str | None, typer.Option("--aware-revision")] = None,
    model: Annotated[str, typer.Option("--model")] = "openai/gpt-5.1",
    reasoning: Annotated[str, typer.Option("--reasoning")] = "high",
    token_limit: Annotated[int | None, typer.Option("--token-limit", help="omit to leave the output limit unset")] = None,
    temperature: Annotated[float, typer.Option("--temperature")] = 1.0,
    concurrency: Annotated[int, typer.Option("--concurrency")] = 6,
    seed_base: Annotated[int, typer.Option("--seed-base")] = 5000,
    root: Annotated[Path, typer.Option("--root")] = DEFAULT_ROOT,
    slug: Annotated[str | None, typer.Option("--slug")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="print the first aware and non-aware meta-prompts")] = False,
) -> None:
    """Framed APE grid: one proposer call per (operator, stance, model-aware) cell at the canonical format."""
    fr = json.loads(framings.read_text())
    if dry_run:
        demos = olmes_demos(5, task)
        op, st = fr["operators"][0], fr["stances"][0]
        typer.echo("=== not aware ===")
        typer.echo(build_framed_meta_prompt(formulation, demos, op["text"], st["text"], None, task))
        if aware_model:
            typer.echo("=== aware ===")
            typer.echo(build_framed_meta_prompt(formulation, demos, op["text"], st["text"], model_card(aware_model, aware_revision), task))
        return
    settings = ProposerSettings(model=model, temperature=temperature, reasoning=reasoning, token_limit=token_limit, concurrency=concurrency)
    out = run_ape_grid(formulation=formulation, framings=fr, aware_model=aware_model, aware_revision=aware_revision,
                       settings=settings, seed_base=seed_base, root=root, slug=slug, task=task)
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    app()
