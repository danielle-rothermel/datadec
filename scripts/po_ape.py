from __future__ import annotations

from typing import Annotated

import typer

from datadec.po.ape import META_PROMPTS, ProposerSettings, build_meta_prompt, run_ape, sample_demos

app = typer.Typer()


@app.command()
def main(
    style: Annotated[str, typer.Option("--style")] = "ape_forward",
    n: Annotated[int, typer.Option("--n")] = 32,
    model: Annotated[str, typer.Option("--model")] = "openai/gpt-5.1",
    temperature: Annotated[float, typer.Option("--temperature")] = 1.0,
    reasoning: Annotated[str, typer.Option("--reasoning")] = "low",
    token_limit: Annotated[int, typer.Option("--token-limit")] = 400,
    concurrency: Annotated[int, typer.Option("--concurrency")] = 8,
    demo_k: Annotated[int, typer.Option("--demo-k")] = 5,
    demo_seed: Annotated[int, typer.Option("--demo-seed")] = 0,
    seed_base: Annotated[int, typer.Option("--seed-base")] = 1000,
    slug: Annotated[str | None, typer.Option("--slug")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Sample instruction candidates from a proposer on OpenRouter; archives all evidence."""
    if style not in META_PROMPTS:
        raise typer.BadParameter(f"style must be one of {sorted(META_PROMPTS)}")
    if dry_run:
        demos = sample_demos(demo_k, demo_seed) if "{demos}" in META_PROMPTS[style] else []
        typer.echo(build_meta_prompt(style, demos))
        return
    settings = ProposerSettings(model=model, temperature=temperature, reasoning=reasoning,
                                token_limit=token_limit, concurrency=concurrency)
    out = run_ape(style=style, n=n, settings=settings, demo_k=demo_k, demo_seed=demo_seed,
                  seed_base=seed_base, slug=slug)
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    app()
