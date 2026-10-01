"""One GEPA run as a trajectory: validation score of every accepted candidate against the metric calls
spent when it was found, with the seed and the best candidate marked, plus a markdown dump of the seed,
the best prompt and the candidate lineage.

Reads ``<run_dir>/gepa_result.json`` and ``<run_dir>/result.json`` written by the fork's local/gepa_arc.py.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated

import matplotlib
import matplotlib.pyplot as plt
import typer

matplotlib.use("Agg")
app = typer.Typer()
COMPONENT = "system_prompt"


def load_run(run_dir: Path) -> dict:
    g = json.loads((run_dir / "gepa_result.json").read_text())
    r = json.loads((run_dir / "result.json").read_text())
    log = (run_dir / "gepa" / "run_log.txt").read_text() if (run_dir / "gepa" / "run_log.txt").exists() else ""
    rejected = len(re.findall(r"not better than old score, skipping|is not better than", log))
    return {"candidates": [c[COMPONENT] for c in g["candidates"]], "parents": g["parents"], "scores": g["val_aggregate_scores"],
            "calls": g["discovery_eval_counts"], "best_idx": g["best_idx"], "total_calls": g["total_metric_calls"],
            "job": r["job"], "reflection_calls": r["reflection_calls"], "rejected": rejected}


def plot(run: dict, out: Path, *, reference: float | None, reference_label: str, title: str | None) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    xs, ys = run["calls"], run["scores"]
    for i, p in enumerate(run["parents"]):
        parent = p[0] if isinstance(p, list) and p else (p if isinstance(p, int) else None)
        if parent is not None:
            ax.plot([xs[parent], xs[i]], [ys[parent], ys[i]], color="0.75", linewidth=1, zorder=1)
    ax.scatter(xs, ys, s=36, color="#3182bd", zorder=3, label="accepted candidate")
    ax.scatter([xs[0]], [ys[0]], s=90, marker="s", color="0.3", zorder=4, label="seed")
    b = run["best_idx"]
    ax.scatter([xs[b]], [ys[b]], s=160, marker="*", color="#d62728", zorder=5, label="best (max val share)")
    if reference is not None:
        ax.axhline(reference, color="0.2", linestyle="--", linewidth=1, label=reference_label)
    for i, (x, y) in enumerate(zip(xs, ys)):
        ax.annotate(str(i), (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8, color="0.3")
    ax.set_xlabel("metric calls (item evaluations) when the candidate was found")
    ax.set_ylabel("validation share (per-char P(correct), 300 dev items)")
    model = run["job"]["model"].split("/")[-1]
    ax.set_title(title or f"GEPA trajectory: {model}, seed {run['job']['seed_instruction']['id']}, {run['reflection_calls']} reflections, {run['rejected']} rejected proposals", fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="0.9")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def markdown(run: dict) -> str:
    job = run["job"]
    b = run["best_idx"]
    lines = [f"# GEPA run: {job['model']} / seed {job['seed_instruction']['id']}", "",
             f"Formulation {job['formulation']}, minibatch {job.get('reflection_minibatch_size')}, budget {job.get('max_metric_calls')} calls"
             + (f", {job['max_proposals']} rounds" if job.get("max_proposals") else "") + f"; {run['reflection_calls']} reflections, "
             f"{len(run['candidates']) - 1} accepted candidates, {run['rejected']} rejected proposals, {run['total_calls']} metric calls.", "",
             f"Seed val share {run['scores'][0]:.4f} -> best {run['scores'][b]:.4f} (candidate {b}, +{run['scores'][b] - run['scores'][0]:.4f}).", "",
             "## Seed prompt", "", "```", job["seed_instruction"]["text"] or "(empty)", "```", "",
             f"## Best prompt (candidate {b}, {len(run['candidates'][b])} characters)", "", "```", run["candidates"][b] or "(empty)", "```", "",
             "## Lineage", "", "| candidate | parent | found at call | val share | chars |", "|---|---|---|---|---|"]
    for i, (c, p, s, x) in enumerate(zip(run["candidates"], run["parents"], run["scores"], run["calls"])):
        parent = p[0] if isinstance(p, list) and p else (p if isinstance(p, int) else None)
        lines.append(f"| {i}{' (best)' if i == b else ''} | {'' if parent is None else parent} | {x} | {s:.4f} | {len(c)} |")
    return "\n".join(lines) + "\n"


@app.command()
def main(
    run_dir: Annotated[Path, typer.Option("--run-dir")],
    out: Annotated[Path, typer.Option("--out", help="output stem; writes <stem>.png and <stem>.md")],
    reference: Annotated[float | None, typer.Option("--reference", help="horizontal reference line, e.g. canonical no-instruction dev share")] = None,
    reference_label: Annotated[str, typer.Option("--reference-label")] = "canonical prompt (no instruction)",
    title: Annotated[str | None, typer.Option("--title")] = None,
) -> None:
    run = load_run(run_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    png, md = Path(str(out) + ".png"), Path(str(out) + ".md")  # stems may contain dots (model sizes)
    plot(run, png, reference=reference, reference_label=reference_label, title=title)
    md.write_text(markdown(run))
    typer.echo(f"{len(run['candidates'])} candidates, best {run['best_idx']} -> {png}")


if __name__ == "__main__":
    app()
