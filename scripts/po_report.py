from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from datadec.po.report import pipeline_report

app = typer.Typer()


def _f(x: float | None, spec: str) -> str:
    return "n/a" if x is None else format(x, spec)


@app.command()
def main(root: Annotated[Path, typer.Argument()], out: Annotated[Path | None, typer.Option("--out")] = None) -> None:
    """Summarize a pipeline root (sweep spreads, baselines, GEPA deltas) as JSON and a short table."""
    rep = pipeline_report(root)
    out = out or (root / "report.json")
    out.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    for s in rep["sweeps"]:
        if s.get("empty"):
            continue
        for form, e in s["formulations"].items():
            pl, pr = e["primary_likelihood"], e["primary"]
            typer.echo(f"{s['sweep']:34s} {form}  pairs={e['n_pairs']:3d}  "
                       f"{e['primary_likelihood_metric']} base={_f(pl.get('baseline'), '.3f')} mean={pl['mean']:.3f} "
                       f"[{pl['min']:.3f},{pl['max']:.3f}] sd={_f(pl['std'], '.3f')}  "
                       f"{e['primary_metric']} base={_f(pr.get('baseline'), '.2f')} mean={pr['mean']:.2f} "
                       f"[{pr['min']:.2f},{pr['max']:.2f}]  fmt_sd={_f(e['format_effect_std'], '.3f')} instr_sd={_f(e['instruction_effect_std'], '.3f')}")
    for name, rows in rep["gepa"].items():
        done = [r for r in rows if r["completed"]]
        typer.echo(f"gepa {name}: {len(done)}/{len(rows)} jobs complete")
        for r in done:
            t = (f"  test seed→gepa primary_likelihood {r['test_seed_primary_likelihood']:.3f}→{r['test_gepa_primary_likelihood']:.3f}"
                 f" primary {r['test_seed_primary']:.2f}→{r['test_gepa_primary']:.2f}") if "test_gepa_primary_likelihood" in r else ""
            typer.echo(f"  {r['group']:5s} r{r['seed_rank']:02d} {r['formulation']} val {r['val_seed']:.3f}→{r['val_best']:.3f} ({r['num_candidates']} cands){t}")
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    app()
