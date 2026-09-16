from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from datadec.po.report import pipeline_report

app = typer.Typer()


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
            cp, ar = e["correct_prob"], e["acc_raw"]
            typer.echo(f"{s['sweep']:34s} {form}  pairs={e['n_pairs']:3d}  correct_prob base={cp.get('baseline', float('nan')):.3f} "
                       f"mean={cp['mean']:.3f} [{cp['min']:.3f},{cp['max']:.3f}] sd={cp['std']:.3f}  acc base={ar.get('baseline', float('nan')):.2f} "
                       f"mean={ar['mean']:.2f}  fmt_sd={e['format_effect_std']:.3f} instr_sd={e['instruction_effect_std']:.3f}")
    for name, rows in rep["gepa"].items():
        done = [r for r in rows if r["completed"]]
        typer.echo(f"gepa {name}: {len(done)}/{len(rows)} jobs complete")
        for r in done:
            t = f"  test seed→gepa correct_prob {r.get('test_seed_correct_prob', float('nan')):.3f}→{r.get('test_gepa_correct_prob', float('nan')):.3f}" if "test_gepa_correct_prob" in r else ""
            typer.echo(f"  {r['group']:5s} r{r['seed_rank']:02d} {r['formulation']} val {r['val_seed']:.3f}→{r['val_best']:.3f} ({r['num_candidates']} cands){t}")
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    app()
