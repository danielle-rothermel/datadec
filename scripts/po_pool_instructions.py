"""Turn a pool report (pool.csv from po_pool_report) into an instruction-set JSON for sweeps.

Keeps the representative of every near-duplicate group for one formulation, names each seed
``<proposer-prefix>-<candidate id>`` (``g51-c003``, ``terra-c010``) and records its pool, proposer and
framing factors, which is the shape of configs/po/instructions/framed-rc-27.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

app = typer.Typer()
PROPOSER_PREFIX = {"openai/gpt-5.1": "g51", "openai/gpt-5.6-terra": "terra"}


def representatives(pool: pd.DataFrame, formulation: str) -> list[dict]:
    rows = pool[(pool["formulation"] == formulation) & pool["representative"].astype(bool)]
    out = []
    for _, r in rows.iterrows():
        prefix = PROPOSER_PREFIX.get(r["proposer"])
        if prefix is None:
            raise ValueError(f"no id prefix for proposer {r['proposer']!r}; add it to PROPOSER_PREFIX")
        out.append({"id": f"{prefix}-{r['id']}", "text": r["text"], "pool": r["pool"], "proposer": r["proposer"],
                    "operator": r["operator"], "stance": r["stance"], "aware": bool(r["aware"]), "words": int(r["words"])})
    return out


@app.command()
def main(
    pool_csv: Annotated[Path, typer.Option("--pool-csv")],
    out: Annotated[Path, typer.Option("--out")],
    formulation: Annotated[str, typer.Option("--formulation")] = "rc",
) -> None:
    seeds = representatives(pd.read_csv(pool_csv), formulation)
    if out.exists():
        raise typer.BadParameter(f"{out} exists; instruction sets are immutable once written")
    out.write_text(json.dumps(seeds, indent=1) + "\n")
    by = pd.Series([s["proposer"] for s in seeds]).value_counts().to_dict()
    typer.echo(f"wrote {len(seeds)} seeds to {out} ({by})")


if __name__ == "__main__":
    app()
