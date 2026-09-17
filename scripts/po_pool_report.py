"""Near-duplicate filtering and rule-based grouping of instruction pools.

Reads one or more APE candidates.jsonl files (framed grids or plain pools), computes pairwise string
similarity (difflib ratio on normalized text, plus token Jaccard), groups near-duplicates above
--threshold (first member by id is the representative), tags every candidate with rule-based categories
(answer-form guidance, elimination, common-sense, no-explanation, science mention, descriptor anchor,
letters listed, modal phrasing, length bucket), and writes pool.csv, groups.json and report.md under
--out. Tags are regex heuristics, listed in TAGS; an LLM-judge definition of sameness is a later pass.
"""

from __future__ import annotations

import difflib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

app = typer.Typer()

TAGS: dict[str, str] = {
    "answer_form_letter": r"\bletter\b",
    "answer_form_text": r"\b(word or phrase|short phrase|phrase|text of|words of|short answer|brief)\b",
    "elimination": r"\b(rule out|eliminat)",
    "commonsense": r"common[- ]sense",
    "no_explanation": r"(no explanation|without explanation|no extra|nothing added|with no other|only its|only the|just that|and nothing)",
    "mentions_science": r"\bscien",
    "descriptor_anchor": r"Answer:",
    "letters_listed": r"A, B, C,? (or|and) D|A[–-]D",
    "modal_or_rule": r"^(You must|Always|Rule:|Never)",
    "mentions_multiple_choice": r"multiple[- ]choice|options?\b|choices?\b",
}


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()


def _ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(_norm(a).split()), set(_norm(b).split())
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


@app.command()
def main(
    candidates: Annotated[list[Path], typer.Option("--candidates", help="candidates.jsonl files")],
    out: Annotated[Path, typer.Option("--out")],
    threshold: Annotated[float, typer.Option("--threshold", help="difflib ratio at or above which two texts are near-duplicates")] = 0.8,
) -> None:
    rows = []
    for p in candidates:
        run = json.loads((p.parent / "run.json").read_text())
        for line in p.read_text().splitlines():
            if line.strip():
                c = json.loads(line)
                if c.get("accepted") and c.get("text"):
                    rows.append({"pool": p.parent.name, "formulation": c.get("formulation") or run.get("formulation"),
                                 "uid": f"{p.parent.name}/{c['id']}", **{k: c.get(k) for k in ("id", "operator", "stance", "aware", "seed")},
                                 "text": c["text"], "words": len(c["text"].split()),
                                 "output_tokens": (c.get("usage") or {}).get("output_tokens")})
    df = pd.DataFrame(rows)
    for tag, pat in TAGS.items():
        df[tag] = df["text"].str.contains(pat, regex=True, flags=re.I if tag != "descriptor_anchor" else 0)
    df["length_bucket"] = pd.cut(df["words"], [0, 14, 30, 10_000], labels=["short", "medium", "long"]).astype(str)
    # near-duplicate groups within each formulation
    df["dup_group"] = None
    df["representative"] = True
    pairs = []
    for form, g in df.groupby("formulation"):
        idx = list(g.index)
        group_of = {}
        for a_i, a in enumerate(idx):
            for b in idx[a_i + 1:]:
                r = _ratio(df.at[a, "text"], df.at[b, "text"])
                j = _jaccard(df.at[a, "text"], df.at[b, "text"])
                pairs.append({"formulation": form, "a": df.at[a, "uid"], "b": df.at[b, "uid"], "ratio": round(r, 3), "jaccard": round(j, 3)})
                if r >= threshold:
                    ga, gb = group_of.get(a), group_of.get(b)
                    gid = ga if ga is not None else (gb if gb is not None else a)
                    for k, v in list(group_of.items()):
                        if v in (ga, gb) and v is not None:
                            group_of[k] = gid
                    group_of[a] = gid
                    group_of[b] = gid
        for k, gid in group_of.items():
            df.at[k, "dup_group"] = f"{form}-g{gid}"
            df.at[k, "representative"] = k == gid
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "pool.csv", index=False)
    pd.DataFrame(pairs).sort_values("ratio", ascending=False).to_csv(out / "pairs.csv", index=False)
    groups = {g: list(sub["uid"]) for g, sub in df[df.dup_group.notna()].groupby("dup_group")}
    (out / "groups.json").write_text(json.dumps(groups, indent=1) + "\n")

    md = [f"# Instruction pool report ({len(df)} candidates, {df.formulation.nunique()} formulations; near-duplicate threshold {threshold})", ""]
    for form, g in df.groupby("formulation"):
        n_groups = g.dup_group.nunique()
        n_dups = int((~g.representative).sum())
        md += [f"## {form.upper()} ({len(g)} candidates; {n_groups} near-duplicate groups absorbing {n_dups} candidates; {len(g) - n_dups} distinct)", ""]
        pr = pd.DataFrame(pairs)
        pr = pr[pr.formulation == form]
        md += [f"Pairwise similarity: median ratio {pr.ratio.median():.2f}, max {pr.ratio.max():.2f}; median Jaccard {pr.jaccard.median():.2f}.", ""]
        md += ["### Tag counts by factor", "", "| tag | all | " + " | ".join(f"op={o}" for o in sorted(g.operator.unique())) + " | " + " | ".join(f"st={s}" for s in sorted(g.stance.unique())) + " | aware | not aware |",
               "|---|---|" + "---|" * (g.operator.nunique() + g.stance.nunique() + 2)]
        for tag in list(TAGS) + ["length_bucket"]:
            if tag == "length_bucket":
                continue
            row = [tag, int(g[tag].sum())]
            row += [int(g[g.operator == o][tag].sum()) for o in sorted(g.operator.unique())]
            row += [int(g[g.stance == s][tag].sum()) for s in sorted(g.stance.unique())]
            row += [int(g[g.aware][tag].sum()), int(g[~g.aware][tag].sum())]
            md.append("| " + " | ".join(str(x) for x in row) + " |")
        md += ["", "Words per instruction: " + ", ".join(f"op={o}: {g[g.operator == o].words.mean():.0f}" for o in sorted(g.operator.unique()))
               + "; " + ", ".join(f"st={s}: {g[g.stance == s].words.mean():.0f}" for s in sorted(g.stance.unique()))
               + f"; aware: {g[g.aware].words.mean():.0f}, not aware: {g[~g.aware].words.mean():.0f}.", ""]
        md += ["### Length buckets", "", " ".join(f"{k}={v}" for k, v in Counter(g.length_bucket).items()), ""]
        if groups:
            md += ["### Near-duplicate groups", ""]
            for gid, members in groups.items():
                if gid.startswith(form):
                    md.append(f"- {gid}: " + ", ".join(m.split('/')[-1] for m in members))
            md.append("")
        md += ["### Candidates", ""]
        for _, r in g.sort_values("id").iterrows():
            flag = "" if r.representative else f" (dup of {r.dup_group})"
            md.append(f"- **{r.id}** [{r.operator}/{r.stance}/{'aware' if r.aware else 'plain'}, {r.words} words]{flag}: {r.text}")
        md.append("")
    (out / "report.md").write_text("\n".join(md) + "\n")
    typer.echo("\n".join(md[:60]))
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    app()
