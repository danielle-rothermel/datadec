"""Run the paper-specific single-scale experiments on local processed OLMES."""

import argparse
from dataclasses import asdict
from hashlib import file_digest
import json
from pathlib import Path

from repro.claims import load_claims
from repro.evaluate import (
    curve_summary,
    evaluate_claims,
    proxy_comparisons,
    recipe_crossovers,
)
from repro.ranking import (
    METRICS,
    SMALL_SEEDS,
    TARGET_SEEDS,
    load_evaluations,
    matched_compute_comparisons,
    observed_checkpoints,
    sweep_rankings,
)


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/repro/ranking"
    )
    parser.add_argument(
        "--matched-compute-tolerance",
        type=float,
        default=0.05,
        help="maximum relative compute undershoot (default: 0.05)",
    )
    args = parser.parse_args()
    if not 0 <= args.matched_compute_tolerance <= 1:
        parser.error("--matched-compute-tolerance must be between 0 and 1")
    source = args.data_dir / "processed/olmes.parquet"
    print(f"Reading {source}", flush=True)
    with source.open("rb") as file:
        source_sha256 = file_digest(file, "sha256").hexdigest()
    evaluations = load_evaluations(args.data_dir)
    sweep = sweep_rankings(
        evaluations, progress=lambda message: print(message, flush=True)
    )
    claims = load_claims(ROOT / "configs/repro_claims/magnusson2025-datadecide.toml")
    evidence = evaluate_claims(sweep)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    sweep.to_parquet(output / "rankings.parquet")
    curve_summary(sweep).to_csv(output / "curves.csv", index=False)
    proxy_comparisons(sweep).to_parquet(
        output / "proxy_comparisons.parquet", index=False
    )
    recipe_crossovers(evaluations).to_csv(output / "recipe_crossovers.csv", index=False)
    matched_compute_comparisons(
        sweep,
        relative_tolerance=args.matched_compute_tolerance,
    ).to_csv(output / "matched_compute.csv", index=False)
    observed_checkpoints(evaluations).to_csv(output / "checkpoints.csv", index=False)
    report = {
        "input": {"path": str(source.resolve()), "sha256": source_sha256},
        "target": {"size": "1B", "step": 69369, "seeds": TARGET_SEEDS},
        "predictor_seeds": {"1B": TARGET_SEEDS, "other_sizes": SMALL_SEEDS},
        "metrics": METRICS,
        "matched_compute_tolerance": args.matched_compute_tolerance,
        "skipped_math_code_claims": [
            "DD-0017",
            "DD-0018",
            "DD-0213",
            "DD-0222",
            "DD-0224",
            "DD-0226",
            "DD-0227",
        ],
        "claims": [
            {
                **asdict(item),
                "statement": claims[item.claim_id].statement,
                "locations": [
                    asdict(location) for location in claims[item.claim_id].locations
                ],
            }
            for item in evidence
        ],
    }
    (output / "claims.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    lines = [
        "# Single-scale claim evidence",
        "",
        "Numerical verdicts concern the observed grid. Qualitative claims require judgment.",
        "",
    ]
    for item in evidence:
        lines.extend(
            [
                f"## {item.claim_id}: {item.status}",
                "",
                claims[item.claim_id].statement,
                "",
                item.summary,
                "",
                f"Remaining judgment / criterion: {item.judgment}",
                "",
            ]
        )
    (output / "claims.md").write_text("\n".join(lines))
    for status in sorted({item.status for item in evidence}):
        print(
            f"{status}: {sum(item.status == status for item in evidence)}", flush=True
        )
    print(f"Wrote claim evidence to {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
