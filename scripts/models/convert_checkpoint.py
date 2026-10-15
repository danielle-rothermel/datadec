from __future__ import annotations

import argparse
import json
from pathlib import Path

from datadec.data.artifacts import DataArtifacts
from datadec.models import DataDecideCheckpoint, convert_and_publish


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--size", required=True)
    parser.add_argument("--seed", type=int, required=True, help="seed ordinal")
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--skip-strict-check",
        action="store_true",
        help="record the strict hf_olmo logit check as skipped",
    )
    args = parser.parse_args()
    checkpoint = DataDecideCheckpoint(
        recipe=args.recipe,
        size=args.size,
        seed_ordinal=args.seed,
        step=args.step,
    )
    result = convert_and_publish(
        checkpoint,
        work_dir=args.work_dir,
        artifacts=DataArtifacts(args.data_dir),
        strict_check=not args.skip_strict_check,
    )
    print(
        json.dumps(
            {
                "repo_id": result.pin.repo_id,
                "revision": result.pin.revision,
                "branch": checkpoint.target_branch,
                "tags": list(result.tags),
                "source": result.source.model_dump(),
                "final_step": result.final.step,
                "final_step_basis": result.final.basis,
                "final_step_disagreements": list(result.final.disagreements),
                "verification": result.verification,
                "weights_sha256": result.conversion.weights_sha256,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
