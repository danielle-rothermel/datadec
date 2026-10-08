"""Score a converted DataDecide checkpoint on ARC-Challenge and compare it
with the published instance-level results of the original checkpoint.

Run from the repository root, for example:

    uv run python scripts/verify_converted_model.py \
        --recipe dclm-baseline --size 150M --seed 0 --step 38157

Needs the processed OLMES detail tables for the recipe under --data-dir
(``uv run datadec download --olmes-details dclm-baseline``). Writes the
Markdown report to --report and, with --json, a machine-readable summary.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from dr_hf import resolve_model_pin
from dr_providers import (
    LocalDevice,
    LocalDtype,
    LocalModelProvider,
    PromptRendering,
    huggingface_config,
)

from datadec.data.artifacts import DataArtifacts
from datadec.eval.verification_report import (
    VerificationRun,
    render_verification_report,
)
from datadec.eval.verify_converted import (
    PublishedKey,
    compare,
    load_arc_challenge_test,
    load_published,
    score_items,
)
from datadec.models import DataDecideCheckpoint

MAX_SEQUENCE_LENGTH = 2048


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--size", required=True)
    parser.add_argument("--seed", type=int, required=True, help="seed ordinal")
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--device", default=LocalDevice.MPS.value)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--limit", type=int, help="score only the first N items (smoke runs)"
    )
    args = parser.parse_args()

    checkpoint = DataDecideCheckpoint(
        recipe=args.recipe,
        size=args.size,
        seed_ordinal=args.seed,
        step=args.step,
    )
    source = load_arc_challenge_test()
    requests = source.requests[: args.limit] if args.limit else source.requests
    model = resolve_model_pin(checkpoint.target_repo_id, ref=checkpoint.target_branch)
    device = LocalDevice(args.device)
    config = huggingface_config(
        model=model.repo_id,
        revision=model.revision,
        device=device,
        dtype=LocalDtype.FLOAT32,
        batch_size=args.batch_size,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        add_bos_token=False,
        prompt_rendering=PromptRendering.FLAT_TEXT,
    )
    published_key = PublishedKey(
        recipe=checkpoint.recipe,
        params=checkpoint.size,
        seed=checkpoint.seed.name,
        step=checkpoint.step,
    )
    published = load_published(DataArtifacts(args.data_dir), published_key)
    if args.limit:
        kept = {request.native_id for request in requests}
        published = {k: v for k, v in published.items() if k in kept}

    with LocalModelProvider(config=config) as provider:
        start = time.perf_counter()
        scoring = score_items(provider, config, requests)
        wall_time = time.perf_counter() - start

    comparison = compare(scoring.items, published)
    execution = scoring.execution
    import platform  # noqa: PLC0415

    run = VerificationRun(
        dataset=source.pin,
        subset=source.subset,
        model=model,
        model_ref=checkpoint.target_branch,
        config_hash=config.identity_hash,
        device=device.value,
        device_detail=f"{platform.machine()} {platform.platform(terse=True)}",
        dtype=LocalDtype.FLOAT32.value,
        batch_size=args.batch_size,
        torch_version=execution.torch_version,
        transformers_version=execution.transformers_version,
        wall_time_seconds=wall_time,
        score_requests=2 * len(requests),
        published=published_key,
        comparison=comparison,
        run_date=datetime.now(UTC).strftime("%Y-%m-%d"),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_verification_report(run))
    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "model": model.model_dump(),
                    "dataset": source.pin.model_dump(),
                    "subset": str(source.subset.key),
                    "items": comparison.items,
                    "wall_time_seconds": wall_time,
                    "comparison": asdict(comparison),
                },
                indent=2,
                default=str,
            )
        )
    print(args.report.read_text())


if __name__ == "__main__":
    main()
