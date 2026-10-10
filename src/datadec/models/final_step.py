from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow.dataset as ds
from dr_hf import get_checkpoint_branches, parse_branch_name

from datadec.config import load_scaling_law_contract
from datadec.config.checkpoints import load_checkpoint_contract
from datadec.data.read import ProcessedTable, read_processed_table

if TYPE_CHECKING:
    from datadec.data.artifacts import DataArtifacts
    from datadec.models.checkpoints import DataDecideCheckpoint

__all__ = [
    "FinalStepDecision",
    "FinalStepEvidence",
    "choose_final_step",
    "final_step_evidence",
    "parse_final_paths",
    "training_data_name",
]

_UNSHARDED_REVISION = re.compile(r"^step(\d+)-unsharded-hf$")
_FETCH_TIMEOUT_SECONDS = 60


@dataclass(frozen=True, slots=True)
class FinalStepEvidence:
    eval_results_last_step: int | None
    final_paths_step: int | None
    final_paths_commit: str
    paper_training_steps: int
    paper_batch_size: int
    paper_source: str
    source_branch_steps: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FinalStepDecision:
    step: int
    basis: str
    disagreements: tuple[str, ...]
    run_completed: bool
    evidence: FinalStepEvidence


def training_data_name(recipe: str) -> str:
    names = [
        name
        for name, key in load_scaling_law_contract().source_group_map.items()
        if key == recipe
    ]
    if len(names) != 1:
        raise ValueError(f"recipe {recipe!r} maps to {len(names)} training data names")
    return names[0]


def parse_final_paths(text: str, model_name: str) -> int | None:
    steps: list[int] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry["model_name"] != model_name:
            continue
        for revision in entry["revisions"]:
            match = _UNSHARDED_REVISION.match(revision)
            if match is None:
                raise ValueError(f"unexpected final revision {revision!r}")
            steps.append(int(match.group(1)))
    if len(steps) > 1:
        raise ValueError(f"{model_name!r} has several final steps: {steps}")
    return steps[0] if steps else None


def _fetch_final_paths(url: str) -> str:
    with urllib.request.urlopen(url, timeout=_FETCH_TIMEOUT_SECONDS) as response:  # noqa: S310
        return response.read().decode("utf-8")


def _eval_results_last_step(
    artifacts: DataArtifacts, checkpoint: DataDecideCheckpoint
) -> int | None:
    table = read_processed_table(
        artifacts,
        ProcessedTable.OLMES,
        filters=(ds.field("data") == checkpoint.recipe_display_name)
        & (ds.field("params") == checkpoint.size)
        & (ds.field("seed") == checkpoint.seed.name),
        columns=["step"],
    )
    if table.num_rows == 0:
        return None
    return int(max(table.column("step").to_pylist()))


def _source_branch_steps(checkpoint: DataDecideCheckpoint) -> tuple[int, ...]:
    steps = []
    for branch in get_checkpoint_branches(checkpoint.source_repo_id):
        info = parse_branch_name(branch)
        if info.seed == checkpoint.seed.branch_slug and info.step is not None:
            steps.append(info.step)
    return tuple(sorted(steps))


def final_step_evidence(
    checkpoint: DataDecideCheckpoint, artifacts: DataArtifacts
) -> FinalStepEvidence:
    contract = load_checkpoint_contract().final_step
    paper = contract.paper.sizes[checkpoint.size]
    final_paths_step = None
    if checkpoint.seed_ordinal == 0:
        model_name = (
            f"{training_data_name(checkpoint.recipe)}-{checkpoint.size}-"
            f"{checkpoint.seed.value}"
        )
        final_paths_step = parse_final_paths(
            _fetch_final_paths(contract.final_paths_url), model_name
        )
    return FinalStepEvidence(
        eval_results_last_step=_eval_results_last_step(artifacts, checkpoint),
        final_paths_step=final_paths_step,
        final_paths_commit=contract.final_paths_commit,
        paper_training_steps=paper.training_steps,
        paper_batch_size=paper.batch_size,
        paper_source=contract.paper.source,
        source_branch_steps=_source_branch_steps(checkpoint),
    )


def choose_final_step(evidence: FinalStepEvidence) -> FinalStepDecision:
    if evidence.final_paths_step is not None:
        step = evidence.final_paths_step
        basis = "authors' final default-seed checkpoint list"
    else:
        step = evidence.paper_training_steps
        basis = "paper model table training steps"
    if step not in evidence.source_branch_steps:
        raise ValueError(f"chosen final step {step} ({basis}) has no source branch")
    disagreements = []
    if evidence.final_paths_step not in (None, step):
        disagreements.append(f"final-paths list names step {evidence.final_paths_step}")
    if evidence.paper_training_steps != step:
        disagreements.append(
            f"paper table lists {evidence.paper_training_steps} training steps"
        )
    if evidence.eval_results_last_step != step:
        disagreements.append(
            f"DataDecide-eval-results ends at step {evidence.eval_results_last_step}"
        )
    return FinalStepDecision(
        step=step,
        basis=basis,
        disagreements=tuple(disagreements),
        run_completed=step >= evidence.paper_training_steps,
        evidence=evidence,
    )
