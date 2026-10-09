"""Persist and load computed measurements and run metadata."""

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, TypeAdapter

from repro.config import EvaluationConfig
from repro.results import ClaimEvidence
from repro.observations import RankingObservation, observations_from_frame


# Explicit persisted link ownership. Ranking measurements come from rankings.parquet.
_TASK_EVIDENCE = pa.struct(
    [
        pa.field("task", pa.string(), nullable=False),
        pa.field("evidence_ids", pa.list_(pa.int64()), nullable=False),
        pa.field("unavailable_ids", pa.list_(pa.int64()), nullable=False),
        pa.field("best_per_seed_evidence_id", pa.int64()),
        pa.field("best_aggregate_evidence_id", pa.int64()),
    ]
)
CLAIM_EVIDENCE_SCHEMA = pa.schema(
    [
        pa.field("claim_id", pa.string(), nullable=False),
        pa.field("tasks", pa.list_(_TASK_EVIDENCE), nullable=False),
        pa.field("related_tables", pa.list_(pa.string()), nullable=False),
    ]
)
_EVIDENCE_ADAPTER = TypeAdapter(tuple[ClaimEvidence, ...])


class InputArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Path
    sha256: str


class RunMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input: InputArtifact
    configuration: EvaluationConfig


def write_claim_evidence(results: tuple[ClaimEvidence, ...], path: Path) -> None:
    table = pa.Table.from_pylist(
        [asdict(result) for result in results], schema=CLAIM_EVIDENCE_SCHEMA
    )
    pq.write_table(table, path)


def read_claim_evidence(path: Path) -> tuple[ClaimEvidence, ...]:
    table = pq.read_table(path)
    if not table.schema.equals(CLAIM_EVIDENCE_SCHEMA, check_metadata=False):
        raise ValueError("unexpected claim-evidence dataset schema")
    return _EVIDENCE_ADAPTER.validate_python(table.to_pylist())


def read_rankings(path: Path) -> Mapping[int, RankingObservation]:
    return observations_from_frame(pd.read_parquet(path))


def write_run_metadata(metadata: RunMetadata, path: Path) -> None:
    path.write_text(metadata.model_dump_json(indent=2) + "\n", encoding="utf-8")


def read_run_metadata(path: Path) -> RunMetadata:
    return RunMetadata.model_validate_json(path.read_text(encoding="utf-8"))


def latest_run(reports_dir: Path, claim_inventory: Path) -> Path | None:
    """Find the newest completed run for this inventory by manifest mtime.

    The evaluator writes run.json last. Directories missing either the manifest
    or either dataset are incomplete; malformed completed runs remain errors.
    """
    manifests = sorted(
        reports_dir.rglob("run.json"),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
        reverse=True,
    )
    for manifest in manifests:
        if not all(
            (manifest.parent / name).is_file()
            for name in ("claim_evidence.parquet", "rankings.parquet")
        ):
            continue
        metadata = read_run_metadata(manifest)
        if (
            metadata.configuration.run.claim_inventory.resolve()
            == claim_inventory.resolve()
        ):
            return manifest.parent
    return None
