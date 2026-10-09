"""Persist and load computed measurements and run metadata."""

from dataclasses import asdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, TypeAdapter

from repro.config import EvaluationConfig
from repro.results import ClaimEvidence


# Explicit persisted column ownership, including nested measurements. Keep this
# independent of dataclass field enumeration so schema changes are deliberate.
_PREDICTION = pa.struct(
    [
        pa.field("evidence_id", pa.int64(), nullable=False),
        pa.field("predictor_size", pa.string(), nullable=False),
        pa.field("predictor_step", pa.int64(), nullable=False),
        pa.field("metric", pa.string(), nullable=False),
        pa.field("compute", pa.float64(), nullable=False),
        pa.field("compute_ratio", pa.float64(), nullable=False),
        pa.field("decision_accuracy", pa.float64(), nullable=False),
        pa.field("decision_accuracy_std", pa.float64(), nullable=False),
        pa.field("seed_accuracies", pa.list_(pa.float64()), nullable=False),
    ]
)
_TASK_MEASUREMENT = pa.struct(
    [
        pa.field("task", pa.string(), nullable=False),
        pa.field("available_comparisons", pa.int64(), nullable=False),
        pa.field("unavailable_comparisons", pa.int64(), nullable=False),
        pa.field("best", _PREDICTION),
    ]
)
CLAIM_EVIDENCE_SCHEMA = pa.schema(
    [
        pa.field("claim_id", pa.string(), nullable=False),
        pa.field("evidence_ids", pa.list_(pa.int64()), nullable=False),
        pa.field("unavailable_ids", pa.list_(pa.int64()), nullable=False),
        pa.field("measurements", pa.list_(_TASK_MEASUREMENT), nullable=False),
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


def write_run_metadata(metadata: RunMetadata, path: Path) -> None:
    path.write_text(metadata.model_dump_json(indent=2) + "\n", encoding="utf-8")


def read_run_metadata(path: Path) -> RunMetadata:
    return RunMetadata.model_validate_json(path.read_text(encoding="utf-8"))


def latest_run(reports_dir: Path, claim_inventory: Path) -> Path | None:
    """Find the newest completed run for this inventory by manifest mtime.

    The evaluator writes run.json last. Directories missing either the manifest
    or claim dataset are incomplete; malformed completed runs remain errors.
    """
    manifests = sorted(
        reports_dir.rglob("run.json"),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
        reverse=True,
    )
    for manifest in manifests:
        if not (manifest.parent / "claim_evidence.parquet").is_file():
            continue
        metadata = read_run_metadata(manifest)
        if (
            metadata.configuration.run.claim_inventory.resolve()
            == claim_inventory.resolve()
        ):
            return manifest.parent
    return None
