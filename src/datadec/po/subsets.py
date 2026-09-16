"""Seeded item subsets so every matched experiment evaluates the same documents."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from datasets import load_dataset

ARC_EASY = ("allenai/ai2_arc", "ARC-Easy")


@dataclass(frozen=True, slots=True)
class ItemSubset:
    dataset_path: str
    dataset_name: str
    split: str
    seed: int
    ids: tuple[str, ...]

    @property
    def n(self) -> int:
        return len(self.ids)

    def to_json(self) -> str:
        return json.dumps(asdict(self) | {"n": self.n}, indent=1) + "\n"

    @classmethod
    def load(cls, path: Path) -> ItemSubset:
        raw = json.loads(Path(path).read_text())
        return cls(
            dataset_path=raw["dataset_path"],
            dataset_name=raw["dataset_name"],
            split=raw["split"],
            seed=int(raw["seed"]),
            ids=tuple(raw["ids"]),
        )


def sample_subset(
    *,
    n: int,
    seed: int,
    dataset_path: str = ARC_EASY[0],
    dataset_name: str = ARC_EASY[1],
    split: str = "test",
    exclude: frozenset[str] = frozenset(),
) -> ItemSubset:
    """Draw n ids uniformly without replacement from the split, deterministically in seed."""
    ds = load_dataset(dataset_path, dataset_name, split=split)
    all_ids = sorted(i for i in ds["id"] if i not in exclude)
    if n > len(all_ids):
        raise ValueError(f"requested {n} ids but only {len(all_ids)} available")
    ids = tuple(random.Random(seed).sample(all_ids, n))
    return ItemSubset(dataset_path, dataset_name, split, seed, ids)


def load_items(subset: ItemSubset) -> list[dict]:
    """Return the full documents for a subset, in subset order."""
    ds = load_dataset(subset.dataset_path, subset.dataset_name, split=subset.split)
    by_id = {row["id"]: row for row in ds}
    return [by_id[i] for i in subset.ids]
