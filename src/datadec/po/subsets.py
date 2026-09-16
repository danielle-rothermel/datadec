"""Seeded item subsets so every matched experiment evaluates the same documents."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from datasets import load_dataset

ARC_EASY = ("allenai/ai2_arc", "ARC-Easy")
DATASET_NAMES = {"arc_easy": "ARC-Easy", "arc_challenge": "ARC-Challenge"}
# The five curated demonstrations OLMES prepends to every ARC-Easy prompt (fewshot_sources.py in the fork).
OLMES_ARC_EASY_FEWSHOT_IDS = ("MCAS_2007_8_5189", "Mercury_SC_401169", "MCAS_2004_8_27", "NYSEDREGENTS_2006_8_10", "Mercury_7013388", "Mercury_7179953", "Mercury_7205118", "MCAS_2016_8_13")
OLMES_ARC_CHALLENGE_FEWSHOT_IDS = ('Mercury_SC_415702', 'MCAS_2009_5_6516', 'Mercury_7233695', 'Mercury_7041615', 'MCAS_1998_4_3', 'Mercury_7041860', 'ACTAAP_2013_5_11', 'MDSA_2008_5_30', 'MEA_2016_8_14', 'Mercury_SC_401653', 'Mercury_7106908')


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
