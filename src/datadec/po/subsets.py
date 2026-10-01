"""Seeded item subsets so every matched experiment evaluates the same documents.

Split policy (decision log, phase 2, 2026-10-01): D = the split DataDecide evaluated (test where its labels are public,
validation otherwise) only ever feeds Test / Test-Reserve; Train and Train-Dev are seeded draws from the train split
with the OLMES few-shot demonstration items removed; Test is topped up from train when the test labels are hidden.
"""

from __future__ import annotations

import importlib.util
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from datasets import load_dataset

OLMES_REPO = Path.home() / "drotherm" / "repos" / "olmes"
ROW_INDEX = "index"  # id_field value meaning "0-based row position within the split" (OLMES `with_indices` tasks)


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    path: str
    name: str | None
    id_field: str  # native id column, or ROW_INDEX
    eval_split: str  # the split DataDecide / OLMES evaluate on (D)
    demo_source: str  # OLMES fewshot source key; demos always come from the train split
    revision: str | None = None  # HF revision for loading; script-based datasets use the parquet conversion branch
    match_fields: tuple[str, ...] = ()  # columns identifying a demo row when it carries no native id


ARC_EASY = ("allenai/ai2_arc", "ARC-Easy")
PARQUET = "refs/convert/parquet"
DATASETS: dict[str, DatasetSpec] = {
    "arc_easy": DatasetSpec("allenai/ai2_arc", "ARC-Easy", "id", "test", "OLMES:ARC-Easy"),
    "arc_challenge": DatasetSpec("allenai/ai2_arc", "ARC-Challenge", "id", "test", "OLMES:ARC-Challenge"),
    "openbookqa": DatasetSpec("allenai/openbookqa", "main", "id", "test", "OLMES:openbookqa"),
    "csqa": DatasetSpec("tau/commonsense_qa", None, "id", "validation", "OLMES:commonsense_qa"),
    "hellaswag": DatasetSpec("allenai/hellaswag", None, "ind", "validation", "OLMES:hellaswag"),  # ids are ints; 433 duplicate validation rows
    "socialiqa": DatasetSpec("allenai/social_i_qa", None, ROW_INDEX, "validation", "OLMES:social_i_qa", PARQUET, ("context", "question")),
    "piqa": DatasetSpec("ybisk/piqa", None, ROW_INDEX, "validation", "OLMES:piqa", PARQUET, ("goal", "sol1")),
    "winogrande": DatasetSpec("allenai/winogrande", "winogrande_xl", ROW_INDEX, "validation", "OLMES:winogrande", None, ("sentence",)),
}
DATASET_NAMES = {k: v.name for k, v in DATASETS.items()}
# The curated demonstrations OLMES prepends to every ARC prompt (fewshot_sources.py in the fork); kept as pinned constants.
OLMES_ARC_EASY_FEWSHOT_IDS = ("MCAS_2007_8_5189", "Mercury_SC_401169", "MCAS_2004_8_27", "NYSEDREGENTS_2006_8_10", "Mercury_7013388", "Mercury_7179953", "Mercury_7205118", "MCAS_2016_8_13")
OLMES_ARC_CHALLENGE_FEWSHOT_IDS = ('Mercury_SC_415702', 'MCAS_2009_5_6516', 'Mercury_7233695', 'Mercury_7041615', 'MCAS_1998_4_3', 'Mercury_7041860', 'ACTAAP_2013_5_11', 'MDSA_2008_5_30', 'MEA_2016_8_14', 'Mercury_SC_401653', 'Mercury_7106908')


def load_split(spec: DatasetSpec, split: str):
    kw = {"revision": spec.revision} if spec.revision else {}
    return load_dataset(spec.path, spec.name, split=split, **kw)


def olmes_fewshot_source(key: str) -> list[dict]:
    """The fork's FEWSHOT_SOURCES[key] (a pure-data module, loaded by path so datadec needs none of oe_eval's deps)."""
    path = OLMES_REPO / "oe_eval" / "tasks" / "fewshot_sources.py"
    mod_spec = importlib.util.spec_from_file_location("olmes_fewshot_sources", path)
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)
    return mod.FEWSHOT_SOURCES[key]


def demo_exclusions(task: str, split: str, ds=None) -> frozenset[str | int]:
    """Ids (or row positions) in `split` of the OLMES demonstration items for `task`, so Train draws never contain a
    demo. Demos come from the train split, so other splits return an empty set. Matching is by native id when the
    demos carry one, else by the spec's match_fields; every demo must be found exactly once."""
    spec = DATASETS[task]
    if split != "train":
        return frozenset()
    demos = olmes_fewshot_source(spec.demo_source)
    ds = ds if ds is not None else load_split(spec, split)
    if spec.id_field != ROW_INDEX and all(spec.id_field in d for d in demos):
        wanted = {d[spec.id_field] for d in demos}
        present = wanted & set(ds[spec.id_field])
        if present != wanted:
            raise ValueError(f"{task}: demo ids not in train: {sorted(wanted - present)}")
        return frozenset(present)
    key = lambda row: tuple(str(row[f]).strip() for f in spec.match_fields)  # noqa: E731
    positions: dict[tuple, list[int]] = {}
    cols = {f: ds[f] for f in spec.match_fields}
    for i in range(len(ds)):
        positions.setdefault(tuple(str(cols[f][i]).strip() for f in spec.match_fields), []).append(i)
    found: set[str | int] = set()
    ids = None if spec.id_field == ROW_INDEX else ds[spec.id_field]
    for d in demos:
        hits = positions.get(key(d), [])
        if len(hits) != 1:
            raise ValueError(f"{task}: demo {key(d)} matched {len(hits)} train rows")
        found.add(hits[0] if ids is None else ids[hits[0]])
    return frozenset(found)


@dataclass(frozen=True, slots=True)
class ItemSubset:
    dataset_path: str
    dataset_name: str | None
    split: str
    seed: int
    ids: tuple[str | int, ...]  # HellaSwag native ids are ints

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
    dataset_name: str | None = ARC_EASY[1],
    split: str = "test",
    exclude: frozenset[str | int] = frozenset(),
    id_field: str = "id",
    revision: str | None = None,
    ds=None,
) -> ItemSubset:
    """Draw n ids uniformly without replacement from the split, deterministically in seed.

    id_field ROW_INDEX means ids are 0-based row positions (datasets whose OLMES tasks index rows). Otherwise only
    ids that occur exactly once in the split are eligible (HellaSwag "ind" repeats for 433 validation rows)."""
    if ds is None:
        ds = load_dataset(dataset_path, dataset_name, split=split, **({"revision": revision} if revision else {}))
    if id_field == ROW_INDEX:
        all_ids = [i for i in range(len(ds)) if i not in exclude]
    else:
        counts = Counter(ds[id_field])
        all_ids = sorted(i for i, c in counts.items() if c == 1 and i not in exclude)
    if n > len(all_ids):
        raise ValueError(f"requested {n} ids but only {len(all_ids)} available")
    ids = tuple(random.Random(seed).sample(all_ids, n))
    return ItemSubset(dataset_path, dataset_name, split, seed, ids)


def spec_for(subset: ItemSubset) -> DatasetSpec | None:
    return next((v for v in DATASETS.values() if (v.path, v.name) == (subset.dataset_path, subset.dataset_name)), None)


def load_items(subset: ItemSubset) -> list[dict]:
    """Return the full documents for a subset, in subset order."""
    spec = spec_for(subset)
    ds = load_split(spec, subset.split) if spec else load_dataset(subset.dataset_path, subset.dataset_name, split=subset.split)
    id_field = spec.id_field if spec else "id"
    if id_field == ROW_INDEX:
        return [ds[int(i)] for i in subset.ids]
    by_id = {row[id_field]: row for row in ds}
    return [by_id[i] for i in subset.ids]
