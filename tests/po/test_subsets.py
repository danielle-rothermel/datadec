"""Subset sampling: row-position ids, uniqueness filter, and OLMES demo exclusion by id or by content."""

from __future__ import annotations

import pytest

from datadec.po import subsets
from datadec.po.subsets import ROW_INDEX, DatasetSpec, demo_exclusions, sample_subset


class FakeDS:
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, k):
        if isinstance(k, str):
            return [r[k] for r in self.rows]
        return self.rows[k]


def test_row_index_mode_samples_positions() -> None:
    ds = FakeDS([{"goal": f"g{i}"} for i in range(10)])
    s = sample_subset(n=4, seed=0, dataset_path="x", dataset_name=None, split="train", id_field=ROW_INDEX, exclude=frozenset({0, 1}), ds=ds)
    assert len(s.ids) == 4 and all(isinstance(i, int) and 2 <= i < 10 for i in s.ids)
    assert s.ids == sample_subset(n=4, seed=0, dataset_path="x", dataset_name=None, split="train", id_field=ROW_INDEX, exclude=frozenset({0, 1}), ds=ds).ids


def test_native_id_mode_drops_duplicate_ids() -> None:
    ds = FakeDS([{"ind": 1}, {"ind": 2}, {"ind": 2}, {"ind": 3}])
    s = sample_subset(n=2, seed=0, dataset_path="x", dataset_name=None, split="validation", id_field="ind", ds=ds)
    assert set(s.ids) == {1, 3}


def test_demo_exclusion_by_content_and_by_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(subsets.DATASETS, "fake_idx", DatasetSpec("x", None, ROW_INDEX, "validation", "SRC", None, ("goal",)))
    monkeypatch.setitem(subsets.DATASETS, "fake_id", DatasetSpec("x", None, "id", "validation", "SRC2"))
    monkeypatch.setattr(subsets, "olmes_fewshot_source", lambda key: [{"goal": "b "}] if key == "SRC" else [{"id": "q2"}])
    ds = FakeDS([{"goal": "a", "id": "q1"}, {"goal": "b", "id": "q2"}, {"goal": "c", "id": "q3"}])
    assert demo_exclusions("fake_idx", "train", ds) == frozenset({1})
    assert demo_exclusions("fake_id", "train", ds) == frozenset({"q2"})
    assert demo_exclusions("fake_idx", "validation", ds) == frozenset()
    monkeypatch.setattr(subsets, "olmes_fewshot_source", lambda key: [{"goal": "zzz"}])
    with pytest.raises(ValueError):
        demo_exclusions("fake_idx", "train", ds)


def test_registry_shapes() -> None:
    for key, spec in subsets.DATASETS.items():
        assert spec.eval_split in ("test", "validation"), key
        assert spec.demo_source.startswith("OLMES:"), key
        if spec.id_field == ROW_INDEX:
            assert spec.match_fields, key
