"""Paired seed-vs-optimized effects from GEPA evaluation sweeps."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location("po_gepa_effects", Path(__file__).parents[2] / "scripts" / "po_gepa_effects.py")
po_gepa_effects = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(po_gepa_effects)


def test_paired_effects_pairs_each_job_with_its_own_seed():
    rows = []
    for job, lift in (("j1", 0.05), ("j2", -0.02)):
        for i in range(20):
            rows.append({"model": "m", "formulation": "rc", "instruction_id": f"seed-{job}", "native_id": f"x{i}", "primary": 0.0, "primary_likelihood": 0.30})
            rows.append({"model": "m", "formulation": "rc", "instruction_id": f"gepa-{job}", "native_id": f"x{i}", "primary": 1.0, "primary_likelihood": 0.30 + lift})
    out = pd.DataFrame(po_gepa_effects.paired_effects(pd.DataFrame(rows), rng=np.random.default_rng(0), resamples=200))
    assert set(out["job"]) == {"j1", "j2"} and set(out["metric"]) == {"primary", "primary_likelihood"}
    share = out[out["metric"] == "primary_likelihood"].set_index("job")
    assert abs(share.loc["j1", "diff"] - 0.05) < 1e-9 and abs(share.loc["j2", "diff"] + 0.02) < 1e-9
    assert (out["n"] == 20).all() and (out["ci_lo"] <= out["diff"]).all() and (out["diff"] <= out["ci_hi"]).all()
