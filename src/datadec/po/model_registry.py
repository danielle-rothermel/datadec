"""Model attributes for the phase-2 tidy tables, derived from (model, revision, dtype, quant).

Every evaluated model resolves to one `ModelInfo`; its `model_key` (`<model>@<revision>@<precision>`, revision
`main` when unpinned) identifies the model in every tidy table, figure and GEPA id. Unknown model names raise:
the registry is closed, so a new family or variant is added here (with its golden test) rather than guessed.

Sources: DataDecide sizes, nominal parameter counts and total training steps come from `configs/catalog.toml`
(through `datadec.data.model_utils`); the intermediate checkpoints are the ones selected in the mid-training and
norm-peak job lists under `configs/po/slurm/` (decision log 2026-10-01 15:01): `arc-mid-*.json` name the
checkpoint nearest 50 % of training, `arc-peak-*.json` the checkpoint at the weight-norm peak. A checkpoint
named in both lists (DataDecide C4 4M, step 2500) is `mid`. Any other pinned step must be a final checkpoint.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from enum import StrEnum, UNIQUE, verify
from functools import cache
from pathlib import Path

from datadec.data.model_utils import create_model_config

SLURM_LISTS = Path(__file__).resolve().parents[3] / "configs" / "po" / "slurm"
CHECKPOINT_LISTS = {"mid": ("arc-mid-datadecide.json", "arc-mid-pythia.json"),
                    "peak": ("arc-peak-datadecide.json", "arc-peak-pythia.json")}
PYTHIA_TOTAL_STEPS = 143000
MAIN_REVISION = "main"
FINAL_FRACTION_MIN = 0.9  # a pinned DataDecide step outside the checkpoint lists must be this close to the end


@verify(UNIQUE)
class Family(StrEnum):
    DATADECIDE = "datadecide"
    PYTHIA = "pythia"
    QWEN3 = "qwen3"
    OLMO2 = "olmo2"
    OLMOE = "olmoe"
    OLMO3 = "olmo3"
    OLMO3_1 = "olmo3.1"


@verify(UNIQUE)
class Variant(StrEnum):
    BASE = "base"
    INSTRUCT = "instruct"
    SFT = "sft"
    DPO = "dpo"
    RLVR = "rlvr"


@verify(UNIQUE)
class Precision(StrEnum):
    FP32_TF32 = "fp32-tf32"
    BF16 = "bf16"
    INT8 = "int8"
    NF4 = "nf4"


@verify(UNIQUE)
class CheckpointKind(StrEnum):
    FINAL = "final"
    MID = "mid"
    PEAK = "peak"


@verify(UNIQUE)
class Recipe(StrEnum):
    DCLM = "dclm"
    DOLMA = "dolma"
    C4 = "c4"


STAGE_INDEX = {Variant.BASE: 0, Variant.SFT: 1, Variant.DPO: 2, Variant.RLVR: 3, Variant.INSTRUCT: 3}
FAMILY_DISPLAY = {Family.DATADECIDE: "DataDecide", Family.PYTHIA: "Pythia", Family.QWEN3: "Qwen3",
                  Family.OLMO2: "OLMo-2", Family.OLMOE: "OLMoE", Family.OLMO3: "OLMo-3", Family.OLMO3_1: "OLMo-3.1"}
RECIPE_DISPLAY = {Recipe.DCLM: "DCLM", Recipe.DOLMA: "Dolma", Recipe.C4: "C4"}
VARIANT_DISPLAY = {Variant.BASE: "", Variant.INSTRUCT: "Instruct", Variant.SFT: "SFT", Variant.DPO: "DPO",
                   Variant.RLVR: "RLVR"}
ERA = {Family.DATADECIDE: 2025, Family.PYTHIA: 2023, Family.QWEN3: 2025, Family.OLMO2: 2025, Family.OLMOE: 2025,
       Family.OLMO3: 2025, Family.OLMO3_1: 2025}

_DD = re.compile(r"^allenai/DataDecide-(?P<recipe>dclm-baseline|dolma1_7|c4)-(?P<size>\d+M|1B)$")
_DD_RECIPE = {"dclm-baseline": Recipe.DCLM, "dolma1_7": Recipe.DOLMA, "c4": Recipe.C4}
_DD_REVISION = re.compile(r"^step(?P<step>\d+)-seed-(?P<seed>[a-z0-9-]+)$")
_PYTHIA = re.compile(r"^EleutherAI/pythia-(?P<size>\d+(?:\.\d+)?[mb])$")
_QWEN3 = re.compile(r"^Qwen/Qwen3-(?P<size>\d+(?:\.\d+)?B)(?P<base>-Base)?$")
_OLMO2 = re.compile(r"^allenai/OLMo-2-0425-(?P<size>\d+B)(?:-(?P<stage>SFT|DPO|RLVR1|Instruct))?$")
_OLMOE = re.compile(r"^allenai/OLMoE-1B-7B-0125(?P<instruct>-Instruct)?$")
_OLMO3_BASE = re.compile(r"^allenai/Olmo-3-(?:1025|1125)-(?P<size>\d+B)$")
_OLMO3_POST = re.compile(r"^allenai/Olmo-(?P<version>3|3\.1)-(?P<size>\d+B)-Instruct(?:-(?P<stage>SFT|DPO))?$")
_STEP = re.compile(r"^step(?P<step>\d+)$")
_STAGE = {None: Variant.INSTRUCT, "SFT": Variant.SFT, "DPO": Variant.DPO, "RLVR1": Variant.RLVR,
          "Instruct": Variant.INSTRUCT}


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """One row of models.csv."""

    model_key: str
    model: str
    revision: str
    family: Family
    display: str
    recipe: Recipe | None
    size_label: str
    params: float
    seed_label: str | None
    variant: Variant
    stage_index: int
    checkpoint_step: int | None
    checkpoint_fraction: float
    checkpoint_kind: CheckpointKind
    precision: Precision
    era: int

    def row(self) -> dict:
        return {k: (str(v) if isinstance(v, StrEnum) else v) for k, v in asdict(self).items()}


def precision(dtype: str | None, quant: str | None) -> Precision:
    """fp32 weights run with TF32 matmuls on the cluster; quantized weights load on top of bfloat16."""
    if quant is not None:
        if dtype != "bfloat16":
            raise ValueError(f"quantized weights expect dtype bfloat16, got {dtype!r}")
        return Precision(quant)
    if dtype == "float32":
        return Precision.FP32_TF32
    if dtype == "bfloat16":
        return Precision.BF16
    raise ValueError(f"unknown dtype {dtype!r} (quant {quant!r})")


def model_key(model: str, revision: str | None, prec: Precision | str) -> str:
    return f"{model}@{revision or MAIN_REVISION}@{prec}"


@cache
def _checkpoint_lists() -> dict[tuple[str, str], CheckpointKind]:
    """(model, revision) -> mid | peak from the job lists; mid wins where both lists name one checkpoint."""
    kinds: dict[tuple[str, str], CheckpointKind] = {}
    for kind in (CheckpointKind.PEAK, CheckpointKind.MID):
        for name in CHECKPOINT_LISTS[kind]:
            for job in json.loads((SLURM_LISTS / name).read_text())["jobs"]:
                kinds[(job["model"], job["revision"])] = kind
    return kinds


def _size_params(size: str) -> float:
    unit = size[-1].lower()
    return float(size[:-1]) * {"m": 1e6, "b": 1e9}[unit]


def _checkpoint(model: str, revision: str | None, step: int | None, total: int | None) -> tuple[CheckpointKind, float]:
    if revision is None or step is None:
        return CheckpointKind.FINAL, 1.0
    kind = _checkpoint_lists().get((model, revision), CheckpointKind.FINAL)
    if total is None:
        raise ValueError(f"{model}@{revision}: no total step count for a pinned checkpoint")
    fraction = step / total
    if kind is CheckpointKind.FINAL:
        if fraction < FINAL_FRACTION_MIN:
            raise ValueError(f"{model}@{revision} is at {fraction:.2f} of training but in neither checkpoint list")
        return kind, 1.0
    return kind, fraction


def _datadecide(model: str, revision: str | None, m: re.Match) -> dict:
    if revision is None or not (r := _DD_REVISION.match(revision)):
        raise ValueError(f"DataDecide revision must be step<N>-seed-<name>: {model}@{revision}")
    recipe, size = _DD_RECIPE[m["recipe"]], m["size"]
    config = create_model_config(size)
    kind, fraction = _checkpoint(model, revision, int(r["step"]), int(config["total_steps"]))
    return dict(family=Family.DATADECIDE, recipe=recipe, size_label=size, params=float(config["nominal_parameter_count"]),
                seed_label=r["seed"], variant=Variant.BASE, checkpoint_step=int(r["step"]),
                checkpoint_kind=kind, checkpoint_fraction=fraction)


def _pythia(model: str, revision: str | None, m: re.Match) -> dict:
    step = None
    if revision is not None:
        if not (r := _STEP.match(revision)):
            raise ValueError(f"Pythia revision must be step<N>: {model}@{revision}")
        step = int(r["step"])
    kind, fraction = _checkpoint(model, revision, step, PYTHIA_TOTAL_STEPS)
    return dict(family=Family.PYTHIA, size_label=m["size"].upper(), params=_size_params(m["size"]),
                variant=Variant.BASE, checkpoint_step=step, checkpoint_kind=kind, checkpoint_fraction=fraction)


def _unpinned(model: str, revision: str | None) -> None:
    if revision is not None:
        raise ValueError(f"{model}: only the released (unpinned) revision is registered, got {revision!r}")


def _attributes(model: str, revision: str | None) -> dict:
    if m := _DD.match(model):
        return _datadecide(model, revision, m)
    if m := _PYTHIA.match(model):
        return _pythia(model, revision, m)
    _unpinned(model, revision)
    if m := _QWEN3.match(model):
        return dict(family=Family.QWEN3, size_label=m["size"], params=_size_params(m["size"]),
                    variant=Variant.BASE if m["base"] else Variant.INSTRUCT)
    if m := _OLMO2.match(model):
        return dict(family=Family.OLMO2, size_label=m["size"], params=_size_params(m["size"]),
                    variant=Variant.BASE if m["stage"] is None else _STAGE[m["stage"]])
    if m := _OLMOE.match(model):
        return dict(family=Family.OLMOE, size_label="1B-7B", params=7e9,
                    variant=Variant.INSTRUCT if m["instruct"] else Variant.BASE)
    if m := _OLMO3_BASE.match(model):
        return dict(family=Family.OLMO3, size_label=m["size"], params=_size_params(m["size"]), variant=Variant.BASE)
    if m := _OLMO3_POST.match(model):
        return dict(family=Family.OLMO3 if m["version"] == "3" else Family.OLMO3_1, size_label=m["size"],
                    params=_size_params(m["size"]), variant=_STAGE[m["stage"]])
    raise ValueError(f"model {model!r} is not in the registry")


def _display(a: dict, prec: Precision) -> str:
    fam = a["family"]
    if fam is Family.DATADECIDE:
        text = f"{RECIPE_DISPLAY[a['recipe']]} {a['size_label']}"
        if a["seed_label"] != "default":
            text += f" ({a['seed_label']})"
    else:
        text = f"{FAMILY_DISPLAY[fam]} {a['size_label']}"
        suffix = "Base" if fam is Family.QWEN3 and a["variant"] is Variant.BASE else VARIANT_DISPLAY[a["variant"]]
        if a["variant"] is Variant.RLVR and fam is Family.OLMO2:
            suffix = "RLVR1"
        text = f"{text} {suffix}".strip()
    if a["checkpoint_kind"] is not CheckpointKind.FINAL:
        text += f" @{a['checkpoint_kind']}"
    if prec is not Precision.FP32_TF32:
        text += f" [{prec}]"
    return text


def describe(model: str, revision: str | None, dtype: str | None, quant: str | None) -> ModelInfo:
    """All models.csv attributes for one evaluated (model, revision, dtype, quant)."""
    prec = precision(dtype, quant)
    a = dict(recipe=None, seed_label=None, checkpoint_step=None, checkpoint_kind=CheckpointKind.FINAL,
             checkpoint_fraction=1.0) | _attributes(model, revision)
    return ModelInfo(model_key=model_key(model, revision, prec), model=model, revision=revision or MAIN_REVISION,
                     family=a["family"], display=_display(a, prec), recipe=a["recipe"], size_label=a["size_label"],
                     params=a["params"], seed_label=a["seed_label"], variant=a["variant"],
                     stage_index=STAGE_INDEX[a["variant"]], checkpoint_step=a["checkpoint_step"],
                     checkpoint_fraction=a["checkpoint_fraction"], checkpoint_kind=a["checkpoint_kind"],
                     precision=prec, era=ERA[a["family"]])
