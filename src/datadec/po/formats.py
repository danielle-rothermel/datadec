"""Prompt-format grid shared with the OLMES fork's oe_eval/tasks/prompt_format.py.

Only option *names* live here; the fork owns the literal strings and the
rendering. The axis and option names must match the fork exactly; the fork
rejects unknown names, so drift fails loudly at run time.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from pathlib import Path

AXES: dict[str, tuple[str, ...]] = {
    "descriptor_pair": ("question_answer", "q_a", "input_output"),
    "descriptor_separator": ("colon_space", "colon_newline", "dash"),
    "descriptor_case": ("title", "upper", "lower"),
    "answer_break": ("newline", "space"),
    "example_separator": ("blank_line", "newline", "rule"),
    "choice_label_style": ("letter_dot", "letter_paren", "letter_rparen", "number_dot", "lower_dot"),
    "choices_header": ("none", "choices", "options"),
}
MC_ONLY_AXES = ("choice_label_style", "choices_header")
CANONICAL: dict[str, str] = {axis: options[0] for axis, options in AXES.items()}


def identity(fmt: dict[str, str]) -> str:
    return "|".join(f"{axis}={fmt[axis]}" for axis in AXES)


def format_id(fmt: dict[str, str]) -> str:
    return hashlib.sha1(identity(fmt).encode()).hexdigest()[:8]


def rc_view(fmt: dict[str, str]) -> dict[str, str]:
    """The same format with MC-only axes reset; RC rendering ignores them."""
    return {axis: (CANONICAL[axis] if axis in MC_ONLY_AXES else fmt[axis]) for axis in AXES}


def validate(fmt: dict[str, str]) -> None:
    for axis, options in AXES.items():
        if fmt.get(axis) not in options:
            raise ValueError(f"format {fmt}: axis {axis!r} must be one of {options}")


def all_formats() -> list[dict[str, str]]:
    return [dict(zip(AXES, combo)) for combo in itertools.product(*AXES.values())]


def sample_formats(n: int, seed: int, *, include_canonical: bool = True) -> list[dict[str, str]]:
    """Uniform sample without replacement over the full grid, canonical first if requested."""
    pool = [f for f in all_formats() if f != CANONICAL]
    k = n - 1 if include_canonical else n
    picked = random.Random(seed).sample(pool, k)
    return ([dict(CANONICAL)] if include_canonical else []) + picked


def write_formats(path: Path, formats: list[dict[str, str]], *, seed: int | None) -> None:
    payload = {"seed": seed, "n": len(formats),
               "formats": [{"id": format_id(f), **f} for f in formats]}
    path.write_text(json.dumps(payload, indent=1) + "\n")


def load_formats(path: Path) -> list[dict[str, str]]:
    raw = json.loads(Path(path).read_text())
    out = []
    for entry in raw["formats"]:
        fmt = {axis: entry[axis] for axis in AXES}
        validate(fmt)
        out.append(fmt)
    return out
