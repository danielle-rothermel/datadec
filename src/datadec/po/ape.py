"""APE-style instruction sampling through dr-providers on OpenRouter.

Every proposer call's full evidence is archived beside the candidates so a
run can be audited or re-scored without touching the network again.

Demonstrations are the task's OLMES few-shot items (the fork's ``FEWSHOT_SOURCES``), rendered exactly as the
canonical RC prompt shows them. ``generate_seed`` produces the one APE starting prompt per task used by the GEPA
cells: a plain instruction-induction meta prompt, GPT-5.6 Sol, and the shared prompt cap (``PromptCap``) enforced
by shortening turns and then sentence-boundary truncation.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import re
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Event

from datasets import load_dataset
from dr_providers import (
    AcceptAllSemanticResponseClassifier,
    GenerationControls,
    HttpProvider,
    MessageRole,
    PromptMessage,
    ProviderCallOutcomeKind,
    ProviderCallRequest,
    ProviderCallState,
    ProviderKind,
    ReasoningEffort,
    StandardProviderCallRetryPolicy,
    Transcript,
    openrouter_chat_config,
    policy_for,
    run_local_provider_call_async,
)

from datadec.po.model_cards import model_card
from datadec.po.subsets import ARC_EASY, DATASETS, olmes_fewshot_source

DEFAULT_ROOT = Path.home() / "drotherm" / "data" / "runs" / "po" / "ape"

META_PROMPTS: dict[str, str] = {
    # Demonstration-conditioned, after Zhou et al. 2022 (forward mode).
    "ape_forward": (
        "I gave a friend an instruction and {k} science questions. The friend read the "
        "instruction and wrote the correct answer to each question. Here are the "
        "question-answer pairs:\n\n{demos}\n\n"
        "Write the instruction I gave my friend. It will be placed once at the top of a "
        "prompt, before several question-answer examples and then a new question, so it "
        "should say how to answer this kind of question in general. Reply with the "
        "instruction only."
    ),
    # Same as ape_forward but demonstrations are shown as lettered choices with a letter answer,
    # so candidates suit the MC formulation (scored continuation is the letter).
    "ape_forward_mc": (
        "I gave a friend an instruction and {k} multiple-choice science questions. The friend read the "
        "instruction and wrote the letter of the correct choice for each question. Here are the "
        "question-answer pairs:\n\n{demos}\n\n"
        "Write the instruction I gave my friend. It will be placed once at the top of a prompt, "
        "before several worked examples and then a new question, so it should say how to answer this "
        "kind of question in general. Reply with the instruction only."
    ),
    "p1_description_mc": (
        "Directly generate an effective system prompt for a language model that answers "
        "grade-school science multiple-choice questions by giving the letter of the correct choice. "
        "The prompt will be placed once above several worked examples and then a new question. "
        "Instruct the model on how to think about and answer such questions. Generate the system prompt only."
    ),
    # Task-description only, after Gao et al. 2026 (no demonstrations shown).
    "p1_description": (
        "Directly generate an effective system prompt for a language model that answers "
        "grade-school science questions. The prompt will be placed once above several "
        "question-answer examples and then a new question. Instruct the model on how to "
        "think about and answer such questions. Generate the system prompt only."
    ),
}


# Framed generation (2026-09-17): demonstrations exactly as the scored prompt renders them, a rewrite
# operator x stance framing pair, and an optional description of the reader model.
FRAMED_META_PROMPTS: dict[str, str] = {  # ARC (science questions); see TASK_FRAMED_META_PROMPTS for other tasks
    "rc": (
        "I gave a friend an instruction and {k} science questions. The friend read the instruction and "
        "wrote the correct answer to each question. Here are the question-answer pairs, exactly as the "
        "friend saw them:\n\n{demos}\n\n{reader}"
        "Write the instruction I gave my friend. It will be placed once at the top of a prompt, before "
        "these examples and then a new question, so it should say how to answer this kind of question in "
        "general. {operator} {stance}Reply with the instruction only."
    ),
    "mc": (
        "I gave a friend an instruction and {k} multiple-choice science questions. The friend read the "
        "instruction and wrote the letter of the correct choice for each question. Here are the "
        "question-answer pairs, exactly as the friend saw them:\n\n{demos}\n\n{reader}"
        "Write the instruction I gave my friend. It will be placed once at the top of a prompt, before "
        "these examples and then a new question, so it should say how to answer this kind of question in "
        "general. {operator} {stance}Reply with the instruction only."
    ),
}
HELLASWAG_FRAMED_META_PROMPTS: dict[str, str] = {
    "rc": (
        "I gave a friend an instruction and {k} short scenarios, each labelled with its activity and cut off "
        "mid-way. The friend read the instruction and wrote the sentence that correctly continues each "
        "scenario. Here are the scenario-continuation pairs, exactly as the friend saw them:\n\n{demos}\n\n{reader}"
        "Write the instruction I gave my friend. It will be placed once at the top of a prompt, before "
        "these examples and then a new scenario, so it should say how to continue this kind of scenario in "
        "general. {operator} {stance}Reply with the instruction only."
    ),
    "mc": (
        "I gave a friend an instruction and {k} short scenarios, each labelled with its activity and cut off "
        "mid-way, with four candidate continuations. The friend read the instruction and wrote the letter of "
        "the continuation that correctly continues each scenario. Here are the scenario-answer pairs, exactly "
        "as the friend saw them:\n\n{demos}\n\n{reader}"
        "Write the instruction I gave my friend. It will be placed once at the top of a prompt, before "
        "these examples and then a new scenario, so it should say how to continue this kind of scenario in "
        "general. {operator} {stance}Reply with the instruction only."
    ),
}
TASK_FRAMED_META_PROMPTS: dict[str, dict[str, str]] = {"arc_easy": FRAMED_META_PROMPTS, "hellaswag": HELLASWAG_FRAMED_META_PROMPTS}
READER_TEMPLATE = (
    "The friend is a language model, not a person: {card}. It never writes anything; it is scored by "
    "the likelihood it assigns to each candidate answer after the prompt, so the instruction can only help "
    "by changing which answer it finds most likely.\n\n"
)


@dataclass(frozen=True, slots=True)
class ProposerSettings:
    model: str = "openai/gpt-5.1"
    temperature: float | None = 1.0  # None omits it (GPT-5.x on OpenRouter rejects temperature and top_p)
    reasoning: str = "low"
    token_limit: int | None = 400  # None leaves the provider's output limit unset
    concurrency: int = 8
    timeout_seconds: float = 180.0
    send_seed: bool = True  # False omits the per-call seed control


# The GEPA-phase proposer and reflector (gepa-run-contract): GPT-5.6 Sol, reasoning medium, 32k output tokens,
# no sampling controls.
SOL_SETTINGS = ProposerSettings(model="openai/gpt-5.6-sol", temperature=None, reasoning="medium", token_limit=32000,
                                concurrency=1, timeout_seconds=900.0, send_seed=False)


def sample_demos(k: int, seed: int) -> list[dict]:
    """k ARC-Easy *train* items, so demonstrations never overlap test subsets."""
    ds = load_dataset(ARC_EASY[0], ARC_EASY[1], split="train")
    rows = list(ds)
    return random.Random(seed).sample(rows, k)


def _hellaswag_preprocess(text: str) -> str:
    """The fork's HellaSwag.preprocess plus a final strip (WikiHow bracket tags removed)."""
    text = text.strip()
    text = re.sub("\\.? \\[title\\]", ". ", text)
    text = re.sub("\\[.*?\\]", "", text)
    return text.replace("  ", " ").strip()


def _arc_demo(d: dict) -> dict:
    return {"id": d["id"], "question": d["question"], "choices": dict(d["choices"]), "answerKey": d["answerKey"]}


def _letters(n: int) -> list[str]:
    return list("ABCDE"[:n])


def _normalize_demo(task: str, d: dict) -> dict:
    """One OLMES few-shot doc in the shared demo shape: id, question, choices {text, label}, answerKey, plus
    `descriptor` (the RC question word) and `cloze` (Winogrande: the prompt is the filled-in sentence).
    Mirrors the fork's :fmt task classes' _process_doc."""
    if task in ("arc_easy", "arc_challenge", "csqa"):
        return _arc_demo(d)
    if task == "openbookqa":
        return {"id": d["id"], "question": d["question_stem"], "choices": dict(d["choices"]), "answerKey": d["answerKey"].strip()}
    if task == "hellaswag":
        endings = [_hellaswag_preprocess(e) for e in d["endings"]]
        return {"id": str(d["ind"]), "question": _hellaswag_preprocess(d["activity_label"] + ": " + d["ctx_a"] + " " + d["ctx_b"].capitalize()),
                "choices": {"text": endings, "label": _letters(len(endings))}, "answerKey": "ABCDE"[int(d["label"])]}
    if task == "socialiqa":
        return {"id": None, "question": d["context"] + " " + d["question"],
                "choices": {"text": [d["answerA"], d["answerB"], d["answerC"]], "label": _letters(3)}, "answerKey": "ABC"[int(d["label"]) - 1]}
    if task == "piqa":
        return {"id": None, "question": d["goal"], "choices": {"text": [d["sol1"], d["sol2"]], "label": _letters(2)},
                "answerKey": "AB"[int(d["label"])], "descriptor": "Goal"}
    if task == "winogrande":
        return {"id": None, "question": d["sentence"], "choices": {"text": [d["option1"], d["option2"]], "label": _letters(2)},
                "answerKey": "AB"[int(d["answer"]) - 1], "cloze": True}
    raise ValueError(f"no demonstration source for task {task!r}")


def olmes_demos(k: int = 5, task: str = "arc_easy") -> list[dict]:
    """The first k OLMES few-shot demonstrations for the task, in OLMES order (what every 5-shot prompt shows)."""
    if task not in DATASETS:
        raise ValueError(f"no demonstration source for task {task!r}")
    raw = olmes_fewshot_source(DATASETS[task].demo_source)
    if len(raw) < k:
        raise ValueError(f"{task}: only {len(raw)} OLMES demonstrations, {k} requested")
    return [_normalize_demo(task, d) for d in raw[:k]]


def _gold_index(d: dict) -> int:
    labels = list(d["choices"]["label"])
    key = d["answerKey"]
    if key not in labels:  # ARC mixes letter and numeric label sets
        key = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E", "A": "1", "B": "2", "C": "3", "D": "4", "E": "5"}[key]
    return labels.index(key)


def _cloze_text(sentence: str, option: str) -> str:
    """Winogrande partial context + target: the sentence with the blank filled (fork Winogrande.partial_*)."""
    blank = sentence.index("_")
    return sentence[:blank] + option + " " + sentence[blank + 1:].strip()


def render_demos(demos: list[dict], formulation: str) -> str:
    """Canonical-format rendering, byte for byte what the scored prompt shows."""
    lines = []
    for d in demos:
        idx = _gold_index(d)
        if d.get("cloze"):
            if formulation != "rc":
                raise ValueError("cloze demonstrations render only in the rc formulation")
            lines.append(_cloze_text(d["question"], d["choices"]["text"][idx]))
        elif formulation == "mc":
            letters = "ABCDE"
            choices = "\n".join(f" {letters[i]}. {t}" for i, t in enumerate(d["choices"]["text"]))
            lines.append(f"Question: {d['question']}\n{choices}\nAnswer: {letters[idx]}")
        else:
            lines.append(f"{d.get('descriptor', 'Question')}: {d['question']}\nAnswer: {d['choices']['text'][idx]}")
    return "\n\n".join(lines)


def render_reader(card: dict[str, str]) -> str:
    parts = [f"{k.replace('_', ' ')}: {v}" for k, v in card.items() if k != "notes"]
    return READER_TEMPLATE.format(card="; ".join(parts))


def build_framed_meta_prompt(formulation: str, demos: list[dict], operator: str, stance: str, card: dict[str, str] | None, task: str = "arc_easy") -> str:
    return TASK_FRAMED_META_PROMPTS[task][formulation].format(
        k=len(demos), demos=render_demos(demos, formulation),
        reader=render_reader(card) if card else "",
        operator=operator, stance=(stance + " ") if stance else "",
    )


def build_meta_prompt(style: str, demos: list[dict]) -> str:
    template = META_PROMPTS[style]
    if "{demos}" not in template:
        return template
    lines = []
    mc = style.endswith("_mc")
    for d in demos:
        labels = list(d["choices"]["label"])
        key = d["answerKey"]
        if key not in labels:  # ARC mixes letter and numeric label sets
            key = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E", "A": "1", "B": "2", "C": "3", "D": "4", "E": "5"}[key]
        idx = labels.index(key)
        if mc:
            letters = "ABCDE"
            choices = "\n".join(f" {letters[i]}. {t}" for i, t in enumerate(d["choices"]["text"]))
            lines.append(f"Question: {d['question']}\n{choices}\nAnswer: {letters[idx]}")
        else:
            lines.append(f"Question: {d['question']}\nAnswer: {d['choices']['text'][idx]}")
    return template.format(k=len(demos), demos="\n\n".join(lines))


def _clean(text: str) -> str:
    t = text.strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'":
        t = t[1:-1].strip()
    return t


def _state(meta_prompt: str | tuple[PromptMessage, ...], settings: ProposerSettings, seed: int | None, classifier_id: str) -> ProviderCallState:
    """meta_prompt is a single user message or a whole conversation (shortening turns)."""
    config = openrouter_chat_config(
        model=settings.model,
        controls=GenerationControls(
            temperature=settings.temperature,
            token_limit=settings.token_limit,
            reasoning=ReasoningEffort(settings.reasoning),
            seed=seed if settings.send_seed else None,
        ),
    )
    messages = (PromptMessage(role=MessageRole.USER, content=meta_prompt),) if isinstance(meta_prompt, str) else meta_prompt
    request = ProviderCallRequest(config=config, transcript=Transcript(messages=messages))
    return ProviderCallState.initial(
        request=request, retry_policy=StandardProviderCallRetryPolicy(), classifier_identifier=classifier_id
    )


async def _gather(provider: HttpProvider, states: list[ProviderCallState], classifier, concurrency: int):
    sem = asyncio.Semaphore(concurrency)

    async def one(state):
        async with sem:
            return await run_local_provider_call_async(
                provider=provider, state=state, classifier=classifier, cancellation=Event()
            )

    return await asyncio.gather(*(one(s) for s in states))


def sample_candidates(
    *,
    meta_prompt: str,
    n: int,
    settings: ProposerSettings,
    seed_base: int,
    out_dir: Path,
) -> list[dict]:
    """Sample n instruction candidates, archiving every call's evidence under out_dir."""
    classifier = AcceptAllSemanticResponseClassifier()
    states = [_state(meta_prompt, settings, seed_base + i, classifier.identifier) for i in range(n)]
    evidence_dir = out_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    with HttpProvider(
        policy=policy_for(
            ProviderKind.OPENROUTER,
            timeout_seconds=settings.timeout_seconds,
            connect_timeout_seconds=30.0,
            idle_timeout_seconds=120.0,
            max_connections=settings.concurrency,
            max_keepalive_connections=settings.concurrency,
            max_request_bytes=1024 * 1024,
            max_response_bytes=8 * 1024 * 1024,
        )
    ) as provider:
        results = asyncio.run(_gather(provider, states, classifier, settings.concurrency))

    candidates = []
    for i, result in enumerate(results):
        cid = f"c{i:03d}"
        (evidence_dir / f"{cid}.json").write_text(json.dumps(result.model_dump(mode="json"), indent=1))
        accepted = result.outcome.kind is ProviderCallOutcomeKind.ACCEPTED
        evidence = result.completed_invocations[-1].observation.evidence if result.completed_invocations else None
        response = evidence.response if evidence is not None else None
        candidates.append(
            {
                "id": cid,
                "seed": seed_base + i,
                "accepted": accepted,
                "text": _clean(response.text) if (accepted and response is not None) else None,
                "stop_reason": getattr(response, "stop_reason", None) if response is not None else None,
                "usage": response.usage.model_dump() if (response is not None and getattr(response, "usage", None)) else None,
                "outcome": str(result.outcome.kind),
            }
        )
    return candidates


def run_ape(
    *,
    style: str,
    n: int,
    settings: ProposerSettings,
    demo_k: int,
    demo_seed: int,
    seed_base: int,
    root: Path = DEFAULT_ROOT,
    slug: str | None = None,
) -> Path:
    started = dt.datetime.now(dt.timezone.utc)
    out_dir = root / f"{started:%Y%m%dT%H%M%SZ}-{slug or style}"
    out_dir.mkdir(parents=True)
    demos = sample_demos(demo_k, demo_seed) if "{demos}" in META_PROMPTS[style] else []
    meta_prompt = build_meta_prompt(style, demos)
    (out_dir / "meta_prompt.txt").write_text(meta_prompt + "\n")
    manifest = {
        "style": style, "n": n, "settings": asdict(settings), "seed_base": seed_base,
        "demo_k": demo_k, "demo_seed": demo_seed, "demo_ids": [d["id"] for d in demos],
        "started_utc": started.isoformat(),
    }
    (out_dir / "run.json").write_text(json.dumps(manifest, indent=1) + "\n")
    candidates = sample_candidates(meta_prompt=meta_prompt, n=n, settings=settings, seed_base=seed_base, out_dir=out_dir)
    with open(out_dir / "candidates.jsonl", "w") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")
    manifest["ended_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest["accepted"] = sum(c["accepted"] for c in candidates)
    manifest["total_tokens"] = sum((c["usage"] or {}).get("total_tokens") or 0 for c in candidates)
    (out_dir / "run.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return out_dir


def run_ape_grid(
    *,
    formulation: str,
    framings: dict,
    aware_model: str | None,
    aware_revision: str | None,
    settings: ProposerSettings,
    seed_base: int,
    root: Path = DEFAULT_ROOT,
    slug: str | None = None,
    task: str = "arc_easy",
) -> Path:
    """One call per (operator, stance, aware) cell; every candidate records its factors."""
    started = dt.datetime.now(dt.timezone.utc)
    out_dir = root / f"{started:%Y%m%dT%H%M%SZ}-{slug or f'framed-{task}-{formulation}'}"
    out_dir.mkdir(parents=True)
    demos = olmes_demos(5, task)
    card = model_card(aware_model, aware_revision) if aware_model else None
    cells = []
    for op in framings["operators"]:
        for st in framings["stances"]:
            for aware in (False, True) if card else (False,):
                cells.append((op, st, aware))
    prompts = [build_framed_meta_prompt(formulation, demos, op["text"], st["text"], card if aware else None, task) for op, st, aware in cells]
    (out_dir / "meta_prompts.jsonl").write_text("".join(json.dumps({"cell": i, "operator": op["id"], "stance": st["id"], "aware": aware, "meta_prompt": pr}) + "\n"
                                                        for i, ((op, st, aware), pr) in enumerate(zip(cells, prompts, strict=True))))
    manifest = {"task": task, "formulation": formulation, "framings": framings, "aware_model": aware_model, "aware_revision": aware_revision,
                "model_card": card, "settings": asdict(settings), "seed_base": seed_base, "demo_ids": [d["id"] for d in demos],
                "n_cells": len(cells), "started_utc": started.isoformat()}
    (out_dir / "run.json").write_text(json.dumps(manifest, indent=1) + "\n")
    classifier = AcceptAllSemanticResponseClassifier()
    states = [_state(pr, settings, seed_base + i, classifier.identifier) for i, pr in enumerate(prompts)]
    evidence_dir = out_dir / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    with HttpProvider(policy=policy_for(ProviderKind.OPENROUTER, timeout_seconds=settings.timeout_seconds, connect_timeout_seconds=30.0,
                                        idle_timeout_seconds=120.0, max_connections=settings.concurrency, max_keepalive_connections=settings.concurrency,
                                        max_request_bytes=1024 * 1024, max_response_bytes=8 * 1024 * 1024)) as provider:
        results = asyncio.run(_gather(provider, states, classifier, settings.concurrency))
    candidates = []
    for i, ((op, st, aware), result) in enumerate(zip(cells, results, strict=True)):
        cid = f"c{i:03d}"
        (evidence_dir / f"{cid}.json").write_text(json.dumps(result.model_dump(mode="json"), indent=1))
        accepted = result.outcome.kind is ProviderCallOutcomeKind.ACCEPTED
        evidence = result.completed_invocations[-1].observation.evidence if result.completed_invocations else None
        response = evidence.response if evidence is not None else None
        candidates.append({
            "id": cid, "seed": seed_base + i, "formulation": formulation, "style": "framed_forward",
            "operator": op["id"], "stance": st["id"], "aware": aware, "aware_model": aware_model if aware else None,
            "accepted": accepted, "text": _clean(response.text) if (accepted and response is not None) else None,
            "stop_reason": getattr(response, "stop_reason", None) if response is not None else None,
            "usage": response.usage.model_dump() if (response is not None and getattr(response, "usage", None)) else None,
            "outcome": str(result.outcome.kind),
        })
    with open(out_dir / "candidates.jsonl", "w") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")
    manifest["ended_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest["accepted"] = sum(c["accepted"] for c in candidates)
    manifest["total_tokens"] = sum((c["usage"] or {}).get("total_tokens") or 0 for c in candidates)
    (out_dir / "run.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return out_dir


def load_instructions(path: Path) -> list[dict]:
    """Accept an APE candidates.jsonl or a JSON list of {id, text}; drop unaccepted."""
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return [{"id": r["id"], "text": r["text"]} for r in rows if r.get("accepted", True) and r.get("text")]
    raw = json.loads(path.read_text())
    return [{"id": r["id"], "text": r["text"]} for r in raw]


# --- APE seed for the GEPA cells (gepa-run-contract: one plain instruction-induction seed per task) -------------

@dataclass(frozen=True, slots=True)
class PromptCap:
    """The instruction length cap shared by the APE seed and every GEPA reflector proposal."""

    words: int = 150  # whitespace-split tokens
    chars: int = 1000
    shortening_turns: int = 2


PROMPT_CAP = PromptCap()


@dataclass(frozen=True, slots=True)
class CapResult:
    text: str
    raw_words: int
    raw_chars: int
    final_words: int
    final_chars: int
    shortening_turns: int
    truncated: bool


def count_words(text: str) -> int:
    return len(text.split())


def within_cap(text: str, cap: PromptCap = PROMPT_CAP) -> bool:
    return count_words(text) <= cap.words and len(text) <= cap.chars


_SENTENCE_END = re.compile(r"[.!?][\"')\]]*(?=\s|$)")
_TOKEN = re.compile(r"\S+")


def truncate_to_cap(text: str, cap: PromptCap = PROMPT_CAP) -> str:
    """The longest prefix ending at a sentence boundary that fits the cap; when no sentence fits, the longest
    whitespace-token prefix that does."""
    text = text.strip()
    if within_cap(text, cap):
        return text
    best = ""
    for m in _SENTENCE_END.finditer(text):
        candidate = text[:m.end()].rstrip()
        if not within_cap(candidate, cap):
            break
        best = candidate
    if best:
        return best
    for m in _TOKEN.finditer(text):
        candidate = text[:m.end()]
        if not within_cap(candidate, cap):
            break
        best = candidate
    return best


def enforce_cap(text: str, shorten: Callable[[str, int], str], cap: PromptCap = PROMPT_CAP) -> CapResult:
    """Up to cap.shortening_turns calls of shorten(current_text, turn) while over the cap, then sentence-boundary
    truncation. shorten returns the rewritten instruction (already cleaned)."""
    raw = text.strip()
    current, turns = raw, 0
    while not within_cap(current, cap) and turns < cap.shortening_turns:
        turns += 1
        current = shorten(current, turns).strip()
    truncated = not within_cap(current, cap)
    final = truncate_to_cap(current, cap) if truncated else current
    return CapResult(text=final, raw_words=count_words(raw), raw_chars=len(raw), final_words=count_words(final),
                     final_chars=len(final), shortening_turns=turns, truncated=truncated)


@dataclass(frozen=True, slots=True)
class TaskWording:
    items: str  # "{k} <items>"
    action: str  # what the friend did with them
    pairs: str  # what the demonstrations are called
    item: str  # one new item
    goal: str  # "say how to <goal> in general"


TASK_WORDING: dict[str, TaskWording] = {
    "arc_easy": TaskWording("grade-school science questions", "wrote the correct answer to each question", "question-answer pairs",
                            "question", "answer this kind of question"),
    "arc_challenge": TaskWording("grade-school science questions", "wrote the correct answer to each question", "question-answer pairs",
                                 "question", "answer this kind of question"),
    "openbookqa": TaskWording("elementary science questions", "wrote the correct answer to each question", "question-answer pairs",
                              "question", "answer this kind of question"),
    "csqa": TaskWording("commonsense questions", "wrote the correct answer to each question", "question-answer pairs",
                        "question", "answer this kind of question"),
    "socialiqa": TaskWording("questions about people in everyday social situations", "wrote the correct answer to each question",
                             "question-answer pairs", "question", "answer this kind of question"),
    "hellaswag": TaskWording("short scenarios, each labelled with its activity and cut off mid-way",
                             "wrote the sentence that correctly continues each scenario", "scenario-continuation pairs",
                             "scenario", "continue this kind of scenario"),
    "piqa": TaskWording("everyday physical goals", "wrote the correct way to achieve each goal", "goal-solution pairs",
                        "goal", "achieve this kind of goal"),
    "winogrande": TaskWording("sentences, each with a blank that one of two candidate names or phrases fills",
                              "completed each sentence by filling its blank with the correct candidate", "completed sentences",
                              "sentence", "fill this kind of blank"),
}

SEED_META_PROMPT = (
    "I gave a friend an instruction and {k} {items}. The friend read the instruction and {action}. Here are the "
    "{pairs}, exactly as the friend saw them:\n\n{demos}\n\n"
    "Write the instruction I gave my friend. It will be placed once at the top of a prompt, before these examples "
    "and then a new {item}, so it should say how to {goal} in general. Write it as a plain statement of what to do, "
    "in at most {words} words. Reply with the instruction only."
)

SHORTEN_PROMPT = (
    "That instruction has {words} words and {chars} characters. Rewrite it within {cap_words} words and "
    "{cap_chars} characters, with the same meaning and the same output format. Reply with the instruction only."
)


def build_seed_meta_prompt(task: str, demos: list[dict], cap: PromptCap = PROMPT_CAP) -> str:
    w = TASK_WORDING[task]
    return SEED_META_PROMPT.format(k=len(demos), items=w.items, action=w.action, pairs=w.pairs,
                                   demos=render_demos(demos, "rc"), item=w.item, goal=w.goal, words=cap.words)


def _call_one(provider: HttpProvider, messages: tuple[PromptMessage, ...], settings: ProposerSettings, evidence_path: Path) -> dict:
    """One proposer call; archives the full evidence and returns {text, usage, cost, stop_reason}. Raises on a
    non-accepted outcome."""
    classifier = AcceptAllSemanticResponseClassifier()
    state = _state(messages, settings, None, classifier.identifier)
    result = asyncio.run(run_local_provider_call_async(provider=provider, state=state, classifier=classifier, cancellation=Event()))
    evidence_path.write_text(json.dumps(result.model_dump(mode="json"), indent=1))
    if result.outcome.kind is not ProviderCallOutcomeKind.ACCEPTED:
        raise RuntimeError(f"proposer call failed ({evidence_path.name}): {result.outcome}")
    response = result.completed_invocations[-1].observation.evidence.response
    return {
        "text": _clean(response.text),
        "usage": response.usage.model_dump() if response.usage is not None else None,
        "cost": response.cost.total_cost if response.cost is not None else None,
        "stop_reason": str(response.stop_reason) if response.stop_reason is not None else None,
    }


def generate_seed(task: str, *, settings: ProposerSettings = SOL_SETTINGS, cap: PromptCap = PROMPT_CAP, k: int = 5,
                  root: Path = DEFAULT_ROOT) -> dict:
    """The APE seed instruction for a task: one call on the plain meta prompt with the task's OLMES demos, then the
    cap enforced (shortening turns continue the conversation). Evidence and a run manifest land in a fresh
    directory under root; returns the seed record written by scripts/po_ape_seed.py."""
    started = dt.datetime.now(dt.timezone.utc)
    out_dir = root / f"{started:%Y%m%dT%H%M%SZ}-seed-{task}"
    evidence_dir = out_dir / "evidence"
    evidence_dir.mkdir(parents=True)
    meta_prompt = build_seed_meta_prompt(task, olmes_demos(k, task), cap)
    (out_dir / "meta_prompt.txt").write_text(meta_prompt + "\n")
    calls: list[dict] = []
    messages: list[PromptMessage] = [PromptMessage(role=MessageRole.USER, content=meta_prompt)]
    policy = policy_for(ProviderKind.OPENROUTER, timeout_seconds=settings.timeout_seconds, connect_timeout_seconds=30.0,
                        idle_timeout_seconds=settings.timeout_seconds, max_connections=1, max_keepalive_connections=1,
                        max_request_bytes=1024 * 1024, max_response_bytes=8 * 1024 * 1024)
    with HttpProvider(policy=policy) as provider:
        first = _call_one(provider, tuple(messages), settings, evidence_dir / "call-00.json")
        calls.append(first | {"kind": "proposal"})
        messages.append(PromptMessage(role=MessageRole.ASSISTANT, content=first["text"]))

        def shorten(current: str, turn: int) -> str:
            messages.append(PromptMessage(role=MessageRole.USER, content=SHORTEN_PROMPT.format(
                words=count_words(current), chars=len(current), cap_words=cap.words, cap_chars=cap.chars)))
            out = _call_one(provider, tuple(messages), settings, evidence_dir / f"call-{turn:02d}.json")
            calls.append(out | {"kind": "shorten"})
            messages.append(PromptMessage(role=MessageRole.ASSISTANT, content=out["text"]))
            return out["text"]

        capped = enforce_cap(first["text"], shorten, cap)
    usage_keys = ("prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens")
    totals = {key: sum((c["usage"] or {}).get(key) or 0 for c in calls) for key in usage_keys}
    costs = [c["cost"] for c in calls if c["cost"] is not None]
    record = {
        "id": f"ape-{task}", "text": capped.text, "meta_prompt": meta_prompt, "model": settings.model,
        "raw_words": capped.raw_words, "raw_chars": capped.raw_chars, "final_words": capped.final_words,
        "final_chars": capped.final_chars, "shortening_turns": capped.shortening_turns, "truncated": capped.truncated,
        "evidence_dir": str(evidence_dir),
    }
    manifest = {"task": task, "settings": asdict(settings), "cap": asdict(cap), "demo_k": k, "started_utc": started.isoformat(),
                "ended_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "calls": calls, "usage": totals,
                "cost_usd": sum(costs) if costs else None, "seed": record}
    (out_dir / "run.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return record
