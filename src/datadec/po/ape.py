"""APE-style instruction sampling through dr-providers on OpenRouter.

Every proposer call's full evidence is archived beside the candidates so a
run can be audited or re-scored without touching the network again.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import re
import random
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
from datadec.po.subsets import ARC_EASY, OLMES_ARC_EASY_FEWSHOT_IDS

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
HELLASWAG_DEMOS = Path(__file__).resolve().parents[3] / "configs" / "po" / "demos-hellaswag-olmes.json"  # OLMES:hellaswag source, train split
READER_TEMPLATE = (
    "The friend is a language model, not a person: {card}. It never writes anything; it is scored by "
    "the likelihood it assigns to each candidate answer after the prompt, so the instruction can only help "
    "by changing which answer it finds most likely.\n\n"
)


@dataclass(frozen=True, slots=True)
class ProposerSettings:
    model: str = "openai/gpt-5.1"
    temperature: float = 1.0
    reasoning: str = "low"
    token_limit: int | None = 400  # None leaves the provider's output limit unset
    concurrency: int = 8
    timeout_seconds: float = 180.0


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


def hellaswag_demos(k: int = 5) -> list[dict]:
    """The first k OLMES HellaSwag demonstrations, normalised to the ARC demo shape (question / choices / answerKey)."""
    raw = json.loads(HELLASWAG_DEMOS.read_text())[:k]
    letters = "ABCDE"
    return [{
        "id": str(d["ind"]),
        "question": _hellaswag_preprocess(d["activity_label"] + ": " + d["ctx_a"] + " " + d["ctx_b"].capitalize()),
        "choices": {"text": [_hellaswag_preprocess(e) for e in d["endings"]], "label": list(letters[: len(d["endings"])])},
        "answerKey": letters[int(d["label"])],
    } for d in raw]


def olmes_demos(k: int = 5, task: str = "arc_easy") -> list[dict]:
    """The first k curated OLMES demonstrations for the task, in OLMES order (ARC-Easy from the train split)."""
    if task == "hellaswag":
        return hellaswag_demos(k)
    if task != "arc_easy":
        raise ValueError(f"no demonstration source for task {task!r}")
    want = list(OLMES_ARC_EASY_FEWSHOT_IDS[:k])
    ds = load_dataset(ARC_EASY[0], ARC_EASY[1], split="train")
    by_id = {r["id"]: r for r in ds if r["id"] in set(want)}
    missing = [i for i in want if i not in by_id]
    if missing:
        raise RuntimeError(f"OLMES demonstration ids not in ARC-Easy train: {missing}")
    return [by_id[i] for i in want]


def render_demos(demos: list[dict], formulation: str) -> str:
    """Canonical-format rendering, byte for byte what the scored prompt shows."""
    lines = []
    for d in demos:
        labels = list(d["choices"]["label"])
        key = d["answerKey"]
        if key not in labels:
            key = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E", "A": "1", "B": "2", "C": "3", "D": "4", "E": "5"}[key]
        idx = labels.index(key)
        if formulation == "mc":
            letters = "ABCDE"
            choices = "\n".join(f" {letters[i]}. {t}" for i, t in enumerate(d["choices"]["text"]))
            lines.append(f"Question: {d['question']}\n{choices}\nAnswer: {letters[idx]}")
        else:
            lines.append(f"Question: {d['question']}\nAnswer: {d['choices']['text'][idx]}")
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


def _state(meta_prompt: str, settings: ProposerSettings, seed: int, classifier_id: str) -> ProviderCallState:
    config = openrouter_chat_config(
        model=settings.model,
        controls=GenerationControls(
            temperature=settings.temperature,
            token_limit=settings.token_limit,
            reasoning=ReasoningEffort(settings.reasoning),
            seed=seed,
        ),
    )
    request = ProviderCallRequest(
        config=config,
        transcript=Transcript(messages=(PromptMessage(role=MessageRole.USER, content=meta_prompt),)),
    )
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
