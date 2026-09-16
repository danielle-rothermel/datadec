"""APE-style instruction sampling through dr-providers on OpenRouter.

Every proposer call's full evidence is archived beside the candidates so a
run can be audited or re-scored without touching the network again.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
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

from datadec.po.subsets import ARC_EASY

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
    # Task-description only, after Gao et al. 2026 (no demonstrations shown).
    "p1_description": (
        "Directly generate an effective system prompt for a language model that answers "
        "grade-school science questions. The prompt will be placed once above several "
        "question-answer examples and then a new question. Instruct the model on how to "
        "think about and answer such questions. Generate the system prompt only."
    ),
}


@dataclass(frozen=True, slots=True)
class ProposerSettings:
    model: str = "openai/gpt-5.1"
    temperature: float = 1.0
    reasoning: str = "low"
    token_limit: int = 400
    concurrency: int = 8
    timeout_seconds: float = 180.0


def sample_demos(k: int, seed: int) -> list[dict]:
    """k ARC-Easy *train* items, so demonstrations never overlap test subsets."""
    ds = load_dataset(ARC_EASY[0], ARC_EASY[1], split="train")
    rows = list(ds)
    return random.Random(seed).sample(rows, k)


def build_meta_prompt(style: str, demos: list[dict]) -> str:
    template = META_PROMPTS[style]
    if "{demos}" not in template:
        return template
    lines = []
    for d in demos:
        labels = list(d["choices"]["label"])
        key = d["answerKey"]
        if key not in labels:  # ARC mixes letter and numeric label sets
            key = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E", "A": "1", "B": "2", "C": "3", "D": "4", "E": "5"}[key]
        answer = d["choices"]["text"][labels.index(key)]
        lines.append(f"Question: {d['question']}\nAnswer: {answer}")
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


def load_instructions(path: Path) -> list[dict]:
    """Accept an APE candidates.jsonl or a JSON list of {id, text}; drop unaccepted."""
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        return [{"id": r["id"], "text": r["text"]} for r in rows if r.get("accepted", True) and r.get("text")]
    raw = json.loads(path.read_text())
    return [{"id": r["id"], "text": r["text"]} for r in raw]
