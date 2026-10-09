from __future__ import annotations

import gzip
import json

import pytest

from datadec.eval.olmes_rc import (
    UNCONDITIONAL_CONTEXT,
    arc_challenge_doc,
    arc_challenge_fewshot_prefix,
    requests_for_doc,
)
from datadec.eval.olmes_shots import ARC_CHALLENGE_SHOTS

# OLMES paper (arXiv 2406.08446), Appendix H, Figure 5: the full 5-shot
# ARC-Challenge cloze prompt and its completion. Transcribed from the figure;
# its line layout is read as one newline between the question and answer
# lines and a blank line between examples.
FIGURE_5_PROMPT = (
    "Question: George wants to warm his hands quickly by rubbing them. "
    "Which skin surface will produce the most heat?\n"
    "Answer: dry palms\n\n"
    "Question: Which of the following statements best explains why magnets "
    "usually stick to a refrigerator door?\n"
    "Answer: The refrigerator door contains iron.\n\n"
    "Question: A fold observed in layers of sedimentary rock most likely "
    "resulted from the\n"
    "Answer: converging of crustal plates.\n\n"
    "Question: Which of these do scientists offer as the most recent "
    "explanation as to why many plants and animals died out at the end of "
    "the Mesozoic era?\n"
    "Answer: impact of an asteroid created dust that blocked the sunlight\n\n"
    "Question: Which of the following is a trait that a dog does NOT inherit "
    "from its parents?\n"
    "Answer: the size of its appetite\n\n"
    "Question: A boat is acted on by a river current flowing north and by "
    "wind blowing on its sails. The boat travels northeast. In which "
    "direction is the wind most likely applying force to the sails of the "
    "boat?\n"
    "Answer:"
)
FIGURE_5_COMPLETION = " east"
# The Figure 5 example item, the sixth entry of OLMES:ARC-Challenge.
FIGURE_5_ITEM = {
    "id": "Mercury_7041860",
    "question": (
        "A boat is acted on by a river current flowing north and by wind "
        "blowing on its sails. The boat travels northeast. In which direction "
        "is the wind most likely applying force to the sails of the boat?"
    ),
    "choices": {
        "text": ["west", "east", "north", "south"],
        "label": ["A", "B", "C", "D"],
    },
    "answerKey": "B",
}


def test_figure_5_prompt_is_reproduced_byte_for_byte() -> None:
    requests = requests_for_doc(arc_challenge_doc(FIGURE_5_ITEM["id"], FIGURE_5_ITEM))

    assert requests.context == FIGURE_5_PROMPT
    assert requests.continuations[requests.gold_index] == FIGURE_5_COMPLETION
    assert requests.continuations == (" west", " east", " north", " south")
    assert requests.unconditional_context == "Answer:"


def test_fewshot_prefix_uses_the_first_five_curated_shots() -> None:
    assert [shot["id"] for shot in ARC_CHALLENGE_SHOTS] == [
        "Mercury_SC_415702",
        "MCAS_2009_5_6516",
        "Mercury_7233695",
        "Mercury_7041615",
        "MCAS_1998_4_3",
    ]
    assert arc_challenge_fewshot_prefix().endswith(
        "Answer: the size of its appetite\n\n"
    )


@pytest.mark.parametrize(
    ("answer_key", "labels", "gold"),
    [
        ("C", ["A", "B", "C", "D"], 2),
        ("3", ["1", "2", "3", "4"], 2),
        ("5", ["1", "2", "3", "4", "5"], 4),
        ("E", ["A", "B", "C", "D", "E"], 4),
    ],
)
def test_answer_keys_map_numeric_to_letters(
    answer_key: str, labels: list[str], gold: int
) -> None:
    fields = {
        "question": "q",
        "choices": {"text": [f"c{i}" for i in labels], "label": labels},
        "answerKey": answer_key,
    }

    assert arc_challenge_doc("x", fields).gold_index == gold


def test_answer_key_beyond_choices_is_refused() -> None:
    fields = {
        "question": "q",
        "choices": {"text": ["a", "b"], "label": ["A", "B"]},
        "answerKey": "C",
    }

    with pytest.raises(ValueError, match="beyond the choices"):
        arc_challenge_doc("x", fields)


EVAL_INSTANCES_REPO = "allenai/DataDecide-eval-instances"
EVAL_INSTANCES_REVISION = "23f3b2e186ca6c39026e3efa00e4af397680c075"
REQUESTS_FILE = "requests/arc_challenge-requests.jsonl.gz"
UNCONDITIONAL_DOC_ID_OFFSET = 1_000_000


@pytest.mark.hub
def test_every_request_string_equals_the_published_requests() -> None:
    from huggingface_hub import hf_hub_download

    from datadec.eval.verify_converted import load_arc_challenge_test

    path = hf_hub_download(
        EVAL_INSTANCES_REPO,
        REQUESTS_FILE,
        repo_type="dataset",
        revision=EVAL_INSTANCES_REVISION,
    )
    with gzip.open(path, "rt") as file:
        published = [json.loads(line) for line in file]
    ours = {
        requests.native_id: requests for requests in load_arc_challenge_test().requests
    }

    seen = set()
    for entry in published:
        requests = ours[entry["native_id"]]
        request = entry["request"]
        unconditional = entry["doc_id"] >= UNCONDITIONAL_DOC_ID_OFFSET
        expected_context = UNCONDITIONAL_CONTEXT if unconditional else requests.context
        assert request["context"] == expected_context, entry["native_id"]
        assert request["continuation"] == requests.continuations[entry["idx"]]
        assert entry["label"] == requests.gold_index
        seen.add((entry["native_id"], unconditional, entry["idx"]))
    expected = {
        (native_id, unconditional, index)
        for native_id, requests in ours.items()
        for unconditional in (False, True)
        for index in range(len(requests.continuations))
    }
    assert seen == expected
    assert len(ours) == 1172
