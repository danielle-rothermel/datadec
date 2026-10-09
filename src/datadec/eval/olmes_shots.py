from __future__ import annotations

from typing import Final

OLMES_SOURCE_REPO: Final = "https://github.com/allenai/olmes"
OLMES_SOURCE_COMMIT: Final = "5a51f502d463b8cdc4a2dcad7d7096c41ff1197e"
OLMES_SOURCE_PATH: Final = "oe_eval/tasks/fewshot_sources.py"
OLMES_SOURCE_LICENSE: Final = "Apache-2.0"

ARC_CHALLENGE_SHOTS: Final[tuple[dict[str, object], ...]] = (
    {
        "id": "Mercury_SC_415702",
        "question": (
            "George wants to warm his hands quickly by rubbing them. "
            "Which skin surface will produce the most heat?"
        ),
        "choices": {
            "text": [
                "dry palms",
                "wet palms",
                "palms covered with oil",
                "palms covered with lotion",
            ],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "A",
    },
    {
        "id": "MCAS_2009_5_6516",
        "question": (
            "Which of the following statements best explains why magnets "
            "usually stick to a refrigerator door?"
        ),
        "choices": {
            "text": [
                "The refrigerator door is smooth.",
                "The refrigerator door contains iron.",
                "The refrigerator door is a good conductor.",
                "The refrigerator door has electric wires in it.",
            ],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "B",
    },
    {
        "id": "Mercury_7233695",
        "question": (
            "A fold observed in layers of sedimentary rock most likely "
            "resulted from the"
        ),
        "choices": {
            "text": [
                "cooling of flowing magma.",
                "converging of crustal plates.",
                "deposition of river sediments.",
                "solution of carbonate minerals.",
            ],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "B",
    },
    {
        "id": "Mercury_7041615",
        "question": (
            "Which of these do scientists offer as the most recent "
            "explanation as to why many plants and animals died out at "
            "the end of the Mesozoic era?"
        ),
        "choices": {
            "text": [
                "worldwide disease",
                "global mountain building",
                "rise of mammals that preyed upon plants and animals",
                "impact of an asteroid created dust that blocked the sunlight",
            ],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "D",
    },
    {
        "id": "MCAS_1998_4_3",
        "question": (
            "Which of the following is a trait that a dog does NOT "
            "inherit from its parents?"
        ),
        "choices": {
            "text": [
                "the length of its fur",
                "the shape of its nose",
                "the size of its appetite",
                "the color of its fur",
            ],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "C",
    },
)
