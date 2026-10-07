import runpy
from pathlib import Path
from typing import cast

from repro.claims import Claim, QuoteLocation, read_quotes


def test_read_quotes_uses_source_lines_and_character_ranges(tmp_path: Path) -> None:
    (tmp_path / "paper.tex").write_text(
        "Paper heading\nThe α model improves.\n", encoding="utf-8"
    )
    (tmp_path / "tables").mkdir()
    (tmp_path / "tables/errors.tex").write_text(
        "Model & Error\nSmall & 2.6\n", encoding="utf-8"
    )
    claims = {
        "DD-0003": Claim(
            "One assertion in two paper sections.",
            (
                QuoteLocation("paper.tex", 2, 4, 11, "Introduction"),
                QuoteLocation("tables/errors.tex", 2, 8, 11, "Error table"),
            ),
        ),
        "DD-0001": Claim(
            "Another assertion in the same sentence.",
            (QuoteLocation("paper.tex", 2, 4, 11, "Introduction"),),
        ),
    }

    quotes = read_quotes(claims, tmp_path)

    assert list(quotes.items()) == [
        ("DD-0003", ("α model", "2.6")),
        ("DD-0001", ("α model",)),
    ]


def test_inventory_groups_restatements_and_preserves_distinct_assertions() -> None:
    script = Path(__file__).resolve().parents[2] / "scripts/repro/claims.py"
    claims = cast(dict[str, Claim], runpy.run_path(str(script))["CLAIMS"])

    assert len(claims) == 64
    assert sum(len(claim.locations) for claim in claims.values()) == 74
    assert tuple(location.section for location in claims["DD-0013"].locations) == (
        "Abstract",
        "Introduction",
        "Figure: scaling-law decision accuracy",
        "Results / Scaling-law comparison",
    )
    assert (
        not {
            "DD-0192",
            "DD-0054",
            "DD-0180",
            "DD-0181",
            "DD-0166",
            "DD-0167",
            "DD-0211",
            "DD-0174",
            "DD-0221",
            "DD-0330",
        }
        & claims.keys()
    )
    for first, second in (("DD-0010", "DD-0011"), ("DD-0014", "DD-0015")):
        assert claims[first].locations[0] == claims[second].locations[0]
        assert claims[first].statement != claims[second].statement
