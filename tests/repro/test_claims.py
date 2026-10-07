from pathlib import Path

from repro.claims import QuoteLocation, read_quotes


def test_read_quotes_uses_source_lines_and_character_ranges(tmp_path: Path) -> None:
    (tmp_path / "paper.tex").write_text(
        "Paper heading\nThe α model improves.\n", encoding="utf-8"
    )
    (tmp_path / "tables").mkdir()
    (tmp_path / "tables/errors.tex").write_text(
        "Model & Error\nSmall & 2.6\n", encoding="utf-8"
    )
    claims = {
        "DD-0003": QuoteLocation("paper.tex", 2, 4, 11),
        "DD-0001": QuoteLocation("tables/errors.tex", 2, 8, 11),
        "DD-0002": QuoteLocation("paper.tex", 2, 4, 11),
    }

    quotes = read_quotes(claims, tmp_path)

    assert list(quotes.items()) == [
        ("DD-0003", "α model"),
        ("DD-0001", "2.6"),
        ("DD-0002", "α model"),
    ]
