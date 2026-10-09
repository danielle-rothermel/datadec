import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from repro.claims import Claim, QuoteLocation, load_claims, read_quotes

CLAIMS_FILE = (
    Path(__file__).resolve().parents[2]
    / "configs/repro_claims/magnusson2025-datadecide.toml"
)


def test_claim_command_uses_bundled_paper_from_any_working_directory(
    tmp_path: Path,
) -> None:
    script = Path(__file__).resolve().parents[2] / "scripts/repro/claims.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--reports-dir",
            str(tmp_path / "no-reports"),
            "--width",
            "120",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )

    lines = result.stdout.splitlines()
    assert sum(line.startswith("DD-") for line in lines) == 64
    assert "DD-0011" in result.stdout
    assert "best models at our larger target scale" in result.stdout
    assert "Saved report: none" in result.stdout
    assert result.stdout.count("Linked ranking rows: 0") == 64
    assert r"\rankingMethod{}" in result.stdout
    assert result.stderr == ""


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
            (
                QuoteLocation("paper.tex", 2, 4, 11),
                QuoteLocation("tables/errors.tex", 2, 8, 11),
            ),
            ("DD-0003",),
        ),
        "DD-0001": Claim(
            (QuoteLocation("paper.tex", 2, 4, 11),),
            ("DD-0001",),
        ),
    }

    quotes = read_quotes(claims, tmp_path)

    assert list(quotes.items()) == [
        ("DD-0003", ("α model", "2.6")),
        ("DD-0001", ("α model",)),
    ]


def test_inventory_preserves_source_groups_and_ids() -> None:
    claims = load_claims(CLAIMS_FILE)

    assert len(claims) == 64
    assert sum(len(claim.locations) for claim in claims.values()) == 74
    assert claims["DD-0013"].original_entry_ids == (
        "DD-0013",
        "DD-0054",
        "DD-0180",
        "DD-0181",
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


def test_load_claims_preserves_order_and_provenance(tmp_path: Path) -> None:
    config = tmp_path / "claims.toml"
    config.write_text(
        """[claims.DD-0003]
original_entry_ids = ["DD-0003", "DD-0008"]
locations = [
    {source_file = "paper.tex", line = 2, start = 4, end = 11},
    {source_file = "table.tex", line = 1, start = 0, end = 3},
]

[claims.DD-0001]
original_entry_ids = ["DD-0001"]
locations = [{source_file = "paper.tex", line = 2, start = 4, end = 11}]
""",
        encoding="utf-8",
    )

    claims = load_claims(config)

    assert list(claims) == ["DD-0003", "DD-0001"]
    assert claims["DD-0003"] == Claim(
        locations=(
            QuoteLocation("paper.tex", 2, 4, 11),
            QuoteLocation("table.tex", 1, 0, 3),
        ),
        original_entry_ids=("DD-0003", "DD-0008"),
    )


@pytest.mark.parametrize(
    ("original", "replacement", "error"),
    [
        ("locations = [", 'statement = "Unquoted text"\nlocations = [', "Extra inputs"),
        ("line = 2", 'line = "second"', "line"),
        ("start = 4", "start = 11", "quote end must be greater than start"),
        ("line = 2", "line = 0", "line"),
        ("locations = [", "unknown = true\nlocations = [", "unknown"),
        (
            'original_entry_ids = ["DD-0001"]',
            "original_entry_ids = []",
            "original_entry_ids",
        ),
    ],
)
def test_load_claims_rejects_invalid_config(
    tmp_path: Path, original: str, replacement: str, error: str
) -> None:
    config = tmp_path / "claims.toml"
    config.write_text(
        """[claims.DD-0001]
original_entry_ids = ["DD-0001"]
locations = [{source_file = "paper.tex", line = 2, start = 4, end = 11}]
""".replace(original, replacement),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match=error):
        load_claims(config)


def test_bundled_selections_are_exact_source_spans():
    claims = load_claims(CLAIMS_FILE)
    paper_dir = CLAIMS_FILE.parents[2] / "docs/papers/2504.11393v2"
    quotes = read_quotes(claims, paper_dir)
    for claim_id, claim in claims.items():
        for location, quote in zip(claim.locations, quotes[claim_id], strict=True):
            line = (
                (paper_dir / location.source_file)
                .read_text()
                .splitlines()[location.line - 1]
            )
            assert 0 <= location.start < location.end <= len(line)
            assert quote == line[location.start : location.end]
            assert len(quote) == location.end - location.start
