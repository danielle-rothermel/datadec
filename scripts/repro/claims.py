"""Print the DataDecide claim inventory and its bundled paper quotes.

Run from the repository root: uv run python scripts/repro/claims.py
"""

from pathlib import Path

from repro.claims import load_claims, read_quotes

REPOSITORY_DIR = Path(__file__).resolve().parents[2]
CLAIMS_FILE = REPOSITORY_DIR / "configs/repro_claims/magnusson2025-datadecide.toml"
PAPER_DIR = REPOSITORY_DIR / "docs/papers/2504.11393v2"


def main() -> None:
    """Print each assertion once, followed by all its paper quotes and sections."""
    claims = load_claims(CLAIMS_FILE)
    for claim_id, quotes in read_quotes(claims, PAPER_DIR).items():
        claim = claims[claim_id]
        print(f"{claim_id}: {claim.statement}")
        for location, quote in zip(claim.locations, quotes, strict=True):
            print(f"  {location.section} ({location.source_file}:{location.line})")
            print(quote)
        print()


if __name__ == "__main__":
    main()
