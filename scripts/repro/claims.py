"""Read and print the paper quotes for 74 primary reproduction claims.

Run: python3 scripts/repro/claims.py

Paper: DataDecide: How to Predict Best Pretraining Data with Small Experiments
Authors: Ian Magnusson et al. (2025). License: CC BY 4.0.
Source download: https://arxiv.org/src/2504.11393v2
Archive SHA-256: 20dc7aa3f920fe465ddf2e12d6f72fff6e8bb3567f53e34f5555a6da138542d1

Extract example_paper.tex and tables/pred_error.tex from that archive into
<repository>/data/raw/repro-paper/2504.11393v2/ before running.

Claim IDs retain the primary-target numbering from PR #43's registry at
commit e56139f8a9539cca4c7d9847279b3b02d77d7f5c. Repeated statements keep
separate IDs. Locations refer to the downloaded v2 source, including LaTeX
macros. Table rows give relative error followed by absolute error, in percent.

Five targets derived from figure readings have no textual quote and are omitted:
DD-0356, DD-0368, DD-0369, DD-0413, DD-0414.
"""

from dataclasses import dataclass
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parents[2] / "data/raw/repro-paper/2504.11393v2"


@dataclass(frozen=True, slots=True)
class QuoteLocation:
    """A source line (1-based) and Python character slice (0-based, end exclusive)."""

    source_file: str
    line: int
    start: int
    end: int


CLAIMS: dict[str, QuoteLocation] = {
    "DD-0010": QuoteLocation("example_paper.tex", 239, 0, 199),
    "DD-0011": QuoteLocation("example_paper.tex", 239, 0, 199),
    "DD-0013": QuoteLocation("example_paper.tex", 240, 0, 187),
    "DD-0014": QuoteLocation("example_paper.tex", 240, 188, 437),
    "DD-0015": QuoteLocation("example_paper.tex", 240, 188, 437),
    "DD-0016": QuoteLocation("example_paper.tex", 240, 188, 437),
    "DD-0017": QuoteLocation("example_paper.tex", 240, 188, 437),
    "DD-0018": QuoteLocation("example_paper.tex", 240, 188, 437),
    "DD-0051": QuoteLocation("example_paper.tex", 285, 39, 130),
    "DD-0052": QuoteLocation("example_paper.tex", 285, 131, 271),
    "DD-0053": QuoteLocation("example_paper.tex", 285, 131, 271),
    "DD-0054": QuoteLocation("example_paper.tex", 286, 43, 170),
    "DD-0055": QuoteLocation("example_paper.tex", 287, 47, 208),
    "DD-0056": QuoteLocation("example_paper.tex", 288, 46, 231),
    "DD-0057": QuoteLocation("example_paper.tex", 288, 46, 231),
    "DD-0098": QuoteLocation("example_paper.tex", 304, 420, 585),
    "DD-0119": QuoteLocation("example_paper.tex", 324, 0, 232),
    "DD-0142": QuoteLocation("example_paper.tex", 345, 573, 689),
    "DD-0148": QuoteLocation("example_paper.tex", 350, 180, 374),
    "DD-0149": QuoteLocation("example_paper.tex", 350, 180, 374),
    "DD-0150": QuoteLocation("example_paper.tex", 350, 180, 374),
    "DD-0164": QuoteLocation("example_paper.tex", 367, 8, 44),
    "DD-0165": QuoteLocation("example_paper.tex", 367, 45, 137),
    "DD-0166": QuoteLocation("example_paper.tex", 367, 138, 213),
    "DD-0167": QuoteLocation("example_paper.tex", 367, 214, 290),
    "DD-0168": QuoteLocation("example_paper.tex", 367, 291, 387),
    "DD-0169": QuoteLocation("example_paper.tex", 370, 0, 217),
    "DD-0174": QuoteLocation("example_paper.tex", 372, 248, 514),
    "DD-0175": QuoteLocation("example_paper.tex", 372, 248, 514),
    "DD-0176": QuoteLocation("example_paper.tex", 372, 248, 514),
    "DD-0177": QuoteLocation("example_paper.tex", 372, 515, 673),
    "DD-0178": QuoteLocation("example_paper.tex", 372, 515, 673),
    "DD-0179": QuoteLocation("example_paper.tex", 372, 515, 673),
    "DD-0180": QuoteLocation("example_paper.tex", 378, 85, 183),
    "DD-0181": QuoteLocation("example_paper.tex", 384, 8, 114),
    "DD-0189": QuoteLocation("example_paper.tex", 389, 400, 469),
    "DD-0192": QuoteLocation("example_paper.tex", 391, 267, 375),
    "DD-0194": QuoteLocation("example_paper.tex", 391, 376, 517),
    "DD-0196": QuoteLocation("example_paper.tex", 400, 8, 238),
    "DD-0197": QuoteLocation("example_paper.tex", 404, 178, 317),
    "DD-0198": QuoteLocation("example_paper.tex", 404, 178, 317),
    "DD-0199": QuoteLocation("example_paper.tex", 406, 0, 117),
    "DD-0202": QuoteLocation("example_paper.tex", 408, 0, 69),
    "DD-0203": QuoteLocation("example_paper.tex", 408, 70, 344),
    "DD-0204": QuoteLocation("example_paper.tex", 408, 70, 344),
    "DD-0205": QuoteLocation("example_paper.tex", 408, 345, 553),
    "DD-0206": QuoteLocation("example_paper.tex", 408, 345, 553),
    "DD-0207": QuoteLocation("example_paper.tex", 408, 554, 732),
    "DD-0208": QuoteLocation("example_paper.tex", 414, 110, 417),
    "DD-0209": QuoteLocation("example_paper.tex", 422, 81, 267),
    "DD-0210": QuoteLocation("example_paper.tex", 422, 81, 267),
    "DD-0211": QuoteLocation("example_paper.tex", 429, 8, 146),
    "DD-0212": QuoteLocation("example_paper.tex", 429, 147, 219),
    "DD-0213": QuoteLocation("example_paper.tex", 429, 220, 336),
    "DD-0218": QuoteLocation("example_paper.tex", 432, 599, 780),
    "DD-0219": QuoteLocation("example_paper.tex", 432, 599, 780),
    "DD-0220": QuoteLocation("example_paper.tex", 432, 781, 903),
    "DD-0221": QuoteLocation("example_paper.tex", 438, 13, 182),
    "DD-0222": QuoteLocation("example_paper.tex", 438, 183, 270),
    "DD-0224": QuoteLocation("example_paper.tex", 442, 245, 351),
    "DD-0225": QuoteLocation("example_paper.tex", 442, 352, 491),
    "DD-0226": QuoteLocation("example_paper.tex", 442, 492, 621),
    "DD-0227": QuoteLocation("example_paper.tex", 442, 622, 823),
    "DD-0301": QuoteLocation("tables/pred_error.tex", 6, 0, 63),
    "DD-0302": QuoteLocation("tables/pred_error.tex", 7, 0, 45),
    "DD-0303": QuoteLocation("tables/pred_error.tex", 8, 0, 62),
    "DD-0304": QuoteLocation("tables/pred_error.tex", 9, 0, 26),
    "DD-0305": QuoteLocation("tables/pred_error.tex", 10, 0, 26),
    "DD-0306": QuoteLocation("tables/pred_error.tex", 11, 0, 41),
    "DD-0307": QuoteLocation("tables/pred_error.tex", 12, 0, 41),
    "DD-0308": QuoteLocation("tables/pred_error.tex", 13, 0, 29),
    "DD-0311": QuoteLocation("example_paper.tex", 512, 157, 377),
    "DD-0312": QuoteLocation("example_paper.tex", 512, 157, 377),
    "DD-0330": QuoteLocation("example_paper.tex", 564, 103, 230),
}


def main() -> None:
    """Read each quote from the paper download and print it after its claim ID."""
    sources = {
        source_file: (PAPER_DIR / source_file).read_text(encoding="utf-8").splitlines()
        for source_file in {location.source_file for location in CLAIMS.values()}
    }
    for claim_id, location in CLAIMS.items():
        line = sources[location.source_file][location.line - 1]
        quote = line[location.start : location.end]
        print(f"{claim_id}\n{quote}\n")


if __name__ == "__main__":
    main()
