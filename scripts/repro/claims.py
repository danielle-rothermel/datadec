"""View paper claims and quotes with the latest locally saved analysis evidence."""

import argparse
from pathlib import Path

from rich.console import Console

from repro.datasets import latest_run
from repro.reporting import load_claim_report, print_claim_report


REPOSITORY_DIR = Path(__file__).resolve().parents[2]
CLAIMS_FILE = REPOSITORY_DIR / "configs/repro_claims/magnusson2025-datadecide.toml"
PAPER_DIR = REPOSITORY_DIR / "docs/papers/2504.11393v2"


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--run-dir", type=Path, help="view a specific saved analysis run"
    )
    selection.add_argument(
        "--reports-dir",
        type=Path,
        default=REPOSITORY_DIR / "outputs/repro",
        help="search this directory recursively for the latest completed run",
    )
    parser.add_argument("--width", type=int, help="override terminal rendering width")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_arguments()
    run_dir = (
        args.run_dir
        if args.run_dir is not None
        else latest_run(args.reports_dir, CLAIMS_FILE)
    )
    report = load_claim_report(CLAIMS_FILE, PAPER_DIR, run_dir)
    print_claim_report(report, Console(width=args.width))


if __name__ == "__main__":
    main()
