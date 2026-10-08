"""Report saved claim measurements and verdicts with quotes from the paper."""

import argparse
import logging
from pathlib import Path

from repro.reporting import load_claim_report, write_claim_report


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="directory containing claim_results.parquet and run.json",
    )
    return parser.parse_args(argv)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> None:
    args = parse_arguments()
    configure_logging()
    report = load_claim_report(args.run_dir)
    write_claim_report(report, args.run_dir)
    logging.info("Wrote claim reports to %s", args.run_dir.resolve())


if __name__ == "__main__":
    main()
