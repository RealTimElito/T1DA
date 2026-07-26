#!/usr/bin/env python3
"""Generate synthetic clinical-format CSV samples for local development."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.ingestion import write_synthetic_sample  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description='Generate synthetic T1DA sample CSVs')
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/sample'),
        help='Directory for generated CSV files.',
    )
    parser.add_argument(
        '--n-patients',
        type=int,
        default=2,
        help='Number of synthetic patients to create.',
    )
    args = parser.parse_args()

    paths = write_synthetic_sample(args.output_dir, n_patients=args.n_patients)
    for path in paths:
        print(path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
