#!/usr/bin/env python3
"""Run the Module 1 data engineering pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as ``python scripts/run_pipeline.py`` from repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.pipeline import run_pipeline, run_synthetic_pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description='T1DA Module 1 data pipeline')
    parser.add_argument(
        '--input',
        type=Path,
        default=None,
        help='Clinical CSV file or directory (HUPA-UCM or canonical format).',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/processed'),
        help='Directory for X_train.npy and y_train.npy.',
    )
    parser.add_argument(
        '--synthetic-only',
        action='store_true',
        help='Skip file ingest; build arrays from in-memory synthetic data.',
    )
    parser.add_argument(
        '--no-sample-fallback',
        action='store_true',
        help='Fail instead of generating synthetic sample data when --input is omitted.',
    )
    args = parser.parse_args()

    if args.synthetic_only:
        summary = run_synthetic_pipeline(args.output_dir)
    else:
        summary = run_pipeline(
            input_path=args.input,
            output_dir=args.output_dir,
            use_sample_if_missing=not args.no_sample_fallback,
        )
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
