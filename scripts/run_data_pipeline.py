#!/usr/bin/env python3
"""Deprecated alias for ``scripts/run_pipeline.py``.

Use ``python scripts/run_pipeline.py`` instead. This wrapper remains for
backward compatibility with earlier scaffold entry points.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.pipeline import run_pipeline, run_synthetic_pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description='T1DA data engineering pipeline')
    parser.add_argument(
        '--input',
        type=Path,
        default=ROOT / 'data' / 'sample',
        help='CSV file or directory of patient CSVs',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=ROOT / 'data' / 'processed',
        help='Directory for X_train.npy and y_train.npy',
    )
    parser.add_argument(
        '--synthetic-only',
        action='store_true',
        help='Skip file ingest; build arrays from in-memory synthetic data',
    )
    args = parser.parse_args()

    if args.synthetic_only:
        metadata = run_synthetic_pipeline(args.output)
    else:
        metadata = run_pipeline(
            input_path=args.input,
            output_dir=args.output,
            use_sample_if_missing=False,
        )

    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == '__main__':
    warnings.warn(
        'run_data_pipeline.py is deprecated; use scripts/run_pipeline.py',
        DeprecationWarning,
        stacklevel=1,
    )
    raise SystemExit(main())
