#!/usr/bin/env python3
"""Validate processed training arrays (NaN and shape checks)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.validation import validate_pipeline_outputs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description='Validate T1DA processed arrays')
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=ROOT / 'data' / 'processed',
    )
    args = parser.parse_args()

    x_path = args.data_dir / 'X_train.npy'
    y_path = args.data_dir / 'y_train.npy'
    if not x_path.exists() or not y_path.exists():
        print(f'Missing arrays in {args.data_dir}', file=sys.stderr)
        return 1

    x_train = np.load(x_path)
    y_train = np.load(y_path)
    report = validate_pipeline_outputs(x_train, y_train)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
