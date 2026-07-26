#!/usr/bin/env python3
"""Evaluate a trained T1DA forecasting model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.dataset import temporal_train_test_split  # noqa: E402
from src.model.evaluate import evaluate_arrays, load_model  # noqa: E402
from src.model.train import save_evaluation_artifacts  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description='Evaluate T1DA forecasting model')
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=ROOT / 'data' / 'processed',
    )
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=ROOT / 'models' / 'glucose_lstm.pt',
    )
    parser.add_argument(
        '--output-prefix',
        type=str,
        default='eval',
        help='Prefix for models/eval_*.json artifacts',
    )
    args = parser.parse_args()

    x_train = np.load(args.data_dir / 'X_train.npy')
    y_train = np.load(args.data_dir / 'y_train.npy')
    _, _, x_test, y_test = temporal_train_test_split(x_train, y_train)

    model = load_model(args.checkpoint)
    report = evaluate_arrays(model, x_test, y_test)
    artifact_paths = save_evaluation_artifacts(
        report,
        args.checkpoint.parent,
        prefix=args.output_prefix,
    )
    print(
        json.dumps(
            {**report, 'artifacts': {k: str(v) for k, v in artifact_paths.items()}},
            indent=2,
        )
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
