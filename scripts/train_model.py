#!/usr/bin/env python3
"""Train the T1DA glucose forecasting LSTM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.pipeline import run_pipeline, run_synthetic_pipeline  # noqa: E402
from src.model.train import save_checkpoint, save_evaluation_artifacts, train_model  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description='Train T1DA forecasting model')
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=ROOT / 'data' / 'processed',
    )
    parser.add_argument(
        '--raw-input',
        type=Path,
        default=None,
        help='HUPA or canonical CSV directory; runs pipeline before training.',
    )
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=ROOT / 'models' / 'glucose_lstm.pt',
    )
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--prepare-data', action='store_true')
    parser.add_argument('--synthetic-only', action='store_true')
    args = parser.parse_args()

    x_path = args.data_dir / 'X_train.npy'
    y_path = args.data_dir / 'y_train.npy'
    if args.raw_input:
        run_pipeline(input_path=args.raw_input, output_dir=args.data_dir, use_sample_if_missing=False)
    elif args.synthetic_only or args.prepare_data or not x_path.exists():
        run_synthetic_pipeline(args.data_dir)

    x_train = np.load(x_path)
    y_train = np.load(y_path)

    result = train_model(
        x_train,
        y_train,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        patience=args.patience,
    )
    ckpt = save_checkpoint(result, args.checkpoint)
    artifact_paths = save_evaluation_artifacts(result, args.checkpoint.parent, prefix='train')
    print(
        json.dumps(
            {
                'checkpoint': str(ckpt),
                'clarke': result['clarke'],
                'confusion_matrix': result['confusion_matrix'],
                'artifacts': {k: str(v) for k, v in artifact_paths.items()},
            },
            indent=2,
        )
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
