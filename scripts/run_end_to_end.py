#!/usr/bin/env python3
"""End-to-end T1DA demo: data → train → parse → forecast → guardrails."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ingestion import generate_synthetic_patient  # noqa: E402
from src.data.pipeline import run_synthetic_pipeline  # noqa: E402
from src.integration import T1DAForecastingSystem  # noqa: E402
from src.model.train import save_checkpoint, train_model  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description='T1DA end-to-end demo')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data' / 'processed')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'models' / 'glucose_lstm.pt')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--patience', type=int, default=7)
    args = parser.parse_args()

    pipeline_meta = run_synthetic_pipeline(args.data_dir)
    x_train = np.load(pipeline_meta['output_paths']['X_train'])
    y_train = np.load(pipeline_meta['output_paths']['y_train'])
    print('Pipeline:', json.dumps(pipeline_meta, indent=2))

    result = train_model(x_train, y_train, epochs=args.epochs, patience=args.patience)
    save_checkpoint(result, args.checkpoint)
    print('Training Clarke:', json.dumps(result['clarke'], indent=2))

    patient = generate_synthetic_patient(patient_id='DEMO', hours=24, seed=7)
    frame = patient.droplevel('patient_id')
    current_bg = float(frame['glucose'].iloc[-1])
    previous_bg = float(frame['glucose'].iloc[-2])

    system = T1DAForecastingSystem(frame, args.checkpoint)
    output = system.process_user_input(
        'Just ate an apple and a slice of pizza, took 4 units of humalog.',
        current_bg=current_bg,
        previous_bg=previous_bg,
        iob_units=0.05,
    )
    print('Integration:', json.dumps(output, indent=2, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
