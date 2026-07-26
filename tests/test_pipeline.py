"""Tests for T1DA Module 1 data engineering."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.constants import FEATURE_COLUMNS, HORIZON_OFFSETS, HORIZON_SIZE, WINDOW_SIZE
from src.data.features import build_feature_matrices, build_windows_from_segment
from src.data.ingestion import generate_synthetic_patient, load_clinical_csv
from src.data.pipeline import run_pipeline
from src.data.resampling import find_valid_segments, resample_clinical_data
from src.data.validation import assert_no_nan, assert_feature_shapes


def test_gap_breaks_long_cgm_outage() -> None:
    frame = generate_synthetic_patient(gap_start_hour=10.0, gap_minutes=30.0)
    segmented = resample_clinical_data(frame)
    patient_segments = segmented['SYNTH001']
    assert len(patient_segments) >= 2


def test_no_nan_in_feature_matrices() -> None:
    frame = generate_synthetic_patient(gap_start_hour=None)
    segmented = resample_clinical_data(frame)
    x_train, y_train = build_feature_matrices(segmented)
    assert x_train.shape[1:] == (WINDOW_SIZE, len(FEATURE_COLUMNS))
    assert y_train.shape[1] == HORIZON_SIZE
    assert_no_nan(x_train)
    assert_no_nan(y_train)


def test_no_future_leakage_in_windows() -> None:
    frame = generate_synthetic_patient(gap_start_hour=None, hours=6, seed=7)
    segmented = resample_clinical_data(frame)
    segment = segmented['SYNTH001'][0]
    x_train, y_train = build_windows_from_segment(segment)

    feature_values = segment[list(FEATURE_COLUMNS)].to_numpy()
    glucose = feature_values[:, 0]
    end_idx = WINDOW_SIZE - 1
    expected_x = feature_values[end_idx - WINDOW_SIZE + 1 : end_idx + 1]
    expected_y = glucose[[end_idx + offset for offset in HORIZON_OFFSETS]]

    np.testing.assert_allclose(x_train[0], expected_x)
    np.testing.assert_allclose(y_train[0], expected_y)


def test_short_gap_is_interpolated() -> None:
    frame = generate_synthetic_patient(gap_start_hour=5.0, gap_minutes=10.0)
    segmented = resample_clinical_data(frame)
    segment = segmented['SYNTH001'][0]
    assert segment['glucose'].isna().sum() == 0


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    output_dir = tmp_path / 'processed'
    summary = run_pipeline(output_dir=output_dir)
    x_train = np.load(output_dir / 'X_train.npy')
    y_train = np.load(output_dir / 'y_train.npy')
    assert_feature_shapes(x_train, y_train)
    assert summary['n_samples'] == len(x_train)


def test_load_canonical_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / 'patient.csv'
    frame = generate_synthetic_patient(hours=2, gap_start_hour=None)
    export = frame.reset_index()[['timestamp', 'glucose', 'basal', 'bolus', 'carbs']]
    export.to_csv(csv_path, index=False)
    loaded = load_clinical_csv(csv_path)
    assert list(loaded.columns) == list(FEATURE_COLUMNS)


def test_find_valid_segments_empty_on_all_nan() -> None:
    frame = generate_synthetic_patient(hours=1)
    resampled = resample_clinical_data(frame)['SYNTH001'][0]
    resampled['glucose'] = np.nan
    assert find_valid_segments(resampled) == []
