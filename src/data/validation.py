"""Data validation helpers with explicit NaN guards."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.constants import FEATURE_COLUMNS, HORIZON_SIZE, WINDOW_SIZE


def assert_no_nan(array: np.ndarray, name: str = 'array') -> None:
    """Assert that a numpy array contains no NaN values."""
    if not isinstance(array, np.ndarray):
        raise TypeError(f'{name} must be a numpy.ndarray, got {type(array)!r}.')
    if np.isnan(array).any():
        nan_count = int(np.isnan(array).sum())
        raise AssertionError(f'{name} contains {nan_count} NaN value(s).')


def assert_feature_shapes(
    x_train: np.ndarray,
    y_train: np.ndarray,
    window_size: int = WINDOW_SIZE,
    n_features: int = len(FEATURE_COLUMNS),
    horizon_size: int = HORIZON_SIZE,
) -> None:
    """Validate feature matrix ranks and trailing dimensions."""
    assert_no_nan(x_train, name='X_train')
    assert_no_nan(y_train, name='y_train')

    if x_train.ndim != 3:
        raise AssertionError(f'X_train must be 3-D, got shape {x_train.shape}.')
    if y_train.ndim != 2:
        raise AssertionError(f'y_train must be 2-D, got shape {y_train.shape}.')
    if x_train.shape[0] != y_train.shape[0]:
        raise AssertionError(
            f'Sample mismatch: X_train={x_train.shape[0]}, y_train={y_train.shape[0]}.'
        )
    if x_train.shape[1:] != (window_size, n_features):
        raise AssertionError(
            f'Expected X_train shape (*, {window_size}, {n_features}), '
            f'got {x_train.shape}.'
        )
    if y_train.shape[1] != horizon_size:
        raise AssertionError(
            f'Expected y_train horizon {horizon_size}, got {y_train.shape[1]}.'
        )


def validate_resampled_segment(segment: pd.DataFrame) -> dict[str, Any]:
    """Run structural checks on one resampled segment.

    Returns:
        Summary dictionary suitable for logging or CLI output.
    """
    if segment.empty:
        raise ValueError('Segment is empty.')

    missing_glucose = int(segment['glucose'].isna().sum())
    report = {
        'steps': len(segment),
        'missing_glucose_steps': missing_glucose,
        'basal_non_negative': bool((segment['basal'] >= 0).all()),
        'bolus_non_negative': bool((segment['bolus'] >= 0).all()),
        'carbs_non_negative': bool((segment['carbs'] >= 0).all()),
    }
    if missing_glucose:
        raise AssertionError(
            f'Segment still has {missing_glucose} missing glucose step(s) after resampling.'
        )
    return report


def validate_pipeline_outputs(
    x_train: np.ndarray,
    y_train: np.ndarray,
    min_samples: int = 1,
) -> dict[str, Any]:
    """Validate final arrays before model handoff."""
    assert_feature_shapes(x_train, y_train)
    if len(x_train) < min_samples:
        raise AssertionError(
            f'Expected at least {min_samples} sample(s), got {len(x_train)}.'
        )

    return {
        'n_samples': int(len(x_train)),
        'x_shape': tuple(x_train.shape),
        'y_shape': tuple(y_train.shape),
        'glucose_min': float(y_train.min()),
        'glucose_max': float(y_train.max()),
    }
