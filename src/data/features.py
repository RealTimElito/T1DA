"""Leak-safe sliding-window feature matrix generation."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from src.data.constants import (
    FEATURE_COLUMNS,
    HORIZON_OFFSETS,
    HORIZON_SIZE,
    WINDOW_SIZE,
)
from src.data.validation import assert_no_nan


def _max_horizon_offset() -> int:
    return max(HORIZON_OFFSETS)


def build_windows_from_segment(
    segment: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    horizon_offsets: Sequence[int] = HORIZON_OFFSETS,
) -> tuple[np.ndarray, np.ndarray]:
    """Create X/y arrays from one contiguous resampled segment.

    Features at index ``t`` use values from ``[t-window_size+1, t]`` inclusive.
    Targets are future glucose readings at configured offsets; samples that would
    read across segment boundaries are skipped.

    Args:
        segment: One gap-safe resampled segment.
        window_size: Number of past 5-minute steps in each sample.
        horizon_offsets: Future step offsets for glucose targets.

    Returns:
        Tuple of ``X`` with shape ``(n, window_size, 4)`` and ``y`` with shape
        ``(n, len(horizon_offsets))``.
    """
    if len(horizon_offsets) != HORIZON_SIZE:
        raise ValueError(
            f'Expected {HORIZON_SIZE} horizon offsets, got {len(horizon_offsets)}.'
        )

    feature_values = segment[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    glucose = feature_values[:, 0]
    n_steps = len(segment)
    max_offset = max(horizon_offsets)

    x_samples: list[np.ndarray] = []
    y_samples: list[np.ndarray] = []

    # ``end_idx`` is the index of the current timestep (inclusive in the window).
    for end_idx in range(window_size - 1, n_steps - max_offset):
        start_idx = end_idx - window_size + 1
        future_indices = [end_idx + offset for offset in horizon_offsets]

        window = feature_values[start_idx : end_idx + 1]
        targets = glucose[future_indices]

        if np.isnan(window).any() or np.isnan(targets).any():
            continue

        x_samples.append(window)
        y_samples.append(targets)

    if not x_samples:
        return (
            np.empty((0, window_size, len(FEATURE_COLUMNS)), dtype=np.float64),
            np.empty((0, HORIZON_SIZE), dtype=np.float64),
        )

    x_array = np.stack(x_samples, axis=0)
    y_array = np.stack(y_samples, axis=0)
    assert_no_nan(x_array, name='X_segment')
    assert_no_nan(y_array, name='y_segment')
    return x_array, y_array


def build_feature_matrices(
    segmented_data: dict[str, list[pd.DataFrame]],
    window_size: int = WINDOW_SIZE,
    horizon_offsets: Sequence[int] = HORIZON_OFFSETS,
) -> tuple[np.ndarray, np.ndarray]:
    """Build training matrices from segmented resampled patient data.

    Args:
        segmented_data: Output of ``resample_clinical_data``.
        window_size: Context window length in 5-minute steps.
        horizon_offsets: Future glucose offsets in 5-minute steps.

    Returns:
        ``X_train`` with shape ``(samples, window_size, 4)`` and ``y_train`` with
        shape ``(samples, horizon_size)``.
    """
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    for segments in segmented_data.values():
        for segment in segments:
            x_seg, y_seg = build_windows_from_segment(
                segment,
                window_size=window_size,
                horizon_offsets=horizon_offsets,
            )
            if len(x_seg):
                x_parts.append(x_seg)
                y_parts.append(y_seg)

    if not x_parts:
        return (
            np.empty((0, window_size, len(FEATURE_COLUMNS)), dtype=np.float64),
            np.empty((0, HORIZON_SIZE), dtype=np.float64),
        )

    x_train = np.concatenate(x_parts, axis=0)
    y_train = np.concatenate(y_parts, axis=0)
    assert_no_nan(x_train, name='X_train')
    assert_no_nan(y_train, name='y_train')
    return x_train, y_train
