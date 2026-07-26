"""Five-minute resampling, CGM interpolation, and gap-aware segmentation."""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd

from src.data.constants import (
    FEATURE_COLUMNS,
    GRID_MINUTES,
    MAX_INTERP_STEPS,
)


def _interpolate_glucose_with_gap_limit(
    glucose: pd.Series,
    max_interp_steps: int = MAX_INTERP_STEPS,
) -> pd.Series:
    """Linearly interpolate CGM gaps shorter than the configured limit.

    Gaps longer than ``max_interp_steps`` remain NaN so downstream logic can
    break sequences instead of inventing long missing blocks.
    """
    if glucose.empty:
        return glucose

    values = glucose.copy()
    is_missing = values.isna()
    if not is_missing.any():
        return values

    # Identify contiguous missing runs.
    group_id = (is_missing != is_missing.shift(fill_value=False)).cumsum()
    for _, group_idx in is_missing.groupby(group_id).groups.items():
        if not is_missing.loc[group_idx].iloc[0]:
            continue
        run_length = len(group_idx)
        if run_length <= max_interp_steps:
            interpolated = values.loc[group_idx].interpolate(method='linear')
            values.loc[group_idx] = interpolated

    return values


def resample_patient_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Align one patient record onto a strict 5-minute grid.

    - Glucose: mean within bin, then linear interpolation for gaps < 15 minutes.
    - Basal: last observation carried forward (rate per interval).
    - Bolus/Carbs: summed within each 5-minute bin (event aggregation).

    Args:
        frame: Canonical patient frame indexed by timestamp.

    Returns:
        Uniformly sampled frame with feature columns only.
    """
    if frame.empty:
        raise ValueError('Cannot resample an empty frame.')

    aggregated = frame.resample(f'{GRID_MINUTES}min').agg(
        {
            'glucose': 'mean',
            'basal': 'last',
            'bolus': 'sum',
            'carbs': 'sum',
        }
    )
    aggregated.index.name = 'timestamp'

    aggregated['glucose'] = _interpolate_glucose_with_gap_limit(aggregated['glucose'])
    aggregated['basal'] = aggregated['basal'].ffill().fillna(0.0)
    aggregated['bolus'] = aggregated['bolus'].fillna(0.0)
    aggregated['carbs'] = aggregated['carbs'].fillna(0.0)

    return aggregated[list(FEATURE_COLUMNS)]


def find_valid_segments(frame: pd.DataFrame) -> list[tuple[int, int]]:
    """Return contiguous index ranges separated by unrecoverable CGM gaps.

    After interpolation, any remaining NaN glucose block is at least 15 minutes
    and forces a sequence break.

    Args:
        frame: Resampled patient frame.

    Returns:
        List of ``(start_idx, end_idx)`` tuples inclusive on the resampled grid.
    """
    valid = ~frame['glucose'].isna()
    if not valid.any():
        return []

    segments: list[tuple[int, int]] = []
    segment_start: int | None = None
    for idx, is_valid in enumerate(valid.to_numpy()):
        if is_valid:
            if segment_start is None:
                segment_start = idx
            continue
        if segment_start is not None:
            segments.append((segment_start, idx - 1))
            segment_start = None

    if segment_start is not None:
        segments.append((segment_start, len(valid) - 1))
    return segments


def iter_segment_frames(frame: pd.DataFrame) -> Iterator[pd.DataFrame]:
    """Yield gap-separated contiguous sub-frames."""
    for start_idx, end_idx in find_valid_segments(frame):
        yield frame.iloc[start_idx : end_idx + 1].copy()


def resample_clinical_data(frame: pd.DataFrame) -> dict[str, list[pd.DataFrame]]:
    """Resample all patients and split into gap-safe segments.

    Args:
        frame: Combined clinical data, optionally multi-indexed by patient.

    Returns:
        Mapping of patient_id -> list of segment frames.
    """
    if isinstance(frame.index, pd.MultiIndex) and 'patient_id' in frame.index.names:
        patient_groups = {
            patient_id: group.droplevel('patient_id')
            for patient_id, group in frame.groupby(level='patient_id')
        }
    else:
        patient_groups = {'default': frame}

    segmented: dict[str, list[pd.DataFrame]] = {}
    for patient_id, patient_frame in patient_groups.items():
        resampled = resample_patient_frame(patient_frame)
        segments = list(iter_segment_frames(resampled))
        if segments:
            segmented[patient_id] = segments
    return segmented
