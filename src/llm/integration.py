"""Wire parsed LLM events into the clinical data pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from src.data.ingestion import normalize_clinical_frame
from src.data.resampling import resample_patient_frame


def apply_parsed_event_to_frame(
    frame: pd.DataFrame,
    parsed: dict[str, Any],
) -> pd.DataFrame:
    """Inject carbs/bolus from a parsed LLM event into a patient frame.

    The frame must be indexed by timestamp with canonical columns. This function
    does not forecast glucose — it only appends or updates sparse event features.

    Args:
        frame: Canonical clinical frame (one patient).
        parsed: Success schema from ``LLMContextParser``.

    Returns:
        Updated frame with new bolus/carbs at the parsed timestamp.
    """
    if parsed.get('error'):
        raise ValueError(f'Cannot apply error payload: {parsed.get("message")}')

    ts = pd.Timestamp(parsed['timestamp'])
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)

    working = frame.copy()
    if len(working.index) and (ts < working.index.min() or ts > working.index.max()):
        ts = working.index.max()
    if ts in working.index:
        working.loc[ts, 'bolus'] += float(parsed['insulin_units'])
        working.loc[ts, 'carbs'] += float(parsed['carbs_grams'])
    else:
        last_glucose = working['glucose'].dropna().iloc[-1] if len(working) else 100.0
        row = pd.DataFrame(
            {
                'glucose': [last_glucose],
                'basal': [working['basal'].iloc[-1] if len(working) else 0.0],
                'bolus': [float(parsed['insulin_units'])],
                'carbs': [float(parsed['carbs_grams'])],
            },
            index=pd.DatetimeIndex([ts], name='timestamp'),
        )
        working = pd.concat([working, row])
        working = working.sort_index()

    export = working.reset_index()
    if 'timestamp' not in export.columns:
        export = export.rename(columns={export.columns[0]: 'timestamp'})
    return normalize_clinical_frame(export)


def build_live_feature_vector(
    frame: pd.DataFrame,
    parsed: dict[str, Any],
    *,
    window_size: int = 12,
) -> dict[str, Any]:
    """Resample after LLM event injection and return the latest model window.

    Returns:
        Dict with ``feature_window`` numpy array and metadata for forecasting.
    """
    import numpy as np

    from src.data.constants import FEATURE_COLUMNS

    updated = apply_parsed_event_to_frame(frame, parsed)
    resampled = resample_patient_frame(updated)
    if len(resampled) < window_size:
        raise ValueError(f'Need at least {window_size} resampled steps after event.')

    window = resampled[list(FEATURE_COLUMNS)].iloc[-window_size:].to_numpy(dtype=np.float64)
    if np.isnan(window).any():
        raise AssertionError('Live feature window contains NaN.')

    return {
        'feature_window': window,
        'timestamp': parsed['timestamp'],
        'parsed_event': parsed,
    }
