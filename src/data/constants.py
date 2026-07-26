"""Shared constants for the T1DA data engineering pipeline."""

from __future__ import annotations

# Uniform resampling grid.
GRID_MINUTES = 5

# Gaps in CGM shorter than this are linearly interpolated; longer gaps break sequences.
MAX_INTERP_GAP_MINUTES = 15
MAX_INTERP_STEPS = MAX_INTERP_GAP_MINUTES // GRID_MINUTES  # 3 five-minute bins

# Input window: 12 steps * 5 min = 60 minutes of history.
WINDOW_SIZE = 12

# Output horizon: 6 future glucose values (mg/dL) at 5-minute step offsets 1..6,
# i.e. 5, 10, 15, 20, 25, and 30 minutes ahead from the window end time T.
HORIZON_SIZE = 6
HORIZON_OFFSETS = (1, 2, 3, 4, 5, 6)

FEATURE_COLUMNS = ('glucose', 'basal', 'bolus', 'carbs')

# Canonical internal column names after ingestion normalization.
CANONICAL_COLUMNS = {
    'glucose': 'glucose',
    'basal': 'basal',
    'bolus': 'bolus',
    'carbs': 'carbs',
}

# HUPA-UCM preprocessed CSV column aliases (semicolon-separated files).
# Mendeley exports use underscores; some mirrors use dotted names.
HUPA_COLUMN_MAP = {
    'time': 'timestamp',
    'glucose': 'glucose',
    'basal.rate': 'basal',
    'basal_rate': 'basal',
    'bolus.volume.delivered': 'bolus',
    'bolus_volume_delivered': 'bolus',
    'carb.input': 'carbs',
    'carb_input': 'carbs',
}
