"""T1DA Module 1: clinical time-series data engineering."""

from src.data.constants import (
    FEATURE_COLUMNS,
    GRID_MINUTES,
    HORIZON_OFFSETS,
    MAX_INTERP_GAP_MINUTES,
    WINDOW_SIZE,
)
from src.data.features import build_feature_matrices
from src.data.pipeline import run_pipeline, run_synthetic_pipeline

__all__ = [
    'FEATURE_COLUMNS',
    'GRID_MINUTES',
    'HORIZON_OFFSETS',
    'MAX_INTERP_GAP_MINUTES',
    'WINDOW_SIZE',
    'build_feature_matrices',
    'run_pipeline',
    'run_synthetic_pipeline',
]
