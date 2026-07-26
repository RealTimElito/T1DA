"""Clarke Error Grid analysis for glucose predictions."""

from __future__ import annotations

from typing import Any

import numpy as np


def _clarke_zone(reference: float, prediction: float) -> str:
    """Assign one reference/prediction pair to Clarke zone A-E."""
    if reference < 70:
        if prediction < 70:
            return 'A'
        if prediction < 84:
            return 'B'
        if prediction < 180:
            return 'E'
        return 'D'

    error_ratio = abs(prediction - reference) / reference
    if prediction < 70:
        return 'C'
    if error_ratio <= 0.2:
        return 'A'
    if prediction <= 1.2 * reference:
        return 'B'
    if reference >= 180 and 70 <= prediction <= 180:
        return 'C'
    if reference >= 70 and reference <= 290 and prediction >= reference + 110:
        return 'C'
    if reference >= 130 and prediction <= (7 / 5) * reference - 182:
        return 'E'
    return 'D'


def clarke_error_grid_summary(
    references: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    """Compute Clarke Error Grid zone counts and A+B percentage.

    Args:
        references: Ground-truth glucose (mg/dL).
        predictions: Predicted glucose (mg/dL).

    Returns:
        Summary dict including per-zone counts and ``zone_ab_pct``.
    """
    refs = np.asarray(references, dtype=np.float64).ravel()
    preds = np.asarray(predictions, dtype=np.float64).ravel()
    if len(refs) != len(preds):
        raise ValueError('references and predictions must have equal length.')
    if len(refs) == 0:
        raise ValueError('Empty input arrays.')

    zones = [_clarke_zone(float(r), float(p)) for r, p in zip(refs, preds)]
    counts = {zone: zones.count(zone) for zone in ('A', 'B', 'C', 'D', 'E')}
    total = len(zones)
    zone_ab_pct = 100.0 * (counts['A'] + counts['B']) / total

    return {
        'n_points': total,
        'zone_counts': counts,
        'zone_ab_pct': zone_ab_pct,
        'meets_target': zone_ab_pct > 95.0,
    }
