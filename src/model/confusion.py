"""Glucose risk-bin confusion matrix for forecast evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np

RISK_LABELS = ('hypo', 'in_range', 'hyper')
HYPO_THRESHOLD = 70.0
HYPER_THRESHOLD = 180.0


def glucose_risk_bin(value: float) -> str:
    """Classify glucose (mg/dL) into hypo / in_range / hyper."""
    if value < HYPO_THRESHOLD:
        return 'hypo'
    if value > HYPER_THRESHOLD:
        return 'hyper'
    return 'in_range'


def glucose_risk_confusion_matrix(
    references: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    """Build a 3x3 confusion matrix over glucose risk bins.

    Rows are actual (reference) bins; columns are predicted bins.
    """
    refs = np.asarray(references, dtype=np.float64).ravel()
    preds = np.asarray(predictions, dtype=np.float64).ravel()
    if len(refs) != len(preds):
        raise ValueError('references and predictions must have equal length.')
    if len(refs) == 0:
        raise ValueError('Empty input arrays.')

    matrix = {
        actual: {pred: 0 for pred in RISK_LABELS}
        for actual in RISK_LABELS
    }
    for ref_val, pred_val in zip(refs, preds):
        actual = glucose_risk_bin(float(ref_val))
        predicted = glucose_risk_bin(float(pred_val))
        matrix[actual][predicted] += 1

    total = len(refs)
    correct = sum(matrix[label][label] for label in RISK_LABELS)
    accuracy = 100.0 * correct / total

    return {
        'labels': list(RISK_LABELS),
        'matrix': matrix,
        'n_points': total,
        'accuracy_pct': accuracy,
        'meets_target': accuracy >= 70.0,
    }
