"""Model evaluation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.model.clarke import clarke_error_grid_summary
from src.model.confusion import glucose_risk_confusion_matrix
from src.model.lstm import GlucoseLSTM


def load_model(checkpoint_path: str | Path, device: str | None = None) -> GlucoseLSTM:
    """Load a trained GlucoseLSTM from a checkpoint file."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dev = torch.device(device)

    payload = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    config = payload['config']
    model = GlucoseLSTM(
        n_features=4,
        hidden_dim=config['hidden_dim'],
        horizon=config['horizon_size'],
    )
    model.load_state_dict(payload['model_state'])
    model.to(dev)
    model.eval()
    return model


def evaluate_arrays(
    model: GlucoseLSTM,
    x: np.ndarray,
    y: np.ndarray,
    *,
    device: str | None = None,
    horizon_idx: int = 0,
) -> dict[str, Any]:
    """Run inference and Clarke analysis on numpy arrays."""
    assert not np.isnan(x).any(), 'X contains NaN'
    assert not np.isnan(y).any(), 'y contains NaN'

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dev = torch.device(device)

    model.eval()
    with torch.no_grad():
        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=dev)
        preds = model(x_tensor).cpu().numpy()

    mse = float(np.mean((preds - y) ** 2))
    clarke = clarke_error_grid_summary(y[:, horizon_idx], preds[:, horizon_idx])
    confusion = glucose_risk_confusion_matrix(y[:, horizon_idx], preds[:, horizon_idx])

    return {
        'mse': mse,
        'clarke': clarke,
        'confusion_matrix': confusion,
        'n_samples': len(x),
        'horizon_idx': horizon_idx,
    }
