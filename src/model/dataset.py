"""Dataset and leak-safe temporal splitting."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class GlucoseForecastDataset(Dataset):
    """PyTorch dataset wrapping pre-built X/y numpy arrays."""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        assert not np.isnan(x).any(), 'X contains NaN'
        assert not np.isnan(y).any(), 'y contains NaN'
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def temporal_train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    test_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split samples chronologically to avoid future leakage.

    Assumes rows are already in temporal order from the data pipeline.

    Args:
        x: Feature array ``(n, window, features)``.
        y: Target array ``(n, horizon)``.
        test_fraction: Fraction of trailing samples reserved for evaluation.

    Returns:
        ``(x_train, y_train, x_test, y_test)`` tuple.
    """
    assert not np.isnan(x).any(), 'X contains NaN before split'
    assert not np.isnan(y).any(), 'y contains NaN before split'
    if not 0.0 < test_fraction < 1.0:
        raise ValueError('test_fraction must be between 0 and 1.')

    n = len(x)
    split_idx = int(n * (1.0 - test_fraction))
    split_idx = max(1, min(split_idx, n - 1))
    return x[:split_idx], y[:split_idx], x[split_idx:], y[split_idx:]
