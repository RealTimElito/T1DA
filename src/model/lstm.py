"""Stacked unidirectional LSTM forecaster."""

from __future__ import annotations

import torch
from torch import nn


class GlucoseLSTM(nn.Module):
    """Multi-horizon glucose forecaster using stacked unidirectional LSTM."""

    def __init__(
        self,
        n_features: int = 4,
        hidden_dim: int = 64,
        n_layers: int = 2,
        horizon: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict future glucose horizons from past feature windows.

        Args:
            x: Input tensor, shape ``(batch, window, features)``.

        Returns:
            Predictions with shape ``(batch, horizon)``.
        """
        assert not torch.isnan(x).any(), 'input contains NaN before LSTM'
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        assert not torch.isnan(last_hidden).any(), 'LSTM output contains NaN'
        predictions = self.head(last_hidden)
        assert not torch.isnan(predictions).any(), 'head output contains NaN'
        return predictions
