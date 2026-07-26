"""Loss functions for glucose forecasting."""

from __future__ import annotations

import torch
from torch import nn


class HypoglycemiaAwareMSELoss(nn.Module):
    """MSE with 3x penalty when predictions underestimate hypoglycemic values."""

    HYPO_THRESHOLD_MG_DL = 70.0
    UNDERESTIMATE_PENALTY = 3.0

    def __init__(
        self,
        hypo_threshold: float = HYPO_THRESHOLD_MG_DL,
        underestimate_penalty: float = UNDERESTIMATE_PENALTY,
    ) -> None:
        super().__init__()
        self.hypo_threshold = hypo_threshold
        self.underestimate_penalty = underestimate_penalty

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted MSE.

        Args:
            predictions: Model output, shape ``(batch, horizon)``.
            targets: Ground-truth glucose, shape ``(batch, horizon)``.

        Returns:
            Scalar loss tensor.
        """
        assert not torch.isnan(predictions).any(), 'predictions contain NaN'
        assert not torch.isnan(targets).any(), 'targets contain NaN'

        squared_error = (predictions - targets) ** 2
        is_hypo = targets < self.hypo_threshold
        underestimates = predictions < targets
        weights = torch.ones_like(squared_error)
        weights = torch.where(is_hypo & underestimates, self.underestimate_penalty, weights)
        return (weights * squared_error).mean()
