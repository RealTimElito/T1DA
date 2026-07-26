"""End-to-end T1DA integration: LLM → data → model → guardrails."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.guardrails.agent import GuardrailAgent
from src.llm.integration import build_live_feature_vector
from src.llm.parser import LLMContextParser
from src.model.evaluate import load_model


class T1DAForecastingSystem:
    """Orchestrates all four modules for a single inference cycle."""

    def __init__(
        self,
        patient_frame: pd.DataFrame,
        checkpoint_path: str | Path,
        *,
        llm_client: Any | None = None,
    ) -> None:
        self.patient_frame = patient_frame
        self.parser = LLMContextParser(llm_client=llm_client)
        self.guardrails = GuardrailAgent()
        self.model = load_model(checkpoint_path)
        self.device = next(self.model.parameters()).device

    def process_user_input(
        self,
        user_text: str,
        *,
        current_bg: float,
        previous_bg: float | None = None,
        iob_units: float = 0.0,
    ) -> dict[str, Any]:
        """Parse text, update features, forecast, and apply guardrails."""
        parsed = self.parser.parse(user_text)
        if parsed.get('error'):
            return {'parsed': parsed, 'forecast': None, 'guardrails': None}

        live = build_live_feature_vector(self.patient_frame, parsed)
        window = live['feature_window']
        assert not np.isnan(window).any(), 'feature window contains NaN'

        x = torch.as_tensor(window[np.newaxis, ...], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            preds = self.model(x).cpu().numpy()[0]
        assert not np.isnan(preds).any(), 'model predictions contain NaN'

        report = self.guardrails.evaluate(
            current_bg=current_bg,
            previous_bg=previous_bg,
            iob_units=iob_units,
            ml_predictions=preds.tolist(),
        )

        return {
            'parsed': parsed,
            'feature_window_shape': window.shape,
            'forecast_mg_dl': preds.tolist(),
            'guardrails': report.to_dict(),
        }
