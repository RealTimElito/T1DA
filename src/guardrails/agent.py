"""Deterministic guardrail firewall between ML predictions and user alerts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from src.guardrails.alerts import AlertLevel, SafetyAlert
from src.guardrails.velocity import VelocityTracker


@dataclass
class GuardrailReport:
    """Combined guardrail output for one evaluation cycle."""

    alerts: list[SafetyAlert]
    allow_delivery: bool
    ml_predictions: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'allow_delivery': self.allow_delivery,
            'alerts': [a.to_dict() for a in self.alerts],
            'ml_predictions': self.ml_predictions,
        }


class GuardrailAgent:
    """Static threshold and contextual checks independent of the ML model."""

    HYPO_MG_DL = 70.0
    SEVERE_HYPO_MG_DL = 54.0
    HYPER_MG_DL = 250.0
    IOB_NEAR_ZERO_UNITS = 0.1
    FALLING_TREND_MG_DL = -10.0

    def __init__(self) -> None:
        self.velocity_tracker = VelocityTracker()

    def evaluate(
        self,
        *,
        current_bg: float,
        previous_bg: float | None = None,
        delta_minutes: float = 5.0,
        iob_units: float = 0.0,
        ml_predictions: Sequence[float] | None = None,
    ) -> GuardrailReport:
        """Run all guardrail checks on current state and optional ML output.

        Args:
            current_bg: Latest CGM reading (mg/dL).
            previous_bg: Prior CGM reading for velocity/trend checks.
            delta_minutes: Minutes between readings.
            iob_units: Active insulin on board (units).
            ml_predictions: Optional model horizon predictions (mg/dL).

        Returns:
            ``GuardrailReport`` with classified alerts and delivery flag.
        """
        alerts: list[SafetyAlert] = []

        if previous_bg is not None:
            alerts.extend(
                self.velocity_tracker.check(previous_bg, current_bg, delta_minutes)
            )
            trend = current_bg - previous_bg
            if trend < self.FALLING_TREND_MG_DL and iob_units <= self.IOB_NEAR_ZERO_UNITS:
                alerts.append(
                    SafetyAlert(
                        level=AlertLevel.WARNING,
                        code='FALLING_ZERO_IOB',
                        message=(
                            'BG trending down with near-zero IOB — possible '
                            'compression low or missed carb/insulin logging.'
                        ),
                        context={
                            'trend_mg_dl': trend,
                            'iob_units': iob_units,
                        },
                    )
                )

        if current_bg < self.SEVERE_HYPO_MG_DL:
            alerts.append(
                SafetyAlert(
                    level=AlertLevel.CRITICAL,
                    code='SEVERE_HYPO',
                    message=f'Current BG {current_bg:.0f} mg/dL is severely hypoglycemic.',
                    context={'current_bg': current_bg},
                )
            )
        elif current_bg < self.HYPO_MG_DL:
            alerts.append(
                SafetyAlert(
                    level=AlertLevel.WARNING,
                    code='HYPO',
                    message=f'Current BG {current_bg:.0f} mg/dL is below 70 mg/dL.',
                    context={'current_bg': current_bg},
                )
            )

        if current_bg > self.HYPER_MG_DL:
            alerts.append(
                SafetyAlert(
                    level=AlertLevel.WARNING,
                    code='HYPER',
                    message=f'Current BG {current_bg:.0f} mg/dL exceeds 250 mg/dL.',
                    context={'current_bg': current_bg},
                )
            )

        if ml_predictions is not None:
            preds = np.asarray(ml_predictions, dtype=np.float64)
            assert not np.isnan(preds).any(), 'ml_predictions contain NaN'
            if (preds < self.HYPO_MG_DL).any():
                alerts.append(
                    SafetyAlert(
                        level=AlertLevel.INFO,
                        code='ML_HYPO_FORECAST',
                        message='Model forecasts hypoglycemia within horizon.',
                        context={'min_predicted_bg': float(preds.min())},
                    )
                )

        has_critical = any(a.level == AlertLevel.CRITICAL for a in alerts)
        pred_list = list(ml_predictions) if ml_predictions is not None else None

        return GuardrailReport(
            alerts=alerts,
            allow_delivery=not has_critical,
            ml_predictions=pred_list,
        )
