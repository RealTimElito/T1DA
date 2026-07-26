"""Blood glucose velocity tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.guardrails.alerts import AlertLevel, SafetyAlert


@dataclass
class VelocityReading:
    """A single BG velocity sample."""

    mg_dl_per_min: float
    delta_bg: float
    delta_minutes: float


class VelocityTracker:
    """Track ΔBG/Δt and raise alarms on rapid rise or fall."""

    RISE_THRESHOLD_MG_DL_PER_MIN = 3.0
    FALL_THRESHOLD_MG_DL_PER_MIN = -2.0

    def compute_velocity(
        self,
        bg_previous: float,
        bg_current: float,
        delta_minutes: float,
    ) -> VelocityReading:
        """Compute velocity between two CGM readings."""
        if delta_minutes <= 0:
            raise ValueError('delta_minutes must be positive.')
        delta_bg = bg_current - bg_previous
        rate = delta_bg / delta_minutes
        return VelocityReading(
            mg_dl_per_min=rate,
            delta_bg=delta_bg,
            delta_minutes=delta_minutes,
        )

    def check(
        self,
        bg_previous: float,
        bg_current: float,
        delta_minutes: float,
    ) -> list[SafetyAlert]:
        """Return velocity-based alerts, if any."""
        reading = self.compute_velocity(bg_previous, bg_current, delta_minutes)
        alerts: list[SafetyAlert] = []

        if reading.mg_dl_per_min > self.RISE_THRESHOLD_MG_DL_PER_MIN:
            alerts.append(
                SafetyAlert(
                    level=AlertLevel.CRITICAL,
                    code='VELOCITY_RISE',
                    message=(
                        f'Rapid BG rise: {reading.mg_dl_per_min:.1f} mg/dL/min '
                        f'(threshold {self.RISE_THRESHOLD_MG_DL_PER_MIN}).'
                    ),
                    context={
                        'mg_dl_per_min': reading.mg_dl_per_min,
                        'delta_bg': reading.delta_bg,
                        'delta_minutes': reading.delta_minutes,
                    },
                )
            )
        elif reading.mg_dl_per_min < self.FALL_THRESHOLD_MG_DL_PER_MIN:
            alerts.append(
                SafetyAlert(
                    level=AlertLevel.CRITICAL,
                    code='VELOCITY_FALL',
                    message=(
                        f'Rapid BG drop: {reading.mg_dl_per_min:.1f} mg/dL/min '
                        f'(threshold {self.FALL_THRESHOLD_MG_DL_PER_MIN}).'
                    ),
                    context={
                        'mg_dl_per_min': reading.mg_dl_per_min,
                        'delta_bg': reading.delta_bg,
                        'delta_minutes': reading.delta_minutes,
                    },
                )
            )

        return alerts
