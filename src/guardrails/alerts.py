"""Alert severity classification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class AlertLevel(str, Enum):
    """Guardrail alert severity."""

    INFO = 'Info'
    WARNING = 'Warning'
    CRITICAL = 'Critical'


@dataclass
class SafetyAlert:
    """One guardrail finding."""

    level: AlertLevel
    code: str
    message: str
    context: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            'level': self.level.value,
            'code': self.code,
            'message': self.message,
        }
        if self.context:
            payload['context'] = self.context
        return payload
