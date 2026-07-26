"""Rule-based and optional-LLM clinical text parser."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from src.llm.nutrition import estimate_carbs_grams, extract_insulin_units
from src.llm.prompts import SYSTEM_PROMPT


@dataclass
class ParsedEvent:
    """Structured event extracted from user text."""

    carbs_grams: float
    insulin_units: float
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ParseError:
    """Clarification request for ambiguous input."""

    message: str
    missing_fields: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            'error': True,
            'needs_clarification': True,
            'message': self.message,
            'missing_fields': self.missing_fields,
        }


class LLMContextParser:
    """Parse casual diabetes logs into JSON features (no forecasting).

    Uses deterministic rule-based extraction by default. An optional ``llm_client``
    callable may be supplied for Gemma or compatible local models; the client must
    return JSON text only and must not perform glucose prediction.
    """

    def __init__(self, llm_client: Any | None = None) -> None:
        self.llm_client = llm_client

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def parse(self, user_text: str, *, timestamp: datetime | None = None) -> dict[str, Any]:
        """Parse user text into success or error JSON schema."""
        if self.llm_client is not None:
            return self._parse_with_llm(user_text, timestamp=timestamp)
        return self._parse_rule_based(user_text, timestamp=timestamp)

    def _current_iso_timestamp(self, timestamp: datetime | None) -> str:
        if timestamp is not None:
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            return timestamp.astimezone(timezone.utc).isoformat()
        return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

    def _parse_rule_based(
        self,
        user_text: str,
        *,
        timestamp: datetime | None,
    ) -> dict[str, Any]:
        carbs, is_vague = estimate_carbs_grams(user_text)
        insulin = extract_insulin_units(user_text)
        missing: list[str] = []

        if is_vague or carbs is None:
            missing.append('carbs_grams')
        if insulin is None:
            # Spec allows 0.0 when carbs clear but insulin missing.
            if carbs is not None and not is_vague:
                insulin = 0.0
            else:
                missing.append('insulin_units')

        if missing:
            return ParseError(
                message='Please specify food items with quantities and insulin units.',
                missing_fields=missing,
            ).to_dict()

        event = ParsedEvent(
            carbs_grams=float(carbs),
            insulin_units=float(insulin),
            timestamp=self._current_iso_timestamp(timestamp),
        )
        return event.to_dict()

    def _parse_with_llm(
        self,
        user_text: str,
        *,
        timestamp: datetime | None,
    ) -> dict[str, Any]:
        """Delegate to external LLM; validate JSON and forbid forecast fields."""
        assert self.llm_client is not None
        response_text = self.llm_client(self.system_prompt, user_text)
        payload = json.loads(response_text)

        forbidden = {'glucose', 'forecast', 'prediction', 'bg', 'blood_sugar'}
        if forbidden.intersection(payload.keys()):
            raise ValueError('LLM response contains forbidden forecasting fields.')

        if payload.get('error'):
            return payload

        ts = payload.get('timestamp', 'CURRENT_ISO_TIMESTAMP')
        if ts == 'CURRENT_ISO_TIMESTAMP':
            ts = self._current_iso_timestamp(timestamp)

        return {
            'carbs_grams': float(payload['carbs_grams']),
            'insulin_units': float(payload['insulin_units']),
            'timestamp': ts,
        }

    @staticmethod
    def parse_json_response(raw: str) -> dict[str, Any]:
        """Parse model output, stripping optional markdown fences."""
        cleaned = raw.strip()
        cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
        cleaned = re.sub(r'\s*```$', '', cleaned)
        return json.loads(cleaned)
