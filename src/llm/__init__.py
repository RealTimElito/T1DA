"""T1DA Module 3: LLM context parsing (text → structured features only)."""

from src.llm.nutrition import estimate_carbs_grams
from src.llm.parser import LLMContextParser, ParsedEvent, ParseError
from src.llm.prompts import SYSTEM_PROMPT
from src.llm.integration import apply_parsed_event_to_frame

__all__ = [
    'LLMContextParser',
    'ParsedEvent',
    'ParseError',
    'SYSTEM_PROMPT',
    'apply_parsed_event_to_frame',
    'estimate_carbs_grams',
]
