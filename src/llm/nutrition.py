"""Approximate carbohydrate grams from common food descriptions."""

from __future__ import annotations

import re
from typing import Iterable

# Grams of carbohydrate per typical serving (approximate USDA values).
NUTRITION_PROFILES: dict[str, float] = {
    'apple': 25.0,
    'medium apple': 25.0,
    'banana': 27.0,
    'orange': 15.0,
    'slice of bread': 15.0,
    'bread': 15.0,
    'slice of pizza': 35.0,
    'pizza': 35.0,
    'pizza slice': 35.0,
    'cup of rice': 45.0,
    'rice': 45.0,
    'cup of pasta': 43.0,
    'pasta': 43.0,
    'granola bar': 22.0,
    'yogurt': 17.0,
    'milk': 12.0,
    'cookie': 15.0,
    'sandwich': 30.0,
    'bagel': 48.0,
    'cereal': 30.0,
    'juice': 26.0,
    'soda': 39.0,
    'crackers': 20.0,
}

_VAGUE_FOOD = re.compile(
    r'\b(snack|something|some food|a bit|a little|stuff|ate something)\b',
    re.IGNORECASE,
)
_INSULIN_PATTERN = re.compile(
    r'(\d+(?:\.\d+)?)\s*(?:units?|u)\s*(?:of\s*)?'
    r'(?:humalog|novolog|apidra|fiasp|lyumjev|insulin)?',
    re.IGNORECASE,
)
_INSULIN_SIMPLE = re.compile(r'took\s+(\d+(?:\.\d+)?)', re.IGNORECASE)
_CARB_GRAMS_PATTERN = re.compile(r'(\d+(?:\.\d+)?)\s*(?:g|grams?)\s*(?:of\s*)?carb', re.IGNORECASE)


def _normalize_food_key(text: str) -> str:
    return re.sub(r'\s+', ' ', text.strip().lower())


def estimate_carbs_from_foods(food_phrases: Iterable[str]) -> float | None:
    """Sum carbs from known food phrases; return None if any phrase is unknown."""
    total = 0.0
    for phrase in food_phrases:
        key = _normalize_food_key(phrase)
        if key not in NUTRITION_PROFILES:
            return None
        total += NUTRITION_PROFILES[key]
    return total


def extract_food_phrases(text: str) -> list[str]:
    """Pull food items from casual meal descriptions."""
    lowered = text.lower()
    if _VAGUE_FOOD.search(lowered):
        return []

    matched: list[str] = []
    occupied: list[tuple[int, int]] = []
    for profile_key in sorted(NUTRITION_PROFILES, key=len, reverse=True):
        start = 0
        while True:
            idx = lowered.find(profile_key, start)
            if idx < 0:
                break
            end = idx + len(profile_key)
            if not any(not (end <= lo or idx >= hi) for lo, hi in occupied):
                matched.append(profile_key)
                occupied.append((idx, end))
            start = idx + 1
    return matched


def estimate_carbs_grams(text: str) -> tuple[float | None, bool]:
    """Estimate total carbs from free text.

    Returns:
        ``(grams, is_vague)`` — grams is None when clarification is required.
    """
    gram_match = _CARB_GRAMS_PATTERN.search(text)
    if gram_match:
        return float(gram_match.group(1)), False

    phrases = extract_food_phrases(text)
    if not phrases:
        return None, True

    total = estimate_carbs_from_foods(phrases)
    if total is None:
        return None, True
    return total, False


def extract_insulin_units(text: str) -> float | None:
    """Extract rapid-acting insulin dose from text."""
    match = _INSULIN_PATTERN.search(text)
    if match:
        return float(match.group(1))
    simple = _INSULIN_SIMPLE.search(text)
    if simple:
        return float(simple.group(1))
    return None
