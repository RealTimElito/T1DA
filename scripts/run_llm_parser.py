#!/usr/bin/env python3
"""Run Module 3: parse casual diabetes log text to JSON features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm.gemma_client import DEFAULT_MODEL_ID, create_parser  # noqa: E402
from src.llm.parser import LLMContextParser  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description='T1DA LLM context parser')
    parser.add_argument('text', nargs='?', help='User log text to parse')
    parser.add_argument('--file', type=Path, help='Read text from file')
    parser.add_argument(
        '--rule-based',
        action='store_true',
        help='Use deterministic parser instead of Gemma 12B',
    )
    parser.add_argument(
        '--model-id',
        type=str,
        default=DEFAULT_MODEL_ID,
        help='Hugging Face model id (default: google/gemma-3-12b-it)',
    )
    args = parser.parse_args()

    user_text = args.text
    if args.file:
        user_text = args.file.read_text(encoding='utf-8').strip()
    if not user_text:
        user_text = (
            'Just ate an apple and a slice of pizza, took 4 units of humalog.'
        )

    if args.rule_based:
        llm_parser: LLMContextParser = LLMContextParser()
    else:
        llm_parser = create_parser(use_gemma=True, model_id=args.model_id)

    result = llm_parser.parse(user_text)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
