#!/usr/bin/env python3
"""Run Module 4: guardrail safety checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.guardrails.agent import GuardrailAgent  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description='T1DA guardrail agent')
    parser.add_argument('--current-bg', type=float, required=True)
    parser.add_argument('--previous-bg', type=float, default=None)
    parser.add_argument('--delta-minutes', type=float, default=5.0)
    parser.add_argument('--iob', type=float, default=0.0)
    parser.add_argument(
        '--predictions',
        type=float,
        nargs='*',
        help='Optional ML horizon predictions (mg/dL)',
    )
    args = parser.parse_args()

    agent = GuardrailAgent()
    report = agent.evaluate(
        current_bg=args.current_bg,
        previous_bg=args.previous_bg,
        delta_minutes=args.delta_minutes,
        iob_units=args.iob,
        ml_predictions=args.predictions,
    )
    print(json.dumps(report.to_dict(), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
