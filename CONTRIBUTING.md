# Contributing

Thanks for interest in T1DA. This is research software for Type 1 diabetes CGM
forecasting — not a medical device.

## Setup

```bash
conda env create -f environment.yml
conda activate t1da
pytest tests/ -q
```

Alternatively: `pip install -e ".[dev]"` from a Python 3.10+ environment.

## Guidelines

1. Keep changes focused; prefer small PRs.
2. Do not commit files under `data/raw/`, `data/processed/`, or `data/external/`
   (except `.gitkeep` stubs and synthetic samples under `data/sample/`).
3. Do not commit secrets (`.env`, HF tokens, Jaeb credentials).
4. Add or update tests when changing pipeline, model, or guardrail behavior.
5. Follow existing module layout under `src/` and Google-style Python docstrings
   where you add new public functions.

## Pull requests

- Describe the motivation and how you tested.
- Ensure `pytest tests/ -q` passes locally.
- Update the README or CHANGELOG when user-facing behavior changes.

## Dataset access

Restricted datasets (OhioT1DM, T1DEXI via Vivli, Jaeb registrations) remain the
contributor's responsibility. Do not redistribute restricted archives in PRs.
