# T1DA — Type 1 Diabetes Forecasting

Module 1 implements the clinical time-series data engineering pipeline for CGM-based glucose forecasting.

## Quick start

```bash
conda env create -f environment.yml
conda activate t1da

# Run with synthetic sample data (default)
python scripts/run_pipeline.py

# Validate outputs
python scripts/validate_data.py
```

Processed arrays are written to `data/processed/X_train.npy` and `data/processed/y_train.npy`.

## External datasets

Download DiaData, Awesome-CGM, Jaeb, T1DEXI, and OhioT1DM into `data/external/`:

```bash
conda activate t1da
python scripts/download_datasets.py --full
```

Use `--only diadata|awesome-cgm|jaeb` to fetch a subset. Jaeb downloads require your details:

```bash
export JAEB_FULL_NAME="Your Name"
export JAEB_EMAIL="you@institution.edu"
export JAEB_INSTITUTION="Your Institution"
python scripts/download_datasets.py --only jaeb
```

OhioT1DM and T1DEXI (Vivli fallback) need manual access — see `data/external/ohiot1dm/README.md` and `data/external/t1dexi/README.md`.

To refresh dependencies after editing `requirements.txt` or `environment.yml`:

```bash
conda env update -f environment.yml --prune
```

## Using real clinical data

### HUPA-UCM (recommended)

1. Download the preprocessed CSV files from [HUPA-UCM on Mendeley](https://data.mendeley.com/datasets/3hbcscwz44/1).
2. Extract `Preprocessed/HUPA*.csv` into `data/raw/`.
3. Run:

```bash
python scripts/run_pipeline.py --input data/raw/
```

HUPA files use semicolon separators with columns: `time`, `glucose`, `basal.rate`, `bolus.volume.delivered`, `carb.input`.

### Canonical CSV format

Place comma-separated files with columns `timestamp`, `glucose`, `basal`, `bolus`, `carbs` in a directory and pass `--input`.

## Pipeline design

| Stage | Module | Behavior |
|-------|--------|----------|
| Ingestion | `src/data/ingestion.py` | Parse HUPA or canonical CSV; normalize timestamps |
| Resampling | `src/data/resampling.py` | 5-minute grid; CGM linear interp for gaps < 15 min |
| Features | `src/data/features.py` | Sliding windows with no temporal leakage |
| Validation | `src/data/validation.py` | NaN assertions and shape checks |

### Output shapes

- `X_train`: `(samples, 12, 4)` — 60 minutes of `[glucose, basal, bolus, carbs]`
- `y_train`: `(samples, 6)` — future glucose (mg/dL) at **5, 10, 15, 20, 25, 30** minutes

Offsets are five-minute grid steps from the window end (`HORIZON_OFFSETS = (1, 2, 3, 4, 5, 6)` in `src/data/constants.py`).

### Leakage controls

- Feature windows use only timesteps `t-11 … t` when predicting from index `t`.
- Samples crossing CGM gaps ≥ 15 minutes are dropped.
- No forward-fill of glucose beyond the 15-minute interpolation limit.

## Tests

```bash
pytest tests/ -q
```

## End-to-end demo

Canonical data entry point: `scripts/run_pipeline.py` (use `--synthetic-only` for in-memory synthetic arrays).

VS Code / Cursor debug configs are in `.vscode/launch.json` (pipeline, train, eval, Gemma parser, guardrails, pytest).

```bash
# Data only
python scripts/run_pipeline.py --synthetic-only

# Train + evaluate (early stopping patience=7)
python scripts/train_model.py --synthetic-only --epochs 100 --patience 7
python scripts/evaluate_model.py

# Full stack: data → train → LLM parse → forecast → guardrails
python scripts/run_end_to_end.py --epochs 5
```

`run_synthetic_pipeline()` and `run_pipeline()` both return a metadata dict with `output_paths`; load `X_train.npy` / `y_train.npy` from there for training.

## Evaluation

After training or `evaluate_model.py`, plot confusion matrix and Clarke zone charts from the JSON artifacts:

```bash
python scripts/plot_metrics.py --prefix train
python scripts/plot_metrics.py --prefix eval
```

PNG files are written next to the JSON under `models/` (e.g. `train_confusion_matrix.png`).

## Project layout

```
T1DA/
├── src/data/           # Module 1 library code
├── scripts/            # CLI entry points
├── tests/              # Unit tests
├── data/
│   ├── sample/         # Generated synthetic CSVs
│   ├── raw/            # Place downloaded clinical data here
│   ├── processed/      # X_train.npy, y_train.npy
│   └── external/       # DiaData, Awesome-CGM, Jaeb, etc.
├── environment.yml
└── requirements.txt
```
