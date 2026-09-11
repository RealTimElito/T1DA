# T1DA — Type 1 Diabetes Forecasting

Research software for CGM-based Type 1 diabetes glucose forecasting: clinical
time-series pipeline → LSTM model → optional Gemma meal/context parsing →
safety guardrails.

## Disclaimer

**Not for clinical use.** T1DA is research / educational software only. It does
**not** provide medical advice, diagnosis, or treatment. Do not use outputs to
make insulin, carbohydrate, or other care decisions. Always follow guidance from
qualified clinicians.

## Quick start

```bash
conda env create -f environment.yml
conda activate t1da

# Run with synthetic sample data (default)
python scripts/run_pipeline.py

# Validate outputs
python scripts/validate_data.py

# Unit tests
pytest tests/ -q
```

Processed arrays are written to `data/processed/X_train.npy` and
`data/processed/y_train.npy`.

Alternatively: `pip install -e ".[dev]"` (Python 3.10+).

## Train and evaluate

```bash
# Synthetic train + evaluate (early stopping patience=7)
python scripts/train_model.py --synthetic-only --epochs 100 --patience 7
python scripts/evaluate_model.py

# Plot Clarke / confusion artifacts under models/
python scripts/plot_metrics.py --prefix train
python scripts/plot_metrics.py --prefix eval
```

A small demo checkpoint is included at `models/glucose_lstm.pt`.

## Optional LLM parser (Gemma)

The meal/context parser defaults to a rule-based path. The optional Gemma path
needs a Hugging Face token, gated model access, and substantial GPU memory
(~24 GB VRAM for `google/gemma-3-12b-it` in bfloat16).

```bash
export HF_TOKEN="hf_..."   # or HUGGING_FACE_HUB_TOKEN
# Accept the model license on Hugging Face for google/gemma-3-12b-it
python scripts/run_llm_parser.py            # Gemma when available
python scripts/run_llm_parser.py --rule-based
```

## Guardrails and end-to-end

```bash
python scripts/run_guardrails.py --current-bg 90 --previous-bg 110 --iob 0

# Full stack: data → train → LLM parse → forecast → guardrails
python scripts/run_end_to_end.py --epochs 5
```

VS Code / Cursor debug configs live in `.vscode/launch.json`.

## External datasets

Download helpers fetch **public** releases into `data/external/`. You must
comply with each provider’s license / data use agreement. **Do not** commit
raw clinical archives to this repository.

```bash
conda activate t1da
python scripts/download_datasets.py --full
```

Use `--only diadata|awesome-cgm|jaeb` for a subset.

### Dataset terms

| Source | Access | Notes |
|--------|--------|--------|
| DiaData (Zenodo) | Automated | Follow Zenodo / DiaData license on the record |
| Awesome-CGM | Automated (git + Zenodo) | Follow upstream LICENSE in the cloned repo |
| Jaeb Center | Registration required | Set identity env vars before download; honor Jaeb ToS |
| OhioT1DM | Manual DUA | Signed agreement + institutional email; no redistribution here |
| T1DEXI | Jaeb and/or Vivli | Use Jaeb record 589 or Vivli study access; respect DUA |
| HUPA-UCM (Mendeley) | Manual download | Cite / follow Mendeley dataset terms |

Jaeb downloads require your details (no placeholder defaults):

```bash
export JAEB_FULL_NAME="Your Name"
export JAEB_EMAIL="you@institution.edu"
export JAEB_INSTITUTION="Your Institution"
python scripts/download_datasets.py --only jaeb
```

OhioT1DM and T1DEXI (Vivli fallback) need manual access — see
`data/external/ohiot1dm/README.md` and `data/external/t1dexi/README.md` after
running the downloader once (or the stubs written by `--only` all).

To refresh dependencies after editing `requirements.txt` or `environment.yml`:

```bash
conda env update -f environment.yml --prune
```

## Using real clinical data

### HUPA-UCM (recommended)

1. Download the preprocessed CSV files from
   [HUPA-UCM on Mendeley](https://data.mendeley.com/datasets/3hbcscwz44/1).
2. Extract `Preprocessed/HUPA*.csv` into `data/raw/`.
3. Run:

```bash
python scripts/run_pipeline.py --input data/raw/
```

HUPA files use semicolon separators with columns: `time`, `glucose`,
`basal.rate`, `bolus.volume.delivered`, `carb.input`.

### Canonical CSV format

Place comma-separated files with columns `timestamp`, `glucose`, `basal`,
`bolus`, `carbs` in a directory and pass `--input`.

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

Offsets are five-minute grid steps from the window end
(`HORIZON_OFFSETS = (1, 2, 3, 4, 5, 6)` in `src/data/constants.py`).

### Leakage controls

- Feature windows use only timesteps `t-11 … t` when predicting from index `t`.
- Samples crossing CGM gaps ≥ 15 minutes are dropped.
- No forward-fill of glucose beyond the 15-minute interpolation limit.

## Project layout

```
T1DA/
├── src/
│   ├── data/           # Ingest, resample, features, validation
│   ├── model/          # LSTM, training, Clarke / confusion metrics
│   ├── llm/            # Rule-based + optional Gemma parser
│   └── guardrails/     # Velocity / alert agent
├── scripts/            # CLI entry points
├── tests/              # Unit tests
├── models/             # Demo weights + eval artifacts
├── data/
│   ├── sample/         # Generated synthetic CSVs
│   ├── raw/            # Place downloaded clinical data here (gitignored)
│   ├── processed/      # X_train.npy, y_train.npy (gitignored)
│   └── external/       # DiaData, Awesome-CGM, Jaeb, etc. (gitignored)
├── LICENSE
├── SECURITY.md
├── CONTRIBUTING.md
├── CHANGELOG.md
├── environment.yml
├── pyproject.toml
└── requirements.txt
```

## License

MIT — see [LICENSE](LICENSE). Dataset files you download remain under their
providers’ terms and are not covered by this software license.

## Security

See [SECURITY.md](SECURITY.md) for how to report vulnerabilities.
