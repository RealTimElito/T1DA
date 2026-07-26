"""Clinical time-series ingestion and parsing."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.data.constants import CANONICAL_COLUMNS, HUPA_COLUMN_MAP

_REQUIRED_FEATURES = tuple(CANONICAL_COLUMNS.values())


def _assert_required_columns(frame: pd.DataFrame) -> None:
    """Raise if canonical feature columns are missing."""
    missing = [col for col in _REQUIRED_FEATURES if col not in frame.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')


def _normalize_timestamp(series: pd.Series) -> pd.Series:
    """Parse timestamps to timezone-naive UTC pandas datetimes."""
    parsed = pd.to_datetime(series, utc=True, errors='coerce')
    if parsed.isna().all():
        raise ValueError('Unable to parse any timestamps from input.')
    return parsed.dt.tz_convert(None)


def normalize_clinical_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted frame with canonical columns and numeric features.

    Args:
        frame: DataFrame containing at least timestamp + 4 clinical features.

    Returns:
        DataFrame indexed by timestamp with float feature columns.
    """
    working = frame.copy()
    if 'timestamp' not in working.columns:
        raise ValueError("Input must include a 'timestamp' column.")

    working['timestamp'] = _normalize_timestamp(working['timestamp'])
    working = working.dropna(subset=['timestamp'])
    working = working.sort_values('timestamp')

    for column in _REQUIRED_FEATURES:
        working[column] = pd.to_numeric(working[column], errors='coerce')

    # Insulin and carbs are sparse events; missing values are treated as zero dose.
    for column in ('basal', 'bolus', 'carbs'):
        working[column] = working[column].fillna(0.0)

    working = working.drop_duplicates(subset=['timestamp'], keep='last')
    working = working.set_index('timestamp')
    _assert_required_columns(working)
    return working[list(_REQUIRED_FEATURES)]


def detect_csv_format(path: Path) -> str:
    """Detect supported CSV dialect from headers."""
    header = path.read_text(encoding='utf-8', errors='replace').splitlines()[0]
    hupa_markers = (
        ('basal.rate', 'bolus.volume.delivered'),
        ('basal_rate', 'bolus_volume_delivered'),
    )
    if any(a in header and b in header for a, b in hupa_markers):
        return 'hupa'
    if all(col in header for col in ('glucose', 'basal', 'bolus', 'carbs')):
        return 'canonical'
    raise ValueError(f'Unsupported CSV format in {path}')


def load_clinical_csv(path: str | Path) -> pd.DataFrame:
    """Load a single patient CSV in HUPA-UCM or canonical T1DA format.

    HUPA-UCM preprocessed files use semicolon separators and dotted column names.
    Canonical files use comma separators and short column names.

    Args:
        path: Path to one patient record file.

    Returns:
        Normalized clinical frame indexed by timestamp.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    fmt = detect_csv_format(file_path)
    if fmt == 'hupa':
        raw = pd.read_csv(file_path, sep=';')
        renamed = raw.rename(columns=HUPA_COLUMN_MAP)
    else:
        renamed = pd.read_csv(file_path)

    return normalize_clinical_frame(renamed)


def load_clinical_directory(path: str | Path, pattern: str = '*.csv') -> pd.DataFrame:
    """Load and concatenate all patient CSV files in a directory.

    Args:
        path: Directory containing per-patient CSV files.
        pattern: Glob pattern for patient files.

    Returns:
        Combined normalized frame with a ``patient_id`` column.
    """
    directory = Path(path)
    if not directory.is_dir():
        raise NotADirectoryError(directory)

    frames: list[pd.DataFrame] = []
    for file_path in sorted(directory.glob(pattern)):
        patient_frame = load_clinical_csv(file_path)
        patient_frame = patient_frame.reset_index()
        patient_frame['patient_id'] = file_path.stem
        frames.append(patient_frame)

    if not frames:
        raise FileNotFoundError(f'No files matching {pattern!r} in {directory}')

    combined = pd.concat(frames, ignore_index=True)
    combined['timestamp'] = pd.to_datetime(combined['timestamp'])
    return combined.set_index(['patient_id', 'timestamp']).sort_index()


def generate_synthetic_patient(
    patient_id: str = 'SYNTH001',
    start: str = '2024-01-01 00:00:00',
    hours: int = 48,
    seed: int = 42,
    gap_start_hour: float | None = 20.0,
    gap_minutes: float = 30.0,
) -> pd.DataFrame:
    """Create synthetic clinical-format data for local pipeline development.

    Generates realistic-ish CGM variation with sparse bolus/carbs and a continuous
    basal rate. Optionally inserts a CGM outage to exercise gap-breaking logic.

    Args:
        patient_id: Identifier stored in the output frame.
        start: UTC start timestamp.
        hours: Duration of the synthetic record.
        seed: Random seed for reproducibility.
        gap_start_hour: Hour offset where a CGM gap begins; None disables gap.
        gap_minutes: Length of the inserted CGM gap.

    Returns:
        Normalized clinical frame indexed by timestamp.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range(start=start, periods=hours * 12 + 1, freq='5min')

    glucose = 110.0 + np.cumsum(rng.normal(0.0, 1.5, size=len(index)))
    glucose = np.clip(glucose, 70.0, 250.0)

    basal = np.full(len(index), 0.8)
    bolus = np.zeros(len(index))
    carbs = np.zeros(len(index))

    meal_times = index[::48]
    for meal_time in meal_times:
        loc = index.get_indexer([meal_time])[0]
        if loc >= 0:
            carbs[loc] = rng.uniform(30.0, 60.0)
            bolus[loc] = rng.uniform(2.0, 6.0)

    frame = pd.DataFrame(
        {
            'timestamp': index,
            'glucose': glucose,
            'basal': basal,
            'bolus': bolus,
            'carbs': carbs,
        }
    )

    if gap_start_hour is not None:
        gap_start = pd.Timestamp(start) + pd.Timedelta(hours=gap_start_hour)
        gap_end = gap_start + pd.Timedelta(minutes=gap_minutes)
        mask = (frame['timestamp'] >= gap_start) & (frame['timestamp'] < gap_end)
        frame.loc[mask, 'glucose'] = np.nan

    normalized = normalize_clinical_frame(frame)
    normalized['patient_id'] = patient_id
    return normalized.reset_index().set_index(['patient_id', 'timestamp'])


def write_synthetic_sample(output_dir: str | Path, n_patients: int = 2) -> list[Path]:
    """Write canonical CSV samples that mimic public dataset layout.

    Args:
        output_dir: Directory for generated CSV files.
        n_patients: Number of synthetic patients to create.

    Returns:
        Paths to generated CSV files.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for patient_idx in range(n_patients):
        frame = generate_synthetic_patient(
            patient_id=f'SAMPLE{patient_idx + 1:03d}',
            seed=100 + patient_idx,
            gap_start_hour=12.0 + patient_idx * 4 if patient_idx == 0 else None,
        )
        export = frame.reset_index()[['timestamp', 'glucose', 'basal', 'bolus', 'carbs']]
        path = directory / f'SAMPLE{patient_idx + 1:03d}.csv'
        export.to_csv(path, index=False)
        written.append(path)
    return written


def load_clinical_json(path: str | Path) -> pd.DataFrame:
    """Load canonical JSON clinical records (list of timestamped events).

    Expected format: JSON array of objects with timestamp, glucose, basal, bolus, carbs.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    raw = pd.read_json(file_path)
    return normalize_clinical_frame(raw)


def load_clinical_file(path: str | Path) -> pd.DataFrame:
    """Dispatch loader by file extension (.csv, .json)."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == '.json':
        return load_clinical_json(file_path)
    if suffix == '.csv':
        return load_clinical_csv(file_path)
    raise ValueError(f'Unsupported file type: {suffix}')


def list_patient_ids(frame: pd.DataFrame) -> Iterable[str]:
    """Return patient identifiers when present, else a single synthetic id."""
    if isinstance(frame.index, pd.MultiIndex) and 'patient_id' in frame.index.names:
        return frame.index.get_level_values('patient_id').unique()
    return ('default',)
