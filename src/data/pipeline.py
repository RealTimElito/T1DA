"""End-to-end Module 1 data engineering orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.features import build_feature_matrices
from src.data.ingestion import (
    generate_synthetic_patient,
    load_clinical_csv,
    load_clinical_directory,
    write_synthetic_sample,
)
from src.data.resampling import resample_clinical_data
from src.data.validation import validate_pipeline_outputs, validate_resampled_segment


def _load_input_frame(
    input_path: str | Path | None,
    use_sample_if_missing: bool,
) -> pd.DataFrame:
    """Load clinical data from path or generate development samples."""
    if input_path is None:
        if not use_sample_if_missing:
            raise ValueError('input_path is required when use_sample_if_missing=False.')
        sample_dir = Path('data/sample')
        write_synthetic_sample(sample_dir, n_patients=2)
        return load_clinical_directory(sample_dir)

    path = Path(input_path)
    if path.is_dir():
        return load_clinical_directory(path)
    if path.is_file():
        frame = load_clinical_csv(path)
        return frame.reset_index().assign(patient_id=path.stem).set_index(
            ['patient_id', 'timestamp']
        )
    if use_sample_if_missing:
        sample_dir = path.parent / 'sample'
        write_synthetic_sample(sample_dir, n_patients=2)
        return load_clinical_directory(sample_dir)
    raise FileNotFoundError(path)


def run_pipeline(
    input_path: str | Path | None = None,
    output_dir: str | Path = 'data/processed',
    use_sample_if_missing: bool = True,
) -> dict[str, Any]:
    """Execute ingestion, resampling, and feature generation.

    Args:
        input_path: File or directory of clinical CSVs. When None, synthetic
            sample data is generated under ``data/sample/``.
        output_dir: Directory for ``X_train.npy`` and ``y_train.npy``.
        use_sample_if_missing: Generate sample data when ``input_path`` is None.

    Returns:
        Summary metadata including shapes and output paths.
    """
    frame = _load_input_frame(input_path, use_sample_if_missing=use_sample_if_missing)
    segmented = resample_clinical_data(frame)

    segment_reports: list[dict[str, Any]] = []
    for patient_id, segments in segmented.items():
        for segment_idx, segment in enumerate(segments):
            report = validate_resampled_segment(segment)
            report.update({'patient_id': patient_id, 'segment_idx': segment_idx})
            segment_reports.append(report)

    x_train, y_train = build_feature_matrices(segmented)
    summary = validate_pipeline_outputs(x_train, y_train, min_samples=1)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x_path = out_dir / 'X_train.npy'
    y_path = out_dir / 'y_train.npy'
    np.save(x_path, x_train)
    np.save(y_path, y_train)

    summary.update(
        {
            'segment_reports': segment_reports,
            'output_paths': {'X_train': str(x_path), 'y_train': str(y_path)},
            'n_patients': len(segmented),
            'n_segments': sum(len(segs) for segs in segmented.values()),
        }
    )
    return summary


def run_synthetic_pipeline(
    output_dir: str | Path,
    *,
    n_patients: int = 3,
    hours: int = 72,
) -> dict[str, Any]:
    """Build arrays from in-memory synthetic patients (no external files)."""
    frames = [
        generate_synthetic_patient(
            patient_id=f'SYNTH{idx + 1:03d}',
            seed=42 + idx,
            hours=hours,
            gap_start_hour=18.0 if idx == 0 else None,
        )
        for idx in range(n_patients)
    ]
    combined = pd.concat(frames)
    segmented = resample_clinical_data(combined)
    x_train, y_train = build_feature_matrices(segmented)
    summary = validate_pipeline_outputs(x_train, y_train, min_samples=1)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'X_train.npy', x_train)
    np.save(out / 'y_train.npy', y_train)
    summary.update(
        {
            'source': 'synthetic',
            'output_paths': {
                'X_train': str(out / 'X_train.npy'),
                'y_train': str(out / 'y_train.npy'),
            },
            'n_patients': len(segmented),
            'n_segments': sum(len(segs) for segs in segmented.values()),
        }
    )
    return summary
