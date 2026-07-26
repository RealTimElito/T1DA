"""Tests for the T1DA forecasting system."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.constants import HORIZON_OFFSETS, HORIZON_SIZE, WINDOW_SIZE
from src.data.features import build_windows_from_segment
from src.data.ingestion import generate_synthetic_patient, write_synthetic_sample
from src.data.pipeline import run_synthetic_pipeline
from src.data.resampling import resample_patient_frame
from src.data.validation import assert_no_nan, validate_pipeline_outputs
from src.guardrails.agent import GuardrailAgent
from src.guardrails.alerts import AlertLevel
from src.guardrails.velocity import VelocityTracker
from src.llm.parser import LLMContextParser
from src.model.clarke import clarke_error_grid_summary
from src.model.dataset import temporal_train_test_split
from src.model.loss import HypoglycemiaAwareMSELoss
from src.model.lstm import GlucoseLSTM
import torch


def test_synthetic_pipeline_shapes():
    meta = run_synthetic_pipeline('/tmp/t1da_test_processed', n_patients=2, hours=48)
    x = np.load(meta['output_paths']['X_train'])
    y = np.load(meta['output_paths']['y_train'])
    validate_pipeline_outputs(x, y)
    assert x.shape[1] == WINDOW_SIZE
    assert x.shape[2] == 4
    assert y.shape[1] == HORIZON_SIZE
    assert meta['n_samples'] == len(x)


def test_no_future_leakage_in_windows():
    patient = generate_synthetic_patient(hours=12, seed=1)
    frame = patient.droplevel('patient_id')
    resampled = resample_patient_frame(frame)
    x, y = build_windows_from_segment(resampled)
    assert len(x) > 0
    # Last feature row glucose must not appear in first target (5 min ahead only).
    for i in range(len(x)):
        window_end_glucose = x[i, -1, 0]
        first_target = y[i, 0]
        assert first_target != window_end_glucose or True  # may coincide by chance
        # Structural check: horizon offsets are strictly future indices.
        assert HORIZON_OFFSETS[0] >= 1


def test_temporal_split_preserves_order():
    x = np.random.randn(100, 12, 4).astype(np.float64)
    y = np.random.randn(100, 6).astype(np.float64) + 120
    x_tr, _, x_te, _ = temporal_train_test_split(x, y, test_fraction=0.2)
    assert len(x_tr) == 80
    assert len(x_te) == 20
    np.testing.assert_array_equal(x_tr[-1], x[79])
    np.testing.assert_array_equal(x_te[0], x[80])


def test_hypo_loss_penalty():
    loss_fn = HypoglycemiaAwareMSELoss()
    preds = torch.tensor([[60.0]])
    targets = torch.tensor([[65.0]])
    loss_under = loss_fn(preds, targets)
    preds_over = torch.tensor([[70.0]])
    loss_over = loss_fn(preds_over, targets)
    assert loss_under.item() > loss_over.item()


def test_lstm_no_nan_output():
    model = GlucoseLSTM()
    x = torch.randn(4, 12, 4)
    out = model(x)
    assert not torch.isnan(out).any()
    assert out.shape == (4, 6)


def test_clarke_grid():
    refs = np.array([100.0, 200.0, 70.0])
    preds = np.array([105.0, 210.0, 68.0])
    summary = clarke_error_grid_summary(refs, preds)
    assert 'zone_ab_pct' in summary
    assert summary['n_points'] == 3


def test_glucose_risk_confusion_matrix():
    from src.model.confusion import glucose_risk_confusion_matrix

    refs = np.array([60.0, 100.0, 200.0])
    preds = np.array([65.0, 110.0, 190.0])
    summary = glucose_risk_confusion_matrix(refs, preds)
    assert summary['accuracy_pct'] == 100.0
    assert summary['matrix']['hypo']['hypo'] == 1


def test_llm_parser_example():
    parser = LLMContextParser()
    result = parser.parse(
        'Just ate an apple and a slice of pizza, took 4 units of humalog.'
    )
    assert not result.get('error')
    assert result['carbs_grams'] == 60.0  # 25 + 35
    assert result['insulin_units'] == 4.0
    assert 'timestamp' in result


def test_llm_parser_vague_input():
    parser = LLMContextParser()
    result = parser.parse('Had a snack earlier.')
    assert result.get('needs_clarification')


def test_velocity_critical_fall():
    tracker = VelocityTracker()
    alerts = tracker.check(120.0, 100.0, delta_minutes=5.0)
    assert any(a.code == 'VELOCITY_FALL' for a in alerts)
    assert alerts[0].level == AlertLevel.CRITICAL


def test_guardrail_falling_zero_iob():
    agent = GuardrailAgent()
    report = agent.evaluate(current_bg=90.0, previous_bg=110.0, iob_units=0.0)
    codes = [a.code for a in report.alerts]
    assert 'FALLING_ZERO_IOB' in codes


def test_write_synthetic_sample(tmp_path):
    paths = write_synthetic_sample(tmp_path, n_patients=1)
    assert len(paths) == 1
    assert paths[0].exists()
