"""Training loop for the glucose LSTM forecaster."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.constants import HORIZON_SIZE, WINDOW_SIZE
from src.model.clarke import clarke_error_grid_summary
from src.model.confusion import glucose_risk_confusion_matrix
from src.model.dataset import GlucoseForecastDataset, temporal_train_test_split
from src.model.loss import HypoglycemiaAwareMSELoss
from src.model.lstm import GlucoseLSTM


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    hidden_dim: int = 64,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    test_fraction: float = 0.2,
    patience: int = 7,
    device: str | None = None,
) -> dict[str, Any]:
    """Train LSTM and return metrics plus model state dict.

    Uses a chronological train/test split so no future samples leak into training.
    """
    assert not np.isnan(x_train).any(), 'X_train contains NaN'
    assert not np.isnan(y_train).any(), 'y_train contains NaN'

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dev = torch.device(device)

    x_tr, y_tr, x_te, y_te = temporal_train_test_split(
        x_train, y_train, test_fraction=test_fraction
    )

    train_loader = DataLoader(
        GlucoseForecastDataset(x_tr, y_tr),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        GlucoseForecastDataset(x_te, y_te),
        batch_size=batch_size,
        shuffle=False,
    )

    model = GlucoseLSTM(
        n_features=x_train.shape[2],
        hidden_dim=hidden_dim,
        horizon=y_train.shape[1],
    ).to(dev)
    criterion = HypoglycemiaAwareMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history: list[dict[str, float]] = []
    best_test_loss = float('inf')
    best_state: dict[str, Any] | None = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(dev)
            y_batch = y_batch.to(dev)
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
            n_batches += 1

        model.eval()
        test_loss = 0.0
        n_test = 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(dev)
                y_batch = y_batch.to(dev)
                preds = model(x_batch)
                test_loss += float(criterion(preds, y_batch).item())
                n_test += 1

        epoch_test_loss = test_loss / max(n_test, 1)
        history.append(
            {
                'epoch': epoch + 1,
                'train_loss': train_loss / max(n_batches, 1),
                'test_loss': epoch_test_loss,
            }
        )

        if epoch_test_loss < best_test_loss:
            best_test_loss = epoch_test_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    all_preds: list[np.ndarray] = []
    all_refs: list[np.ndarray] = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            preds = model(x_batch.to(dev)).cpu().numpy()
            all_preds.append(preds)
            all_refs.append(y_batch.numpy())

    preds_array = np.concatenate(all_preds, axis=0)
    refs_array = np.concatenate(all_refs, axis=0)
    # Evaluate first horizon (5 min ahead) for Clarke grid and risk confusion matrix.
    clarke = clarke_error_grid_summary(refs_array[:, 0], preds_array[:, 0])
    confusion = glucose_risk_confusion_matrix(refs_array[:, 0], preds_array[:, 0])

    return {
        'history': history,
        'clarke': clarke,
        'confusion_matrix': confusion,
        'model_state': model.state_dict(),
        'config': {
            'hidden_dim': hidden_dim,
            'window_size': WINDOW_SIZE,
            'horizon_size': HORIZON_SIZE,
            'n_train': len(x_tr),
            'n_test': len(x_te),
            'device': str(dev),
            'patience': patience,
            'best_epoch': best_epoch,
            'epochs_run': len(history),
            'best_test_loss': best_test_loss,
        },
    }


def save_checkpoint(result: dict[str, Any], path: str | Path) -> Path:
    """Persist model weights and training metadata."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'model_state': result['model_state'],
        'config': result['config'],
        'clarke': result['clarke'],
        'confusion_matrix': result.get('confusion_matrix'),
        'history': result['history'],
    }
    torch.save(payload, out)
    meta_path = out.with_suffix('.json')
    meta_path.write_text(
        json.dumps(
            {
                'config': result['config'],
                'clarke': result['clarke'],
                'confusion_matrix': result.get('confusion_matrix'),
                'history': result['history'],
            },
            indent=2,
        ),
        encoding='utf-8',
    )
    return out


def save_evaluation_artifacts(
    result: dict[str, Any],
    models_dir: str | Path,
    *,
    prefix: str = 'evaluation',
) -> dict[str, Path]:
    """Write Clarke and confusion-matrix reports under ``models/``."""
    out_dir = Path(models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        'clarke': result.get('clarke'),
        'confusion_matrix': result.get('confusion_matrix'),
        'config': result.get('config'),
        'mse': result.get('mse'),
        'n_samples': result.get('n_samples'),
    }
    report_path = out_dir / f'{prefix}_report.json'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')

    clarke_path = out_dir / f'{prefix}_clarke.json'
    clarke_path.write_text(
        json.dumps(result.get('clarke', {}), indent=2),
        encoding='utf-8',
    )

    confusion_path = out_dir / f'{prefix}_confusion_matrix.json'
    confusion_path.write_text(
        json.dumps(result.get('confusion_matrix', {}), indent=2),
        encoding='utf-8',
    )

    return {
        'report': report_path,
        'clarke': clarke_path,
        'confusion_matrix': confusion_path,
    }
