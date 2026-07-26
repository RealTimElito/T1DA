#!/usr/bin/env python3
"""Plot confusion matrix and Clarke error grid metrics from JSON artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

CLARKE_ZONE_COLORS = {
    'A': '#2ca02c',
    'B': '#98df8a',
    'C': '#ffbb78',
    'D': '#ff7f0e',
    'E': '#d62728',
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _matrix_array(data: dict) -> tuple[list[str], np.ndarray]:
    labels = data['labels']
    matrix = data['matrix']
    arr = np.array(
        [[matrix[actual][pred] for pred in labels] for actual in labels],
        dtype=np.int64,
    )
    return labels, arr


def plot_confusion_matrix(
    data: dict,
    title: str,
    output_path: Path,
    show: bool,
) -> None:
    labels, arr = _matrix_array(data)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(arr, cmap='Blues')

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)

    threshold = arr.max() / 2 if arr.max() > 0 else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = arr[i, j]
            color = 'white' if val > threshold else 'black'
            ax.text(j, i, f'{val:,}', ha='center', va='center', color=color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    acc = data.get('accuracy_pct')
    n = data.get('n_points')
    if acc is not None and n is not None:
        fig.text(0.5, 0.02, f'n={n:,}  accuracy={acc:.2f}%', ha='center', fontsize=9)

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def plot_clarke_zones(
    data: dict,
    title: str,
    output_path: Path,
    show: bool,
) -> None:
    zones = ['A', 'B', 'C', 'D', 'E']
    counts = [data['zone_counts'].get(zone, 0) for zone in zones]
    colors = [CLARKE_ZONE_COLORS[zone] for zone in zones]
    total = data['n_points']
    pcts = [100.0 * count / total for count in counts]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(zones, counts, color=colors)

    ax.set_xlabel('Clarke zone')
    ax.set_ylabel('Count')
    ax.set_title(title)

    for bar, pct in zip(bars, pcts):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{int(height):,}\n({pct:.2f}%)',
            ha='center',
            va='bottom',
            fontsize=8,
        )

    ab_pct = data.get('zone_ab_pct')
    if ab_pct is not None:
        fig.text(
            0.5,
            0.02,
            f'n={total:,}  zones A+B={ab_pct:.2f}%',
            ha='center',
            fontsize=9,
        )

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Plot evaluation metrics from JSON artifacts',
    )
    parser.add_argument(
        '--models-dir',
        type=Path,
        default=ROOT / 'models',
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='train',
        choices=['train', 'eval'],
        help='Artifact prefix (train or eval)',
    )
    parser.add_argument(
        '--confusion',
        type=Path,
        default=None,
        help='Confusion matrix JSON path (default: models/<prefix>_confusion_matrix.json)',
    )
    parser.add_argument(
        '--clarke',
        type=Path,
        default=None,
        help='Clarke JSON path (default: models/<prefix>_clarke.json)',
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display plots interactively',
    )
    args = parser.parse_args()

    confusion_path = args.confusion or args.models_dir / f'{args.prefix}_confusion_matrix.json'
    clarke_path = args.clarke or args.models_dir / f'{args.prefix}_clarke.json'

    confusion_out = confusion_path.with_suffix('.png')
    clarke_out = clarke_path.with_suffix('.png')

    plot_confusion_matrix(
        _load_json(confusion_path),
        f'{args.prefix} confusion matrix',
        confusion_out,
        args.show,
    )
    plot_clarke_zones(
        _load_json(clarke_path),
        f'{args.prefix} Clarke error grid zones',
        clarke_out,
        args.show,
    )

    print(confusion_out)
    print(clarke_out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
