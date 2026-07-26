#!/usr/bin/env python3
"""Download HUPA-UCM preprocessed CSV files into data/raw/."""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MENDELEY_DATASET_ID = '3hbcscwz44'
MENDELEY_API = (
    f'https://data.mendeley.com/public-api/datasets/{MENDELEY_DATASET_ID}?fields=files'
)
HUPA_PATTERN = re.compile(r'HUPA\d+[a-zA-Z]\.csv', re.IGNORECASE)


def _fetch_file_list() -> list[dict]:
    """Query Mendeley public API for downloadable files."""
    import requests

    headers = {
        'Accept': 'application/json',
        'User-Agent': 't1da-hupa-downloader/1.0',
    }
    response = requests.get(MENDELEY_API, headers=headers, timeout=60)
    if response.status_code == 403:
        raise RuntimeError(
            'Mendeley API returned 403. Download manually from '
            'https://data.mendeley.com/datasets/3hbcscwz44/1 '
            'and extract Preprocessed/HUPA*.csv into data/raw/.'
        )
    response.raise_for_status()
    data = response.json()
    files = []
    for entry in data.get('files', []):
        download_url = entry.get('download_url')
        if not download_url and 'content_details' in entry:
            download_url = entry['content_details'].get('download_url')
        if download_url:
            files.append({
                'filename': entry['filename'],
                'url': download_url,
            })
    return files


def _download_hupa_csvs(output_dir: Path) -> list[Path]:
    """Download HUPA patient CSVs via Mendeley public API."""
    import requests

    output_dir.mkdir(parents=True, exist_ok=True)
    files = _fetch_file_list()
    hupa_files = [f for f in files if HUPA_PATTERN.fullmatch(f['filename'])]
    if not hupa_files:
        raise RuntimeError(
            'No HUPA*.csv files found via Mendeley API. '
            'Manual download required (see README).'
        )

    saved: list[Path] = []
    for entry in hupa_files:
        dest = output_dir / entry['filename']
        if dest.exists():
            saved.append(dest)
            continue
        print(f'Downloading {entry["filename"]}...')
        resp = requests.get(entry['url'], timeout=600)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        saved.append(dest)
    return saved


def _extract_local_zip(zip_path: Path, output_dir: Path) -> list[Path]:
    """Extract HUPA CSVs from a local Mendeley zip archive."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            base = Path(name).name
            if HUPA_PATTERN.fullmatch(base):
                target = output_dir / base
                with archive.open(name) as src, target.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
                saved.append(target)
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(description='Download HUPA-UCM preprocessed CSVs')
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=ROOT / 'data' / 'raw',
        help='Destination for HUPA*.csv files',
    )
    parser.add_argument(
        '--zip',
        type=Path,
        default=None,
        help='Local Mendeley zip (skip API download)',
    )
    args = parser.parse_args()

    try:
        if args.zip:
            paths = _extract_local_zip(args.zip, args.output_dir)
        else:
            paths = _download_hupa_csvs(args.output_dir)
    except Exception as exc:  # noqa: BLE001 — CLI surfaces actionable message
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1

    print(f'Downloaded {len(paths)} HUPA CSV file(s) to {args.output_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
