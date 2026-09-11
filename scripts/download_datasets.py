#!/usr/bin/env python3
"""Download external T1D/CGM datasets into data/external/."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / 'data' / 'external'

ZENODO_DIADATA_RECORD = '17285631'
ZENODO_AWESOME_CGM_RECORD = '14541646'

# Jaeb public.jaeb.org record IDs (from DiaData, babelbetes, and JCHR index).
JAEB_DATASETS: dict[int, str] = {
    536: 'T1D_Exchange_Registry',
    537: 'Severe_Hypoglycemia_Older_Adults',
    546: 'ReplaceBG',
    560: 'Loop',
    563: 'JDRF_CGM_RCT',
    564: 'WISDM',
    565: 'CITY',
    566: 'FLAIR',
    573: 'DCLP3',
    589: 'T1DEXI',
    591: 'CLVer',
    599: 'PEDAP',
}

# Known direct S3 URLs (from prior JCHR registrations). Used when the API is blocked.
JAEB_KNOWN_URLS: dict[int, str] = {
    565: (
        'https://live-jchrpublicdatasets.s3.amazonaws.com/Diabetes/'
        'Public%20Datasets/CITYPublicDataset-344bea7d-8085-4deb-8038-6cb747a744e3.zip'
    ),
}

OHIO_T1DM_INFO = """\
# OhioT1DM

OhioT1DM requires a signed Data Use Agreement (DUA) and an institutional email.
Do **not** redistribute the dataset via this repository or public forks.

1. Complete the DUA form linked from:
   https://webpages.charlotte.edu/rbunescu/data/ohiot1dm/OhioT1DM-dataset.html
2. Email the signed form to the Ohio University coordinator (see that page).
3. Place the decrypted dataset under: `data/external/ohiot1dm/`

Alternatively, email razvan.bunescu@charlotte.edu with subject "OhioT1DM Request".
"""

T1DEXI_VIVLI_INFO = """\
# T1DEXI

T1DEXI access is governed by Jaeb and/or Vivli terms. Honor any DUA you sign.
Do **not** commit or redistribute restricted archives in this repo.

T1DEXI is mirrored on Jaeb (record 589) as "T1DEXI - DATA FOR UPLOAD.zip".

```bash
export JAEB_FULL_NAME="Your Name"
export JAEB_EMAIL="you@institution.edu"
export JAEB_INSTITUTION="Your Institution"
python scripts/download_datasets.py --only jaeb --jaeb-ids 589
```

If Jaeb is unavailable, request access via Vivli:
https://search.vivli.org/doiLanding/studies/PR00008428/isLanding
"""


def _download_file(url: str, dest: Path, session) -> Path:
    """Stream-download a file with resume support."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {}
    mode = 'wb'
    if dest.exists() and dest.stat().st_size > 0:
        headers['Range'] = f'bytes={dest.stat().st_size}-'
        mode = 'ab'

    print(f'Downloading {dest.name}...')
    with session.get(url, headers=headers, stream=True, timeout=600) as resp:
        if resp.status_code == 416:
            print(f'  already complete: {dest}')
            return dest
        resp.raise_for_status()
        with dest.open(mode) as handle:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return dest


def _zenodo_files(record_id: str, session) -> list[dict]:
    resp = session.get(
        f'https://zenodo.org/api/records/{record_id}',
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json().get('files', [])


def download_diadata(output_dir: Path, session, *, full: bool = False) -> list[Path]:
    """Download DiaData release from Zenodo."""
    out = output_dir / 'diadata'
    out.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for entry in _zenodo_files(ZENODO_DIADATA_RECORD, session):
        name = entry['key']
        if not full and name.endswith('.zip'):
            # Skip multi-GB archives unless --full.
            print(f'Skipping {name} (use --full to download).')
            continue
        dest = out / name
        if dest.exists() and dest.stat().st_size == entry.get('size', -1):
            saved.append(dest)
            continue
        url = entry['links']['self']
        saved.append(_download_file(url, dest, session))
    return saved


def download_awesome_cgm(output_dir: Path, session) -> list[Path]:
    """Clone Awesome-CGM and fetch the Zenodo v2.0.0 release archive."""
    import requests

    out = output_dir / 'awesome-cgm'
    out.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    repo_dir = out / 'repo'
    if not (repo_dir / '.git').exists():
        print('Cloning Awesome-CGM repository...')
        subprocess.run(
            ['git', 'clone', '--depth', '1',
             'https://github.com/IrinaStatsLab/Awesome-CGM.git', str(repo_dir)],
            check=True,
        )
        saved.append(repo_dir)

    for entry in _zenodo_files(ZENODO_AWESOME_CGM_RECORD, session):
        dest = out / Path(entry['key']).name
        if dest.exists():
            saved.append(dest)
            continue
        saved.append(_download_file(entry['links']['self'], dest, session))
    return saved


def _require_jaeb_identity() -> dict[str, str]:
    """Return Jaeb registration fields from env; refuse placeholder defaults."""
    required = ('JAEB_FULL_NAME', 'JAEB_EMAIL', 'JAEB_INSTITUTION')
    missing = [name for name in required if not os.environ.get(name, '').strip()]
    if missing:
        raise RuntimeError(
            'Jaeb downloads require real registration details. Set: '
            + ', '.join(missing)
            + '. Example:\n'
            '  export JAEB_FULL_NAME="Your Name"\n'
            '  export JAEB_EMAIL="you@institution.edu"\n'
            '  export JAEB_INSTITUTION="Your Institution"'
        )
    return {
        'fullname': os.environ['JAEB_FULL_NAME'].strip(),
        'email': os.environ['JAEB_EMAIL'].strip(),
        'institution': os.environ['JAEB_INSTITUTION'].strip(),
        'purpose': os.environ.get(
            'JAEB_PURPOSE',
            'Type 1 diabetes CGM forecasting research (T1DA project).',
        ).strip(),
    }


def _jaeb_request_url(rec_id: int, protocol: str, session) -> str | None:
    """Resolve a Jaeb dataset download URL via /getprotfile."""
    identity = _require_jaeb_identity()
    payload = {
        'recId': str(rec_id),
        'protDs': protocol,
        **identity,
    }
    resp = session.post(
        'https://public.jaeb.org/getprotfile',
        data=payload,
        timeout=120,
    )
    text = resp.text.strip()
    if 'Validation request' in text or 'captcha' in text.lower():
        return None
    if text.startswith('"') and text.endswith('"'):
        return json.loads(text)
    if text.startswith('http'):
        return text
    return None


def _jaeb_protocol(rec_id: int, session) -> str:
    resp = session.get(f'https://public.jaeb.org/dataset/{rec_id}', timeout=60)
    resp.raise_for_status()
    match = re.search(
        r'id="ctl00_CphMain_ctl00_ProtocolDescription"[^>]*>(.*?)</span>',
        resp.text,
        re.S,
    )
    if not match:
        return JAEB_DATASETS.get(rec_id, f'dataset_{rec_id}')
    return re.sub(r'<[^>]+>', '', match.group(1)).strip()


def download_jaeb(
    output_dir: Path,
    session,
    *,
    datasets: list[int] | None = None,
) -> list[Path]:
    """Download Jaeb Center public diabetes datasets.

    API registration requires ``JAEB_FULL_NAME``, ``JAEB_EMAIL``, and
    ``JAEB_INSTITUTION``. Known direct URLs (if any) still require you to
    comply with Jaeb terms of use.
    """
    out = output_dir / 'jaeb'
    out.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    ids = datasets or sorted(JAEB_DATASETS)
    blocked = False
    needs_api = any(rec_id not in JAEB_KNOWN_URLS for rec_id in ids)
    if needs_api:
        _require_jaeb_identity()

    for rec_id in ids:
        slug = JAEB_DATASETS.get(rec_id, f'dataset_{rec_id}')
        dest = out / f'{rec_id}_{slug}.zip'

        url = JAEB_KNOWN_URLS.get(rec_id)
        if url is None and not blocked:
            try:
                protocol = _jaeb_protocol(rec_id, session)
                url = _jaeb_request_url(rec_id, protocol, session)
            except RuntimeError:
                raise
            except Exception as exc:  # noqa: BLE001
                print(f'  Jaeb lookup failed for {rec_id}: {exc}')
                url = None
            if url is None:
                blocked = True
                print(
                    '  Jaeb API blocked (captcha/rate limit). '
                    'Retry later with JAEB_FULL_NAME, JAEB_EMAIL, '
                    'JAEB_INSTITUTION set.'
                )

        if url is None:
            continue
        if dest.exists():
            saved.append(dest)
            continue

        filename = Path(unquote(urlparse(url).path)).name
        dest = out / f'{rec_id}_{slug}_{filename}'
        try:
            saved.append(_download_file(url, dest, session))
        except Exception as exc:  # noqa: BLE001
            print(f'  failed {rec_id}: {exc}')

    readme = out / 'README.md'
    if blocked and not readme.exists():
        readme.write_text(
            '# Jaeb downloads\n\n'
            'Some datasets require passing the Jaeb captcha. Re-run:\n\n'
            '```bash\n'
            'export JAEB_FULL_NAME="Your Name"\n'
            'export JAEB_EMAIL="you@institution.edu"\n'
            'export JAEB_INSTITUTION="Your Institution"\n'
            'python scripts/download_datasets.py --only jaeb\n'
            '```\n',
            encoding='utf-8',
        )
    return saved


def write_manual_instructions(output_dir: Path) -> None:
    """Write README stubs for datasets that need manual access."""
    ohio = output_dir / 'ohiot1dm'
    ohio.mkdir(parents=True, exist_ok=True)
    (ohio / 'README.md').write_text(OHIO_T1DM_INFO, encoding='utf-8')

    t1dexi = output_dir / 't1dexi'
    t1dexi.mkdir(parents=True, exist_ok=True)
    (t1dexi / 'README.md').write_text(T1DEXI_VIVLI_INFO, encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser(description='Download external T1D datasets')
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=EXTERNAL,
        help='Root directory for external datasets',
    )
    parser.add_argument(
        '--only',
        choices=['diadata', 'awesome-cgm', 'jaeb', 'all'],
        default='all',
        help='Subset of datasets to download',
    )
    parser.add_argument(
        '--full',
        action='store_true',
        help='Download large DiaData zip archives (~3 GB total)',
    )
    parser.add_argument(
        '--jaeb-ids',
        type=int,
        nargs='*',
        help='Specific Jaeb record IDs (default: all known diabetes datasets)',
    )
    args = parser.parse_args()

    import requests

    session = requests.Session()
    session.headers['User-Agent'] = 't1da-dataset-downloader/1.0'

    targets = (
        ['diadata', 'awesome-cgm', 'jaeb']
        if args.only == 'all'
        else [args.only]
    )

    results: dict[str, list[Path]] = {}
    try:
        if 'diadata' in targets:
            results['diadata'] = download_diadata(
                args.output_dir, session, full=args.full,
            )
        if 'awesome-cgm' in targets:
            results['awesome-cgm'] = download_awesome_cgm(args.output_dir, session)
        if 'jaeb' in targets:
            results['jaeb'] = download_jaeb(
                args.output_dir, session, datasets=args.jaeb_ids,
            )
        if args.only == 'all':
            write_manual_instructions(args.output_dir)
    except Exception as exc:  # noqa: BLE001
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1

    for name, paths in results.items():
        print(f'{name}: {len(paths)} file(s) in {args.output_dir / name}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
