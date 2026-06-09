# -*- coding: utf-8 -*-
"""
bccwj-textmerge.py

Joins per-subject events.tsv files with event.tsv to add surface
and related text columns.

Join strategy:
    EEG/MEG : match section_num (subject) → subsection_num (master),
              pair rows sequentially within each section
    fMRI    : match run-N (filename) → section_num (master),
              pair rows sequentially within each run

Usage:
    python bccwj-textmerge.py --modality eeg|meg|fmri [--subject sub-01] [--dry-run]
"""

import argparse
import re
import pandas as pd
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
MASTER_FILE = SCRIPT_DIR / 'event.tsv'
TASK        = 'BCCWJreading'
OUT_SUFFIX  = '_with_surface'

ROOT_DIR = SCRIPT_DIR.parent

MODALITY_CONFIG = {
    # 'eeg':  {'root': ROOT_DIR / 'ds007753',  'subdir': 'eeg',  'join_key': 'subsection_num'},
    # 'meg':  {'root': ROOT_DIR / 'ds007763',  'subdir': 'meg',  'join_key': 'subsection_num'},
    # 'fmri': {'root': ROOT_DIR / 'ds007752', 'subdir': 'func', 'join_key': 'section_num'},
    'eeg':  {'root': ROOT_DIR / 'BCCWJ-EEG',  'subdir': 'eeg',  'join_key': 'subsection_num'},
    'meg':  {'root': ROOT_DIR / 'BCCWJ-MEG',  'subdir': 'meg',  'join_key': 'subsection_num'},
    'fmri': {'root': ROOT_DIR / 'BCCWJ-fMRI', 'subdir': 'func', 'join_key': 'section_num'},
}

# ─────────────────────────────────────────────────────────────────────────────

def load_master(path: Path) -> pd.DataFrame:
    master = pd.read_csv(path, sep='\t')
    required = {'surface', 'section_num', 'subsection_num'}
    missing  = required - set(master.columns)
    if missing:
        raise ValueError(f"Master event.tsv missing columns: {missing}\n"
                         f"Available: {list(master.columns)}")
    master['section_num']    = master['section_num'].astype(str)
    master['subsection_num'] = master['subsection_num'].astype(str)
    return master


def merge_eeg_meg(df: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    if 'section_num' not in df.columns:
        raise ValueError("'section_num' column missing in subject file")

    master_cols = [c for c in master.columns if c not in ('section_num', 'subsection_num')]
    df = df.copy()
    df['section_num'] = df['section_num'].astype(str)
    for col in master_cols:
        df[col] = None

    for section, subj_group in df.groupby('section_num', sort=False):
        master_group = master[master['subsection_num'] == section].reset_index(drop=True)
        if master_group.empty:
            print(f"    WARNING: no master rows for section {section}")
            continue
        n_subj, n_master = len(subj_group), len(master_group)
        if n_subj != n_master:
            print(f"    WARNING: section {section} — "
                  f"{n_subj} subject rows vs {n_master} master rows")
        for col in master_cols:
            for subj_idx, val in zip(subj_group.index, master_group[col].values):
                df.at[subj_idx, col] = val

    return df


def merge_fmri(df: pd.DataFrame, master: pd.DataFrame, run_num: str) -> pd.DataFrame:
    master_cols    = [c for c in master.columns if c not in ('section_num', 'subsection_num')]
    master_section = master[master['section_num'] == run_num].reset_index(drop=True)
    if master_section.empty:
        raise ValueError(f"No master rows for section_num={run_num}")

    n_subj, n_master = len(df), len(master_section)
    if n_subj != n_master:
        print(f"    WARNING: {n_subj} subject rows vs {n_master} master rows")

    df = df.copy()
    for col in master_cols:
        df[col] = master_section[col].values[:n_subj]
    return df


def reconstruct(file: Path, master: pd.DataFrame, modality: str, dry_run: bool) -> None:
    bids_id  = file.stem.split('_')[0]
    # Save into derivatives/text_events/<modality>/<subject>/
    modality_dir = 'fmri' if file.parent.name == 'func' else file.parent.name
    deriv_dir = file.parents[3] / 'derivatives' / 'text_events' / modality_dir / bids_id
    out_file  = deriv_dir / (file.stem + OUT_SUFFIX + '.tsv')

    print(f"  {bids_id}  {file.name}")
    print(f"         -> {out_file}")

    if dry_run:
        return

    deriv_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(file, sep='\t')

    if modality in ('eeg', 'meg'):
        merged = merge_eeg_meg(df, master)
    else:
        m = re.search(r'run-(\d+)', file.name)
        if not m:
            print(f"    [SKIP] could not parse run number from {file.name}")
            return
        merged = merge_fmri(df, master, m.group(1))

    n_matched = merged['surface'].notna().sum()
    print(f"    matched {n_matched}/{len(merged)} rows")
    merged.to_csv(out_file, sep='\t', index=False)
    print(f"    saved -> {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--modality', choices=['eeg', 'meg', 'fmri'], required=True,
                        help='Modality to process: eeg, meg, or fmri')
    parser.add_argument('--subject',
                        help='Process only this subject (e.g. sub-01)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print planned actions without writing any files')
    args = parser.parse_args()

    cfg    = MODALITY_CONFIG[args.modality]
    prefix = '[DRY RUN] ' if args.dry_run else ''

    print(f"{prefix}Modality   : {args.modality.upper()}")
    print(f"{prefix}Master     : {MASTER_FILE}")
    print(f"{prefix}Subjects   : {cfg['root']}\n")

    if not MASTER_FILE.exists():
        raise SystemExit(f"Master file not found: {MASTER_FILE}")

    master = load_master(MASTER_FILE)
    print(f"Master loaded: {len(master)} rows, columns: {list(master.columns)}\n")

    sub = args.subject or 'sub-*'
    if args.modality == 'fmri':
        pattern = f'{sub}/{cfg["subdir"]}/{sub}_task-{TASK}_run-*_events.tsv'
    else:
        pattern = f'{sub}/{cfg["subdir"]}/{sub}_task-{TASK}_events.tsv'

    files = sorted(cfg['root'].glob(pattern))

    if not files:
        raise SystemExit(f"No events.tsv files found under {cfg['root']}")

    print(f"Found {len(files)} file(s)\n")

    for f in files:
        reconstruct(f, master, args.modality, dry_run=args.dry_run)

    print(f"\n{'Dry run complete. Re-run without --dry-run to write files.' if args.dry_run else 'Done.'}")


if __name__ == '__main__':
    main()
