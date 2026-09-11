#!/usr/bin/env python3
"""Onset-correct the continuous EEG derivatives (BCCWJ-EEG v2.0.0).

The recording trigger was sent at each word's OFFSET. The RAW recordings
(.eeg/.vhdr/.vmrk) keep that offset trigger as recorded. This script shifts the
word markers in the continuous derivative files (derivatives/sub-*/eeg/*_raw.fif)
by -0.5 s so they sit at the word ONSET (onset = offset - 500 ms), matching the
onset-locked evoked (*_ave.fif). Users can then epoch a continuous derivative
directly (no manual shift needed).

"""
import os
import sys
from pathlib import Path

import numpy as np
import mne

mne.set_log_level("ERROR")

ROOT = Path(os.environ.get("BCCWJ_EEG", "../ds007753")).resolve()
SHIFT = 0.500
N_WORDS = 1642


def word_onsets(raw):
    descr = np.asarray(raw.annotations.description)
    counts = {d: int((descr == d).sum()) for d in set(descr)}
    words = [d for d, n in counts.items() if n == N_WORDS]
    if len(words) != 1:
        raise ValueError(f"no unique {N_WORDS}-count word marker: {counts}")
    desc = words[0]
    return desc, (descr == desc)


def process(fif: Path, dry: bool) -> str:
    sub = fif.name.split("_")[0]
    raw_vhdr = mne.io.read_raw_brainvision(
        ROOT / sub / "eeg" / f"{sub}_task-BCCWJreading_eeg.vhdr", verbose=False)
    t_offset = np.sort(raw_vhdr.annotations.onset[word_onsets(raw_vhdr)[1]])

    raw = mne.io.read_raw_fif(fif, preload=False)
    ann = raw.annotations
    desc, mask = word_onsets(raw)
    first = np.sort(ann.onset[mask])[0]
    off0, on0 = t_offset[0], t_offset[0] - SHIFT
    if abs(first - on0) < 0.01:
        return f"[skip] {fif.name}: already onset-corrected"
    if abs(first - off0) >= 0.01:
        return f"[skip] {fif.name}: unexpected marker time {first:.3f} (raw offset {off0:.3f}); left unchanged"
    if dry:
        return f"[dry ] {fif.name}: would shift {int(mask.sum())} {desc!r} by -{SHIFT}s"
    new = ann.onset.copy()
    new[mask] -= SHIFT
    raw.set_annotations(mne.Annotations(new, ann.duration, ann.description, ann.orig_time))
    tmp = fif.with_name(fif.stem + ".tmp_raw.fif")
    raw.save(tmp, overwrite=True)
    os.replace(tmp, fif)
    return f"[done] {fif.name}: shifted {int(mask.sum())} {desc!r} to onset (offset -0.5 s)"


def main() -> None:
    dry = "--dry-run" in sys.argv
    files = sorted((ROOT / "derivatives").glob("sub-*/eeg/*_raw.fif"))
    if not files:
        raise SystemExit(f"No continuous derivatives under {ROOT}. Set BCCWJ_EEG or edit ROOT.")
    print(f"{len(files)} continuous derivative files\n")
    for f in files:
        print(process(f, dry), flush=True)


if __name__ == "__main__":
    main()
