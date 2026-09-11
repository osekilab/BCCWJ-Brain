"""Grand-average for BCCWJ-EEG.

Loads every preprocessed evoked derivative
(derivatives/sub-*/eeg/*_ave.fif), computes the grand average across subjects,
and saves a butterfly + GFP + topomap figure.

The evoked derivatives are time-locked to word ONSET. (In dataset v1.0.0 the
presentation trigger was mistakenly recorded at the stimulus offset; this was
corrected to onset in v2.0.0, so this figure should be reproduced from the
v2.0.0 derivatives.)

Note on event timing
---------------------
The presentation trigger was recorded at each word's OFFSET. The RAW recordings
(.eeg/.vhdr/.vmrk) keep that offset trigger, exactly as recorded. The DERIVATIVES
are onset-corrected: the continuous *_raw.fif word markers and the evoked
*_ave.fif are time-locked to word ONSET (onset = offset - 500 ms). So you can
epoch a continuous derivative directly, no shift needed (see evoked_from_continuous
below). Only the raw files and events.tsv carry the offset.

Requires MNE-Python and matplotlib.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import mne

mne.set_log_level("WARNING")


def evoked_from_continuous(continuous_fif, tmin=-0.1, tmax=1.0, baseline=(-0.1, 0)):
    """Average a continuous derivative (*_raw.fif) into an onset-locked evoked.

    The continuous derivatives are onset-corrected, so their word markers already
    sit at the word ONSET -- just epoch and average, no shift needed. This
    reproduces the published *_ave.fif (which additionally applied manual artifact
    rejection, so its nave is slightly lower). Returns an mne.Evoked.

    Example
    -------
    >>> ev = evoked_from_continuous(
    ...     "derivatives/sub-01/eeg/sub-01_task-BCCWJreading_eeg_reref0.1-40-ica_raw.fif")
    >>> ev.plot(spatial_colors=True, gfp=True)
    """
    raw = mne.io.read_raw_fif(continuous_fif, preload=True, verbose=False)
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    # the word marker is the event code that occurs once per word (1642x)
    word_code = [c for c in event_id.values() if (events[:, 2] == c).sum() == 1642][0]
    words = events[events[:, 2] == word_code]
    epochs = mne.Epochs(raw, words, tmin=tmin, tmax=tmax, baseline=baseline,
                        preload=True, verbose=False)
    return epochs.average()

# Directory containing per-subject evoked derivatives.
# Override with the BCCWJ_DERIV environment variable if your layout differs.
main_dir = os.environ.get("BCCWJ_DERIV", "../ds007753/derivatives")

montage = mne.channels.make_standard_montage("easycap-M1")

# Collect all *_ave.fif files from sub-XX/eeg/ subdirectories
ave_files = sorted(
    os.path.join(main_dir, sub, "eeg", f)
    for sub in os.listdir(main_dir)
    if sub.startswith("sub-")
    for f in os.listdir(os.path.join(main_dir, sub, "eeg"))
    if f.endswith("_ave.fif")
)

evokeds = []
for evoked_file_path in ave_files:
    try:
        evoked = mne.read_evokeds(evoked_file_path, baseline=(-0.1, 0))[0]
        evoked.set_montage(montage, on_missing="ignore")
        evokeds.append(evoked)
        print(f"Loaded: {evoked_file_path}")
    except Exception as e:
        print(f"Error loading {evoked_file_path}: {e}")

if not evokeds:
    raise ValueError("No evoked data could be loaded. Please check the files and paths.")

print(f"\n{len(evokeds)} subjects loaded")

grand_avg = mne.grand_average(evokeds)
grand_avg.set_montage(montage, on_missing="ignore")

fig = grand_avg.plot_joint(
    times=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    title=f"BCCWJ-EEG grand-average ERP (onset-locked, N={len(evokeds)} subjects)",
    ts_args={"gfp": True},
    topomap_args={"cmap": "turbo"},
    show=False,
)
fig.savefig("grand_average_EEG.png", dpi=600, bbox_inches="tight")
print("Saved: grand_average_EEG.png")
