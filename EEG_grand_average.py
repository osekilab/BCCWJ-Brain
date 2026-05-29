import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne

mne.set_log_level("WARNING")

# Directory containing per-subject evoked files
main_dir = '../ds007753/derivatives'

# Collect all *_ave.fif files from sub-XX/eeg/ subdirectories
ave_files = sorted([
    os.path.join(main_dir, sub, 'eeg', f)
    for sub in os.listdir(main_dir)
    if sub.startswith('sub-')
    for f in os.listdir(os.path.join(main_dir, sub, 'eeg'))
    if f.endswith('_ave.fif')
])

evokeds = []

for evoked_file_path in ave_files:
    try:
        evoked = mne.read_evokeds(evoked_file_path, baseline=(-0.1, 0))
        evokeds.append(evoked[0])
        print(f"Loaded: {evoked_file_path}")
    except Exception as e:
        print(f"Error loading {evoked_file_path}: {e}")

if not evokeds:
    raise ValueError("No evoked data could be loaded. Please check the files and conditions.")

print(f"\n{len(evokeds)} subjects loaded")

grand_avg = mne.combine_evoked(evokeds, weights=[1 / len(evokeds)] * len(evokeds))

fig = plt.figure(figsize=(22, 12))
grand_avg.plot(spatial_colors=True, gfp=True, show=False)

grand_avg.plot_joint(
    times=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    ts_args={'gfp': True},
    topomap_args={'cmap': 'turbo'},
    show=False,
)

plt.savefig('grand_average_EEG.png', dpi=600, bbox_inches='tight')
plt.show()
