import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne

mne.set_log_level("WARNING")

# Directory containing per-subject evoked files
main_dir = '../ds007763/derivatives'

# Collect all *_ave.fif files from sub-XX/meg/ subdirectories
ave_files = sorted([
    os.path.join(main_dir, sub, 'meg', f)
    for sub in os.listdir(main_dir)
    if sub.startswith('sub-')
    for f in os.listdir(os.path.join(main_dir, sub, 'meg'))
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

median_evoked = mne.combine_evoked(evokeds, weights=[1 / len(evokeds)] * len(evokeds))

fig = plt.figure(figsize=(22, 12))
median_evoked.plot(spatial_colors=True, gfp=True, show=False)

median_evoked.plot_joint(
    times=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    ts_args={'gfp': True},
    show=False,
)

plt.savefig('grand_average_BCCWJ_MEG.png', dpi=600, bbox_inches='tight')
plt.show()
