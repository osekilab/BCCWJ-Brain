"""
Inter-Subject Correlation (ISC) analysis — leave-one-out, section-wise.

Matches the reference implementation:
  For each section separately:
    1. Compute the LOO mean timeseries for each subject (mean of all others).
    2. Correlate subject i's timeseries with that mean, voxel-wise (Pearson r).
  Average the per-section ISC maps across sections → one ISC map per subject.
  Group result: median ISC across subjects.

Memory-efficient two-pass per section:
  Pass 1 — accumulate sum across subjects for that section.
  Pass 2 — reload each subject, compute LOO mean, correlate.
Peak RAM per section ≈ 3 × (n_TRs_sec × n_voxels × 4 bytes) ≈ 700 MB.

Outputs (data/isc/):
    isc_median.nii        — group median ISC (Pearson r)
    isc_tstat.nii         — one-sample t-stat (df = N-1)
    isc_zstat.nii         — equivalent z-stat
    isc_pmap.nii          — uncorrected two-sided p-value map
    isc_visualization.png — glass brain + axial slices of median ISC

Run:
    ./isc_analysis.py
"""

import gc
import io
import os
import glob
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.colorbar import ColorbarBase
from natsort import natsorted
from scipy import stats
from nilearn import datasets, plotting
from nilearn.maskers import NiftiMasker

# ── Config ────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DERIV_DIR   = "../ds007752/derivatives"
OUT_DIR     = os.path.join(_SCRIPT_DIR, "data", "isc")
os.makedirs(OUT_DIR, exist_ok=True)

SECTION_ORDER = ["A", "B", "C", "D"]  # Originally we assigned these labels in Sugimoto et al., 2024 these correspond to 1,2,3, and 4.
RUN_NUM       = {"A": 1, "B": 2, "C": 3, "D": 4}

# ── Visualization config ───────────────────────────────────────────────────────
VIZ_OUTPUT = os.path.join(OUT_DIR, "isc_visualization.png")
Z_COORDS   = [-10, 10, 20, 30, 40]
R_THRESH   = 0.25
R_VMAX     = 0.5
R_CMAP     = LinearSegmentedColormap.from_list("isc_r", ["black", "red", "yellow"], N=256)

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_section(sub, sec_letter, masker):
    """Load one section for one subject → (n_TRs, n_voxels) float32."""
    run = RUN_NUM[sec_letter]
    nii = os.path.join(DERIV_DIR, sub, "func",
                       f"{sub}_task-BCCWJreading_run-{run}_preproc_bold.nii.gz")
    return masker.transform(nii).astype(np.float32)

def pearson_r_cols(A, B):
    """Vectorised Pearson r column-wise (no pre-z-scoring needed)."""
    A = A - A.mean(0); B = B - B.mean(0)
    num   = (A * B).sum(0)
    denom = np.sqrt((A * A).sum(0) * (B * B).sum(0)) + 1e-12
    return (num / denom).astype(np.float32)

def save_map(arr, path, masker):
    masker.inverse_transform(arr.astype(np.float32)).to_filename(path)
    print(f"  Saved: {path}")

def glass_to_img(path, cmap, threshold, vmax):
    display = plotting.plot_glass_brain(
        path, display_mode="lzr", threshold=threshold, vmax=vmax,
        cmap=cmap, colorbar=False, plot_abs=False, black_bg=False, alpha=0.8,
    )
    buf = io.BytesIO()
    display.savefig(buf, dpi=150)
    display.close()
    buf.seek(0)
    return plt.imread(buf)

# ── Discover subjects ─────────────────────────────────────────────────────────
sub_dirs = natsorted(glob.glob(os.path.join(DERIV_DIR, "sub-*")))
subjects = [os.path.basename(d) for d in sub_dirs]
N        = len(subjects)
print(f"Found {N} subjects")

# ── Brain mask (Yeo 2011 liberal mask, resampled to 2mm functional space) ─────
MASK_FILE = os.path.join(_SCRIPT_DIR, "data", "isc", "yeo_liberal_mask_2mm.nii")

if not os.path.exists(MASK_FILE):
    print("Yeo mask not found — downloading and building ...")
    yeo        = datasets.fetch_atlas_yeo_2011()
    atlas_img  = nib.load(yeo.thick_17)
    mask_data  = (atlas_img.get_fdata(dtype=np.float32) > 0).astype(np.uint8)
    mask_img   = nib.Nifti1Image(mask_data, atlas_img.affine, atlas_img.header)
    mask_img.set_data_dtype(np.uint8)
    nib.save(mask_img, MASK_FILE)
    print(f"  Saved: {MASK_FILE}")

masker = NiftiMasker(mask_img=MASK_FILE, standardize=False,
                     detrend=False, t_r=2.0).fit()
n_vox     = int(masker.mask_img_.get_fdata().sum())
print(f"Brain mask: {n_vox} voxels  (Yeo 2011 liberal mask, 2mm)")

# ── Section-wise ISC (two-pass per section) ───────────────────────────────────
# isc_maps[i] = mean ISC across sections for subject i
isc_maps = np.zeros((N, n_vox), dtype=np.float32)

for sec in SECTION_ORDER:
    print(f"\n── Section {sec} ──────────────────────────────────────")

    # Pass 1: accumulate sum across subjects for this section
    print("  Pass 1: accumulating sum ...")
    sum_sec = None
    for sub in subjects:
        data = load_section(sub, sec, masker)
        if sum_sec is None:
            sum_sec = data.copy()
        else:
            sum_sec += data
        del data; gc.collect()

    # Pass 2: LOO ISC per subject for this section
    print("  Pass 2: computing ISC ...")
    for i, sub in enumerate(subjects):
        data = load_section(sub, sec, masker)
        loo  = (sum_sec - data) / (N - 1)        # LOO mean of all others
        isc_maps[i] += pearson_r_cols(data, loo)  # accumulate across sections
        del data, loo; gc.collect()
        print(f"    {subjects[i]}: section {sec} ISC mean = {isc_maps[i].mean():.4f}")

    del sum_sec; gc.collect()

# Average over sections (matches reference: mean of section-wise correlations)
isc_maps /= len(SECTION_ORDER)

# ── Group statistics ──────────────────────────────────────────────────────────
isc_median = np.median(isc_maps, axis=0)

# Fisher z-transform → one-sample t-test (H0: ISC = 0)
z_maps         = np.arctanh(np.clip(isc_maps, -0.9999, 0.9999))
t_vals, p_vals = stats.ttest_1samp(z_maps, 0, axis=0)
z_stat         = stats.norm.ppf(1 - p_vals / 2) * np.sign(t_vals)

print(f"\nGroup median ISC: mean={isc_median.mean():.4f}, max={isc_median.max():.4f}")
print(f"T-stat: max={t_vals.max():.2f},  voxels t>3: {(t_vals > 3).sum()}")

# ── Save outputs ──────────────────────────────────────────────────────────────
print("\nSaving ...")
save_map(isc_median, f"{OUT_DIR}/isc_median.nii", masker)
save_map(t_vals,     f"{OUT_DIR}/isc_tstat.nii",  masker)
save_map(z_stat,     f"{OUT_DIR}/isc_zstat.nii",  masker)
save_map(p_vals,     f"{OUT_DIR}/isc_pmap.nii",   masker)

print(f"\n── ISC Summary (df = {N-1}) ──────────────────────────")
print(f"  Subjects         : {N}")
print(f"  Sections         : {len(SECTION_ORDER)}")
print(f"  Brain voxels     : {n_vox}")
print(f"  Median ISC mean  : {isc_median.mean():.4f}")
print(f"  Median ISC max   : {isc_median.max():.4f}")
print(f"  Voxels ISC > 0.1 : {(isc_median > 0.1).sum()}")
print(f"  Voxels ISC > 0.2 : {(isc_median > 0.2).sum()}")
print(f"  Voxels t > 3.0   : {(t_vals > 3.0).sum()}")
print(f"\nOutputs in: {OUT_DIR}/")

# ── Visualization ─────────────────────────────────────────────────────────────
print("\nRendering visualization ...")

MEDIAN_PATH = f"{OUT_DIR}/isc_median.nii"
ROWS = [
    ("ISC\n(r)", MEDIAN_PATH, R_CMAP, R_THRESH, R_VMAX, "ISC  (Pearson r)"),
]

n_slices = len(Z_COORDS)
w_ratios = [0.45, 3.5] + [1.3] * n_slices + [0.3]

fig = plt.figure(figsize=(26, 4.5 * len(ROWS)), facecolor="white")
gs  = gridspec.GridSpec(
    len(ROWS), 2 + n_slices + 1,
    figure=fig,
    width_ratios=w_ratios,
    hspace=0.1, wspace=0.04,
)

for row, (row_label, map_path, cmap, threshold, vmax, cbar_label) in enumerate(ROWS):
    ax_lbl = fig.add_subplot(gs[row, 0])
    ax_lbl.text(0.5, 0.5, row_label, ha="center", va="center",
                fontsize=12, fontweight="bold", color="black", rotation=90,
                transform=ax_lbl.transAxes)
    ax_lbl.axis("off")

    ax_glass = fig.add_subplot(gs[row, 1])
    ax_glass.imshow(glass_to_img(map_path, cmap, threshold, vmax),
                    interpolation="bilinear")
    ax_glass.axis("off")
    if row == 0:
        ax_glass.set_title("Glass brain  (L | axial | R)",
                            fontsize=10, pad=4, color="black")

    for i, z in enumerate(Z_COORDS):
        ax_sl = fig.add_subplot(gs[row, 2 + i])
        plotting.plot_stat_map(
            map_path, display_mode="z", cut_coords=[z],
            threshold=threshold, vmax=vmax, cmap=cmap,
            colorbar=False, draw_cross=False, annotate=False,
            black_bg=False, axes=ax_sl,
        )
        if row == 0:
            ax_sl.set_title(f"z = {z}", fontsize=9, pad=3, color="black")

    ax_cb = fig.add_subplot(gs[row, -1])
    norm  = Normalize(vmin=threshold, vmax=vmax)
    cb    = ColorbarBase(ax_cb, cmap=cmap, norm=norm, orientation="vertical")
    cb.set_ticks([threshold, (threshold + vmax) / 2, vmax])
    cb.set_ticklabels([f"{threshold:.2f}", f"{(threshold + vmax) / 2:.2f}", f"{vmax:.1f}"])
    cb.ax.tick_params(labelsize=8, colors="black")
    cb.outline.set_edgecolor("black")
    cb.set_label(cbar_label, rotation=270, labelpad=18, fontsize=8, color="black")

fig.suptitle("Inter-Subject Correlation (ISC)",
             fontsize=14, fontweight="bold", color="black", y=1.02)
plt.savefig(VIZ_OUTPUT, dpi=200, bbox_inches="tight",
            facecolor="white", pad_inches=0.15)
plt.close()
print(f"  Saved: {VIZ_OUTPUT}")
print("Done.")
