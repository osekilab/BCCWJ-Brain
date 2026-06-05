import io
import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.colorbar import ColorbarBase
from natsort import natsorted
from scipy import stats
from nilearn import plotting

from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm import threshold_stats_img

# ── Paths ─────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
#DERIV_DIR   = "../ds007752/derivatives"
DERIV_DIR = "/Users/osekilab1/BCCWJ-fMRI/derivatives"
PRED_CSV    = os.path.join(_SCRIPT_DIR, "fMRIpredictors.csv")  #predictors that are already convolved with HRF.
OUT_1ST     = os.path.join(_SCRIPT_DIR, "data", "glm", "word_rate_length", "first_level")
OUT_2ND     = os.path.join(_SCRIPT_DIR, "data", "glm", "word_rate_length")

os.makedirs(OUT_1ST, exist_ok=True)
os.makedirs(OUT_2ND, exist_ok=True)

TR        = 2.0
SECTIONS  = ["A", "B", "C", "D"]    
SEC_NUM   = {"A": 1, "B": 2, "C": 3, "D": 4}
RUN_NUM   = {"A": 1, "B": 2, "C": 3, "D": 4}   # run-1=A, run-2=B, run-3=C, run-4=D
RP_COLS   = ["rp_dx", "rp_dy", "rp_dz", "rp_rx", "rp_ry", "rp_rz"]
CONTRASTS = ["word_rate", "word_length"]

# ── Load predictor file ────────────────────────────────────────────────────────
pred_all = pd.read_csv(PRED_CSV)


def make_design_matrix(section_letter, rp_file):
    """Return z-scored [word_rate | word_length | 6 motion params | constant]."""
    sec    = SEC_NUM[section_letter]
    sec_df = pred_all[pred_all["section_number"] == sec]

    wr = sec_df["word_rate"].values.copy()
    wl = sec_df["word_length"].values.copy()
    rp = np.loadtxt(rp_file)   # (n_TR, 6)

    assert len(wr) == len(rp), (
        f"Length mismatch: predictors={len(wr)}, rp={len(rp)} for section {section_letter}"
    )

    def zscore(x):
        return (x - x.mean()) / (x.std() + 1e-12)

    rp_z = (rp - rp.mean(axis=0)) / (rp.std(axis=0) + 1e-12)

    dm = pd.DataFrame(rp_z, columns=RP_COLS)
    dm.insert(0, "word_length", zscore(wl))
    dm.insert(0, "word_rate",   zscore(wr))
    dm["constant"] = 1.0
    return dm


# ── Discover subjects ─────────────────────────────────────────────────────────
subject_dirs = natsorted(glob.glob(os.path.join(DERIV_DIR, "sub-*")))
subjects     = [os.path.basename(d) for d in subject_dirs]
print(f"Found {len(subjects)} subjects: {subjects}")

# ── First-level GLM ───────────────────────────────────────────────────────────
first_level_maps = {c: [] for c in CONTRASTS}

for sub in subjects:
    func_dir = os.path.join(DERIV_DIR, sub, "func")

    nii_files = []
    dm_list   = []
    ok        = True

    for sec in SECTIONS:
        run = RUN_NUM[sec]
        nii = os.path.join(func_dir, f"{sub}_task-BCCWJreading_run-{run}_preproc_bold.nii.gz")
        rp  = os.path.join(func_dir, f"{sub}_task-BCCWJreading_run-{run}_preproc_rp.txt")

        if not os.path.exists(nii):
            print(f"  [{sub}] missing NII for section {sec} (run-{run}) — skipping subject")
            ok = False
            break
        if not os.path.exists(rp):
            print(f"  [{sub}] missing RP for section {sec} (run-{run}) — skipping subject")
            ok = False
            break

        nii_files.append(nii)
        dm_list.append(make_design_matrix(sec, rp))

    if not ok:
        continue

    print(f"[{sub}] fitting GLM ...")

    glm = FirstLevelModel(
        t_r            = TR,
        hrf_model      = None,   # predictors are already HRF-convolved
        noise_model    = "ar1",
        standardize    = True,
        smoothing_fwhm = None,
        n_jobs         = 1,
        verbose        = 0,
    )

    try:
        glm.fit(nii_files, design_matrices=dm_list)

        for contrast in CONTRASTS:
            eff_map  = glm.compute_contrast(contrast_def=contrast, output_type="effect_size")
            out_path = f"{OUT_1ST}/{sub}_{contrast}_eff.nii"
            eff_map.to_filename(out_path)
            first_level_maps[contrast].append(out_path)

        print(f"  saved: word_rate + word_length maps")

    except Exception as e:
        print(f"  [{sub}] ERROR: {e}")

print(f"\nFirst level done:")
for c in CONTRASTS:
    print(f"  {c}: {len(first_level_maps[c])} maps")

# ── Second-level GLM (one-sample t-test, per contrast) ───────────────────────
for contrast in CONTRASTS:
    maps = first_level_maps[contrast]
    if len(maps) < 2:
        print(f"\nNot enough maps for group analysis of {contrast}.")
        continue

    print(f"\nGroup analysis: {contrast}  ({len(maps)} subjects) ...")

    design_matrix_2nd = pd.DataFrame(np.ones(len(maps)), columns=["intercept"])

    second_level = SecondLevelModel(smoothing_fwhm=None, n_jobs=1)
    second_level.fit(maps, design_matrix=design_matrix_2nd)

    z_map = second_level.compute_contrast(
        second_level_contrast="intercept", output_type="z_score"
    )
    z_path = f"{OUT_2ND}/group_{contrast}_zstat.nii"
    z_map.to_filename(z_path)
    print(f"  Saved: {z_path}")

    t_map = second_level.compute_contrast(
        second_level_contrast="intercept", output_type="stat"
    )
    t_path = f"{OUT_2ND}/group_{contrast}_tstat.nii"
    t_map.to_filename(t_path)
    print(f"  Saved: {t_path}")

# ── FWE (Bonferroni) thresholding of group maps ───────────────────────────────
n_subjects = max(len(first_level_maps[c]) for c in CONTRASTS)
df         = n_subjects - 1   # one-sample t-test df

print(f"\n── FWE Bonferroni thresholding (p < 0.05, df = {df}) ──")
print(f"{'contrast':<14} {'z_thresh':>10} {'t_thresh':>10}")
print("-" * 36)

for contrast in CONTRASTS:
    z_path = f"{OUT_2ND}/group_{contrast}_zstat.nii"
    t_path = f"{OUT_2ND}/group_{contrast}_tstat.nii"

    if not os.path.exists(z_path):
        print(f"  {contrast}: z-stat map not found, skipping.")
        continue

    # FWE Bonferroni threshold on the z-stat map
    _, z_thresh = threshold_stats_img(
        z_path, alpha=0.05, height_control="bonferroni", cluster_threshold=50
    )

    # Convert z threshold → equivalent t threshold for the same p-value
    p_thresh = stats.norm.sf(z_thresh)          # one-sided p at z_thresh
    t_thresh = stats.t.ppf(1.0 - p_thresh, df)  # equivalent t for df

    print(f"{contrast:<14} {z_thresh:>10.4f} {t_thresh:>10.4f}")

    # Save FWE-thresholded t-stat map
    t_img  = nib.load(t_path)
    t_data = t_img.get_fdata().copy()
    t_data[np.abs(t_data) < t_thresh] = 0
    nib.save(nib.Nifti1Image(t_data, t_img.affine, t_img.header),
             f"{OUT_2ND}/group_{contrast}_tstat_fwe05.nii")
    print(f"  Saved: {OUT_2ND}/group_{contrast}_tstat_fdr05.nii")

print("\nDone.")

# ── Visualization ─────────────────────────────────────────────────────────────
print("\nRendering visualization ...")

Z_COORDS = [-18, -6, 6, 16, 28, 48]
VIZ_OUT  = os.path.join(OUT_2ND, "glm_word_rate_length_visualization.png")

CONTRAST_VIZ = {
    "word_rate":   {"label": "word\nrate",   "cmap": LinearSegmentedColormap.from_list("wr", ["red",  "yellow"], N=256)},
    "word_length": {"label": "word\nlength", "cmap": LinearSegmentedColormap.from_list("wl", ["blue", "cyan"],   N=256)},
}

def get_tmap_range(path):
    d  = nib.load(path).get_fdata()
    nz = np.abs(d[d != 0])
    return (float(nz.min()), float(nz.max())) if len(nz) else (2.0, 8.0)

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

ROWS = []
for contrast, cfg in CONTRAST_VIZ.items():
    map_path        = f"{OUT_2ND}/group_{contrast}_tstat_fwe05.nii"
    t_thresh, t_max = get_tmap_range(map_path)
    ROWS.append((cfg["label"], map_path, cfg["cmap"], t_thresh, t_max))

n_slices = len(Z_COORDS)
w_ratios = [0.5, 3.2] + [1.2] * n_slices + [0.35]

fig = plt.figure(figsize=(28, 4.5 * len(ROWS)), facecolor="white")
gs  = gridspec.GridSpec(
    len(ROWS), 2 + n_slices + 1,
    figure=fig, width_ratios=w_ratios,
    hspace=0.15, wspace=0.04,
)

for row, (row_label, map_path, cmap, t_thresh, t_vmax) in enumerate(ROWS):
    ax_lbl = fig.add_subplot(gs[row, 0])
    ax_lbl.text(0.5, 0.5, row_label, ha="center", va="center",
                fontsize=13, fontweight="bold", color="black",
                transform=ax_lbl.transAxes)
    ax_lbl.axis("off")

    ax_glass = fig.add_subplot(gs[row, 1])
    ax_glass.imshow(glass_to_img(map_path, cmap, t_thresh, t_vmax), interpolation="bilinear")
    ax_glass.axis("off")
    ax_glass.set_title(
        f"FWE p<0.05 (Bonferroni)  |t| > {t_thresh:.2f},  k > 50   (L | axial | R)",
        fontsize=9, pad=4, color="black",
    )

    for i, z in enumerate(Z_COORDS):
        ax_sl = fig.add_subplot(gs[row, 2 + i])
        plotting.plot_stat_map(
            map_path, display_mode="z", cut_coords=[z],
            threshold=t_thresh, vmax=t_vmax, cmap=cmap,
            colorbar=False, draw_cross=False, annotate=False,
            black_bg=False, axes=ax_sl,
        )
        ax_sl.set_title(f"z={z}", fontsize=9, pad=3, color="black")

    ax_cb = fig.add_subplot(gs[row, -1])
    norm  = Normalize(vmin=t_thresh, vmax=t_vmax)
    cb    = ColorbarBase(ax_cb, cmap=cmap, norm=norm, orientation="vertical")
    cb.set_ticks([t_thresh, t_vmax])
    cb.set_ticklabels([f"{t_thresh:.2f}", f"{t_vmax:.1f}"])
    cb.ax.tick_params(labelsize=8, colors="black")
    cb.outline.set_edgecolor("black")
    cb.set_label("t-stat\n(FWE p<0.05, k>50)", rotation=270, labelpad=22, fontsize=8, color="black")

fig.suptitle("GLM: Word Rate & Word Length (FWE p<0.05, Bonferroni)",
             fontsize=14, fontweight="bold", color="black", y=1.02)
plt.savefig(VIZ_OUT, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.15)
plt.close()
print(f"  Saved: {VIZ_OUT}")

# ── LaTeX cluster tables ───────────────────────────────────────────────────────
print("\nGenerating LaTeX cluster tables ...")

from nilearn.reporting import get_clusters_table
from nilearn import datasets as nl_datasets

ho        = nl_datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
ho_img    = ho.maps
ho_data   = ho_img.get_fdata()
ho_labels = ho.labels

def atlas_label(x, y, z):
    vox = np.round(np.linalg.inv(ho_img.affine) @ [x, y, z, 1])[:3].astype(int)
    vox = np.clip(vox, 0, np.array(ho_data.shape) - 1)
    idx = int(ho_data[vox[0], vox[1], vox[2]])
    side = "L." if x < 0 else "R."
    label = ho_labels[idx - 1] if idx > 0 else "unlabeled"
    return f"{side}{label}"

VOXEL_VOL = 8   # mm³ per voxel (2 mm isotropic)
FWE_T     = 6.53
TABLE_DIR = OUT_2ND

CAPTIONS = {
    "word_rate":   r"Word rate — FWE ($p<0.05$, Bonferroni, $k>50$ voxels, $t>6.53$).",
    "word_length": r"Word length — FWE ($p<0.05$, Bonferroni, $k>50$ voxels, $t>6.53$).",
}
LABELS = {
    "word_rate":   "tab:clusters_word_rate",
    "word_length": "tab:clusters_word_length",
}

all_tex = []
for contrast in CONTRASTS:
    map_path = f"{OUT_2ND}/group_{contrast}_tstat_fwe05.nii"
    img      = nib.load(map_path)
    table    = get_clusters_table(img, stat_threshold=FWE_T, cluster_threshold=50)
    main     = table[~table["Cluster ID"].astype(str).str.contains("[a-z]")]

    rows = []
    for i, (_, row) in enumerate(main.iterrows(), start=1):
        vox    = int(row["Cluster Size (mm3)"] / VOXEL_VOL)
        peak   = f"{row['Peak Stat']:.2f}"
        x, y, z = row["X"], row["Y"], row["Z"]
        pval   = stats.t.sf(float(row["Peak Stat"]), df) * 2
        p_str  = r"$< .001$" if pval < 0.001 else f"${pval:.3f}$"
        region = atlas_label(x, y, z)
        mni    = f"({int(x)}, {int(y)}, {int(z)})"
        rows.append(f"        {i} & {vox} & {peak} & {p_str} & {mni} & {region} \\\\")

    body = "\n".join(rows)
    tex = (
        r"\begin{table}[h]\centering" "\n"
        r"\resizebox{16cm}{!}{" "\n"
        r"\begin{tabular}{rrrcl l}" "\n"
        r"\toprule" "\n"
        r"Cluster & Voxels & Peak $t$ & $p$ & MNI (x,y,z) & Region \\" "\n"
        r"\midrule" "\n"
        f"{body}\n"
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"}" "\n"
        f"\\caption{{{CAPTIONS[contrast]}}}\\label{{{LABELS[contrast]}}}\n"
        r"\end{table}"
    )
    all_tex.append(tex)
    print(f"\n{'─'*60}\n{tex}")

tex_path = os.path.join(OUT_2ND, "cluster_tables.tex")
with open(tex_path, "w") as f:
    f.write("\n\n".join(all_tex) + "\n")
print(f"\nSaved: {tex_path}")
