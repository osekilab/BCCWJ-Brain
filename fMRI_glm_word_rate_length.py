"""
GLM analysis: word_rate + word_length jointly (first + second level).

Both predictors are fit in the same GLM so their effects are estimated
while controlling for each other.

Design matrix per run:
    word_rate, word_length (HRF-conv, z-scored) + 6 motion params + constant

First level:  effect_size maps per subject
    → data/glm/word_rate_length/first_level/{sub}_word_rate_eff.nii
    → data/glm/word_rate_length/first_level/{sub}_word_length_eff.nii

Second level: one-sample t-test (group) for each contrast
    → data/glm/word_rate_length/group_word_rate_zstat.nii
    → data/glm/word_rate_length/group_word_rate_tstat.nii
    → data/glm/word_rate_length/group_word_length_zstat.nii
    → data/glm/word_rate_length/group_word_length_tstat.nii

"""

import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from natsort import natsorted
from scipy import stats

from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm import threshold_stats_img

# ── Paths ─────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DERIV_DIR   = "../ds007752/derivatives"
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

# ── FDR thresholding of group maps ───────────────────────────────────────────
n_subjects = max(len(first_level_maps[c]) for c in CONTRASTS)
df         = n_subjects - 1   # one-sample t-test df

print(f"\n── FDR thresholding (q < 0.05, df = {df}) ──")
print(f"{'contrast':<14} {'z_thresh':>10} {'t_thresh':>10}")
print("-" * 36)

for contrast in CONTRASTS:
    z_path = f"{OUT_2ND}/group_{contrast}_zstat.nii"
    t_path = f"{OUT_2ND}/group_{contrast}_tstat.nii"

    if not os.path.exists(z_path):
        print(f"  {contrast}: z-stat map not found, skipping.")
        continue

    # FDR threshold on the z-stat map (nilearn standard)
    _, z_thresh = threshold_stats_img(
        z_path, alpha=0.05, height_control="fdr", cluster_threshold=0
    )

    # Convert z threshold → equivalent t threshold for the same p-value
    p_thresh = stats.norm.sf(z_thresh)          # one-sided p at z_thresh
    t_thresh = stats.t.ppf(1.0 - p_thresh, df)  # equivalent t for df

    print(f"{contrast:<14} {z_thresh:>10.4f} {t_thresh:>10.4f}")

    # Save FDR-thresholded t-stat map
    t_img  = nib.load(t_path)
    t_data = t_img.get_fdata().copy()
    t_data[np.abs(t_data) < t_thresh] = 0
    nib.save(nib.Nifti1Image(t_data, t_img.affine, t_img.header),
             f"{OUT_2ND}/group_{contrast}_tstat_fdr05.nii")
    print(f"  Saved: {OUT_2ND}/group_{contrast}_tstat_fdr05.nii")

print("\nDone.")
