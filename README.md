# BCCWJ-Brain

Analysis pipelines for EEG, MEG, and fMRI data collected during naturalistic reading of the Balanced Corpus of Contemporary Written Japanese (BCCWJ).

---

## Download

Download the datasets from OpenNeuro.

| Modality | Dataset |
|---|---|
| fMRI | [ds007752](https://openneuro.org/datasets/ds007752) |
| EEG  | [ds007753](https://openneuro.org/datasets/ds007753) |
| MEG  | [ds007763](https://openneuro.org/datasets/ds007763) |

---

## Dependencies

See [environment.yml](environment.yml) for the full list.

```bash
conda env create -f environment.yml
conda activate bccwj-brain
```

---

## EEG / MEG

| Script | Description |
|---|---|
| [EEG_grand_average.py](EEG_grand_average.py) | Loads per-subject evoked `.fif` files (baseline −100–0 ms) and computes the grand-average EEG response. Produces a butterfly + GFP plot and a joint topographic plot. |
| [MEG_grand_average.py](MEG_grand_average.py) | Same workflow for MEG data. |

**Expected input structure:**

```
../ds007753/derivatives/sub-XX/eeg/*_ave.fif   # EEG
../ds007763/derivatives/sub-XX/meg/*_ave.fif   # MEG
```

**Outputs:** `grand_average_EEG.png`, `grand_average_BCCWJ_MEG.png`

---

## fMRI

### [fMRI_glm_word_rate_length.py](fMRI_glm_word_rate_length.py)

First- and second-level GLM estimating the effects of **word rate** and **word length** on BOLD signal, fit jointly so each predictor controls for the other.

- Design matrix per run: `word_rate | word_length | 6 motion params | constant` (all z-scored; predictors are pre-convolved with HRF)
- First level: AR(1) noise model, one effect-size map per subject per contrast
- Second level: one-sample t-test across subjects
- Thresholding: FWE correction (Bonferroni, p < 0.05), cluster extent k > 50 voxels
- Visualization: glass brain (L | axial | R) + 6 axial slices; warm colormap (red→yellow) for word rate, cold (blue→cyan) for word length
- Cluster tables: automatically exported as LaTeX (`cluster_tables.tex`) with Harvard-Oxford atlas region labels

**Required inputs:**

```
../ds007752/derivatives/sub-*/func/
    {sub}_task-BCCWJreading_run-{1..4}_preproc_bold.nii.gz
    {sub}_task-BCCWJreading_run-{1..4}_preproc_rp.txt   # SPM realignment parameters (6 columns)
fMRIpredictors.csv   # columns: section_number, word_rate, word_length (HRF-convolved)
```

**Outputs (`data/glm/word_rate_length/`):**

```
first_level/{sub}_word_rate_eff.nii
first_level/{sub}_word_length_eff.nii
group_word_rate_zstat.nii
group_word_rate_tstat.nii
group_word_rate_tstat_fwe05.nii        # FWE-thresholded
group_word_length_zstat.nii
group_word_length_tstat.nii
group_word_length_tstat_fwe05.nii      # FWE-thresholded
glm_word_rate_length_visualization.png
cluster_tables.tex                     # LaTeX cluster tables with region labels
```

---

### [fMRI_isc_analysis.py](fMRI_isc_analysis.py)

Leave-one-out **Inter-Subject Correlation (ISC)** analysis, computed section-wise and averaged across sections. Matches the reference implementation in Nastase et al. (2019).

- Brain mask: Yeo 2011 liberal cortical mask at 2 mm (auto-downloaded if absent at `data/isc/yeo_liberal_mask_2mm.nii`)
- Memory-efficient two-pass algorithm per section (~700 MB peak RAM)
- Group statistic: median ISC; Fisher z-transform → one-sample t-test
- Thresholding: FWE correction (Bonferroni, p < 0.05)
- Visualization: two-row figure — median ISC (Pearson r) and FWE-thresholded z-stat map; glass brain (L | axial | R) + 5 axial slices

**Required inputs:**

```
../ds007752/derivatives/sub-*/func/
    {sub}_task-BCCWJreading_run-{1..4}_preproc_bold.nii.gz
```

**Outputs (`data/isc/`):**

```
isc_median.nii           # group median Pearson r
isc_tstat.nii            # one-sample t-stat (df = N−1)
isc_zstat.nii            # equivalent z-stat
isc_pmap.nii             # uncorrected two-sided p-values
isc_zstat_fwe05.nii      # FWE-thresholded z-stat map
isc_visualization.png    # glass brain + axial slice figure (r and FWE z-stat)
```

---

## Text–Imaging Event Merge

### [bccwj-textmerge.py](bccwj-textmerge.py)

Merges text annotations from the BCCWJ master event file into per-subject BIDS event files. Adds `surface` (word form), `word_length`, `sent_id`, `bunsetsu_pos`, and `count_ave_log` columns by sequentially pairing rows within each section. Source files are never modified; outputs get a `_with_surface` suffix.

**Join strategy:**
- EEG/MEG: match `section_num` (subject) → `subsection_num` (master), pair rows sequentially
- fMRI: match run-N (filename) → `section_num` (master), pair rows sequentially

**Required inputs:**

```
event.tsv                                                        # master file
../ds007753/sub-*/eeg/sub-*_task-BCCWJreading_events.tsv         # EEG
../ds007763/sub-*/meg/sub-*_task-BCCWJreading_events.tsv         # MEG
../ds007752/sub-*/func/sub-*_task-BCCWJreading_run-*_events.tsv  # fMRI
```

**Outputs (written to `derivatives/text_events/` within each dataset):**

```
# EEG
../ds007753/derivatives/text_events/{sub}/eeg/{sub}_task-BCCWJreading_events_with_surface.tsv

# MEG
../ds007763/derivatives/text_events/{sub}/meg/{sub}_task-BCCWJreading_events_with_surface.tsv

# fMRI
../ds007752/derivatives/text_events/{sub}/func/{sub}_task-BCCWJreading_run-*_events_with_surface.tsv
```

**Usage:**

```bash
python bccwj-textmerge.py --modality eeg|meg|fmri             # all subjects
python bccwj-textmerge.py --modality eeg --subject sub-01     # single subject
python bccwj-textmerge.py --modality fmri --dry-run           # preview only
```
