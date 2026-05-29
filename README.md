# BCCWJ-Brain

Data validation pipelines for EEG, MEG, and fMRI data collected during naturalistic reading of the Balanced Corpus of Contemporary Written Japanese (BCCWJ).
---

## Download
Download the dataset from Openneuro.

- BCCWJ-fMRI: [https://openneuro.org/datasets/ds007752](https://openneuro.org/datasets/ds007752)
- BCCWJ-MEG: [https://openneuro.org/datasets/ds007763](https://openneuro.org/datasets/ds007763)
- BCCWJ-EEG:[https://openneuro.org/datasets/ds007753](https://openneuro.org/datasets/ds007753)


## Dependencies

See [environment.yml](environment.yml) for the full list.

Create and activate the conda environment with:

```bash
conda env create -f environment.yml
conda activate bccwj-brain
```

### EEG / MEG

| Script | Description |
|---|---|
| [EEG_grand_average.py](EEG_grand_average.py) | Loads per-subject evoked `.fif` files (baseline −100–0 ms) and computes the grand-average EEG response across participants. Produces a butterfly + GFP plot and a joint topographic plot. |
| [MEG_grand_average.py](MEG_grand_average.py) | Same workflow for MEG data. |

Both scripts expect BIDS-like derivatives directories:

```
../BCCWJ-Brain/BCCWJ-EEG/derivatives/sub-XX/eeg/*_ave.fif
../BCCWJ-Brain/BCCWJ-MEG/derivatives/sub-XX/meg/*_ave.fif
```

Outputs: `grand_average_EEG.png`, `grand_average_BCCWJ_MEG.png`

---

### fMRI

#### [fMRI_glm_word_rate_length.py](fMRI_glm_word_rate_length.py)

First- and second-level GLM estimating the effects of **word rate** and **word length** on BOLD signal while controlling for each other.

- Design matrix per run: `word_rate | word_length | 6 motion params | constant` (all z-scored; predictors are pre-convolved with HRF)
- First level: AR(1) noise model, one effect-size map per subject per contrast
- Second level: one-sample t-test across subjects
- FDR correction (q < 0.05) on z-stat maps; equivalent t-threshold saved

**Required inputs:**

```
../BCCWJ-fMRI/derivatives/sub-*/func/
    {sub}_task-BCCWJreading_run-{1..4}_preproc_bold.nii.gz
    {sub}_task-BCCWJreading_run-{1..4}_preproc_rp.txt   # SPM realignment parameters (6 columns)
fMRIpredictors.csv   # columns: section_number, word_rate, word_length (HRF-convolved)
```

**Outputs (`data/glm/word_rate_length/`):**

```
first_level/{sub}_word_rate_eff.nii
first_level/{sub}_word_length_eff.nii
group_word_rate_zstat.nii / tstat.nii / tstat_fdr05.nii
group_word_length_zstat.nii / tstat.nii / tstat_fdr05.nii
```

---

#### [fMRI_isc_analysis.py](fMRI_isc_analysis.py)

Leave-one-out **Inter-Subject Correlation (ISC)** analysis, computed section-wise then averaged. Also generates the ISC visualization.

- Brain mask: Yeo 2011 liberal cortical mask at 2 mm (auto-downloaded if not present at `data/isc/yeo_liberal_mask_2mm.nii`)
- Group statistic: median ISC; Fisher z-transform → one-sample t-test
- Visualization: glass brain (L | axial | R) + 5 axial slices (z = −10 to +40 mm), thresholded at r > 0.25

**Required inputs:**

```
../BCCWJ-fMRI/derivatives/sub-*/func/
    {sub}_task-BCCWJreading_run-{1..4}_preproc_bold.nii.gz
```

**Outputs (`data/isc/`):**

```
isc_median.nii          # group median Pearson r
isc_tstat.nii           # one-sample t-stat (df = N−1)
isc_zstat.nii           # equivalent z-stat
isc_pmap.nii            # uncorrected two-sided p-values
isc_visualization.png   # glass brain + axial slice figure
```

---

## Text–Imaging Event Merge

These scripts merge  text information into the brain imaging event files, producing the annotated event tables used in the main analysis. They are not part of the data validation pipeline. The event files are not freely available. Please check BCCWJ.


### [bccwj-meegtextmerge.py](bccwj-meegtextmerge.py)

Reconstructs the `surface` (word form) column in each per-subject MEG/EEG events file by sequentially pairing subject rows with rows in the master event file (`MEG-EEGevents.tsv`) within each section. Originals are never modified.

Supports `--dry-run` to preview planned output files without writing anything.

**Required inputs:**

```
MEG-EEGevents.tsv                                          # master file with subsection_num and surface columns
sub-*/eeg/sub-*_task-BCCWJreading_events.tsv               # per-subject event files with section_num column
```

**Output (written alongside each source file):**

```
sub-*_task-BCCWJreading_events_with_surface.tsv
```

**Usage:**

```bash
python bccwj-meegtextmerge.py            # write output files
python bccwj-meegtextmerge.py --dry-run  # preview only
```