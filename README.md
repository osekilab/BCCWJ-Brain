# BCCWJ-Brain

Data valiation pipelines for EEG, MEG, and fMRI data collected during naturalistic reading of the Balanced Corpus of Contemporary Written Japanese (BCCWJ).

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
 [EEG_grand_average.py](EEG_grand_average.py):  Loads per-subject evoked `.fif` files (baseline −100–0 ms) and computes the grand-average EEG response. Produces a butterfly + GFP plot and a joint topographic plot. 
 
 [MEG_grand_average.py](MEG_grand_average.py):  Same workflow for MEG data. 



## fMRI

### [fMRI_glm_word_rate_length.py](fMRI_glm_word_rate_length.py)

First- and second-level GLM estimating the effects of **word rate** and **word length** on BOLD signal, fit jointly so each predictor controls for the other.

- Design matrix per run: `word_rate | word_length | 6 motion params | constant` (all z-scored; predictors are pre-convolved with HRF)
- First level: AR(1) noise model, one effect-size map per subject per contrast
- Second level: one-sample t-test across subjects
- Thresholding: FWE correction (Bonferroni, p < 0.05), cluster extent k > 50 voxels

### [fMRI_isc_analysis.py](fMRI_isc_analysis.py)

Leave-one-out **Inter-Subject Correlation (ISC)** analysis, computed section-wise and averaged across sections. 

- Brain mask: Yeo 2011 liberal cortical mask at 2 mm (auto-downloaded if absent at `data/isc/yeo_liberal_mask_2mm.nii`)
- Visualization: median ISC (Pearson r) 


## Text–Imaging Event Merge

### [bccwj-textmerge.py](bccwj-textmerge.py)

The texts are not avaiable publicly. Please obtain a BCCWJ license first (check [https://clrd.ninjal.ac.jp/bccwj/en/subscription.html](https://clrd.ninjal.ac.jp/bccwj/en/subscription.html)).

Merges text annotations from the BCCWJ master event file into per-subject BIDS event files. Adds `surface` (word form), `word_length`, `sent_id`, and `bunsetsu_pos` columns by sequentially pairing rows within each section. Source files are never modified; outputs get a `_with_surface` suffix.
