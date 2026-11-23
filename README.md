# Label-Free Recalibration of Skin Cancer Referral Thresholds

Code for the paper:

> **Label-free recalibration of skin cancer referral thresholds under site shift: a multi-cohort external validation**
> Arshad Saeed, Saira Afzal, Tahira Afzal, Mubashir Hussain (submitted, 2026)

**Corresponding author:** Arshad Saeed, School of Information Sciences and Engineering, Bay Atlantic University, Washington, DC, USA (asaeed@stu.bau.edu)

---

## Main result

At α = 5%, a threshold calibrated on HAM10000 missed **17.3%** of malignant lesions on average at the three external cohorts and broke its promise in **96%** of model–seed–site runs (78–100% per model; 6 models × 3 seeds × 3 sites). Weighted conformal recalibration, using no labels from the new site, lowered the **average** miss rate to **4.6%**, but this varied by site (PAD-UFES-20 1.5%, BCN20000 4.6%, MILK10k 7.6%) and came with a referral rate of 93%. Melanoma-specific misses under WCP stayed above 5% (5.2–13.4%). Quantile transfer made misses worse (26–28%).

Decisions are made **per image**; confidence intervals resample whole lesions. A lesion-level sensitivity analysis (a lesion is referred if any of its images is referred) is in `notebooks/skin_referral_lesion_level_kaggle.ipynb`.

## What this repository does

A skin cancer classifier is trained on HAM10000 and used as a **referral tool**: it auto-clears low-risk lesions and refers the rest to a dermatologist, with the promise that no more than α (for example 5%) of malignant lesions are auto-cleared.

We test whether that promise still holds when the same model is deployed at **new sites** (BCN20000, MILK10k with anatomical-site subgroups, PAD-UFES-20 smartphone images). We then compare **label-free recalibration methods** that use only unlabeled images from the new site:

| Method | Uses target labels? | Idea |
| --- | --- | --- |
| `SRC` | No | Split-conformal threshold from the source site only (baseline) |
| `WCP` | No | Weighted conformal prediction with a density ratio from a domain classifier; refers the case when the evidence is too weak ("certify or refuse") |
| `QTC` | No | Keeps the source auto-clear fraction on the target site |
| `QTC_LS` | No | QTC with EM label-shift (prevalence) correction |
| `ORACLE_n` | Yes, n = 20 or 50 malignant cases | Few-label reference |

The study also includes a **leakage audit** (image-level vs lesion-level splits of HAM10000).

## Repository layout

```
src/
  data_prep.py          Step 1: download, resize (512 px), label harmonization, lesion-level splits, overlap removal
  train.py              Step 2: 5 CNNs x 3 seeds + leakage-audit runs + frozen DINOv2 probe
  referral_lib.py       Referral rule and recalibration methods (SRC, WCP, QTC, QTC_LS, ORACLE)
  analysis.py           Step 3: discrimination, leakage audit, referral experiment tables
  figures.py            Step 4: figures, bootstrap CIs, skin-tone and artifact audits
  fig1_study_design.py  Figure 1
notebooks/
  skin_referral_kaggle.ipynb   Full pipeline in one notebook (Kaggle, GPU T4, Internet ON)
splits/                 Lesion-level split files with SHA-256 checksums (added after the final run)
figures/                Output figures
```

## Data

All datasets are public. The code downloads them automatically.

| Dataset | Source | Licence |
| --- | --- | --- |
| HAM10000 | ISIC Archive, collection 212 | CC BY-NC 4.0 |
| BCN20000 | ISIC Archive, collection 249 | CC BY-NC 4.0 |
| MILK10k | ISIC Archive, collection 425 (doi:10.34970/648456) | CC BY-NC 4.0 |
| PAD-UFES-20 | Mendeley Data (doi:10.17632/zr7vgbcyr2.1) | CC BY 4.0 |

Please cite the original dataset papers when you use them. Licences listed here are as stated by the providers at download time; check the source pages before reuse.

## How to run

**Option A, Kaggle (recommended):** upload `notebooks/skin_referral_kaggle.ipynb`, set *Accelerator = GPU T4* and *Internet = ON*, then *Save & Run All*. The full run takes about 3 to 4 hours.

**Option B, local or Colab:**

```bash
pip install -r requirements.txt
export PROJ=./outputs/SkinCancer_Referral_Q1_2026
export LOCAL=./data_512
python src/data_prep.py     # ~40 min, needs internet
python src/train.py         # needs a CUDA GPU
python src/analysis.py
python src/figures.py
```

## Reproducibility

- Lesion-level 70/15/15 splits (5 split seeds generated, 42–46; seeds 42–44 used for training), stratified by class, grouped by `lesion_id`. A check asserts that no lesion appears in two subsets.
- External images are removed if they share an image ID or lesion ID with HAM10000, or are near-duplicates (perceptual hash, Hamming distance ≤ 4).
- Fixed seeds: 42, 43, 44 for training; split seeds 42–46.

## Citation

See `CITATION.cff`. The citation will be updated when the paper is published.

## License

Code: MIT License, © 2026 Arshad Saeed, Bay Atlantic University. Datasets keep their original licences.

## Notebooks (Kaggle)

| Notebook | What it does |
| --- | --- |
| `notebooks/skin_referral_kaggle.ipynb` | Full pipeline: data download and cleaning, training of all models, predictions (GPU) |
| `notebooks/skin_referral_effnet_fp32_kaggle.ipynb` | Re-runs EfficientNet-B3 with full-precision inference (fp16 overflowed for some images) |
| `notebooks/skin_referral_analysis_v2_kaggle.ipynb` | CPU analysis: discrimination, leakage audit, referral safety, paired tests, figures, audits |
| `notebooks/skin_referral_lesion_level_kaggle.ipynb` | Lesion-level sensitivity analysis and SHA-256 checksums of the split manifests |

`src/make_results.py` turns the result CSVs into manuscript-ready tables; `src/fig_results.py`, `src/fig_extra.py` and `src/fig1_study_design.py` draw the figures.
