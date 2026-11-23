# Split files (lesion-level manifests)

The manifests used in the paper were produced by `src/data_prep.py` (Kaggle notebook
`notebooks/skin_referral_kaggle.ipynb`). Each manifest has one row per image, with `image_id`,
`lesion_id`, labels and, for HAM10000, the split columns `split_s42` ... `split_s46`
(lesion-level 70/15/15; seeds 42-44 were used for training) and `leaky_s42` ... `leaky_s46`
(conventional image-level splits used only for the leakage audit).

SHA-256 checksums of the exact files used (computed in `notebooks/skin_referral_lesion_level_kaggle.ipynb`):

| File | Rows | SHA-256 |
|---|---|---|
| ham10000.csv | 11,719 | 3a3dffe7b84a377d691fa8936371bc955eb91d81ff77d60c325983b4d19eab03 |
| bcn20000.csv | 17,411 | a487274b5ae59a72cbc891ca95da30d5a01667d750e0ae4380c35ea7e47d233c |
| milk10k.csv | 4,149 | 52bfe0c4b92cf789da1fd6f46a4a39f2276c31c101546861f71a68ac930d00e7 |
| padufes20.csv | 2,283 | b7770416d9c8d78cc72e4a33729bb7673b3834fa4784387ee188b952fe0a1d32 |

Re-running `src/data_prep.py` regenerates the splits deterministically; compare your files with
these checksums. The CSV files themselves are available from the corresponding author on request
and will be added here upon acceptance.

Author: Arshad Saeed, Bay Atlantic University (asaeed@stu.bau.edu)
