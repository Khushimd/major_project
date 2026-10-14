CLEAN DICM-ONLY MELANOMA VS BENIGN DATASET
==================================================
This directory is a NEW experiment dataset.
The frozen 16,529-image melanoma_benign experiment was not modified.
No model was trained.

SOURCE DATASET
--------------
ISIC-DICM-17K official metadata only:
  dataset/isic_dicm_17k/train-set-metadata.csv
  dataset/isic_dicm_17k/valid-set-metadata.csv
Official source rows: 17,060
Images resolved under the project images/ directory.
Official DICM class was NOT used as the label.

FILTERING RULES
---------------
1. Keep melanoma only when diagnosis == "melanoma"
   AND benign_malignant is in {"malignant", "indeterminate/malignant"}.
2. Keep benign only when diagnosis is in the pre-specified benign list
   AND benign_malignant == "benign".
3. Do not use official DICM class=1.
4. Exclude BCC, SCC, AK, AIMP/AMP, atypical Spitz, melanoma metastasis,
   scar, "other", missing diagnosis, indeterminate, and all other
   ambiguous diagnoses.

MELANOMA DEFINITION
-------------------
diagnosis == melanoma
AND benign_malignant in {malignant, indeterminate/malignant}

Kept melanoma: 6,787
  malignant: 6,773
  indeterminate/malignant: 14
All kept melanoma rows have diagnosis exactly "melanoma".

BENIGN DEFINITION
-----------------
diagnosis in the list below AND benign_malignant == "benign":
  nevus
  seborrheic keratosis
  solar lentigo
  acrochordon
  lichenoid keratosis
  pigmented benign keratosis
  lentigo NOS
  dermatofibroma
  verruca
  vascular lesion
  angioma
  lentigo simplex
  neurofibroma
  angiokeratoma
  sebaceous hyperplasia
  angiofibroma or fibrous papule
  cafe-au-lait macule
  sebaceous adenoma
  clear cell acanthoma

Kept benign: 4,586
  nevus 3544
  seborrheic keratosis 244
  solar lentigo 179
  acrochordon 167
  lichenoid keratosis 137
  lentigo NOS 97
  verruca 62
  dermatofibroma 54
  angioma 32
  vascular lesion 23
  lentigo simplex 21
  neurofibroma 13
  angiokeratoma 5
  sebaceous hyperplasia 3
  angiofibroma or fibrous papule 2
  sebaceous adenoma 1
  cafe-au-lait macule 1
  clear cell acanthoma 1
  pigmented benign keratosis 0
    (112 rows with this diagnosis existed but were excluded because
     benign_malignant was not exactly "benign")

TOTALS
------
Total images: 11,373
Melanoma: 6,787
Benign: 4,586
Unique non-empty patient IDs: 2,434
Unique non-empty lesion IDs: 4,834
Missing patient_id: 7,058
Missing lesion_id: 3,396

EXCLUDED RECORDS
----------------
Official rows excluded: 5,687

Excluded by diagnosis:
  missing diagnosis                         2708
  basal cell carcinoma                      1498
  squamous cell carcinoma                    676
  actinic keratosis                          294
  pigmented benign keratosis                 112
  seborrheic keratosis                       105
  atypical melanocytic proliferation          80
  AIMP                                        77
  melanoma metastasis                         36
  solar lentigo                               25
  vascular lesion                             21
  dermatofibroma                              18
  nevus                                       10
  scar                                         9
  other                                        8
  clear cell acanthoma                         4
  melanoma                                     3
  atypical spitz tumor                         2
  lentigo NOS                                  1

Excluded by benign_malignant:
  benign                          3094
  malignant                       1687
  missing                          748
  indeterminate                     73
  indeterminate/malignant           56
  indeterminate/benign              29

Note: some rows with a listed benign diagnosis were excluded because
benign_malignant was not exactly "benign". Three melanoma-diagnosis
rows were excluded because benign_malignant was not malignant or
indeterminate/malignant.

GROUPING STRATEGY
-----------------
Primary: patient_id
If patient_id is missing: lesion_id
If both are missing: image_id
Random seed: 42
Target: train 70% / validation 15% / test 15%
Percentages are approximate because groups are never split.

Grouping used (image counts):
  patient: 4,315
  lesion fallback: 5,257
  image fallback: 1,801

NEW SPLITS
----------
Train:
  total 7961
  melanoma 4747
  benign 3214
  patients 1706
  lesions 3418

Validation:
  total 1706
  melanoma 1016
  benign 690
  patients 384
  lesions 707

Test:
  total 1706
  melanoma 1024
  benign 682
  patients 344
  lesions 709

LEAKAGE AUDIT (NEW EXPERIMENT ONLY)
-----------------------------------
No image ID appears in more than one new split.
No lesion ID appears in more than one new split.
No non-empty patient ID appears in more than one new split.
No duplicate image path appears across splits.
No same-size same-content image files were detected.
No test image appears in the new train or validation set.

OVERLAP WITH OLD 16,529-IMAGE EXPERIMENT
----------------------------------------
Old files were not modified.
This overlap is reported only.

Clean images overlapping old manifest: 6,930
Clean images overlapping old train: 4,834
Clean images overlapping old validation: 981
Clean images overlapping old test: 1,115

CROSS-DATASET PATIENT AUDIT
---------------------------
Non-empty patient IDs present in both the clean DICM pool and the old
experiment: 1,403

Those 1,403 patients contribute:
  2,581 images in the new clean DICM dataset
  4,823 images in the old experiment

Old-experiment split membership of those patients:
  train only: 971
  validation only: 217
  test only: 215
  multiple old splits: 0

New-experiment split membership of those patients:
  train only: 974
  validation only: 228
  test only: 201
  multiple new splits: 0

Any patient with images in multiple NEW splits: NO

The full per-patient list (patient ID, DICM image count, old image
count, old split membership) is in leakage_report.json.

LABELS
------
label = 0 benign
label = 1 melanoma
source = ISIC-DICM-17K

FILES
-----
manifest.csv
train.csv
val.csv
test.csv
split_detail.json
leakage_report.json
dataset_report.json
CLEAN_DATASET_README.txt
CLEAN_DATASET_PREPARATION_REPORT.txt

TRAINING STARTED: NO
