# ISIC 2020 Melanoma-vs-Benign Dataset Audit

TRAINING PERFORMED: NO
MODELS MODIFIED: NO
PRODUCTION APP MODIFIED: NO
OLD DATASET MODIFIED: NO
CLEAN DICM DATASET MODIFIED: NO
NEW SPLITS CREATED: NO

This is an audit of files already in the repository. No labels were changed,
no images were deleted, and no new splits were created.

## 1. Dataset identification

- Metadata file: `C:\Projects\major-project\dataset\raw\train.csv`
- Image directory: `C:\Projects\major-project\dataset\raw\train`
- Metadata rows: 33126
- Unique metadata image_name values: 33126
- Image files in the ISIC 2020 directory: 8584
- Unique local filenames: 8584

This is the SIIM-ISIC 2020 Challenge training CSV (`train.csv`), copied into
`dataset/raw/` by `scripts/extract_isic.py`. The extractor default was **all
target=1 images plus a seeded 8,000-image benign sample**, not the full 33,126 JPEGs.
That matches the local file count: 8584.

No separate `ISIC_2020_Training_GroundTruth.csv` is stored in the repo besides
`C:\Projects\major-project\dataset\raw\train.csv`. No `diagnosis_confirm_type` archive metadata for ISIC 2020 was found
in the project.

## 2. Metadata fields

Columns actually present:

| column | non-null | unique |
|---|---:|---:|
| image_name | 33126 | 33126 |
| patient_id | 33126 | 2056 |
| sex | 33061 | 2 |
| age_approx | 33058 | 18 |
| anatom_site_general_challenge | 32599 | 6 |
| diagnosis | 33126 | 9 |
| benign_malignant | 33126 | 2 |
| target | 33126 | 2 |

Derived for this audit: `image_id` (normalized `image_name`), `local_image`.

Requested fields **not present** in this CSV:

- `diagnosis_1`
- `diagnosis_2`
- `diagnosis_3`
- `diagnosis_confirm_type`
- `lesion_id`
- `image_type`
- `attribution`
- `acquisition_year`
- `dermoscopic_type`
- `image_manipulation`

Therefore this audit **cannot** report ISIC 2020 confirmation type, hierarchical
diagnosis_1/2/3, lesion_id, image type, attribution, year, or device from local metadata.

## 3. Official label interpretation

Do not assume `target` is melanoma without checking. Observed values:

### diagnosis

| diagnosis | count |
|---|---:|
| unknown | 27124 |
| nevus | 5193 |
| melanoma | 584 |
| seborrheic keratosis | 135 |
| lentigo NOS | 44 |
| lichenoid keratosis | 37 |
| solar lentigo | 7 |
| cafe-au-lait macule | 1 |
| atypical melanocytic proliferation | 1 |

### benign_malignant

| benign_malignant | count |
|---|---:|
| benign | 32542 |
| malignant | 584 |

### target

| target | count |
|---|---:|
| 0 | 32542 |
| 1 | 584 |

### diagnosis × benign_malignant

| diagnosis \ benign_malignant | benign | malignant |
|---|---|---|
| atypical melanocytic proliferation | 1 | 0 |
| cafe-au-lait macule | 1 | 0 |
| lentigo NOS | 44 | 0 |
| lichenoid keratosis | 37 | 0 |
| melanoma | 0 | 584 |
| nevus | 5193 | 0 |
| seborrheic keratosis | 135 | 0 |
| solar lentigo | 7 | 0 |
| unknown | 27124 | 0 |

### diagnosis × target

| diagnosis \ target | 0 | 1 |
|---|---|---|
| atypical melanocytic proliferation | 1 | 0 |
| cafe-au-lait macule | 1 | 0 |
| lentigo NOS | 44 | 0 |
| lichenoid keratosis | 37 | 0 |
| melanoma | 0 | 584 |
| nevus | 5193 | 0 |
| seborrheic keratosis | 135 | 0 |
| solar lentigo | 7 | 0 |
| unknown | 27124 | 0 |

### benign_malignant × target

| benign_malignant \ target | 0 | 1 |
|---|---|---|
| benign | 32542 | 0 |
| malignant | 0 | 584 |

**Interpretation from this file only:**

- `target==1` coincides exactly with `diagnosis==melanoma` and with `benign_malignant==malignant` in this CSV.
- Melanoma rows (all three agree): 584
- Disagreements among those three fields: 0
- `target==0` is **not** the same as a named benign diagnosis. It includes `unknown`, `atypical melanocytic proliferation`, and named benign diagnoses.
- `benign_malignant` in this CSV is only `benign` or `malignant`. There is no `indeterminate` value here.
- There are **no non-melanoma malignant diagnoses** (no BCC/SCC) in this ISIC 2020 challenge CSV.
- Confirmation method is **not recorded**. This audit does not claim histopathology, consensus, or serial imaging.

## 4. Melanoma candidate definition

Proposed strict rule (aligned with the clean DICM melanoma rule, adapted to available fields):

- `diagnosis == melanoma`
- AND `benign_malignant == malignant`

`target==1` matches this rule in this file, but the rule is stated from diagnosis + benign_malignant, not from target alone.

Rule counts (full metadata / local images):

| rule | metadata rows | with local image |
|---|---:|---:|
| diagnosis==melanoma | 584 | 584 |
| benign_malignant==malignant | 584 | 584 |
| target==1 | 584 | 584 |
| diagnosis==melanoma AND benign_malignant==malignant | 584 | 584 |
| diagnosis==melanoma AND target==1 | 584 | 584 |
| strict melanoma AND local image | 584 | 584 |

### Melanoma diagnosis distribution (strict metadata pool)

| diagnosis | count |
|---|---:|
| melanoma | 584 |

### Melanoma confirmation-type distribution

`diagnosis_confirm_type` is not in the available ISIC 2020 CSV. Count unknown: all proposed melanoma rows.
- confirmation type missing: 584 / 584

Local-image melanoma candidates: 584 (all target=1 melanoma JPEGs were extracted).

## 5. Benign candidate definition

Do **not** use `target==0` as benign. `target==0` includes unknown and atypical melanocytic proliferation.

Proposed strict rule (aligned with the clean DICM benign list, using only diagnoses that actually occur here):

- `benign_malignant == benign`
- AND `diagnosis` in the pre-specified named benign list used for the clean DICM design
- Named diagnoses present in ISIC 2020 that satisfy this: nevus, seborrheic keratosis, lentigo NOS,
  lichenoid keratosis, solar lentigo, cafe-au-lait macule

Excluded from strict benign even if `benign_malignant==benign`:
- `unknown` (no named diagnosis)
- `atypical melanocytic proliferation` (ambiguous melanocytic lesion)

| rule | metadata rows | with local image |
|---|---:|---:|
| benign_malignant==benign | 32542 | 8000 |
| target==0 | 32542 | 8000 |
| benign_malignant==benign AND target==0 | 32542 | 8000 |
| benign_malignant==benign AND diagnosis in explicit benign list | 5417 | 1335 |
| strict benign AND local image | 1335 | 1335 |
| official-benign including unknown, local image | 8000 | 8000 |
| official-benign local minus atypical AMP | 7999 | 7999 |

### Official-benign rows excluded by the strict named-diagnosis rule

| diagnosis | count |
|---|---:|
| unknown | 27124 |
| atypical melanocytic proliferation | 1 |

Strict benign metadata candidates: 5417
Strict benign with a local JPEG: 1335

The previous frozen experiment used a **looser** local rule: `benign_malignant==benign`
and drop only `ISIC_4225838` (atypical melanocytic proliferation), **keeping `diagnosis=unknown`**.
That looser local pool is reported above as `official-benign local minus atypical AMP`.
This audit's proposed scientific candidate is the **strict named-diagnosis** pool.

## 6. Diagnosis purity

### MELANOMA (strict metadata pool)

| diagnosis | count |
|---|---:|
| melanoma | 584 |

Confirmation type: not available (all missing).

### BENIGN (strict metadata pool)

| diagnosis | count |
|---|---:|
| nevus | 5193 |
| seborrheic keratosis | 135 |
| lentigo NOS | 44 |
| lichenoid keratosis | 37 |
| solar lentigo | 7 |
| cafe-au-lait macule | 1 |

Confirmation type: not available (all missing).

Diagnoses that should **not** enter a strict melanoma-vs-benign experiment:

| diagnosis | metadata count | reason |
|---|---:|---|
| unknown | 27124 | No named diagnosis; cannot assign a clean class |
| atypical melanocytic proliferation | 1 | Ambiguous melanocytic lesion; not confirmed melanoma or benign |

No BCC, SCC, AK, metastasis, or other non-melanoma cancers appear in this ISIC 2020 challenge CSV.

## 7. Confirmation-type analysis

The available ISIC 2020 training CSV does not include `diagnosis_confirm_type`.
This audit therefore **cannot** count histopathology vs expert consensus vs serial imaging.
Any claim that these melanoma rows are biopsy-proven would be an external assumption,
not a field observed in this repository.

## 8. Image availability

- Metadata rows: 33126
- Available images in `C:\Projects\major-project\dataset\raw\train`: 8584
- Metadata rows whose JPEG is present locally: 8584
- Metadata rows missing a local JPEG: 24542
- Local files whose stem is not in metadata: 0

Missing images are expected: the extractor did not unpack the full 33,126 JPEGs.

| subset | metadata | local JPEG | missing JPEG |
|---|---:|---:|---:|
| strict melanoma | 584 | 584 | 0 |
| strict benign | 5417 | 1335 | 4082 |
| official benign including unknown | 32542 | 8000 | 24542 |

## 9. Duplicate analysis

- Duplicate metadata image_name: 0
- Duplicate metadata image_id: 0
- Duplicate local filenames: 0
- Duplicate local image IDs: 0

## 10. DICM image overlap

Clean DICM manifest was read only. It was not modified.
Clean DICM rows: 11373 (melanoma 6787, benign 4586).

Full-metadata ISIC candidate pools vs clean DICM:

| overlap | count |
|---|---:|
| ISIC melanoma ∩ DICM melanoma | 581 |
| ISIC melanoma ∩ DICM benign | 0 |
| ISIC benign ∩ DICM melanoma | 0 |
| ISIC benign ∩ DICM benign | 532 |
| local ISIC melanoma ∩ DICM melanoma | 581 |
| local ISIC melanoma ∩ DICM benign | 0 |
| local ISIC benign ∩ DICM melanoma | 0 |
| local ISIC benign ∩ DICM benign | 143 |

Image IDs were compared after ISIC ID normalization. Overlaps were **not** removed.

## 11. DICM patient overlap

- Unique non-empty patient IDs in strict ISIC melanoma+benign metadata pool: 1223
- Unique non-empty patient IDs in clean DICM: 2434
- Shared patient IDs: 741
- Strict ISIC pool images from shared patients: 4157
- Clean DICM images from shared patients: 1135

Patient IDs were not removed or remapped.

## 12. DICM lesion overlap

ISIC 2020 challenge CSV has **no `lesion_id` column**.
The old experiment stored empty `lesion_id` for ISIC 2020 rows.
- ISIC 2020 lesion IDs available: 0
- Clean DICM non-empty lesion IDs: 4834
- Shared lesion IDs: 0

Lesion-level overlap **cannot be audited from this ISIC 2020 file**.

## 13. Old-experiment overlap

Old frozen experiment `dataset/processed/melanoma_benign/manifest.csv` was read only.

| item | count |
|---|---:|
| old manifest rows | 16529 |
| ISIC melanoma ∩ old images | 581 |
| ISIC benign ∩ old images | 1335 |
| local ISIC melanoma ∩ old | 581 |
| local ISIC benign ∩ old | 1335 |
| all ISIC metadata ∩ old | 8580 |
| shared patient IDs (strict ISIC pool vs old) | 1144 |
| strict ISIC images from those patients | 5631 |
| old-experiment images from those patients | 5333 |
| shared lesion IDs | 0 |

The old experiment used DICM official class=1 as melanoma and local ISIC 2020
`benign_malignant==benign` (minus one AMP image) as benign. That is a different
label definition from this audit's strict pools.

## 14. Source/domain confounding signals

These are **potential source/domain confounding signals**, not proof of model bias.
ISIC 2020 challenge CSV has no year, camera, attribution, dermoscopic type, or image_type.
Compared fields: sex, age_approx, anatom_site_general_challenge.

### Sex (strict metadata melanoma vs benign)

Melanoma:
| sex | count |
|---|---:|
| male | 364 |
| female | 220 |

Benign:
| sex | count |
|---|---:|
| male | 3007 |
| female | 2410 |

### Anatomical site

Melanoma:
| site | count |
|---|---:|
| torso | 257 |
| lower extremity | 124 |
| upper extremity | 111 |
| head/neck | 74 |
| <MISSING> | 9 |
| palms/soles | 5 |
| oral/genital | 4 |

Benign:
| site | count |
|---|---:|
| torso | 3087 |
| lower extremity | 1484 |
| upper extremity | 608 |
| head/neck | 186 |
| <MISSING> | 49 |
| palms/soles | 2 |
| oral/genital | 1 |

### Age (age_approx)

- Melanoma mean/median: 58.13 / 60.0 (n=584)
- Benign mean/median: 50.30 / 50.0 (n=5414)

Additional signals:
- Local ISIC JPEGs are a **non-random subset** of benign rows (seeded sample of 8,000 official-benign, including unknown).
- Strict named benign images on disk are therefore a sample of a sample, not the full ISIC 2020 named-benign set.
- Melanoma on disk is complete relative to this CSV (all 584 melanoma JPEGs are present).
- Combining with DICM would mix two acquisition sources unless overlap and domain shift are handled explicitly.

## 15. Final usable counts

Using the strict rules and **images actually on disk**:

- Strict ISIC melanoma with local JPEG: **584**
- Strict ISIC benign with local JPEG: **1335**
- Strict local total: **1919**

If the remaining ISIC 2020 JPEGs were extracted (not done in this audit):

- Strict ISIC melanoma in metadata: **584**
- Strict ISIC benign in metadata: **5417**
- Strict metadata total: **6001**

Clean DICM (unchanged, for comparison): melanoma 6,787; benign 4,586; total 11,373.

## 16. Excluded counts and reasons

| reason | metadata rows |
|---|---:|
| not strict melanoma and not strict benign | 27125 |
| diagnosis=unknown | 27124 |
| atypical melanocytic proliferation | 1 |
| strict melanoma missing local JPEG | 0 |
| strict benign missing local JPEG | 4082 |

## 17. Limitations

- Confirmation type is absent; melanoma/benign cannot be verified as histopathology-confirmed from this file.
- Lesion IDs are absent; lesion-level leakage vs DICM cannot be measured.
- Most ISIC 2020 JPEGs are not on disk; local benign named-diagnosis images are a subset of a seeded sample.
- `diagnosis=unknown` dominates official-benign rows (27,124 / 32,542).
- Patient IDs exist and can overlap DICM even when image IDs do not.
- This challenge CSV contains no non-melanoma cancers, so it cannot represent keratinocyte malignancy exclusion the way DICM metadata can.
- No splits were designed. No model was trained.

## 18. Recommendation for the NEXT DATASET DESIGN

This section compares options. It does **not** select one.

### Option A. ISIC 2020 verified melanoma + verified benign only

| item | local images on disk | full metadata if remaining JPEGs were added |
|---|---:|---:|
| melanoma | 584 | 584 |
| benign | 1335 | 5417 |
| total | 1919 | 6001 |

- Confirmation: not recorded in local metadata.
- Overlap: many named-benign ISIC IDs also appear in clean DICM benign (see section 10); melanoma image overlap with DICM melanoma is listed above.
- Source/domain: single challenge source, but local benign is an incomplete sample; class sizes are small on disk (584 melanoma / 1,335 named benign).
- Major limitations: tiny local strict pool; missing confirmation and lesion IDs; unknown diagnoses cannot be used if a clean label is required.

### Option B. Clean DICM verified melanoma + verified benign only

| item | count |
|---|---:|
| melanoma | 6787 |
| benign | 4586 |
| total | 11373 |

- Confirmation: DICM official metadata included `diagnosis_confirm_type`; the clean melanoma pool was previously reported as histopathology-confirmed in the DICM design document. That is a DICM-metadata property, not repeated here as a new claim about ISIC 2020.
- Overlap: none with a second source if used alone. Internal leakage-safe splits already exist and are frozen.
- Source/domain: single DICM source; previous DICM source-bias audit still applies (hospital/camera confounding possible).
- Major limitations: DICM-only domain; missing patient IDs required grouping fallbacks; already trained on in the clean DICM experiment.

### Option C. Clean DICM + verified local ISIC 2020 cases after removing image overlaps

Keep all clean DICM rows. Add local strict ISIC images whose image_id is not already in clean DICM (melanoma or benign).

- Additional local ISIC melanoma not in DICM: 3
- Additional local ISIC benign not in DICM: 1192
- Combined melanoma (DICM ∪ extra ISIC): 6790
- Combined benign (DICM ∪ extra ISIC): 5778
- Combined total: 12568

- Confirmation: mixed. DICM has confirmation-type fields; ISIC 2020 local CSV does not.
- Overlap: image-level overlap can be dropped as above, but patient-level overlap would remain unless also excluded. Lesion overlap cannot be checked for ISIC 2020.
- Source/domain: two sources (DICM archive vs SIIM-ISIC 2020 challenge photography). Class-conditional source differences are a potential confounding signal.
- Major limitations: mixed confirmation quality; patient leakage risk across sources; local ISIC benign add-on is small; would require **new** leakage-safe splits (not created here).

No option is selected. No dataset is built. No training is started.

---

TRAINING PERFORMED: NO
MODELS MODIFIED: NO
PRODUCTION APP MODIFIED: NO
OLD DATASET MODIFIED: NO
CLEAN DICM DATASET MODIFIED: NO
NEW SPLITS CREATED: NO

