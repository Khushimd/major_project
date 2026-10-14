FINAL MELANOMA VS BENIGN RESEARCH DATASET
No model training was performed during this dataset-construction stage.
Production models, the application, the old frozen experiment, and the clean DICM directory were not modified.

Melanoma: clean DICM diagnosis==melanoma AND benign_malignant in {malignant, indeterminate/malignant}. 6787 images.
Benign: clean DICM named-benign (4586) + ISIC 2020 named-benign (5417 metadata) minus 532 duplicate image IDs already in DICM benign.
ISIC 2020 melanoma was NOT added.

Final: melanoma=6787 benign=9467 total=16254 ratio_benign:melanoma=1.395:1
Train {'total': 11379, 'melanoma': 4703, 'benign': 6676, 'patients': 2051, 'lesions': 3340, 'missing_patient_id': 4865, 'missing_lesion_id': 5893, 'group_levels': {'patient': 6514, 'lesion': 3594, 'image': 1271}, 'by_source': {"('ISIC-2020', 0)": 3487, "('ISIC-DICM-17K', 1)": 4703, "('ISIC-DICM-17K', 0)": 3189}}
Val {'total': 2438, 'melanoma': 1043, 'benign': 1395, 'patients': 440, 'lesions': 753, 'missing_patient_id': 1131, 'missing_lesion_id': 1168, 'group_levels': {'patient': 1307, 'lesion': 846, 'image': 285}, 'by_source': {"('ISIC-2020', 0)": 673, "('ISIC-DICM-17K', 1)": 1043, "('ISIC-DICM-17K', 0)": 722}}
Test {'total': 2437, 'melanoma': 1041, 'benign': 1396, 'patients': 425, 'lesions': 741, 'missing_patient_id': 1062, 'missing_lesion_id': 1216, 'group_levels': {'patient': 1375, 'lesion': 817, 'image': 245}, 'by_source': {"('ISIC-2020', 0)": 721, "('ISIC-DICM-17K', 1)": 1041, "('ISIC-DICM-17K', 0)": 675}}
Seed=42. Grouping: patient_id -> lesion_id -> image_id.
Leakage across new splits: 0 patient / 0 lesion / 0 image.
