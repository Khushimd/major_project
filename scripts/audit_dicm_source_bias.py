"""
Read-only audit of ISIC-DICM-17K metadata vs the current ISIC 2020 benign pool.

Does not train, and does not modify datasets, models, splits, or production files.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

import config
from training.dataset_loader import normalize_isic_id

REPORT = config.DATASET_DIR / "isic_dicm_17k" / "DICM_SOURCE_BIAS_AUDIT.txt"
ARCHIVE = config.BASE_DIR / "metadata.csv"
TRAIN_META = config.ISIC_DICM_LABEL_DIR / "train-set-metadata.csv"
VALID_META = config.ISIC_DICM_LABEL_DIR / "valid-set-metadata.csv"
ISIC2020 = config.RAW_DATA_DIR / "train.csv"


def _counts(series: pd.Series, limit: int | None = None) -> list[str]:
    vc = series.fillna("<MISSING>").astype(str).str.strip()
    vc = vc.replace("", "<EMPTY>")
    vc = vc.value_counts(dropna=False)
    if limit:
        shown = vc.head(limit)
        extra = "" if len(vc) <= limit else f"\n    ... {len(vc) - limit} more values"
        lines = [f"    {k}: {int(v)}" for k, v in shown.items()]
        return lines + ([extra] if extra else [])
    return [f"    {k}: {int(v)}" for k, v in vc.items()]


def _nunique(series: pd.Series) -> int:
    cleaned = series.astype(str).str.strip()
    cleaned = cleaned[~cleaned.isin(["", "nan", "None", "<NA>"])]
    return int(cleaned.nunique())


def _id_set(series: pd.Series) -> set[str]:
    values = series.dropna().astype(str).str.strip()
    return {v for v in values if v and v.lower() not in {"nan", "none"}}


def main() -> None:
    archive = pd.read_csv(ARCHIVE)
    official_train = pd.read_csv(TRAIN_META)
    official_valid = pd.read_csv(VALID_META)
    official = pd.concat([official_train, official_valid], ignore_index=True)
    official["class"] = pd.to_numeric(official["class"], errors="coerce")
    official["image_id"] = official["isic_id"].map(normalize_isic_id)
    official["patient_id_clean"] = official["patient_id"].astype(str).str.strip()
    official["lesion_id_clean"] = official["lesion_id"].astype(str).str.strip()

    archive["image_id"] = archive["isic_id"].map(normalize_isic_id)
    merged = official.merge(
        archive,
        on="image_id",
        how="left",
        suffixes=("_official", "_archive"),
    )

    class0 = official[official["class"] == 0].copy()
    class1 = official[official["class"] == 1].copy()
    class0_m = merged[merged["class"] == 0]
    class1_m = merged[merged["class"] == 1]

    isic = pd.read_csv(ISIC2020)
    isic["image_id"] = isic["image_name"].map(normalize_isic_id)
    isic["patient_id_clean"] = isic["patient_id"].astype(str).str.strip()
    isic_benign = isic[isic["benign_malignant"].astype(str).str.strip().str.lower() == "benign"]
    isic_used = isic_benign[isic_benign["image_id"] != normalize_isic_id("ISIC_4225838")]

    dicm_ids = set(official["image_id"])
    isic_ids = set(isic["image_id"])
    isic_benign_ids = set(isic_used["image_id"])
    image_overlap_all = sorted(dicm_ids & isic_ids)
    image_overlap_benign = sorted(dicm_ids & isic_benign_ids)
    image_overlap_class = official[official["image_id"].isin(image_overlap_all)]["class"].value_counts(dropna=False).to_dict()

    dicm_patients = _id_set(official["patient_id_clean"])
    isic_patients = _id_set(isic["patient_id_clean"])
    patient_overlap = sorted(dicm_patients & isic_patients)

    dicm_ip = {p for p in dicm_patients if p.upper().startswith("IP_")}
    isic_ip = {p for p in isic_patients if p.upper().startswith("IP_")}
    ip_overlap = sorted(dicm_ip & isic_ip)

    # Image dimension sample from on-disk files if present (read-only).
    dim_note = "Not measured from pixels in this audit. Official metadata has no width/height columns."
    try:
        import cv2

        samples = []
        for source, paths in (
            (
                "DICM class=1 sample",
                class1["isic_id"].head(8).map(lambda x: config.ISIC_DICM_IMAGE_DIR / f"{normalize_isic_id(x)}.jpg"),
            ),
            (
                "DICM class=0 sample",
                class0["isic_id"].head(8).map(lambda x: config.ISIC_DICM_IMAGE_DIR / f"{normalize_isic_id(x)}.jpg"),
            ),
            (
                "ISIC 2020 benign sample",
                isic_used["image_name"].head(8).map(lambda x: config.RAW_DATA_DIR / "train" / f"{x}.jpg"),
            ),
        ):
            sizes = []
            for path in paths:
                img = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if img is not None:
                    sizes.append(f"{img.shape[1]}x{img.shape[0]}")
            if sizes:
                samples.append(f"  {source}: {', '.join(sizes)}")
        if samples:
            dim_note = "Small read-only pixel sample (width x height):\n" + "\n".join(samples)
    except Exception as exc:
        dim_note = f"Pixel-size sample unavailable: {exc}"

    lines = [
        "DICM SOURCE-BIAS AUDIT",
        "Read-only. No training. No dataset/model/split/production files were modified.",
        "",
        "Current experimental construction (context, not changed here):",
        "  Melanoma  = 8,530 images from ISIC-DICM-17K official class=1",
        "  Benign    = 7,999 local ISIC 2020 images with benign_malignant=benign",
        "  DICM official class=0 (non-melanoma) was NOT used as the benign class.",
        "",
        "Files inspected:",
        f"  {ARCHIVE}",
        f"  {TRAIN_META}  rows={len(official_train)}",
        f"  {VALID_META}  rows={len(official_valid)}",
        f"  {ISIC2020}  rows={len(isic)}",
        "",
        "============================================================",
        "1. COLUMNS",
        "============================================================",
        "",
        f"Official train/valid metadata ({len(official_train.columns)} columns):",
        *[f"  - {c}" for c in official_train.columns],
        "",
        f"Archive metadata.csv ({len(archive.columns)} columns; archive extras not in official labels):",
        *[f"  - {c}" for c in archive.columns],
        "",
        "Archive-only fields (not in official train/valid CSVs):",
        *[f"  - {c}" for c in sorted(set(archive.columns) - set(official_train.columns) - {"image_id"})],
        "",
        "Official-only fields (not in archive metadata.csv):",
        *[f"  - {c}" for c in sorted(set(official_train.columns) - set(archive.columns) - {"image_id"})],
        "",
        "============================================================",
        "2. OFFICIAL BINARY class COLUMN",
        "============================================================",
        "",
        "class is the official DICM melanoma vs non-melanoma label.",
        "It is NOT a benign vs malignant label.",
        "",
        f"Official train rows: {len(official_train)}",
        f"Official valid rows: {len(official_valid)}",
        f"Official combined rows: {len(official)}",
        "",
        "Combined class counts:",
        *_counts(official["class"]),
        "",
        "Train class counts:",
        *_counts(official_train["class"]),
        "",
        "Valid class counts:",
        *_counts(official_valid["class"]),
        "",
        "============================================================",
        "3. class=0 (NON-MELANOMA) DIAGNOSIS FIELDS",
        "============================================================",
        "",
        f"n class=0: {len(class0)}",
        "",
        "IMPORTANT: class=0 means NON-MELANOMA, not necessarily benign.",
        "",
        "Official diagnosis:",
        *_counts(class0["diagnosis"]),
        "",
        "Official benign_malignant:",
        *_counts(class0["benign_malignant"]),
        "",
        "Official melanocytic:",
        *_counts(class0["melanocytic"]),
        "",
        "Official nevus_type:",
        *_counts(class0["nevus_type"]),
        "",
        "Official diagnosis_confirm_type:",
        *_counts(class0["diagnosis_confirm_type"]),
        "",
        "Archive diagnosis_1 (merged on isic_id):",
        *_counts(class0_m["diagnosis_1"]),
        "",
        "Archive diagnosis_2:",
        *_counts(class0_m["diagnosis_2"]),
        "",
        "Archive diagnosis_3:",
        *_counts(class0_m["diagnosis_3"]),
        "",
        "============================================================",
        "4. class=1 (MELANOMA) DIAGNOSIS FIELDS",
        "============================================================",
        "",
        f"n class=1: {len(class1)}",
        "",
        "Official diagnosis:",
        *_counts(class1["diagnosis"]),
        "",
        "Official benign_malignant:",
        *_counts(class1["benign_malignant"]),
        "",
        "Official mel_class:",
        *_counts(class1["mel_class"]),
        "",
        "Official mel_type:",
        *_counts(class1["mel_type"]),
        "",
        "Official melanocytic:",
        *_counts(class1["melanocytic"]),
        "",
        "Archive diagnosis_1:",
        *_counts(class1_m["diagnosis_1"]),
        "",
        "============================================================",
        "5. DIAGNOSIS CATEGORIES INSIDE DICM NON-MELANOMA (class=0)",
        "============================================================",
        "",
        "These are the official `diagnosis` values among class=0 images.",
        "They include both benign diagnoses and other non-melanoma disease.",
        "",
        *_counts(class0["diagnosis"]),
        "",
        "benign_malignant among class=0 (do NOT equate this to 'benign class'):",
        *_counts(class0["benign_malignant"]),
        "",
        "============================================================",
        "6. PATIENT IDs",
        "============================================================",
        "",
        f"Official patient_id non-empty unique, class=0: {_nunique(class0['patient_id'])}",
        f"Official patient_id non-empty unique, class=1: {_nunique(class1['patient_id'])}",
        f"Official patient_id missing/empty, class=0: {int((class0['patient_id_clean'].isin(['', 'nan'])).sum())}",
        f"Official patient_id missing/empty, class=1: {int((class1['patient_id_clean'].isin(['', 'nan'])).sum())}",
        f"Archive patient_id unique among merged class=0: {_nunique(class0_m['patient_id_archive']) if 'patient_id_archive' in class0_m.columns else _nunique(class0_m.get('patient_id', pd.Series(dtype=str)))}",
        "",
        "============================================================",
        "7. LESION IDs",
        "============================================================",
        "",
        f"Official lesion_id non-empty unique, class=0: {_nunique(class0['lesion_id'])}",
        f"Official lesion_id non-empty unique, class=1: {_nunique(class1['lesion_id'])}",
        f"Official lesion_id missing/empty, class=0: {int((class0['lesion_id_clean'].isin(['', 'nan'])).sum())}",
        f"Official lesion_id missing/empty, class=1: {int((class1['lesion_id_clean'].isin(['', 'nan'])).sum())}",
        "",
        "============================================================",
        "8. IMAGE ID OVERLAP WITH ISIC 2020",
        "============================================================",
        "",
        f"DICM official unique image IDs: {len(dicm_ids)}",
        f"ISIC 2020 train.csv unique image IDs: {len(isic_ids)}",
        f"ISIC 2020 official-benign IDs after dropping ISIC_4225838: {len(isic_benign_ids)}",
        f"Image-ID overlap DICM intersect all ISIC 2020: {len(image_overlap_all)}",
        f"Image-ID overlap DICM intersect ISIC 2020 benign-used pool: {len(image_overlap_benign)}",
        f"Overlap by DICM class: {image_overlap_class}",
        "",
        "Note: any overlap of DICM melanoma (class=1) with ISIC 2020 is expected if the",
        "same ISIC image appears in both collections. Overlap of DICM class=0 with the",
        "current benign pool would be more concerning for leakage.",
        "",
        "============================================================",
        "9. PATIENT ID OVERLAP DICM vs ISIC 2020",
        "============================================================",
        "",
        f"DICM unique non-empty patient_id: {len(dicm_patients)}",
        f"ISIC 2020 unique non-empty patient_id: {len(isic_patients)}",
        f"Exact string overlap of all patient_id values: {len(patient_overlap)}",
        f"DICM IDs starting with IP_: {len(dicm_ip)}",
        f"ISIC 2020 IDs starting with IP_: {len(isic_ip)}",
        f"Overlap of IP_ patient IDs: {len(ip_overlap)}",
        "",
        "Identifier compatibility:",
        "  ISIC 2020 uses IP_######## patient IDs.",
        "  DICM official patient_id is present for most rows, but the namespace is mixed.",
        "  Exact string match is the only safe overlap test used here.",
        "  No attempt was made to invent a crosswalk for non-IP identifiers.",
        "",
        "============================================================",
        "10. ACQUISITION / SOURCE FIELDS",
        "============================================================",
        "",
        "Official year (collection/challenge year — a source-shift signal):",
        *_counts(official["year"]),
        "",
        "Official year by class=0:",
        *_counts(class0["year"]),
        "",
        "Official year by class=1:",
        *_counts(class1["year"]),
        "",
        "Official image_type:",
        *_counts(official["image_type"]),
        "",
        "Official dermoscopic_type:",
        *_counts(official["dermoscopic_type"]),
        "",
        "Official anatom_site_general:",
        *_counts(official["anatom_site_general"]),
        "",
        "Archive attribution (dataset/contributor — strongest source field):",
        *_counts(merged["attribution"]),
        "",
        "Archive attribution by class=0:",
        *_counts(class0_m["attribution"]),
        "",
        "Archive attribution by class=1:",
        *_counts(class1_m["attribution"]),
        "",
        "Archive image_type:",
        *_counts(merged["image_type_archive"] if "image_type_archive" in merged.columns else merged.get("image_type", pd.Series(dtype=str))),
        "",
        "Archive image_manipulation:",
        *_counts(merged["image_manipulation"]),
        "",
        "No camera/device column exists in official or archive metadata.",
        "No explicit width/height columns exist.",
        dim_note,
        "",
        "============================================================",
        "11. NON-MELANOMA vs BENIGN",
        "============================================================",
        "",
        "DICM class=0 is NON-MELANOMA.",
        "It is not safe to treat class=0 as a benign class.",
        "",
        "Reasons from this audit:",
        "  - The official task is melanoma vs non-melanoma.",
        "  - class=0 diagnosis values can include inflammatory, keratinocytic,",
        "    and other non-melanoma diseases, not only nevi.",
        "  - benign_malignant is a separate field and is not identical to class.",
        "  - The current experiment correctly used only class=1 as melanoma and",
        "    used ISIC 2020 benign_malignant=benign as the other class.",
        "",
        "That construction avoids mislabeling non-melanoma disease as benign,",
        "but it creates a SOURCE confound:",
        "  melanoma images come from the DICM 17K collection;",
        "  benign images come from ISIC 2020.",
        "A model can partly learn collection/camera/year style instead of",
        "melanoma biology. This audit does not measure how large that effect is.",
        "",
        "============================================================",
        "12. IMPLICATIONS (NO TRAINING)",
        "============================================================",
        "",
        "No model was trained or changed.",
        "The frozen melanoma_benign splits were not modified.",
        "Production files were not modified.",
        "",
        "If a later experiment addresses source confounding, do it as a new",
        "controlled study. Do not silently rewrite the frozen test set.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {REPORT}")


if __name__ == "__main__":
    main()
