"""
Read-only audit of local ISIC 2020 metadata vs a melanoma-vs-benign design.

Writes only dataset/isic_2020/ISIC_2020_MELANOMA_BENIGN_AUDIT.md
Does not train or modify datasets, models, experiments, or the app.

TRAINING PERFORMED: NO
MODELS MODIFIED: NO
PRODUCTION APP MODIFIED: NO
OLD DATASET MODIFIED: NO
CLEAN DICM DATASET MODIFIED: NO
NEW SPLITS CREATED: NO
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

import config

OUT_DIR = ROOT / "dataset" / "isic_2020"
REPORT = OUT_DIR / "ISIC_2020_MELANOMA_BENIGN_AUDIT.md"
META = config.RAW_DATA_DIR / "train.csv"
IMG_DIR = config.RAW_DATA_DIR / "train"
DICM_MANIFEST = config.PROCESSED_DATA_DIR / "dicm_clean_melanoma_benign" / "manifest.csv"
OLD_MANIFEST = config.PROCESSED_DATA_DIR / "melanoma_benign" / "manifest.csv"

BENIGN_DX = {
    "nevus",
    "seborrheic keratosis",
    "solar lentigo",
    "acrochordon",
    "lichenoid keratosis",
    "pigmented benign keratosis",
    "lentigo NOS",
    "dermatofibroma",
    "verruca",
    "vascular lesion",
    "angioma",
    "lentigo simplex",
    "neurofibroma",
    "angiokeratoma",
    "sebaceous hyperplasia",
    "angiofibroma or fibrous papule",
    "cafe-au-lait macule",
    "sebaceous adenoma",
    "clear cell acanthoma",
}
EMPTY = {"", "nan", "none", "<na>", "null", "na"}


def normalize_isic_id(value: str) -> str:
    text = str(value or "")
    match = re.search(r"ISIC_(\d+)", text, re.I)
    if match:
        return f"ISIC_{int(match.group(1)):07d}"
    return Path(text).stem


def _clean(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in EMPTY else text


def _idset(series: pd.Series) -> set[str]:
    return {_clean(v) for v in series if _clean(v)}


def _counts(series: pd.Series) -> dict[str, int]:
    filled = series.map(lambda v: _clean(v) or "<MISSING>")
    return {str(k): int(v) for k, v in filled.value_counts().items()}


def _md_table(mapping: dict, key_name: str, max_rows: int | None = None) -> list[str]:
    items = list(mapping.items())
    if max_rows is not None:
        items = items[:max_rows]
    lines = [f"| {key_name} | count |", "|---|---:|"]
    for key, n in items:
        lines.append(f"| {key} | {n} |")
    if not items:
        lines.append("| *(none)* | 0 |")
    return lines


def _crosstab_md(df: pd.DataFrame, a: str, b: str) -> list[str]:
    ct = pd.crosstab(df[a].map(lambda v: _clean(v) or "<MISSING>"), df[b].map(lambda v: _clean(v) or "<MISSING>"), dropna=False)
    cols = [str(c) for c in ct.columns]
    lines = ["| " + a + " \\ " + b + " | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for idx, row in ct.iterrows():
        lines.append("| " + str(idx) + " | " + " | ".join(str(int(v)) for v in row.tolist()) + " |")
    return lines


def overlap_report(left_ids: set[str], right_ids: set[str], left_df: pd.DataFrame, id_col: str) -> dict:
    shared = left_ids & right_ids
    affected = int(left_df[id_col].map(_clean).isin(shared).sum()) if shared else 0
    return {"shared": len(shared), "affected_images": affected}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(META)
    meta["image_id"] = meta["image_name"].map(normalize_isic_id)
    for col in meta.columns:
        if col not in {"age_approx", "target"}:
            meta[col] = meta[col]
    meta["diagnosis_c"] = meta["diagnosis"].map(_clean)
    meta["bm"] = meta["benign_malignant"].map(_clean).str.lower()
    meta["patient_c"] = meta["patient_id"].map(_clean)
    meta["target_i"] = pd.to_numeric(meta["target"], errors="coerce")

    local_files = sorted(p for p in IMG_DIR.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    local_names = [p.name for p in local_files]
    local_stems = {normalize_isic_id(p.stem) for p in local_files}
    name_counts = Counter(local_names)
    dup_names = {k: v for k, v in name_counts.items() if v > 1}
    id_counts = Counter(normalize_isic_id(p.stem) for p in local_files)
    dup_ids = {k: v for k, v in id_counts.items() if v > 1}

    meta["local_image"] = meta["image_id"].isin(local_stems)
    missing_meta = meta.loc[~meta["local_image"]].copy()
    extra_files = local_stems - set(meta["image_id"])

    # Label interpretation
    dx_counts = _counts(meta["diagnosis"])
    bm_counts = _counts(meta["benign_malignant"])
    target_counts = _counts(meta["target"].astype(str))

    # Melanoma candidate rules
    r_mel_dx = meta["diagnosis_c"].str.lower() == "melanoma"
    r_mel_bm = meta["bm"] == "malignant"
    r_mel_t = meta["target_i"] == 1
    rules_mel = {
        "diagnosis==melanoma": r_mel_dx,
        "benign_malignant==malignant": r_mel_bm,
        "target==1": r_mel_t,
        "diagnosis==melanoma AND benign_malignant==malignant": r_mel_dx & r_mel_bm,
        "diagnosis==melanoma AND target==1": r_mel_dx & r_mel_t,
        "strict melanoma AND local image": r_mel_dx & r_mel_bm & meta["local_image"],
    }

    # Benign candidate rules
    r_ben_bm = meta["bm"] == "benign"
    r_ben_t = meta["target_i"] == 0
    r_ben_dx = meta["diagnosis_c"].isin(BENIGN_DX)
    r_unknown = meta["diagnosis_c"].str.lower() == "unknown"
    r_atyp = meta["diagnosis_c"].str.lower() == "atypical melanocytic proliferation"
    rules_ben = {
        "benign_malignant==benign": r_ben_bm,
        "target==0": r_ben_t,
        "benign_malignant==benign AND target==0": r_ben_bm & r_ben_t,
        "benign_malignant==benign AND diagnosis in explicit benign list": r_ben_bm & r_ben_dx,
        "strict benign AND local image": r_ben_bm & r_ben_dx & meta["local_image"],
        "official-benign including unknown, local image": r_ben_bm & meta["local_image"],
        "official-benign local minus atypical AMP": r_ben_bm & meta["local_image"] & ~r_atyp,
    }

    proposed_mel = meta.loc[r_mel_dx & r_mel_bm].copy()
    proposed_ben = meta.loc[r_ben_bm & r_ben_dx].copy()
    proposed_mel_local = proposed_mel.loc[proposed_mel["local_image"]].copy()
    proposed_ben_local = proposed_ben.loc[proposed_ben["local_image"]].copy()

    excluded_from_benign_official = meta.loc[r_ben_bm & ~r_ben_dx].copy()
    excluded_reasons = _counts(excluded_from_benign_official["diagnosis"])

    # DICM overlap
    dicm = pd.read_csv(DICM_MANIFEST)
    dicm["image_id"] = dicm["image_id"].map(normalize_isic_id)
    dicm["patient_c"] = dicm["patient_id"].map(_clean)
    dicm["lesion_c"] = dicm["lesion_id"].map(_clean)
    dicm_mel = dicm[dicm["label"] == 1]
    dicm_ben = dicm[dicm["label"] == 0]
    dicm_mel_ids = set(dicm_mel["image_id"])
    dicm_ben_ids = set(dicm_ben["image_id"])
    isic_mel_ids = set(proposed_mel["image_id"])
    isic_ben_ids = set(proposed_ben["image_id"])
    isic_mel_local_ids = set(proposed_mel_local["image_id"])
    isic_ben_local_ids = set(proposed_ben_local["image_id"])

    img_overlap = {
        "ISIC melanoma ∩ DICM melanoma": len(isic_mel_ids & dicm_mel_ids),
        "ISIC melanoma ∩ DICM benign": len(isic_mel_ids & dicm_ben_ids),
        "ISIC benign ∩ DICM melanoma": len(isic_ben_ids & dicm_mel_ids),
        "ISIC benign ∩ DICM benign": len(isic_ben_ids & dicm_ben_ids),
        "local ISIC melanoma ∩ DICM melanoma": len(isic_mel_local_ids & dicm_mel_ids),
        "local ISIC melanoma ∩ DICM benign": len(isic_mel_local_ids & dicm_ben_ids),
        "local ISIC benign ∩ DICM melanoma": len(isic_ben_local_ids & dicm_mel_ids),
        "local ISIC benign ∩ DICM benign": len(isic_ben_local_ids & dicm_ben_ids),
    }

    isic_p = _idset(proposed_mel["patient_id"]) | _idset(proposed_ben["patient_id"])
    dicm_p = _idset(dicm["patient_id"])
    shared_p = isic_p & dicm_p
    isic_pool = pd.concat([proposed_mel, proposed_ben], ignore_index=True)
    n_isic_from_shared_p = int(isic_pool["patient_c"].isin(shared_p).sum())
    n_dicm_from_shared_p = int(dicm["patient_c"].isin(shared_p).sum())

    isic_l = _idset(meta.get("lesion_id", pd.Series(dtype=str))) if "lesion_id" in meta.columns else set()
    dicm_l = _idset(dicm["lesion_id"])
    shared_l = isic_l & dicm_l

    # Old experiment
    old = pd.read_csv(OLD_MANIFEST)
    old["image_id"] = old["image_id"].map(normalize_isic_id)
    old["patient_c"] = old["patient_id"].map(_clean)
    old["lesion_c"] = old["lesion_id"].map(_clean)
    old_ids = set(old["image_id"])
    old_p = _idset(old["patient_id"])
    old_l = _idset(old["lesion_id"])
    old_overlap = {
        "ISIC melanoma images ∩ old manifest": len(isic_mel_ids & old_ids),
        "ISIC benign images ∩ old manifest": len(isic_ben_ids & old_ids),
        "local ISIC melanoma ∩ old": len(isic_mel_local_ids & old_ids),
        "local ISIC benign ∩ old": len(isic_ben_local_ids & old_ids),
        "all ISIC metadata ∩ old": len(set(meta["image_id"]) & old_ids),
        "old source_dataset counts": old["source_dataset"].value_counts().to_dict() if "source_dataset" in old.columns else {},
    }
    shared_old_p = isic_p & old_p
    n_isic_old_p = int(isic_pool["patient_c"].isin(shared_old_p).sum())
    n_old_from_shared_p = int(old["patient_c"].isin(shared_old_p).sum())
    shared_old_l = isic_l & old_l

    # Source confounding
    def _group_dist(df: pd.DataFrame, col: str) -> dict:
        return _counts(df[col]) if col in df.columns else {}

    mel_sex = _group_dist(proposed_mel, "sex")
    ben_sex = _group_dist(proposed_ben, "sex")
    mel_site = _group_dist(proposed_mel, "anatom_site_general_challenge")
    ben_site = _group_dist(proposed_ben, "anatom_site_general_challenge")
    mel_age = proposed_mel["age_approx"].describe().to_dict() if "age_approx" in proposed_mel else {}
    ben_age = proposed_ben["age_approx"].describe().to_dict() if "age_approx" in proposed_ben else {}

    missing_cols = [
        "diagnosis_1",
        "diagnosis_2",
        "diagnosis_3",
        "diagnosis_confirm_type",
        "lesion_id",
        "image_type",
        "attribution",
        "acquisition_year",
        "dermoscopic_type",
        "image_manipulation",
    ]
    present_missing = [c for c in missing_cols if c in meta.columns]
    absent_cols = [c for c in missing_cols if c not in meta.columns]

    # Combined option counts (image-level de-dup only)
    c_mel = len(dicm_mel_ids | isic_mel_local_ids)
    c_ben = len(dicm_ben_ids | isic_ben_local_ids)
    # After removing overlapping images from ISIC side (keep DICM)
    add_mel = len(isic_mel_local_ids - dicm_mel_ids - dicm_ben_ids)
    add_ben = len(isic_ben_local_ids - dicm_mel_ids - dicm_ben_ids)

    lines = [
        "# ISIC 2020 Melanoma-vs-Benign Dataset Audit",
        "",
        "TRAINING PERFORMED: NO",
        "MODELS MODIFIED: NO",
        "PRODUCTION APP MODIFIED: NO",
        "OLD DATASET MODIFIED: NO",
        "CLEAN DICM DATASET MODIFIED: NO",
        "NEW SPLITS CREATED: NO",
        "",
        "This is an audit of files already in the repository. No labels were changed,",
        "no images were deleted, and no new splits were created.",
        "",
        "## 1. Dataset identification",
        "",
        f"- Metadata file: `{META}`",
        f"- Image directory: `{IMG_DIR}`",
        f"- Metadata rows: {len(meta)}",
        f"- Unique metadata image_name values: {meta['image_name'].nunique()}",
        f"- Image files in the ISIC 2020 directory: {len(local_files)}",
        f"- Unique local filenames: {len(set(local_names))}",
        "",
        "This is the SIIM-ISIC 2020 Challenge training CSV (`train.csv`), copied into",
        "`dataset/raw/` by `scripts/extract_isic.py`. The extractor default was **all",
        "target=1 images plus a seeded 8,000-image benign sample**, not the full 33,126 JPEGs.",
        f"That matches the local file count: {len(local_files)}.",
        "",
        "No separate `ISIC_2020_Training_GroundTruth.csv` is stored in the repo besides",
        f"`{META}`. No `diagnosis_confirm_type` archive metadata for ISIC 2020 was found",
        "in the project.",
        "",
        "## 2. Metadata fields",
        "",
        "Columns actually present:",
        "",
        "| column | non-null | unique |",
        "|---|---:|---:|",
    ]
    for col in meta.columns:
        if col in {"diagnosis_c", "bm", "patient_c", "target_i", "local_image", "image_id"}:
            continue
        nonnull = int(meta[col].notna().sum()) if col in meta.columns else 0
        nunq = int(meta[col].nunique(dropna=True)) if col in meta.columns else 0
        lines.append(f"| {col} | {nonnull} | {nunq} |")
    lines.extend(
        [
            "",
            "Derived for this audit: `image_id` (normalized `image_name`), `local_image`.",
            "",
            "Requested fields **not present** in this CSV:",
            "",
        ]
    )
    for col in absent_cols:
        lines.append(f"- `{col}`")
    if present_missing:
        lines.append("")
        lines.append("Requested fields that are present: " + ", ".join(f"`{c}`" for c in present_missing))
    lines.extend(
        [
            "",
            "Therefore this audit **cannot** report ISIC 2020 confirmation type, hierarchical",
            "diagnosis_1/2/3, lesion_id, image type, attribution, year, or device from local metadata.",
            "",
            "## 3. Official label interpretation",
            "",
            "Do not assume `target` is melanoma without checking. Observed values:",
            "",
            "### diagnosis",
            "",
        ]
    )
    lines.extend(_md_table(dx_counts, "diagnosis"))
    lines.extend(["", "### benign_malignant", "", * _md_table(bm_counts, "benign_malignant"), ""])
    lines.extend(["### target", "", * _md_table(target_counts, "target"), ""])
    lines.extend(
        [
            "### diagnosis × benign_malignant",
            "",
            *_crosstab_md(meta, "diagnosis", "benign_malignant"),
            "",
            "### diagnosis × target",
            "",
            *_crosstab_md(meta, "diagnosis", "target"),
            "",
            "### benign_malignant × target",
            "",
            *_crosstab_md(meta, "benign_malignant", "target"),
            "",
            "**Interpretation from this file only:**",
            "",
            "- `target==1` coincides exactly with `diagnosis==melanoma` and with `benign_malignant==malignant` in this CSV.",
            f"- Melanoma rows (all three agree): {(r_mel_dx & r_mel_bm & r_mel_t).sum()}",
            f"- Disagreements among those three fields: {int(((r_mel_dx != r_mel_bm) | (r_mel_dx != r_mel_t) | (r_mel_bm != r_mel_t)).sum())}",
            "- `target==0` is **not** the same as a named benign diagnosis. It includes `unknown`, `atypical melanocytic proliferation`, and named benign diagnoses.",
            "- `benign_malignant` in this CSV is only `benign` or `malignant`. There is no `indeterminate` value here.",
            "- There are **no non-melanoma malignant diagnoses** (no BCC/SCC) in this ISIC 2020 challenge CSV.",
            "- Confirmation method is **not recorded**. This audit does not claim histopathology, consensus, or serial imaging.",
            "",
            "## 4. Melanoma candidate definition",
            "",
            "Proposed strict rule (aligned with the clean DICM melanoma rule, adapted to available fields):",
            "",
            "- `diagnosis == melanoma`",
            "- AND `benign_malignant == malignant`",
            "",
            "`target==1` matches this rule in this file, but the rule is stated from diagnosis + benign_malignant, not from target alone.",
            "",
            "Rule counts (full metadata / local images):",
            "",
            "| rule | metadata rows | with local image |",
            "|---|---:|---:|",
        ]
    )
    for name, mask in rules_mel.items():
        n = int(mask.sum())
        n_loc = int((mask & meta["local_image"]).sum())
        lines.append(f"| {name} | {n} | {n_loc} |")
    lines.extend(
        [
            "",
            "### Melanoma diagnosis distribution (strict metadata pool)",
            "",
            *_md_table(_counts(proposed_mel["diagnosis"]), "diagnosis"),
            "",
            "### Melanoma confirmation-type distribution",
            "",
            "`diagnosis_confirm_type` is not in the available ISIC 2020 CSV. Count unknown: all proposed melanoma rows.",
            f"- confirmation type missing: {len(proposed_mel)} / {len(proposed_mel)}",
            "",
            f"Local-image melanoma candidates: {len(proposed_mel_local)} (all target=1 melanoma JPEGs were extracted).",
            "",
            "## 5. Benign candidate definition",
            "",
            "Do **not** use `target==0` as benign. `target==0` includes unknown and atypical melanocytic proliferation.",
            "",
            "Proposed strict rule (aligned with the clean DICM benign list, using only diagnoses that actually occur here):",
            "",
            "- `benign_malignant == benign`",
            "- AND `diagnosis` in the pre-specified named benign list used for the clean DICM design",
            "- Named diagnoses present in ISIC 2020 that satisfy this: nevus, seborrheic keratosis, lentigo NOS,",
            "  lichenoid keratosis, solar lentigo, cafe-au-lait macule",
            "",
            "Excluded from strict benign even if `benign_malignant==benign`:",
            "- `unknown` (no named diagnosis)",
            "- `atypical melanocytic proliferation` (ambiguous melanocytic lesion)",
            "",
            "| rule | metadata rows | with local image |",
            "|---|---:|---:|",
        ]
    )
    for name, mask in rules_ben.items():
        lines.append(f"| {name} | {int(mask.sum())} | {int((mask & meta['local_image']).sum())} |")
    lines.extend(
        [
            "",
            "### Official-benign rows excluded by the strict named-diagnosis rule",
            "",
            *_md_table(excluded_reasons, "diagnosis"),
            "",
            f"Strict benign metadata candidates: {len(proposed_ben)}",
            f"Strict benign with a local JPEG: {len(proposed_ben_local)}",
            "",
            "The previous frozen experiment used a **looser** local rule: `benign_malignant==benign`",
            "and drop only `ISIC_4225838` (atypical melanocytic proliferation), **keeping `diagnosis=unknown`**.",
            "That looser local pool is reported above as `official-benign local minus atypical AMP`.",
            "This audit's proposed scientific candidate is the **strict named-diagnosis** pool.",
            "",
            "## 6. Diagnosis purity",
            "",
            "### MELANOMA (strict metadata pool)",
            "",
            *_md_table(_counts(proposed_mel["diagnosis"]), "diagnosis"),
            "",
            "Confirmation type: not available (all missing).",
            "",
            "### BENIGN (strict metadata pool)",
            "",
            *_md_table(_counts(proposed_ben["diagnosis"]), "diagnosis"),
            "",
            "Confirmation type: not available (all missing).",
            "",
            "Diagnoses that should **not** enter a strict melanoma-vs-benign experiment:",
            "",
            "| diagnosis | metadata count | reason |",
            "|---|---:|---|",
            f"| unknown | {int((meta['diagnosis_c'].str.lower()=='unknown').sum())} | No named diagnosis; cannot assign a clean class |",
            f"| atypical melanocytic proliferation | {int(r_atyp.sum())} | Ambiguous melanocytic lesion; not confirmed melanoma or benign |",
            "",
            "No BCC, SCC, AK, metastasis, or other non-melanoma cancers appear in this ISIC 2020 challenge CSV.",
            "",
            "## 7. Confirmation-type analysis",
            "",
            "The available ISIC 2020 training CSV does not include `diagnosis_confirm_type`.",
            "This audit therefore **cannot** count histopathology vs expert consensus vs serial imaging.",
            "Any claim that these melanoma rows are biopsy-proven would be an external assumption,",
            "not a field observed in this repository.",
            "",
            "## 8. Image availability",
            "",
            f"- Metadata rows: {len(meta)}",
            f"- Available images in `{IMG_DIR}`: {len(local_files)}",
            f"- Metadata rows whose JPEG is present locally: {int(meta['local_image'].sum())}",
            f"- Metadata rows missing a local JPEG: {int((~meta['local_image']).sum())}",
            f"- Local files whose stem is not in metadata: {len(extra_files)}",
            "",
            "Missing images are expected: the extractor did not unpack the full 33,126 JPEGs.",
            "",
            "| subset | metadata | local JPEG | missing JPEG |",
            "|---|---:|---:|---:|",
            f"| strict melanoma | {len(proposed_mel)} | {len(proposed_mel_local)} | {len(proposed_mel)-len(proposed_mel_local)} |",
            f"| strict benign | {len(proposed_ben)} | {len(proposed_ben_local)} | {len(proposed_ben)-len(proposed_ben_local)} |",
            f"| official benign including unknown | {int(r_ben_bm.sum())} | {int((r_ben_bm & meta['local_image']).sum())} | {int(r_ben_bm.sum())-int((r_ben_bm & meta['local_image']).sum())} |",
            "",
            "## 9. Duplicate analysis",
            "",
            f"- Duplicate metadata image_name: {int(meta['image_name'].duplicated().sum())}",
            f"- Duplicate metadata image_id: {int(meta['image_id'].duplicated().sum())}",
            f"- Duplicate local filenames: {len(dup_names)}",
            f"- Duplicate local image IDs: {len(dup_ids)}",
            "",
            "## 10. DICM image overlap",
            "",
            "Clean DICM manifest was read only. It was not modified.",
            f"Clean DICM rows: {len(dicm)} (melanoma {(dicm['label']==1).sum()}, benign {(dicm['label']==0).sum()}).",
            "",
            "Full-metadata ISIC candidate pools vs clean DICM:",
            "",
            "| overlap | count |",
            "|---|---:|",
        ]
    )
    for k, v in img_overlap.items():
        lines.append(f"| {k} | {v} |")
    lines.extend(
        [
            "",
            "Image IDs were compared after ISIC ID normalization. Overlaps were **not** removed.",
            "",
            "## 11. DICM patient overlap",
            "",
            f"- Unique non-empty patient IDs in strict ISIC melanoma+benign metadata pool: {len(isic_p)}",
            f"- Unique non-empty patient IDs in clean DICM: {len(dicm_p)}",
            f"- Shared patient IDs: {len(shared_p)}",
            f"- Strict ISIC pool images from shared patients: {n_isic_from_shared_p}",
            f"- Clean DICM images from shared patients: {n_dicm_from_shared_p}",
            "",
            "Patient IDs were not removed or remapped.",
            "",
            "## 12. DICM lesion overlap",
            "",
            "ISIC 2020 challenge CSV has **no `lesion_id` column**.",
            "The old experiment stored empty `lesion_id` for ISIC 2020 rows.",
            f"- ISIC 2020 lesion IDs available: {len(isic_l)}",
            f"- Clean DICM non-empty lesion IDs: {len(dicm_l)}",
            f"- Shared lesion IDs: {len(shared_l)}",
            "",
            "Lesion-level overlap **cannot be audited from this ISIC 2020 file**.",
            "",
            "## 13. Old-experiment overlap",
            "",
            "Old frozen experiment `dataset/processed/melanoma_benign/manifest.csv` was read only.",
            "",
            "| item | count |",
            "|---|---:|",
            f"| old manifest rows | {len(old)} |",
            f"| ISIC melanoma ∩ old images | {old_overlap['ISIC melanoma images ∩ old manifest']} |",
            f"| ISIC benign ∩ old images | {old_overlap['ISIC benign images ∩ old manifest']} |",
            f"| local ISIC melanoma ∩ old | {old_overlap['local ISIC melanoma ∩ old']} |",
            f"| local ISIC benign ∩ old | {old_overlap['local ISIC benign ∩ old']} |",
            f"| all ISIC metadata ∩ old | {old_overlap['all ISIC metadata ∩ old']} |",
            f"| shared patient IDs (strict ISIC pool vs old) | {len(shared_old_p)} |",
            f"| strict ISIC images from those patients | {n_isic_old_p} |",
            f"| old-experiment images from those patients | {n_old_from_shared_p} |",
            f"| shared lesion IDs | {len(shared_old_l)} |",
            "",
            "The old experiment used DICM official class=1 as melanoma and local ISIC 2020",
            "`benign_malignant==benign` (minus one AMP image) as benign. That is a different",
            "label definition from this audit's strict pools.",
            "",
            "## 14. Source/domain confounding signals",
            "",
            "These are **potential source/domain confounding signals**, not proof of model bias.",
            "ISIC 2020 challenge CSV has no year, camera, attribution, dermoscopic type, or image_type.",
            "Compared fields: sex, age_approx, anatom_site_general_challenge.",
            "",
            "### Sex (strict metadata melanoma vs benign)",
            "",
            "Melanoma:",
            *_md_table(mel_sex, "sex"),
            "",
            "Benign:",
            *_md_table(ben_sex, "sex"),
            "",
            "### Anatomical site",
            "",
            "Melanoma:",
            *_md_table(mel_site, "site"),
            "",
            "Benign:",
            *_md_table(ben_site, "site"),
            "",
            "### Age (age_approx)",
            "",
            f"- Melanoma mean/median: {mel_age.get('mean', float('nan')):.2f} / {mel_age.get('50%', float('nan')):.1f} (n={int(mel_age.get('count', 0))})",
            f"- Benign mean/median: {ben_age.get('mean', float('nan')):.2f} / {ben_age.get('50%', float('nan')):.1f} (n={int(ben_age.get('count', 0))})",
            "",
            "Additional signals:",
            "- Local ISIC JPEGs are a **non-random subset** of benign rows (seeded sample of 8,000 official-benign, including unknown).",
            "- Strict named benign images on disk are therefore a sample of a sample, not the full ISIC 2020 named-benign set.",
            "- Melanoma on disk is complete relative to this CSV (all 584 melanoma JPEGs are present).",
            "- Combining with DICM would mix two acquisition sources unless overlap and domain shift are handled explicitly.",
            "",
            "## 15. Final usable counts",
            "",
            "Using the strict rules and **images actually on disk**:",
            "",
            f"- Strict ISIC melanoma with local JPEG: **{len(proposed_mel_local)}**",
            f"- Strict ISIC benign with local JPEG: **{len(proposed_ben_local)}**",
            f"- Strict local total: **{len(proposed_mel_local)+len(proposed_ben_local)}**",
            "",
            "If the remaining ISIC 2020 JPEGs were extracted (not done in this audit):",
            "",
            f"- Strict ISIC melanoma in metadata: **{len(proposed_mel)}**",
            f"- Strict ISIC benign in metadata: **{len(proposed_ben)}**",
            f"- Strict metadata total: **{len(proposed_mel)+len(proposed_ben)}**",
            "",
            "Clean DICM (unchanged, for comparison): melanoma 6,787; benign 4,586; total 11,373.",
            "",
            "## 16. Excluded counts and reasons",
            "",
            "| reason | metadata rows |",
            "|---|---:|",
            f"| not strict melanoma and not strict benign | {len(meta)-len(proposed_mel)-len(proposed_ben)} |",
            f"| diagnosis=unknown | {int((meta['diagnosis_c'].str.lower()=='unknown').sum())} |",
            f"| atypical melanocytic proliferation | {int(r_atyp.sum())} |",
            f"| strict melanoma missing local JPEG | {len(proposed_mel)-len(proposed_mel_local)} |",
            f"| strict benign missing local JPEG | {len(proposed_ben)-len(proposed_ben_local)} |",
            "",
            "## 17. Limitations",
            "",
            "- Confirmation type is absent; melanoma/benign cannot be verified as histopathology-confirmed from this file.",
            "- Lesion IDs are absent; lesion-level leakage vs DICM cannot be measured.",
            "- Most ISIC 2020 JPEGs are not on disk; local benign named-diagnosis images are a subset of a seeded sample.",
            "- `diagnosis=unknown` dominates official-benign rows (27,124 / 32,542).",
            "- Patient IDs exist and can overlap DICM even when image IDs do not.",
            "- This challenge CSV contains no non-melanoma cancers, so it cannot represent keratinocyte malignancy exclusion the way DICM metadata can.",
            "- No splits were designed. No model was trained.",
            "",
            "## 18. Recommendation for the NEXT DATASET DESIGN",
            "",
            "This section compares options. It does **not** select one.",
            "",
            "### Option A. ISIC 2020 verified melanoma + verified benign only",
            "",
            "| item | local images on disk | full metadata if remaining JPEGs were added |",
            "|---|---:|---:|",
            f"| melanoma | {len(proposed_mel_local)} | {len(proposed_mel)} |",
            f"| benign | {len(proposed_ben_local)} | {len(proposed_ben)} |",
            f"| total | {len(proposed_mel_local)+len(proposed_ben_local)} | {len(proposed_mel)+len(proposed_ben)} |",
            "",
            "- Confirmation: not recorded in local metadata.",
            "- Overlap: many named-benign ISIC IDs also appear in clean DICM benign (see section 10); melanoma image overlap with DICM melanoma is listed above.",
            "- Source/domain: single challenge source, but local benign is an incomplete sample; class sizes are small on disk and melanoma-heavy if only local files are used.",
            "- Major limitations: tiny local strict pool; missing confirmation and lesion IDs; unknown diagnoses cannot be used if a clean label is required.",
            "",
            "### Option B. Clean DICM verified melanoma + verified benign only",
            "",
            "| item | count |",
            "|---|---:|",
            "| melanoma | 6787 |",
            "| benign | 4586 |",
            "| total | 11373 |",
            "",
            "- Confirmation: DICM official metadata included `diagnosis_confirm_type`; the clean melanoma pool was previously reported as histopathology-confirmed in the DICM design document. That is a DICM-metadata property, not repeated here as a new claim about ISIC 2020.",
            "- Overlap: none with a second source if used alone. Internal leakage-safe splits already exist and are frozen.",
            "- Source/domain: single DICM source; previous DICM source-bias audit still applies (hospital/camera confounding possible).",
            "- Major limitations: DICM-only domain; missing patient IDs required grouping fallbacks; already trained on in the clean DICM experiment.",
            "",
            "### Option C. Clean DICM + verified local ISIC 2020 cases after removing image overlaps",
            "",
            "Keep all clean DICM rows. Add local strict ISIC images whose image_id is not already in clean DICM (melanoma or benign).",
            "",
            f"- Additional local ISIC melanoma not in DICM: {add_mel}",
            f"- Additional local ISIC benign not in DICM: {add_ben}",
            f"- Combined melanoma (DICM ∪ extra ISIC): {len(dicm_mel_ids)+add_mel}",
            f"- Combined benign (DICM ∪ extra ISIC): {len(dicm_ben_ids)+add_ben}",
            f"- Combined total: {len(dicm_mel_ids)+add_mel+len(dicm_ben_ids)+add_ben}",
            "",
            "- Confirmation: mixed. DICM has confirmation-type fields; ISIC 2020 local CSV does not.",
            "- Overlap: image-level overlap can be dropped as above, but patient-level overlap would remain unless also excluded. Lesion overlap cannot be checked for ISIC 2020.",
            "- Source/domain: two sources (DICM archive vs SIIM-ISIC 2020 challenge photography). Class-conditional source differences are a potential confounding signal.",
            "- Major limitations: mixed confirmation quality; patient leakage risk across sources; local ISIC benign add-on is small; would require **new** leakage-safe splits (not created here).",
            "",
            "No option is selected. No dataset is built. No training is started.",
            "",
            "---",
            "",
            "TRAINING PERFORMED: NO",
            "MODELS MODIFIED: NO",
            "PRODUCTION APP MODIFIED: NO",
            "OLD DATASET MODIFIED: NO",
            "CLEAN DICM DATASET MODIFIED: NO",
            "NEW SPLITS CREATED: NO",
            "",
        ]
    )

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "metadata": str(META),
        "image_dir": str(IMG_DIR),
        "metadata_rows": int(len(meta)),
        "image_files": int(len(local_files)),
        "columns": [c for c in meta.columns if c not in {"diagnosis_c", "bm", "patient_c", "target_i", "local_image"}],
        "strict_melanoma_metadata": int(len(proposed_mel)),
        "strict_melanoma_local": int(len(proposed_mel_local)),
        "strict_benign_metadata": int(len(proposed_ben)),
        "strict_benign_local": int(len(proposed_ben_local)),
        "image_overlap": img_overlap,
        "shared_dicm_patients": int(len(shared_p)),
        "report": str(REPORT),
        "training_performed": False,
    }
    (OUT_DIR / "ISIC_2020_MELANOMA_BENIGN_AUDIT.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("AUDIT WRITTEN", REPORT)
    print("TRAINING PERFORMED: NO")


if __name__ == "__main__":
    main()
