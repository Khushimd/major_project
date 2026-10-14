"""
Phase A/B audit and freeze of a Derm7pt external-validation cohort
for the FROZEN old Exp2-C production ensemble.

Does NOT:
- retrain, fine-tune, or load model weights for inference
- change preprocessing, ensemble weights, or the 0.30 threshold
- modify production artifacts, the old experiment, or frozen CSVs
- add Derm7pt to any training dataset
- tune anything on Derm7pt

TRAINING PERFORMED: NO
INFERENCE PERFORMED: NO
MODELS MODIFIED: NO
PRODUCTION APP MODIFIED: NO
OLD EXPERIMENT MODIFIED: NO
FROZEN CSVS MODIFIED: NO
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd

import config


def average_hash_hex(path: str, size: int = 8) -> str:
    """8x8 average hash. Local copy so this audit does not import training/TF."""
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return ""
    arr = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    bits = (arr > float(arr.mean())).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"

DERM7PT_DIR = ROOT / "dataset" / "external" / "derm7pt"
META_CSV = DERM7PT_DIR / "meta" / "meta.csv"
IMAGE_DIR = DERM7PT_DIR / "images"
INDEX_FILES = {
    "train": DERM7PT_DIR / "meta" / "train_indexes.csv",
    "valid": DERM7PT_DIR / "meta" / "valid_indexes.csv",
    "test": DERM7PT_DIR / "meta" / "test_indexes.csv",
}

OLD_SPLIT_DIR = config.PROCESSED_DATA_DIR / "melanoma_benign"
OLD_CSVS = {
    "manifest": OLD_SPLIT_DIR / "manifest.csv",
    "train": OLD_SPLIT_DIR / "train.csv",
    "val": OLD_SPLIT_DIR / "val.csv",
    "test": OLD_SPLIT_DIR / "test.csv",
}

EFF_PATH = (
    ROOT
    / "models"
    / "experiments"
    / "internal_optimization"
    / "exp2_loss_comparison"
    / "exp2c_bce_smooth"
    / "efficientnetb0_bce_smooth.keras"
)
MOB_PATH = (
    ROOT
    / "models"
    / "experiments"
    / "internal_optimization"
    / "exp2_loss_comparison"
    / "exp2c_bce_smooth"
    / "mobilenetv2_bce_smooth.keras"
)
FROZEN_INTERNAL = ROOT / "models" / "evaluation" / "final_test" / "FINAL_TEST_EVALUATION.json"

OUT_DIR = ROOT / "models" / "experiments" / "old_model_derm7pt_external"

EMPTY = {"", "nan", "none", "<na>", "null", "na"}

# Clinically defensible melanoma vs clearly non-melanoma mapping.
# Inspected from meta.csv before this dict was finalized.
POSITIVE_DIAGNOSES = {
    "melanoma",
    "melanoma (in situ)",
    "melanoma (less than 0.76 mm)",
    "melanoma (0.76 to 1.5 mm)",
    "melanoma (more than 1.5 mm)",
    "melanoma metastasis",
}
NEGATIVE_DIAGNOSES = {
    "basal cell carcinoma",
    "blue nevus",
    "clark nevus",
    "combined nevus",
    "congenital nevus",
    "dermal nevus",
    "dermatofibroma",
    "lentigo",
    "recurrent nevus",
    "reed or spitz nevus",
    "seborrheic keratosis",
    "vascular lesion",
}
EXCLUDED_DIAGNOSES = {
    "melanosis": (
        "ambiguous_not_melanoma_not_clearly_benign: melanosis is not a melanoma "
        "diagnosis and is not automatically assigned to the non-melanoma class"
    ),
    "miscellaneous": (
        "ambiguous_unspecified_mixed_category: miscellaneous is not a named "
        "diagnosis and cannot be defensibly assigned"
    ),
}

FROZEN_ENSEMBLE = {
    "efficientnet_weight": 0.75,
    "mobilenet_weight": 0.25,
    "threshold": 0.30,
    "source": "models/evaluation/final_test/FINAL_TEST_EVALUATION.json",
}


def _clean(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() in EMPTY else text


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _norm_name(value: str) -> str:
    return Path(_clean(value).replace("\\", "/")).name.lower()


def _norm_stem(value: str) -> str:
    return Path(_norm_name(value)).stem.lower()


def _hamming_hex(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def resolve_image(relpath: str) -> Path | None:
    rel = _clean(relpath).replace("\\", "/")
    if not rel:
        return None
    candidate = IMAGE_DIR / rel
    if candidate.is_file():
        return candidate.resolve()
    # Case-insensitive fallback (Windows usually finds the file anyway).
    parts = Path(rel).parts
    cursor = IMAGE_DIR
    for part in parts:
        if not cursor.is_dir():
            return None
        match = next((p for p in cursor.iterdir() if p.name.lower() == part.lower()), None)
        if match is None:
            return None
        cursor = match
    return cursor.resolve() if cursor.is_file() else None


def load_official_splits(n_rows: int) -> dict[int, str]:
    membership: dict[int, str] = {}
    seen: dict[str, set[int]] = {}
    for split, path in INDEX_FILES.items():
        idx = pd.read_csv(path)
        values = [int(v) for v in idx.iloc[:, 0].tolist()]
        seen[split] = set(values)
        for value in values:
            if value in membership:
                raise RuntimeError(f"Derm7pt official index {value} appears in multiple splits")
            if value < 0 or value >= n_rows:
                raise RuntimeError(f"Derm7pt official index {value} out of range for {n_rows} rows")
            membership[value] = split
    if len(membership) != n_rows:
        missing = sorted(set(range(n_rows)) - set(membership))
        raise RuntimeError(f"Official Derm7pt indexes do not cover all rows: missing {missing[:10]}")
    return membership


def map_diagnosis(diagnosis: str) -> dict:
    if diagnosis in POSITIVE_DIAGNOSES:
        return {
            "binary_label": 1,
            "binary_label_name": "melanoma",
            "exclusion_status": "included",
            "exclusion_reason": "",
            "mapping_bucket": "positive_melanoma",
        }
    if diagnosis in NEGATIVE_DIAGNOSES:
        return {
            "binary_label": 0,
            "binary_label_name": "non_melanoma",
            "exclusion_status": "included",
            "exclusion_reason": "",
            "mapping_bucket": "negative_clearly_non_melanoma",
        }
    if diagnosis in EXCLUDED_DIAGNOSES:
        return {
            "binary_label": "",
            "binary_label_name": "",
            "exclusion_status": "excluded",
            "exclusion_reason": EXCLUDED_DIAGNOSES[diagnosis],
            "mapping_bucket": "excluded_ambiguous",
        }
    return {
        "binary_label": "",
        "binary_label_name": "",
        "exclusion_status": "excluded",
        "exclusion_reason": f"unmapped_diagnosis:{diagnosis}",
        "mapping_bucket": "excluded_unmapped",
    }


def fingerprint_path(path: Path) -> dict:
    return {
        "path": str(path.resolve()) if path.is_file() else str(path),
        "exists": bool(path.is_file()),
        "size": int(path.stat().st_size) if path.is_file() else None,
        "sha256": _sha256(path) if path.is_file() else None,
    }


def load_old_development() -> pd.DataFrame:
    frames = []
    for split in ("train", "val", "test"):
        df = pd.read_csv(OLD_CSVS[split])
        df["old_split"] = split
        frames.append(df)
    old = pd.concat(frames, ignore_index=True)
    manifest = pd.read_csv(OLD_CSVS["manifest"])
    if len(old) != len(manifest):
        raise RuntimeError(
            f"Old split union {len(old)} != manifest {len(manifest)}; refusing to continue"
        )
    old["image_path"] = old["image_path"].astype(str)
    old["image_id"] = old["image_id"].map(_clean)
    old["isic_id"] = old.get("isic_id", pd.Series([""] * len(old))).map(_clean)
    old["patient_id"] = old.get("patient_id", pd.Series([""] * len(old))).map(_clean)
    old["lesion_id"] = old.get("lesion_id", pd.Series([""] * len(old))).map(_clean)
    old["group_id"] = old.get("group_id", pd.Series([""] * len(old))).map(_clean)
    old["filename"] = old["image_path"].map(_norm_name)
    old["stem"] = old["image_path"].map(_norm_stem)
    missing = [p for p in old["image_path"] if not Path(p).is_file()]
    if missing:
        raise RuntimeError(f"STOP: {len(missing)} old development images missing, e.g. {missing[:5]}")
    return old


def hash_old_images(old: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, row in old.iterrows():
        path = Path(row["image_path"])
        rec = {
            "old_split": row["old_split"],
            "image_id": row["image_id"],
            "isic_id": row["isic_id"],
            "patient_id": row["patient_id"],
            "lesion_id": row["lesion_id"],
            "group_id": row["group_id"],
            "image_path": str(path),
            "filename": row["filename"],
            "stem": row["stem"],
            "size": int(path.stat().st_size),
            "sha256": _sha256(path),
            "ahash": average_hash_hex(str(path)) or "",
        }
        rows.append(rec)
        if (len(rows) % 2000) == 0:
            print(f"Hashed {len(rows)}/{len(old)} old development images", flush=True)
    print(f"Hashed {len(rows)}/{len(old)} old development images", flush=True)
    return pd.DataFrame(rows)


def build_audit() -> None:
    started = datetime.now(timezone.utc).isoformat()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not META_CSV.is_file():
        raise RuntimeError(f"STOP: Derm7pt meta.csv not found: {META_CSV}")
    if not IMAGE_DIR.is_dir():
        raise RuntimeError(f"STOP: Derm7pt images directory not found: {IMAGE_DIR}")
    for path in (*OLD_CSVS.values(), EFF_PATH, MOB_PATH, FROZEN_INTERNAL):
        if not Path(path).is_file():
            raise RuntimeError(f"STOP: required frozen file missing: {path}")

    meta = pd.read_csv(META_CSV)
    n_rows = len(meta)
    official_split = load_official_splits(n_rows)
    diagnosis_counts = {
        str(k): int(v) for k, v in meta["diagnosis"].fillna("<MISSING>").value_counts().items()
    }
    missing_values = {col: int(meta[col].isna().sum()) for col in meta.columns}
    case_ids = [_clean(v) for v in meta["case_id"].tolist()]
    nonempty_case_ids = [v for v in case_ids if v]

    print("Derm7pt diagnosis distribution:")
    for name, count in diagnosis_counts.items():
        print(f"  {count:4d}  {name}")
    print("Proposed mapping:")
    print("  POSITIVE (melanoma):", sorted(POSITIVE_DIAGNOSES))
    print("  NEGATIVE (clearly non-melanoma):", sorted(NEGATIVE_DIAGNOSES))
    print("  EXCLUDED:", EXCLUDED_DIAGNOSES)

    unknown = sorted(
        set(diagnosis_counts) - POSITIVE_DIAGNOSES - NEGATIVE_DIAGNOSES - set(EXCLUDED_DIAGNOSES)
    )
    if unknown:
        raise RuntimeError(f"STOP: unmapped Derm7pt diagnoses found: {unknown}")

    print("Loading old development splits (read-only)...", flush=True)
    old = load_old_development()
    print("Hashing old development images...", flush=True)
    old_fp = hash_old_images(old)

    old_sha = defaultdict(list)
    old_ahash = defaultdict(list)
    old_stem = defaultdict(list)
    old_name = defaultdict(list)
    old_id = defaultdict(list)
    old_records = old_fp.to_dict(orient="records")
    for rec in old_records:
        old_sha[rec["sha256"]].append(rec)
        if rec["ahash"]:
            old_ahash[rec["ahash"]].append(rec)
        old_stem[rec["stem"]].append(rec)
        old_name[rec["filename"]].append(rec)
        if rec["image_id"]:
            old_id[rec["image_id"].lower()].append(rec)
        if rec["isic_id"]:
            old_id[rec["isic_id"].lower()].append(rec)

    old_patient_ids = {v for v in old["patient_id"] if v}
    old_lesion_ids = {v for v in old["lesion_id"] if v}
    old_group_ids = {v for v in old["group_id"] if v}
    old_image_ids = {v for v in old["image_id"] if v} | {v for v in old["isic_id"] if v}
    old_id_lower = {v.lower() for v in old_image_ids}

    derm_name_counts = Counter()
    derm_hash_counts = Counter()
    image_rows = []
    overlap_rows = []

    print("Auditing Derm7pt dermoscopic images...", flush=True)
    for row_index, row in meta.iterrows():
        diagnosis = _clean(row["diagnosis"])
        mapping = map_diagnosis(diagnosis)
        derm_rel = _clean(row["derm"]).replace("\\", "/")
        clinic_rel = _clean(row["clinic"]).replace("\\", "/")
        derm_path = resolve_image(derm_rel)
        clinic_path = resolve_image(clinic_rel)
        image_exists = bool(derm_path and derm_path.is_file())
        sha256 = _sha256(derm_path) if image_exists else ""
        ahash = average_hash_hex(str(derm_path)) if image_exists else ""
        size = int(derm_path.stat().st_size) if image_exists else None
        derm_name = _norm_name(derm_rel)
        derm_stem = _norm_stem(derm_rel)
        if derm_name:
            derm_name_counts[derm_name] += 1
        if sha256:
            derm_hash_counts[sha256] += 1

        overlap_hits = []
        if derm_stem:
            for rec in old_id.get(derm_stem, []):
                overlap_hits.append(("filename_or_id", rec, None))
            for rec in old_stem.get(derm_stem, []):
                overlap_hits.append(("filename_or_id", rec, None))
        if derm_name:
            for rec in old_name.get(derm_name, []):
                overlap_hits.append(("filename", rec, None))
        if sha256 and sha256 in old_sha:
            for rec in old_sha[sha256]:
                overlap_hits.append(("sha256", rec, 0))
        if ahash:
            for old_hash, recs in old_ahash.items():
                dist = _hamming_hex(ahash, old_hash)
                if dist <= 6:
                    kind = "phash_exact" if dist == 0 else "phash_near"
                    for rec in recs:
                        overlap_hits.append((kind, rec, dist))

        # Deduplicate overlap records by old path + kind.
        uniq = {}
        for kind, rec, dist in overlap_hits:
            key = (kind, rec["image_path"])
            if key not in uniq or (dist is not None and uniq[key][2] is not None and dist < uniq[key][2]):
                uniq[key] = (kind, rec, dist)

        exclude_overlap = []
        report_only = []
        for kind, rec, dist in uniq.values():
            item = {
                "derm7pt_row_index": int(row_index),
                "case_num": int(row["case_num"]),
                "derm_relpath": derm_rel,
                "original_diagnosis": diagnosis,
                "match_type": kind,
                "hamming": dist,
                "old_split": rec["old_split"],
                "old_image_id": rec["image_id"],
                "old_image_path": rec["image_path"],
                "old_sha256": rec["sha256"],
                "old_ahash": rec["ahash"],
                "old_size": rec["size"],
                "derm7pt_sha256": sha256,
                "derm7pt_ahash": ahash,
                "derm7pt_size": size,
            }
            high_conf = False
            if kind in {"sha256", "filename", "filename_or_id"}:
                high_conf = True
            elif kind == "phash_exact" and size is not None and rec["size"] == size:
                high_conf = True
            item["high_confidence"] = high_conf
            item["action"] = "exclude_from_external_eval" if high_conf else "report_only"
            overlap_rows.append(item)
            if high_conf:
                exclude_overlap.append(item)
            else:
                report_only.append(item)

        overlap_status = "none"
        overlap_reason = ""
        if exclude_overlap:
            kinds = sorted({x["match_type"] for x in exclude_overlap})
            overlap_status = "high_confidence_duplicate"
            overlap_reason = (
                "excluded_overlap:"
                + ",".join(kinds)
                + "; old_ids="
                + ",".join(sorted({x["old_image_id"] for x in exclude_overlap}))
            )
        elif report_only:
            overlap_status = "phash_candidate_not_excluded"
            overlap_reason = "perceptual_candidate_hamming_" + ",".join(
                sorted({str(x["hamming"]) for x in report_only})
            )

        inclusion = "yes"
        inclusion_blockers = []
        if not image_exists:
            inclusion = "no"
            inclusion_blockers.append("missing_dermoscopic_image")
        if mapping["exclusion_status"] == "excluded":
            inclusion = "no"
            inclusion_blockers.append(mapping["exclusion_reason"])
        if overlap_status == "high_confidence_duplicate":
            inclusion = "no"
            inclusion_blockers.append(overlap_reason)

        image_rows.append(
            {
                "derm7pt_row_index": int(row_index),
                "case_num": int(row["case_num"]),
                "official_split": official_split[int(row_index)],
                "case_id": _clean(row["case_id"]),
                "derm_filename": Path(derm_rel).name,
                "derm_relpath": derm_rel,
                "derm_abspath": str(derm_path) if derm_path else "",
                "clinic_relpath": clinic_rel,
                "clinic_used_for_evaluation": "no",
                "original_diagnosis": diagnosis,
                "binary_label": mapping["binary_label"],
                "binary_label_name": mapping["binary_label_name"],
                "mapping_bucket": mapping["mapping_bucket"],
                "exclusion_status": mapping["exclusion_status"]
                if mapping["exclusion_status"] == "excluded"
                else ("excluded" if inclusion == "no" else "included"),
                "exclusion_reason": " | ".join(inclusion_blockers),
                "overlap_status": overlap_status,
                "overlap_reason": overlap_reason,
                "final_evaluation_inclusion": inclusion,
                "image_exists": image_exists,
                "file_size_bytes": size if size is not None else "",
                "sha256": sha256,
                "ahash": ahash or "",
                "sex": _clean(row["sex"]),
                "location": _clean(row["location"]),
                "elevation": _clean(row["elevation"]),
                "management": _clean(row["management"]),
                "seven_point_score": row["seven_point_score"],
                "notes": _clean(row["notes"]),
            }
        )

    manifest = pd.DataFrame(image_rows)
    overlap_df = pd.DataFrame(overlap_rows)
    if overlap_df.empty:
        overlap_df = pd.DataFrame(
            columns=[
                "derm7pt_row_index",
                "case_num",
                "derm_relpath",
                "original_diagnosis",
                "match_type",
                "hamming",
                "high_confidence",
                "action",
                "old_split",
                "old_image_id",
                "old_image_path",
            ]
        )

    duplicate_filenames = {k: int(v) for k, v in derm_name_counts.items() if v > 1}
    duplicate_hashes = {k: int(v) for k, v in derm_hash_counts.items() if v > 1}
    missing_images = manifest.loc[~manifest["image_exists"], "derm_relpath"].tolist()

    included = manifest[manifest["final_evaluation_inclusion"] == "yes"]
    excluded = manifest[manifest["final_evaluation_inclusion"] == "no"]
    n_melanoma = int((included["binary_label"] == 1).sum())
    n_non = int((included["binary_label"] == 0).sum())

    derm_case_ids = {v for v in manifest["case_id"] if v}
    patient_id_overlap = sorted(derm_case_ids & old_patient_ids)
    lesion_id_overlap = sorted(derm_case_ids & old_lesion_ids)
    group_id_overlap = sorted(derm_case_ids & old_group_ids)
    image_id_overlap = sorted({v.lower() for v in derm_case_ids} & old_id_lower)

    diagnosis_breakdown = []
    for diagnosis, count in diagnosis_counts.items():
        chunk = manifest[manifest["original_diagnosis"] == diagnosis]
        diagnosis_breakdown.append(
            {
                "diagnosis": diagnosis,
                "total_cases": int(count),
                "included": int((chunk["final_evaluation_inclusion"] == "yes").sum()),
                "excluded": int((chunk["final_evaluation_inclusion"] == "no").sum()),
                "mapping": map_diagnosis(diagnosis)["mapping_bucket"],
                "exclusion_reasons": sorted(
                    {r for r in chunk["exclusion_reason"].tolist() if r}
                ),
            }
        )

    frozen_internal = json.loads(FROZEN_INTERNAL.read_text(encoding="utf-8-sig"))
    frozen_csv_snapshot = {name: fingerprint_path(path) for name, path in OLD_CSVS.items()}
    model_snapshot = {
        "efficientnetb0": fingerprint_path(EFF_PATH),
        "mobilenetv2": fingerprint_path(MOB_PATH),
    }
    expected_eff = "02755e9c69e9f885b71696a278f807c9a93c29b88d59f7d252002f8bbbb62f87"
    expected_mob = "ad049149423ad1a5a73f9b1bd6dabd0765968ae532472b8afd5757eb7f33146f"
    if model_snapshot["efficientnetb0"]["sha256"] != expected_eff:
        raise RuntimeError("STOP: EfficientNet file hash does not match the frozen 92.28% evaluation")
    if model_snapshot["mobilenetv2"]["sha256"] != expected_mob:
        raise RuntimeError("STOP: MobileNet file hash does not match the frozen 92.28% evaluation")

    label_mapping = {
        "positive_class": 1,
        "positive_name": "melanoma",
        "negative_class": 0,
        "negative_name": "non_melanoma",
        "image_used": "dermoscopic_only",
        "clinical_image_used": False,
        "positive_diagnoses": sorted(POSITIVE_DIAGNOSES),
        "negative_diagnoses": sorted(NEGATIVE_DIAGNOSES),
        "excluded_diagnoses": EXCLUDED_DIAGNOSES,
        "notes": [
            "Melanoma in situ is included as melanoma.",
            "All listed melanoma thickness categories are included as melanoma.",
            "Melanoma metastasis (n=4) is included as melanoma because it is a melanoma diagnosis; it is reported separately because it is not a primary cutaneous lesion.",
            "Melanosis is excluded: not melanoma and not automatically treated as non-melanoma/benign.",
            "Miscellaneous is excluded: unspecified mixed category.",
            "Basal cell carcinoma is non-melanoma (negative), not benign in the clinical sense.",
            "The old development labels used DICM official class=1 as the positive class, which is not a pure melanoma label.",
        ],
    }

    high_conf_overlap_cases = sorted(
        {
            int(r["derm7pt_row_index"])
            for r in overlap_rows
            if r.get("high_confidence")
        }
    )
    phash_only_cases = sorted(
        {
            int(r["derm7pt_row_index"])
            for r in overlap_rows
            if not r.get("high_confidence")
        }
        - set(high_conf_overlap_cases)
    )

    audit = {
        "title": "Derm7pt external-validation audit for the frozen old Exp2-C model",
        "phase": "A_and_B_audit_and_cohort_freeze",
        "inference_performed": False,
        "training_performed": False,
        "models_modified": False,
        "started_utc": started,
        "finished_utc": None,
        "derm7pt": {
            "root": str(DERM7PT_DIR),
            "meta_csv": str(META_CSV),
            "image_dir": str(IMAGE_DIR),
            "n_rows": n_rows,
            "columns": list(meta.columns),
            "diagnosis_column": "diagnosis",
            "dermoscopic_image_column": "derm",
            "clinical_image_column": "clinic",
            "case_identifiers": {
                "case_num": "sequential case number in meta.csv, values 1..N",
                "case_id": "sparse internal identifier; 27 nonempty values; not an ISIC patient_id",
                "official_indexes": "0-based row indexes in train/valid/test_indexes.csv",
            },
            "missing_values": missing_values,
            "diagnosis_counts": diagnosis_counts,
            "official_split_counts": {
                split: int((manifest["official_split"] == split).sum())
                for split in ("train", "valid", "test")
            },
            "dermoscopic_images_found": int(manifest["image_exists"].sum()),
            "dermoscopic_images_missing": missing_images,
            "duplicate_derm_filenames": duplicate_filenames,
            "duplicate_derm_sha256": duplicate_hashes,
            "notes_nonempty": int(manifest["notes"].astype(str).str.len().gt(0).sum()),
        },
        "label_mapping": label_mapping,
        "cohort": {
            "total_derm7pt_cases": n_rows,
            "eligible_melanoma": int(
                (
                    (manifest["mapping_bucket"] == "positive_melanoma")
                    & (manifest["image_exists"])
                ).sum()
            ),
            "eligible_non_melanoma": int(
                (
                    (manifest["mapping_bucket"] == "negative_clearly_non_melanoma")
                    & (manifest["image_exists"])
                ).sum()
            ),
            "excluded_ambiguous": int(
                (manifest["mapping_bucket"] == "excluded_ambiguous").sum()
            ),
            "excluded_for_overlap": int(
                (manifest["overlap_status"] == "high_confidence_duplicate").sum()
            ),
            "excluded_total": int(len(excluded)),
            "evaluated_if_inference_runs": int(len(included)),
            "evaluated_melanoma": n_melanoma,
            "evaluated_non_melanoma": n_non,
        },
        "diagnosis_breakdown": diagnosis_breakdown,
        "old_development": {
            "dataset": "dataset/processed/melanoma_benign",
            "n": int(len(old)),
            "train": int((old["old_split"] == "train").sum()),
            "val": int((old["old_split"] == "val").sum()),
            "test": int((old["old_split"] == "test").sum()),
            "sources": {str(k): int(v) for k, v in old["source_dataset"].value_counts().items()}
            if "source_dataset" in old.columns
            else {},
            "patient_ids_nonempty": int(len(old_patient_ids)),
            "lesion_ids_nonempty": int(len(old_lesion_ids)),
            "label_note": (
                "Old positive class is DICM official class=1 plus ISIC 2020 official-benign "
                "negatives. This is not a clean melanoma-vs-benign label."
            ),
        },
        "overlap": {
            "exact_filename_or_id_matches": int(
                sum(1 for r in overlap_rows if r["match_type"] in {"filename", "filename_or_id"})
            ),
            "exact_sha256_matches": int(sum(1 for r in overlap_rows if r["match_type"] == "sha256")),
            "phash_exact_same_size": int(
                sum(
                    1
                    for r in overlap_rows
                    if r["match_type"] == "phash_exact" and r.get("high_confidence")
                )
            ),
            "phash_candidates_not_excluded": int(
                sum(1 for r in overlap_rows if r.get("action") == "report_only")
            ),
            "high_confidence_duplicate_cases": high_conf_overlap_cases,
            "phash_candidate_cases_not_excluded": phash_only_cases,
            "independent_cases_after_exclusions": int(len(included)),
            "patient_identifier_overlap": {
                "derm7pt_case_id_vs_old_patient_id": patient_id_overlap,
                "derm7pt_case_id_vs_old_lesion_id": lesion_id_overlap,
                "derm7pt_case_id_vs_old_group_id": group_id_overlap,
                "derm7pt_case_id_vs_old_image_id": image_id_overlap,
                "assessment": (
                    "Patient-level overlap could not be completely assessed. Derm7pt has no "
                    "ISIC-style patient_id. The sparse case_id field does not share a namespace "
                    "with old patient/lesion/image identifiers. No identifier overlap was found "
                    "among the nonempty Derm7pt case_id values. Identity was not inferred from "
                    "sex, location, or other unrelated fields."
                ),
            },
            "independence_claim": (
                "Derm7pt is treated as an external cohort after filename, SHA-256, and "
                "high-confidence perceptual-hash checks against the frozen old development "
                "images. Patient-level independence is not fully proven."
            ),
        },
        "frozen_old_internal_result_not_recomputed": {
            "test_images": 2473,
            "accuracy": 0.9228,
            "sensitivity": 0.9524,
            "specificity": 0.8886,
            "precision": 0.9078,
            "f1": 0.9296,
            "roc_auc": 0.9799,
            "pr_auc": 0.9836,
            "tp": 1261,
            "tn": 1021,
            "fp": 128,
            "fn": 63,
            "ensemble_weights": {"efficientnet": 0.75, "mobilenet": 0.25},
            "threshold": 0.30,
            "label_caveat": (
                "This internal result used DICM class=1 as the positive class and is not a "
                "clean melanoma-vs-benign result. It was not recomputed in this audit."
            ),
            "source": str(FROZEN_INTERNAL),
            "integrity_from_file": frozen_internal.get("integrity", {}),
        },
        "frozen_model_files": model_snapshot,
        "frozen_csv_snapshot": frozen_csv_snapshot,
        "stop_condition": (
            "STOP: audit and cohort freeze completed. Model inference was not run."
        ),
    }

    run_config = {
        "task": "old_model_derm7pt_external_validation",
        "phase_completed": "audit_and_cohort_freeze",
        "inference_run": False,
        "training_run": False,
        "model_files": model_snapshot,
        "preprocessing_implementation": (
            "preprocessing.image_preprocess.ImagePreprocessor.process_for_model "
            "(hair removal + light denoise + resize 224 + /255; no CLAHE; no lesion crop; "
            "ImageNet preprocess inside the saved models). Not executed in this audit."
        ),
        "ensemble_weights": {
            "efficientnet": FROZEN_ENSEMBLE["efficientnet_weight"],
            "mobilenet": FROZEN_ENSEMBLE["mobilenet_weight"],
        },
        "threshold": FROZEN_ENSEMBLE["threshold"],
        "threshold_optimization_on_derm7pt": False,
        "ensemble_weight_optimization_on_derm7pt": False,
        "calibration_on_derm7pt": False,
        "derm7pt_dataset_path": str(DERM7PT_DIR),
        "old_development_path": str(OLD_SPLIT_DIR),
        "n_input_cases": n_rows,
        "n_excluded_cases": int(len(excluded)),
        "n_evaluated_cases": int(len(included)),
        "label_mapping_file": str(OUT_DIR / "label_mapping.json"),
        "datetime_utc": datetime.now(timezone.utc).isoformat(),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
        },
        "old_internal_result_recomputed": False,
        "production_or_live_app_modified": False,
        "old_experiment_modified": False,
        "frozen_csvs_modified": False,
    }

    audit["finished_utc"] = datetime.now(timezone.utc).isoformat()
    audit["run_config"] = run_config

    md = render_report(audit, manifest)
    _json_dump(OUT_DIR / "audit_report.json", audit)
    (OUT_DIR / "audit_report.md").write_text(md, encoding="utf-8")
    _json_dump(OUT_DIR / "derm7pt_audit_report.json", audit)
    (OUT_DIR / "derm7pt_audit_report.md").write_text(md, encoding="utf-8")
    _json_dump(OUT_DIR / "label_mapping.json", label_mapping)
    _json_dump(OUT_DIR / "run_config.json", run_config)
    _json_dump(
        OUT_DIR / "overlap_report.json",
        {
            "n_rows": int(len(overlap_df)),
            "high_confidence_cases": high_conf_overlap_cases,
            "phash_candidate_cases_not_excluded": phash_only_cases,
            "patient_identifier_assessment": audit["overlap"]["patient_identifier_overlap"],
            "independence_claim": audit["overlap"]["independence_claim"],
        },
    )
    overlap_df.to_csv(OUT_DIR / "overlap_report.csv", index=False)
    manifest.to_csv(OUT_DIR / "external_manifest.csv", index=False)
    shutil.copy2(OUT_DIR / "external_manifest.csv", OUT_DIR / "derm7pt_external_manifest.csv")

    status = (
        "# Derm7pt external validation — STOPPED BEFORE INFERENCE\n\n"
        "The audit and frozen external cohort are complete.\n\n"
        "Model inference was **not** run.\n\n"
        "No threshold scan, ensemble-weight search, calibration, or retraining was performed.\n"
    )
    (OUT_DIR / "EXTERNAL_VALIDATION_STATUS.md").write_text(status, encoding="utf-8")

    print("Wrote", OUT_DIR)
    print("Total cases:", n_rows)
    print("Included for later evaluation:", len(included), "melanoma", n_melanoma, "non-melanoma", n_non)
    print("Excluded:", len(excluded))
    print("High-confidence overlaps:", len(high_conf_overlap_cases))
    print("STOP: do not run model inference from this script.")


def render_report(audit: dict, manifest: pd.DataFrame) -> str:
    d7 = audit["derm7pt"]
    cohort = audit["cohort"]
    overlap = audit["overlap"]
    old = audit["old_development"]
    mapping = audit["label_mapping"]
    internal = audit["frozen_old_internal_result_not_recomputed"]
    lines = [
        "# Derm7pt audit for the frozen old Exp2-C model",
        "",
        "The old production model was **not** evaluated on Derm7pt in this step. "
        "This document freezes the external cohort only.",
        "",
        "Statement for the later evaluation (not yet executed):",
        "",
        "> The old production model was evaluated on Derm7pt without retraining, "
        "fine-tuning, threshold optimization, ensemble-weight optimization, or calibration.",
        "",
        "That statement will apply only after Phase C inference is run against this frozen manifest.",
        "",
        "## 1. Objective",
        "",
        "Prepare an independent Derm7pt dermoscopic cohort for external validation of the "
        "already-frozen old Exp2-C ensemble (EfficientNetB0 0.75 + MobileNetV2 0.25, threshold 0.30).",
        "",
        "## 2. Frozen model specification",
        "",
        "- Model: EfficientNetB0 + MobileNetV2, soft-voting ensemble",
        "- EfficientNet weight: 0.75",
        "- MobileNet weight: 0.25",
        "- Operating threshold: 0.30 (frozen; not to be retuned on Derm7pt)",
        f"- EfficientNet file: `{audit['frozen_model_files']['efficientnetb0']['path']}`",
        f"- EfficientNet SHA-256: `{audit['frozen_model_files']['efficientnetb0']['sha256']}`",
        f"- MobileNet file: `{audit['frozen_model_files']['mobilenetv2']['path']}`",
        f"- MobileNet SHA-256: `{audit['frozen_model_files']['mobilenetv2']['sha256']}`",
        "- Preprocessing to use later: existing `ImagePreprocessor.process_for_model`",
        "- Scores are model-predicted melanoma probability/score estimates, not calibrated clinical probabilities",
        "",
        "## 3. OLD INTERNAL TEST RESULT (frozen, not recomputed)",
        "",
        "This is **not** the Derm7pt result. It is quoted from the existing frozen evaluation.",
        "",
        f"- Test images: {internal['test_images']}",
        f"- Accuracy: {internal['accuracy']:.2%}",
        f"- Sensitivity: {internal['sensitivity']:.2%}",
        f"- Specificity: {internal['specificity']:.2%}",
        f"- Precision: {internal['precision']:.2%}",
        f"- F1: {internal['f1']:.2%}",
        f"- ROC-AUC: {internal['roc_auc']:.2%}",
        f"- PR-AUC: {internal['pr_auc']:.2%}",
        f"- TP {internal['tp']} / TN {internal['tn']} / FP {internal['fp']} / FN {internal['fn']}",
        "",
        internal["label_caveat"],
        "",
        "## 4. Derm7pt dataset",
        "",
        f"- Path: `{d7['root']}`",
        f"- Rows in meta.csv: {d7['n_rows']}",
        f"- Columns: {', '.join(d7['columns'])}",
        f"- Diagnosis column: `{d7['diagnosis_column']}`",
        f"- Dermoscopic image column: `{d7['dermoscopic_image_column']}`",
        f"- Clinical image column: `{d7['clinical_image_column']}` (not used)",
        f"- Dermoscopic images found: {d7['dermoscopic_images_found']}",
        f"- Missing dermoscopic images: {len(d7['dermoscopic_images_missing'])}",
        f"- Duplicate derm filenames: {len(d7['duplicate_derm_filenames'])}",
        f"- Duplicate derm SHA-256 values: {len(d7['duplicate_derm_sha256'])}",
        "",
        "Official original index files were used. No new random split was created.",
        "",
        f"- Official train indexes: {d7['official_split_counts']['train']}",
        f"- Official valid indexes: {d7['official_split_counts']['valid']}",
        f"- Official test indexes: {d7['official_split_counts']['test']}",
        "",
        "The later external evaluation cohort is **all eligible independent cases**, not only the official test split, because this model was not trained on Derm7pt. Official split membership is stored on every row.",
        "",
        "### Diagnosis distribution (complete, inspected from meta.csv)",
        "",
        "| Diagnosis | Count | Mapping |",
        "|---|---:|---|",
    ]
    for item in audit["diagnosis_breakdown"]:
        lines.append(
            f"| {item['diagnosis']} | {item['total_cases']} | {item['mapping']} |"
        )
    lines.extend(
        [
            "",
            "### Missing values",
            "",
            "| Column | Missing |",
            "|---|---:|",
        ]
    )
    for col, n_miss in d7["missing_values"].items():
        lines.append(f"| {col} | {n_miss} |")
    lines.extend(
        [
            "",
            "## 5. Label definition",
            "",
            "Binary mapping for this external validation only:",
            "",
            "Positive (1) = melanoma diagnoses:",
            "",
        ]
    )
    for name in mapping["positive_diagnoses"]:
        lines.append(f"- `{name}`")
    lines.extend(["", "Negative (0) = clearly non-melanoma diagnoses:", ""])
    for name in mapping["negative_diagnoses"]:
        lines.append(f"- `{name}`")
    lines.extend(["", "Notes:", ""])
    for note in mapping["notes"]:
        lines.append(f"- {note}")
    lines.extend(
        [
            "",
            "## 6. Exclusion criteria",
            "",
            "A case is excluded from the frozen evaluation cohort if any of the following apply:",
            "",
            "- the dermoscopic image is missing",
            "- the diagnosis is ambiguous (`melanosis`, `miscellaneous`, or any later unmapped label)",
            "- a high-confidence overlap with old development data is found (exact filename/ID, exact SHA-256, or identical average-hash **and** identical file size)",
            "",
            "Excluded / ambiguous diagnosis reasons:",
            "",
        ]
    )
    for name, reason in mapping["excluded_diagnoses"].items():
        n = next(
            item["total_cases"]
            for item in audit["diagnosis_breakdown"]
            if item["diagnosis"] == name
        )
        lines.append(f"- `{name}` (n={n}): {reason}")
    lines.extend(
        [
            "",
            "No case was silently discarded. Every row is in `external_manifest.csv` with an inclusion flag and reason.",
            "",
            "## 7. Overlap / leakage audit",
            "",
            f"Old development dataset: `{old['dataset']}` (n={old['n']}; train {old['train']}, val {old['val']}, test {old['test']}).",
            "",
            f"- Exact filename/ID match rows: {overlap['exact_filename_or_id_matches']}",
            f"- Exact SHA-256 match rows: {overlap['exact_sha256_matches']}",
            f"- High-confidence same-size perceptual exact matches: {overlap['phash_exact_same_size']}",
            f"- Perceptual candidates reported but not excluded: {overlap['phash_candidates_not_excluded']}",
            f"- High-confidence duplicate Derm7pt cases excluded: {len(overlap['high_confidence_duplicate_cases'])}",
            f"- Independent cases remaining: {overlap['independent_cases_after_exclusions']}",
            "",
            overlap["independence_claim"],
            "",
            "### Patient / lesion identifiers",
            "",
            overlap["patient_identifier_overlap"]["assessment"],
            "",
            f"- Derm7pt case_id ∩ old patient_id: {len(overlap['patient_identifier_overlap']['derm7pt_case_id_vs_old_patient_id'])}",
            f"- Derm7pt case_id ∩ old lesion_id: {len(overlap['patient_identifier_overlap']['derm7pt_case_id_vs_old_lesion_id'])}",
            f"- Derm7pt case_id ∩ old group_id: {len(overlap['patient_identifier_overlap']['derm7pt_case_id_vs_old_group_id'])}",
            f"- Derm7pt case_id ∩ old image_id: {len(overlap['patient_identifier_overlap']['derm7pt_case_id_vs_old_image_id'])}",
            "",
            "## 8. External evaluation cohort (frozen)",
            "",
            f"- Total Derm7pt cases: {cohort['total_derm7pt_cases']}",
            f"- Eligible melanoma (label-mapped, image present): {cohort['eligible_melanoma']}",
            f"- Eligible non-melanoma (label-mapped, image present): {cohort['eligible_non_melanoma']}",
            f"- Excluded ambiguous diagnoses: {cohort['excluded_ambiguous']}",
            f"- Excluded for overlap: {cohort['excluded_for_overlap']}",
            f"- Excluded total: {cohort['excluded_total']}",
            f"- Frozen evaluation n: {cohort['evaluated_if_inference_runs']} "
            f"({cohort['evaluated_melanoma']} melanoma / {cohort['evaluated_non_melanoma']} non-melanoma)",
            "",
            "This manifest is now frozen. Do not tune weights or thresholds on it.",
            "",
            "## 9. External performance",
            "",
            "Not computed. Inference was not run.",
            "",
            "## 10. Confusion matrix",
            "",
            "Not computed. Inference was not run.",
            "",
            "## 11. Diagnosis distribution",
            "",
            "| Diagnosis | Total | Included | Excluded |",
            "|---|---:|---:|---:|",
        ]
    )
    for item in audit["diagnosis_breakdown"]:
        lines.append(
            f"| {item['diagnosis']} | {item['total_cases']} | {item['included']} | {item['excluded']} |"
        )
    lines.extend(
        [
            "",
            "## 12. Limitations",
            "",
            "- Derm7pt and the old development set come from different sources, cameras, and labeling protocols.",
            "- The old model’s positive class is not pure melanoma (DICM `class=1`). Derm7pt negatives include BCC and other non-melanoma diagnoses.",
            "- Patient-level overlap cannot be completely ruled out.",
            "- 8×8 average-hash candidates with Hamming distance 1–6 were reported and not auto-excluded unless they also had exact SHA-256 or same-size exact hash.",
            "- Official Derm7pt train/valid/test indexes were recorded but were not used to drop independent cases.",
            "- No claim of clinical validation is made.",
            "",
            "## 13. Reproducibility information",
            "",
            f"- Audit started (UTC): {audit['started_utc']}",
            f"- Audit finished (UTC): {audit['finished_utc']}",
            f"- Python: {audit['run_config']['software']['python']}",
            f"- Platform: {audit['run_config']['software']['platform']}",
            f"- pandas: {audit['run_config']['software']['pandas']}",
            "- Script: `scripts/audit_old_model_derm7pt_external.py`",
            "- Outputs: `models/experiments/old_model_derm7pt_external/`",
            "",
            "Reproduce the cohort by re-running the audit script. It only reads frozen files and writes this directory.",
            "",
            "## 14. STOP",
            "",
            audit["stop_condition"],
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    build_audit()
