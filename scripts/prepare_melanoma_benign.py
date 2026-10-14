"""
Build Melanoma vs Benign manifests and a frozen 70/15/15 grouped split.

Melanoma: ISIC-DICM-17K official class == 1 only.
Benign:   local ISIC 2020 images with benign_malignant == "benign",
          minus ISIC_4225838 (atypical melanocytic proliferation).

Does not train models, does not copy images, and does not touch
production weights or the live application.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from training.data_integrity import audit_split_leakage, save_leakage_report
from training.dataset_loader import normalize_isic_id

OUT_DIR = config.PROCESSED_DATA_DIR / "melanoma_benign"
ATYPICAL_IMAGE_ID = "ISIC_4225838"
EMPTY = {"", "nan", "none", "na", "null"}
RECORD_COLUMNS = [
    "image_path",
    "image_id",
    "isic_id",
    "label",
    "class_name",
    "patient_id",
    "lesion_id",
    "group_id",
    "source_dataset",
]


def _clean_id(value) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in EMPTY else text


def _filled(series: pd.Series) -> Set[str]:
    return {v for v in series.fillna("").astype(str).str.strip() if v.lower() not in EMPTY}


def _overlap_block(left: Set[str], right: Set[str], examples: int = 50) -> Dict:
    inter = sorted(left & right)
    return {
        "left_n": len(left),
        "right_n": len(right),
        "overlap_n": len(inter),
        "overlap_ids": inter[:examples],
        "overlap_all": inter,
    }


def _index_images(roots: Iterable[Path]) -> Dict[str, str]:
    index: Dict[str, str] = {}
    seen: Set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        resolved = str(root.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        iterator = root.rglob("*") if root.is_dir() else []
        for path in iterator:
            if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            stored = str(path.resolve())
            stem = path.stem
            index.setdefault(stem, stored)
            index.setdefault(stem.lower(), stored)
            index.setdefault(normalize_isic_id(stem), stored)
    return index


def _resolve_path(name: str, index: Dict[str, str]) -> Optional[str]:
    keys = [
        str(name or ""),
        Path(str(name or "")).stem,
        normalize_isic_id(name),
    ]
    for key in keys:
        if not key or key.lower() in EMPTY:
            continue
        path = index.get(key) or index.get(key.lower()) or index.get(normalize_isic_id(key))
        if path:
            return path
    return None


def _assign_group_id(row: pd.Series) -> str:
    patient_id = _clean_id(row.get("patient_id"))
    lesion_id = _clean_id(row.get("lesion_id"))
    image_id = _clean_id(row.get("isic_id") or row.get("image_id"))
    if patient_id:
        return f"patient:{patient_id}"
    if lesion_id:
        return f"lesion:{lesion_id}"
    return f"image:{image_id}"


def _load_archive_ids() -> pd.DataFrame:
    archive = config.BASE_DIR / "metadata.csv"
    if not archive.exists():
        return pd.DataFrame(columns=["image_id", "patient_id", "lesion_id"])
    extra = pd.read_csv(archive)
    keep = [c for c in extra.columns if c.lower() in {"isic_id", "patient_id", "lesion_id"}]
    extra = extra[keep].copy()
    rename = {c: c.lower() for c in extra.columns}
    extra = extra.rename(columns=rename)
    extra["image_id"] = extra["isic_id"].map(normalize_isic_id)
    extra["patient_id"] = extra.get("patient_id", "").map(_clean_id)
    extra["lesion_id"] = extra.get("lesion_id", "").map(_clean_id)
    extra = extra.drop(columns=["isic_id"], errors="ignore")
    return extra.drop_duplicates(subset=["image_id"], keep="first")


def _enrich_ids(frame: pd.DataFrame, archive: pd.DataFrame) -> pd.DataFrame:
    if archive.empty or frame.empty:
        return frame
    merged = frame.merge(archive, on="image_id", how="left", suffixes=("", "_archive"))
    for col in ("patient_id", "lesion_id"):
        archive_col = f"{col}_archive"
        if archive_col not in merged.columns:
            continue
        current = merged[col].map(_clean_id)
        incoming = merged[archive_col].map(_clean_id)
        merged[col] = np.where(current.eq(""), incoming, current)
        merged.drop(columns=[archive_col], inplace=True)
    return merged


def load_dicm_melanoma() -> Tuple[pd.DataFrame, Dict]:
    label_dir = config.ISIC_DICM_LABEL_DIR
    train_csv = label_dir / "train-set-metadata.csv"
    valid_csv = label_dir / "valid-set-metadata.csv"
    if not train_csv.exists() or not valid_csv.exists():
        raise FileNotFoundError(
            f"Official ISIC-DICM-17K labels not found: {train_csv} / {valid_csv}"
        )

    official = pd.concat(
        [pd.read_csv(train_csv), pd.read_csv(valid_csv)],
        ignore_index=True,
    )
    official["class"] = pd.to_numeric(official["class"], errors="coerce")
    class_counts = official["class"].value_counts(dropna=False).to_dict()
    melanoma_meta = official[official["class"] == 1].copy()

    index = _index_images(
        [
            config.ISIC_DICM_IMAGE_DIR,
            config.BASE_DIR / "images",
            config.DATASET_DIR / "isic_dicm_17k",
        ]
    )
    rows: List[dict] = []
    missing = 0
    for _, row in melanoma_meta.iterrows():
        isic_id = normalize_isic_id(row.get("isic_id") or row.get("image") or "")
        path = _resolve_path(row.get("isic_id") or row.get("image") or "", index)
        if path is None:
            missing += 1
            continue
        rows.append(
            {
                "image_path": path,
                "image_id": isic_id,
                "isic_id": isic_id,
                "label": 1,
                "class_name": "Melanoma",
                "patient_id": _clean_id(row.get("patient_id")),
                "lesion_id": _clean_id(row.get("lesion_id")),
                "source_dataset": "ISIC-DICM-17K",
                "diagnosis": _clean_id(row.get("diagnosis")),
                "official_class": 1,
            }
        )
    frame = pd.DataFrame(rows)
    frame = frame.drop_duplicates(subset=["image_id"], keep="first").reset_index(drop=True)
    stats = {
        "official_rows_total": int(len(official)),
        "official_class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "official_class_1": int(len(melanoma_meta)),
        "matched_images": int(len(frame)),
        "missing_images": int(missing),
        "duplicate_ids_dropped": int(len(melanoma_meta) - missing - len(frame)),
        "label_source": [str(train_csv), str(valid_csv)],
        "image_dir": str(config.ISIC_DICM_IMAGE_DIR),
    }
    if len(frame) != 8530:
        stats["warning"] = f"Expected 8530 melanoma images, found {len(frame)}."
    return frame, stats


def load_isic2020_benign() -> Tuple[pd.DataFrame, Dict]:
    csv_path = config.RAW_DATA_DIR / "train.csv"
    image_dir = config.RAW_DATA_DIR / "train"
    if not csv_path.exists():
        raise FileNotFoundError(f"ISIC 2020 train.csv not found: {csv_path}")

    meta = pd.read_csv(csv_path)
    bm = meta["benign_malignant"].astype(str).str.strip().str.lower()
    official_benign = meta[bm == "benign"].copy()
    official_malignant = int((bm == "malignant").sum())
    missing_label = int((~bm.isin(["benign", "malignant"])).sum())

    index = _index_images([image_dir])
    local_rows: List[dict] = []
    for _, row in official_benign.iterrows():
        image_name = str(row.get("image_name") or "")
        isic_id = normalize_isic_id(image_name)
        path = _resolve_path(image_name, index)
        if path is None:
            continue
        local_rows.append(
            {
                "image_path": path,
                "image_id": isic_id,
                "isic_id": isic_id,
                "label": 0,
                "class_name": "Benign",
                "patient_id": _clean_id(row.get("patient_id")),
                "lesion_id": "",
                "source_dataset": "ISIC-2020",
                "diagnosis": _clean_id(row.get("diagnosis")),
                "benign_malignant": _clean_id(row.get("benign_malignant")),
                "target": int(row.get("target")) if pd.notna(row.get("target")) else 0,
            }
        )
    local = pd.DataFrame(local_rows).drop_duplicates(subset=["image_id"], keep="first")
    before_clean = int(len(local))
    atypical_mask = local["image_id"].map(normalize_isic_id) == normalize_isic_id(ATYPICAL_IMAGE_ID)
    atypical = local[atypical_mask].copy()
    cleaned = local[~atypical_mask].copy().reset_index(drop=True)
    stats = {
        "metadata": str(csv_path),
        "image_dir": str(image_dir),
        "official_train_rows": int(len(meta)),
        "official_benign": int(len(official_benign)),
        "official_malignant": official_malignant,
        "missing_unknown_binary_label": missing_label,
        "local_benign_before_cleaning": before_clean,
        "atypical_case_removed": {
            "image_id": ATYPICAL_IMAGE_ID,
            "diagnosis": "atypical melanocytic proliferation",
            "removed_n": int(len(atypical)),
            "rows": atypical[["image_id", "diagnosis", "patient_id"]].to_dict("records"),
        },
        "local_benign_after_cleaning": int(len(cleaned)),
        "diagnosis_distribution_after_cleaning": {
            str(k): int(v)
            for k, v in cleaned["diagnosis"].value_counts(dropna=False).to_dict().items()
        },
        "unknown_diagnosis_kept": int((cleaned["diagnosis"] == "unknown").sum()),
        "note": (
            "diagnosis=unknown is kept because benign_malignant=benign and target=0. "
            "Only ISIC_4225838 was removed."
        ),
    }
    return cleaned, stats


def load_old_dataset_ids() -> Dict[str, Dict]:
    siim_meta = pd.read_csv(config.RAW_DATA_DIR / "train.csv")
    siim_images = _index_images([config.RAW_DATA_DIR / "train"])
    siim_ids = {normalize_isic_id(name) for name in siim_meta["image_name"].astype(str)}
    siim_on_disk = {normalize_isic_id(stem) for stem in siim_images}
    siim_malignant = {
        normalize_isic_id(row.image_name)
        for row in siim_meta.itertuples(index=False)
        if str(getattr(row, "benign_malignant", "")).strip().lower() == "malignant"
    }
    siim_benign = {
        normalize_isic_id(row.image_name)
        for row in siim_meta.itertuples(index=False)
        if str(getattr(row, "benign_malignant", "")).strip().lower() == "benign"
    }

    ham_meta = pd.read_csv(config.BASE_DIR / "HAM10000_metadata.csv")
    ham_ids = {normalize_isic_id(x) for x in ham_meta["image_id"].astype(str)}
    ham_lesions = _filled(ham_meta["lesion_id"])
    ham_mel = {
        normalize_isic_id(row.image_id)
        for row in ham_meta.itertuples(index=False)
        if str(getattr(row, "dx", "")).strip().lower() == "mel"
    }
    ham_nonmel = ham_ids - ham_mel

    ph2_ids: Set[str] = set()
    ph2_dir = config.BASE_DIR / "testing_data" / "PH2 Dataset images"
    ph2_labels = config.BASE_DIR / "testing_data" / "PH2_dataset.txt"
    if ph2_labels.exists():
        for line in ph2_labels.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "IMD" not in line:
                continue
            for token in line.replace("|", " ").split():
                if token.startswith("IMD"):
                    ph2_ids.add(token.strip())
    if ph2_dir.exists():
        for path in ph2_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                ph2_ids.add(path.stem.split("_")[0] if path.stem.startswith("IMD") else path.stem)

    return {
        "siim": {
            "metadata_ids": siim_ids,
            "on_disk_ids": siim_on_disk,
            "malignant_ids": siim_malignant,
            "benign_ids": siim_benign,
            "lesion_ids": set(),
            "patient_ids": _filled(siim_meta["patient_id"]),
        },
        "ham": {
            "image_ids": ham_ids,
            "lesion_ids": ham_lesions,
            "melanoma_ids": ham_mel,
            "non_melanoma_ids": ham_nonmel,
            "patient_ids": set(),
        },
        "ph2": {
            "image_ids": ph2_ids,
            "lesion_ids": set(),
            "patient_ids": set(),
        },
    }


def audit_and_clean(
    melanoma: pd.DataFrame,
    benign: pd.DataFrame,
    old: Dict[str, Dict],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    mel_ids = set(melanoma["image_id"].map(normalize_isic_id))
    ben_ids = set(benign["image_id"].map(normalize_isic_id))
    mel_stems = {Path(p).stem for p in melanoma["image_path"].astype(str)}
    ben_stems = {Path(p).stem for p in benign["image_path"].astype(str)}
    mel_norm_stems = {normalize_isic_id(s) for s in mel_stems}
    ben_norm_stems = {normalize_isic_id(s) for s in ben_stems}
    mel_lesions = _filled(melanoma["lesion_id"])
    ben_lesions = _filled(benign["lesion_id"])
    mel_patients = _filled(melanoma["patient_id"])
    ben_patients = _filled(benign["patient_id"])

    image_overlap = _overlap_block(mel_ids, ben_ids)
    stem_overlap = _overlap_block(mel_norm_stems, ben_norm_stems)
    lesion_overlap = _overlap_block(mel_lesions, ben_lesions)
    patient_overlap = _overlap_block(mel_patients, ben_patients)

    exclude_image_ids = set(image_overlap["overlap_all"]) | set(stem_overlap["overlap_all"])
    exclude_lesion_ids = set(lesion_overlap["overlap_all"])

    def _drop_conflicts(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        lesion = df["lesion_id"].map(_clean_id)
        drop = df["image_id"].map(normalize_isic_id).isin(exclude_image_ids) | lesion.isin(
            exclude_lesion_ids
        )
        return df[~drop].copy().reset_index(drop=True), int(drop.sum())

    melanoma_clean, dropped_mel = _drop_conflicts(melanoma)
    benign_clean, dropped_ben = _drop_conflicts(benign)

    # Patient overlap is reported but not used as a drop key: one patient can
    # have both a melanoma and a separate benign lesion. Grouped splitting
    # keeps that patient inside a single split.
    new_ids = set(melanoma_clean["image_id"]).union(set(benign_clean["image_id"]))
    new_lesions = _filled(melanoma_clean["lesion_id"]) | _filled(benign_clean["lesion_id"])
    new_patients = _filled(melanoma_clean["patient_id"]) | _filled(benign_clean["patient_id"])

    siim = old["siim"]
    ham = old["ham"]
    ph2 = old["ph2"]
    siim_image = _overlap_block(new_ids, siim["on_disk_ids"] or siim["metadata_ids"])
    ham_image = _overlap_block(new_ids, ham["image_ids"])
    ph2_image = _overlap_block(new_ids, {normalize_isic_id(x) for x in ph2["image_ids"]} | ph2["image_ids"])
    ham_lesion = _overlap_block(new_lesions, ham["lesion_ids"])

    overlap_report = {
        "pools_before_overlap_removal": {
            "dicm_melanoma": int(len(melanoma)),
            "isic2020_benign_after_atypical_removal": int(len(benign)),
        },
        "melanoma_vs_benign": {
            "image_id": {k: v for k, v in image_overlap.items() if k != "overlap_all"},
            "filename_stem": {k: v for k, v in stem_overlap.items() if k != "overlap_all"},
            "lesion_id": {k: v for k, v in lesion_overlap.items() if k != "overlap_all"},
            "patient_id": {k: v for k, v in patient_overlap.items() if k != "overlap_all"},
            "exclusion_policy": {
                "image_id_or_filename": (
                    "Excluded from both classes. The same image cannot be Melanoma "
                    "and Benign."
                ),
                "lesion_id": (
                    "Excluded from both classes. The same lesion cannot be Melanoma "
                    "and Benign."
                ),
                "patient_id": (
                    "Reported only, not excluded. A patient may have one melanoma "
                    "and a different benign lesion. group_id uses patient_id so "
                    "that patient stays in one split."
                ),
            },
            "excluded_image_ids": sorted(exclude_image_ids),
            "excluded_lesion_ids": sorted(exclude_lesion_ids),
            "melanoma_rows_dropped": dropped_mel,
            "benign_rows_dropped": dropped_ben,
        },
        "old_datasets": {
            "siim_isic_2020": {
                "images_on_disk": len(siim["on_disk_ids"]),
                "metadata_ids": len(siim["metadata_ids"]),
                "image_overlap": {k: v for k, v in siim_image.items() if k != "overlap_all"},
                "overlap_with_siim_malignant": len(new_ids & siim["malignant_ids"]),
                "overlap_with_siim_benign": len(new_ids & siim["benign_ids"]),
                "note": (
                    "The Benign class is the local ISIC 2020 / SIIM train subset, "
                    "so SIIM overlap is expected and is not removed. Melanoma IDs "
                    "that also appear in SIIM malignant are the same melanoma "
                    "images and are kept once, from DICM."
                ),
            },
            "ham10000": {
                "metadata_ids": len(ham["image_ids"]),
                "image_overlap": {k: v for k, v in ham_image.items() if k != "overlap_all"},
                "lesion_overlap": {k: v for k, v in ham_lesion.items() if k != "overlap_all"},
                "overlap_with_ham_melanoma": len(new_ids & ham["melanoma_ids"]),
                "overlap_with_ham_non_melanoma": len(new_ids & ham["non_melanoma_ids"]),
                "note": (
                    "HAM10000 is not part of this training set. Overlap is recorded "
                    "for later external-validation exclusion and is not dropped here."
                ),
            },
            "ph2_testing": {
                "ids": len(ph2["image_ids"]),
                "image_overlap": {k: v for k, v in ph2_image.items() if k != "overlap_all"},
                "note": (
                    "PH2 uses IMD* identifiers, not ISIC IDs. Overlap is recorded "
                    "and is not dropped."
                ),
            },
        },
        "final_counts_after_conflict_removal": {
            "melanoma": int(len(melanoma_clean)),
            "benign": int(len(benign_clean)),
            "total": int(len(melanoma_clean) + len(benign_clean)),
        },
    }
    return melanoma_clean, benign_clean, overlap_report


def grouped_split(
    frame: pd.DataFrame, seed: int = config.SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = frame.groupby("group_id", as_index=False).agg(
        label=("label", "max"),
        n=("label", "size"),
    )
    stratify = groups["label"] if groups["label"].nunique() > 1 else None
    try:
        g_trainval, g_test = train_test_split(
            groups, test_size=0.15, stratify=stratify, random_state=seed
        )
        g_train, g_val = train_test_split(
            g_trainval,
            test_size=0.15 / 0.85,
            stratify=g_trainval["label"] if stratify is not None else None,
            random_state=seed,
        )
    except ValueError:
        g_trainval, g_test = train_test_split(groups, test_size=0.15, random_state=seed)
        g_train, g_val = train_test_split(g_trainval, test_size=0.15 / 0.85, random_state=seed)

    train = frame[frame["group_id"].isin(set(g_train["group_id"]))].copy()
    val = frame[frame["group_id"].isin(set(g_val["group_id"]))].copy()
    test = frame[frame["group_id"].isin(set(g_test["group_id"]))].copy()
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def _split_counts(df: pd.DataFrame) -> Dict:
    n = len(df)
    mel = int((df["label"] == 1).sum())
    ben = int((df["label"] == 0).sum())
    return {
        "total": n,
        "melanoma": mel,
        "benign": ben,
        "melanoma_pct": round(100.0 * mel / max(n, 1), 2),
        "benign_pct": round(100.0 * ben / max(n, 1), 2),
        "patients": len(_filled(df["patient_id"])),
        "lesions": len(_filled(df["lesion_id"])),
        "groups": len(_filled(df["group_id"])),
    }


def audit_split_detail(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> Dict:
    frames = {"train": train, "val": val, "test": test}

    def _set(df: pd.DataFrame, col: str) -> Set[str]:
        return _filled(df[col]) if col in df.columns else set()

    report: Dict = {"splits": {}, "overlaps": {}}
    for name, df in frames.items():
        report["splits"][name] = _split_counts(df)
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    for left, right in pairs:
        key = f"{left}_vs_{right}"
        report["overlaps"][key] = {
            "patient_id": {
                "overlap_n": len(_set(frames[left], "patient_id") & _set(frames[right], "patient_id")),
                "overlap_ids": sorted(
                    _set(frames[left], "patient_id") & _set(frames[right], "patient_id")
                )[:20],
            },
            "lesion_id": {
                "overlap_n": len(_set(frames[left], "lesion_id") & _set(frames[right], "lesion_id")),
                "overlap_ids": sorted(
                    _set(frames[left], "lesion_id") & _set(frames[right], "lesion_id")
                )[:20],
            },
            "image_id": {
                "overlap_n": len(_set(frames[left], "image_id") & _set(frames[right], "image_id")),
                "overlap_ids": sorted(
                    _set(frames[left], "image_id") & _set(frames[right], "image_id")
                )[:20],
            },
            "group_id": {
                "overlap_n": len(_set(frames[left], "group_id") & _set(frames[right], "group_id")),
            },
        }
    report["patient_overlap"] = sum(v["patient_id"]["overlap_n"] for v in report["overlaps"].values())
    report["lesion_overlap"] = sum(v["lesion_id"]["overlap_n"] for v in report["overlaps"].values())
    report["image_overlap"] = sum(v["image_id"]["overlap_n"] for v in report["overlaps"].values())
    report["clean"] = (
        report["patient_overlap"] == 0
        and report["lesion_overlap"] == 0
        and report["image_overlap"] == 0
    )
    report["test_frozen"] = True
    report["test_must_not_be_used_for"] = [
        "threshold tuning",
        "ensemble weight selection",
        "hyperparameter tuning",
        "model selection",
        "augmentation decisions",
    ]
    return report


def _records(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["image_id"] = out["image_id"].map(normalize_isic_id)
    out["isic_id"] = out["isic_id"].map(normalize_isic_id)
    out["label"] = out["label"].astype(int)
    out["class_name"] = np.where(out["label"] == 1, "Melanoma", "Benign")
    out["patient_id"] = out["patient_id"].map(_clean_id)
    out["lesion_id"] = out["lesion_id"].map(_clean_id)
    out["group_id"] = out.apply(_assign_group_id, axis=1)
    extra = [c for c in ("diagnosis", "benign_malignant", "target", "official_class") if c in out.columns]
    return out[RECORD_COLUMNS + extra]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    archive = _load_archive_ids()

    melanoma, melanoma_stats = load_dicm_melanoma()
    melanoma = _enrich_ids(melanoma, archive)
    benign, benign_stats = load_isic2020_benign()
    benign = _enrich_ids(benign, archive)

    old = load_old_dataset_ids()
    melanoma, benign, overlap_report = audit_and_clean(melanoma, benign, old)

    manifest = _records(pd.concat([melanoma, benign], ignore_index=True))
    if manifest["image_id"].duplicated().any():
        dups = manifest.loc[manifest["image_id"].duplicated(keep=False), "image_id"].tolist()
        raise ValueError(f"Duplicate image_id values remain after cleaning: {sorted(set(dups))[:20]}")
    missing_files = [p for p in manifest["image_path"].astype(str) if not Path(p).is_file()]
    if missing_files:
        raise FileNotFoundError(f"Missing {len(missing_files)} image files, e.g. {missing_files[0]}")
    if set(manifest["label"].unique()) - {0, 1}:
        raise ValueError(f"Labels must be 0/1, found {sorted(manifest['label'].unique())}")
    if set(manifest["class_name"].unique()) - {"Benign", "Melanoma"}:
        raise ValueError(f"Unexpected class_name values: {sorted(manifest['class_name'].unique())}")
    if ((manifest["label"] == 0) & (manifest["class_name"] != "Benign")).any():
        raise ValueError("Label 0 must be Benign.")
    if ((manifest["label"] == 1) & (manifest["class_name"] != "Melanoma")).any():
        raise ValueError("Label 1 must be Melanoma.")
    if (manifest["source_dataset"] == "ISIC-DICM-17K").sum() != int((manifest["label"] == 1).sum()):
        raise ValueError("All melanoma rows must come from ISIC-DICM-17K.")
    if (manifest["source_dataset"] == "ISIC-2020").sum() != int((manifest["label"] == 0).sum()):
        raise ValueError("All benign rows must come from ISIC 2020.")

    train, val, test = grouped_split(manifest)
    leakage = audit_split_leakage(train, val, test, check_near_duplicates=False)
    detail = audit_split_detail(train, val, test)
    if not detail["clean"]:
        raise RuntimeError(
            "Split leakage detected: "
            f"patient={detail['patient_overlap']} "
            f"lesion={detail['lesion_overlap']} "
            f"image={detail['image_overlap']}"
        )

    manifest.to_csv(OUT_DIR / "manifest.csv", index=False)
    train.to_csv(OUT_DIR / "train.csv", index=False)
    val.to_csv(OUT_DIR / "val.csv", index=False)
    test.to_csv(OUT_DIR / "test.csv", index=False)

    save_leakage_report(leakage, path=OUT_DIR / "leakage_report.json")
    (OUT_DIR / "overlap_report.json").write_text(
        json.dumps(overlap_report, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "split_detail.json").write_text(json.dumps(detail, indent=2), encoding="utf-8")

    report = {
        "dataset": "Melanoma vs Benign",
        "labels": {"0": "Benign", "1": "Melanoma"},
        "melanoma_source": "ISIC-DICM-17K",
        "benign_source": "ISIC 2020",
        "melanoma_rule": "official class == 1 only; DICM class 0 was not used",
        "benign_rule": (
            "local images with benign_malignant == benign; "
            "ISIC_4225838 removed; diagnosis=unknown kept"
        ),
        "melanoma_stats": melanoma_stats,
        "benign_stats": benign_stats,
        "counts": {
            "melanoma": int((manifest["label"] == 1).sum()),
            "benign": int((manifest["label"] == 0).sum()),
            "total": int(len(manifest)),
        },
        "training": _split_counts(train),
        "validation": _split_counts(val),
        "test": _split_counts(test),
        "grouping": {
            "priority": ["patient_id", "lesion_id", "isic_id"],
            "groups": int(manifest["group_id"].nunique()),
            "seed": config.SEED,
            "ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
        },
        "overlap_summary": {
            "melanoma_vs_benign_image": overlap_report["melanoma_vs_benign"]["image_id"]["overlap_n"],
            "melanoma_vs_benign_lesion": overlap_report["melanoma_vs_benign"]["lesion_id"]["overlap_n"],
            "melanoma_vs_benign_patient": overlap_report["melanoma_vs_benign"]["patient_id"]["overlap_n"],
            "siim_isic_2020": overlap_report["old_datasets"]["siim_isic_2020"]["image_overlap"]["overlap_n"],
            "ham10000": overlap_report["old_datasets"]["ham10000"]["image_overlap"]["overlap_n"],
            "ph2_testing": overlap_report["old_datasets"]["ph2_testing"]["image_overlap"]["overlap_n"],
        },
        "split_leakage": {
            "patient_overlap": detail["patient_overlap"],
            "lesion_overlap": detail["lesion_overlap"],
            "image_overlap": detail["image_overlap"],
            "clean": detail["clean"],
        },
        "no_oversample": True,
        "no_image_duplication": True,
        "test_frozen": True,
        "production_models_untouched": [
            str(config.EFFICIENTNET_PATH),
            str(config.MOBILENET_PATH),
        ],
        "active_dataset_unchanged": config.ACTIVE_DATASET,
        "old_splits_untouched": [
            str(config.SPLIT_DIR),
            str(config.ISIC_DICM_SPLIT_DIR),
        ],
        "output_dir": str(OUT_DIR),
    }
    (OUT_DIR / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("===== FINAL MELANOMA vs BENIGN DATASET =====")
    print(f"Melanoma images: {report['counts']['melanoma']}")
    print(f"Benign images: {report['counts']['benign']}")
    print(f"Total images: {report['counts']['total']}")
    print(f"TRAIN {report['training']}")
    print(f"VAL {report['validation']}")
    print(f"TEST {report['test']}")
    print(f"Wrote files to {OUT_DIR}")
    print("Training was not started. Production models were not changed.")


if __name__ == "__main__":
    main()
