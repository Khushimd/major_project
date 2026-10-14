"""
Train / validation / test leakage checks for the melanoma pipeline.

Looks for:
- identical image IDs across splits
- identical patient / lesion group IDs across splits
- HAM10000 vs SIIM-ISIC 2020 image-id overlap
- cheap perceptual near-duplicates (8x8 average hash)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from PIL import Image

import config
from utils.logger import setup_logger

logger = setup_logger("data_integrity")


def _nonempty(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.strip()
    return text[~text.str.lower().isin({"", "nan", "none", "na"})]


def _split_set(df: pd.DataFrame, column: str) -> set:
    if column not in df.columns:
        return set()
    return set(_nonempty(df[column]).tolist())


def _overlap_report(name: str, a: set, b: set) -> Dict:
    inter = a & b
    return {
        "left": name.split("_vs_")[0],
        "right": name.split("_vs_")[-1],
        "left_n": len(a),
        "right_n": len(b),
        "overlap_n": len(inter),
        "overlap_examples": sorted(list(inter))[:20],
    }


def average_hash_hex(path: str, size: int = 8) -> Optional[str]:
    """8x8 average hash used as a cheap near-duplicate fingerprint."""
    try:
        with Image.open(path) as im:
            arr = np.asarray(im.convert("L").resize((size, size)), dtype=np.float32)
    except Exception:
        return None
    bits = (arr > float(arr.mean())).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def audit_source_overlap(df: pd.DataFrame) -> Dict:
    """HAM10000 and SIIM-ISIC 2020 can share ISIC image identifiers."""
    if "source" not in df.columns or "image_id" not in df.columns:
        return {"skipped": True, "reason": "source/image_id columns missing"}

    by_source = {
        str(source): set(_nonempty(chunk["image_id"]).tolist())
        for source, chunk in df.groupby(df["source"].astype(str))
    }
    ham = by_source.get("ham10000", set())
    isic = by_source.get("siim_isic_2020", set())
    overlap = sorted(ham & isic)
    report = {
        "ham10000_ids": len(ham),
        "siim_isic_2020_ids": len(isic),
        "shared_image_ids": len(overlap),
        "shared_examples": overlap[:30],
    }
    logger.info(
        "Source image-id overlap: HAM=%d ISIC=%d shared=%d",
        report["ham10000_ids"],
        report["siim_isic_2020_ids"],
        report["shared_image_ids"],
    )
    return report


def audit_split_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    check_near_duplicates: bool = True,
    near_dup_limit: Optional[int] = None,
) -> Dict:
    """Return a JSON-serializable leakage report for the three splits."""
    frames = {"train": train_df, "val": val_df, "test": test_df}
    report: Dict = {
        "counts": {},
        "image_id_overlap": {},
        "group_id_overlap": {},
        "path_overlap": {},
        "near_duplicates": {"enabled": bool(check_near_duplicates)},
    }

    for name, df in frames.items():
        n_mal = int(df["label"].sum()) if "label" in df.columns else None
        report["counts"][name] = {
            "rows": int(len(df)),
            "malignant": n_mal,
            "benign": int(len(df) - n_mal) if n_mal is not None else None,
            "unique_image_ids": len(_split_set(df, "image_id")),
            "unique_groups": len(_split_set(df, "group_id")),
            "unique_paths": len(_split_set(df, "image_path")),
        }

    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    for left, right in pairs:
        key = f"{left}_vs_{right}"
        report["image_id_overlap"][key] = _overlap_report(
            key, _split_set(frames[left], "image_id"), _split_set(frames[right], "image_id")
        )
        report["group_id_overlap"][key] = _overlap_report(
            key, _split_set(frames[left], "group_id"), _split_set(frames[right], "group_id")
        )
        report["path_overlap"][key] = _overlap_report(
            key, _split_set(frames[left], "image_path"), _split_set(frames[right], "image_path")
        )

    leak_groups = sum(v["overlap_n"] for v in report["group_id_overlap"].values())
    leak_ids = sum(v["overlap_n"] for v in report["image_id_overlap"].values())
    if leak_groups or leak_ids:
        logger.warning(
            "LEAKAGE DETECTED: group overlap=%d, image-id overlap=%d",
            leak_groups,
            leak_ids,
        )
    else:
        logger.info("No patient/lesion/image-id overlap across train/val/test.")

    if check_near_duplicates:
        report["near_duplicates"] = _audit_near_duplicates(
            frames, limit=near_dup_limit
        )

    report["clean"] = (
        leak_groups == 0
        and leak_ids == 0
        and report["near_duplicates"].get("cross_split_hash_collisions", 0) == 0
    )
    return report


def _audit_near_duplicates(
    frames: Dict[str, pd.DataFrame],
    limit: Optional[int] = None,
) -> Dict:
    hash_to_splits: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    hashed = 0
    failed = 0
    for split, df in frames.items():
        paths: Iterable[str] = df["image_path"].astype(str).tolist()
        if limit:
            paths = paths[: int(limit)]
        for path in paths:
            digest = average_hash_hex(path)
            if digest is None:
                failed += 1
                continue
            hashed += 1
            hash_to_splits[digest][split].append(path)

    collisions = []
    for digest, by_split in hash_to_splits.items():
        if len(by_split) < 2:
            continue
        collisions.append(
            {
                "hash": digest,
                "splits": {k: v[:3] for k, v in by_split.items()},
                "n_images": sum(len(v) for v in by_split.values()),
            }
        )

    logger.info(
        "Near-duplicate scan: hashed=%d failed=%d cross-split hash collisions=%d",
        hashed,
        failed,
        len(collisions),
    )
    return {
        "enabled": True,
        "hashed": hashed,
        "failed": failed,
        "cross_split_hash_collisions": len(collisions),
        "examples": collisions[:25],
        "note": (
            "8x8 average-hash collisions are potential near-duplicates, not "
            "proof that two clinical images are the same lesion."
        ),
    }


def _filled_ids(df: pd.DataFrame, column: str) -> set:
    return _split_set(df, column)


def audit_dicm_split_detail(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict:
    """Patient/lesion/image leakage counts for the ISIC-DICM-17K split."""
    frames = {"train": train_df, "val": val_df, "test": test_df}
    report: Dict = {"splits": {}, "overlaps": {}}
    for name, df in frames.items():
        n_mel = int((df["label"] == 1).sum())
        n_non = int((df["label"] == 0).sum())
        report["splits"][name] = {
            "images": int(len(df)),
            "melanoma": n_mel,
            "non_melanoma": n_non,
            "melanoma_pct": round(100.0 * n_mel / max(len(df), 1), 2),
            "patients": len(_filled_ids(df, "patient_id")),
            "lesions": len(_filled_ids(df, "lesion_id")),
            "groups": len(_filled_ids(df, "group_id")),
        }
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    for left, right in pairs:
        key = f"{left}_vs_{right}"
        report["overlaps"][key] = {
            "patient_id": _overlap_report(key, _filled_ids(frames[left], "patient_id"), _filled_ids(frames[right], "patient_id")),
            "lesion_id": _overlap_report(key, _filled_ids(frames[left], "lesion_id"), _filled_ids(frames[right], "lesion_id")),
            "image_id": _overlap_report(key, _filled_ids(frames[left], "image_id"), _filled_ids(frames[right], "image_id")),
            "image_path": _overlap_report(key, _filled_ids(frames[left], "image_path"), _filled_ids(frames[right], "image_path")),
            "group_id": _overlap_report(key, _filled_ids(frames[left], "group_id"), _filled_ids(frames[right], "group_id")),
        }
    report["patient_overlap"] = sum(v["patient_id"]["overlap_n"] for v in report["overlaps"].values())
    report["lesion_overlap"] = sum(v["lesion_id"]["overlap_n"] for v in report["overlaps"].values())
    report["image_overlap"] = sum(v["image_id"]["overlap_n"] for v in report["overlaps"].values())
    report["clean"] = (
        report["patient_overlap"] == 0
        and report["lesion_overlap"] == 0
        and report["image_overlap"] == 0
    )
    return report


def audit_old_dataset_overlap(dicm_ids: Iterable[str]) -> Dict:
    """Record SIIM / HAM overlap for later external-validation exclusion. Does not delete anything."""
    dicm = {str(x) for x in dicm_ids}
    siim_dir = config.RAW_DATA_DIR / "train"
    siim = {p.stem for p in siim_dir.glob("*.jpg")} if siim_dir.exists() else set()
    ham: set = set()
    for part in (
        config.BASE_DIR / "HAM10000_images_part_1",
        config.BASE_DIR / "HAM10000_images_part_2",
    ):
        if part.exists():
            ham.update(p.stem for p in part.glob("*.jpg"))
    report = {
        "dicm_ids": len(dicm),
        "siim_isic_2020": {
            "images_on_disk": len(siim),
            "overlap_n": len(dicm & siim),
            "overlap_examples": sorted(dicm & siim)[:20],
        },
        "ham10000": {
            "images_on_disk": len(ham),
            "overlap_n": len(dicm & ham),
            "overlap_examples": sorted(dicm & ham)[:20],
        },
        "note": (
            "Overlaps are expected because ISIC-DICM-17K is curated from the ISIC archive. "
            "Old datasets stay unused in the new training set. Use these IDs later if "
            "an external validation must exclude shared images."
        ),
    }
    logger.info(
        "Old-dataset overlap: SIIM=%d HAM=%d (kept unused for DICM training)",
        report["siim_isic_2020"]["overlap_n"],
        report["ham10000"]["overlap_n"],
    )
    return report


def save_leakage_report(report: Dict, path: Optional[Path] = None) -> Path:
    path = Path(path or (config.MODELS_DIR / "experiments" / "leakage_report.json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Leakage report saved: %s", path)
    return path
