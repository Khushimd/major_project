"""
Create a NEW clean DICM-only melanoma-vs-benign dataset and grouped splits.

Writes only under dataset/processed/dicm_clean_melanoma_benign/.
Does not train and does not modify the frozen melanoma_benign experiment.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import config

OUT_DIR = config.PROCESSED_DATA_DIR / "dicm_clean_melanoma_benign"
OLD_DIR = config.PROCESSED_DATA_DIR / "melanoma_benign"
TRAIN_META = config.ISIC_DICM_LABEL_DIR / "train-set-metadata.csv"
VALID_META = config.ISIC_DICM_LABEL_DIR / "valid-set-metadata.csv"
EXPECTED = {"melanoma": 6787, "benign": 4586, "total": 11373}
SEED = 42
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
MALIGNANT_OK = {"malignant", "indeterminate/malignant"}
REQUIRED_FILES = [
    "manifest.csv",
    "train.csv",
    "val.csv",
    "test.csv",
    "split_detail.json",
    "leakage_report.json",
    "dataset_report.json",
    "CLEAN_DATASET_README.txt",
    "CLEAN_DATASET_PREPARATION_REPORT.txt",
]


def normalize_isic_id(value: str) -> str:
    text = str(value or "")
    match = re.search(r"ISIC_(\d+)", text, re.I)
    if match:
        return f"ISIC_{int(match.group(1)):07d}"
    match = re.search(r"ISIC20\d{2}_(\d+)", text, re.I)
    if match:
        return f"ISIC_{int(match.group(1)):07d}"
    match = re.search(r"ISIC(\d{6,})", text, re.I)
    if match:
        return f"ISIC_{int(match.group(1)):07d}"
    return Path(text).stem


def _clean(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "<na>", "null", "na"} else text


def _nunique(series: pd.Series) -> int:
    return int(series.map(_clean).replace("", np.nan).nunique())


def _idset(series: pd.Series) -> set[str]:
    return {v for v in series.map(_clean) if v}


def _json_dump(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _index_images() -> dict[str, str]:
    index: dict[str, str] = {}
    roots = [
        config.ISIC_DICM_IMAGE_DIR,
        config.BASE_DIR / "images",
        config.DATASET_DIR / "isic_dicm_17k",
    ]
    seen: set[str] = set()
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        resolved = str(root.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            stored = str(path.resolve())
            stem = path.stem
            index.setdefault(stem, stored)
            index.setdefault(stem.lower(), stored)
            index.setdefault(normalize_isic_id(stem), stored)
    return index


def _resolve(name: str, index: dict[str, str]) -> str | None:
    for key in (str(name or ""), Path(str(name or "")).stem, normalize_isic_id(name)):
        if not key:
            continue
        path = index.get(key) or index.get(key.lower()) or index.get(normalize_isic_id(key))
        if path:
            return path
    return None


def _group_key(patient_id: str, lesion_id: str, image_id: str) -> tuple[str, str]:
    if patient_id:
        return f"patient:{patient_id}", "patient"
    if lesion_id:
        return f"lesion:{lesion_id}", "lesion"
    return f"image:{image_id}", "image"


def filter_official(official: pd.DataFrame) -> pd.DataFrame:
    dx = official["diagnosis"].map(_clean)
    bm = official["benign_malignant"].map(_clean).str.lower()
    melanoma = (dx == "melanoma") & bm.isin(MALIGNANT_OK)
    benign = dx.isin(BENIGN_DX) & (bm == "benign")
    keep = official.loc[melanoma | benign].copy()
    keep["diagnosis"] = dx.loc[keep.index]
    keep["benign_malignant"] = bm.loc[keep.index]
    keep["label"] = np.where(melanoma.loc[keep.index], 1, 0)
    keep["image_id"] = keep["isic_id"].map(normalize_isic_id)
    keep["patient_id"] = keep["patient_id"].map(_clean)
    keep["lesion_id"] = keep["lesion_id"].map(_clean)
    keep["source"] = "ISIC-DICM-17K"
    keep = keep.drop_duplicates(subset=["image_id"], keep="first").reset_index(drop=True)
    return keep


def assign_splits(pool: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pool = pool.copy()
    keys = []
    levels = []
    for row in pool.itertuples(index=False):
        key, level = _group_key(row.patient_id, row.lesion_id, row.image_id)
        keys.append(key)
        levels.append(level)
    pool["group_id"] = keys
    pool["group_level"] = levels
    groups = []
    for gid, g in pool.groupby("group_id"):
        groups.append(
            {
                "group_id": gid,
                "n": int(len(g)),
                "stratum": "melanoma_group" if int((g["label"] == 1).sum()) else "benign_only",
            }
        )
    gdf = pd.DataFrame(groups)
    assigned = {}
    for _, sub in gdf.groupby("stratum"):
        order = np.array(sub.index.to_numpy(), copy=True)
        rng.shuffle(order)
        sub = sub.loc[order]
        target = {
            "train": 0.70 * int(sub["n"].sum()),
            "val": 0.15 * int(sub["n"].sum()),
            "test": 0.15 * int(sub["n"].sum()),
        }
        filled = {"train": 0.0, "val": 0.0, "test": 0.0}
        for row in sub.itertuples(index=False):
            remaining = {k: target[k] - filled[k] for k in target}
            split = max(remaining, key=remaining.get)
            assigned[row.group_id] = split
            filled[split] += row.n
    pool["split"] = pool["group_id"].map(assigned)
    if pool["split"].isna().any():
        raise RuntimeError("Some rows were not assigned a split.")
    return pool


def _split_stats(df: pd.DataFrame) -> dict:
    return {
        "total": int(len(df)),
        "melanoma": int((df["label"] == 1).sum()),
        "benign": int((df["label"] == 0).sum()),
        "patients": _nunique(df["patient_id"]),
        "lesions": _nunique(df["lesion_id"]),
        "missing_patient_id": int((df["patient_id"] == "").sum()),
        "missing_lesion_id": int((df["lesion_id"] == "").sum()),
        "group_levels": {k: int(v) for k, v in df["group_level"].value_counts().items()},
    }


def main() -> None:
    if OUT_DIR.exists():
        existing = [name for name in REQUIRED_FILES if (OUT_DIR / name).exists()]
        if existing:
            raise RuntimeError(f"Refusing to overwrite existing files: {existing}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    official = pd.concat([pd.read_csv(TRAIN_META), pd.read_csv(VALID_META)], ignore_index=True)
    excluded_n = len(official)
    pool = filter_official(official)
    n_mel = int((pool["label"] == 1).sum())
    n_ben = int((pool["label"] == 0).sum())
    if n_mel != EXPECTED["melanoma"] or n_ben != EXPECTED["benign"] or len(pool) != EXPECTED["total"]:
        raise RuntimeError(
            f"STOP: clean filter counts differ from design. "
            f"got melanoma={n_mel} benign={n_ben} total={len(pool)}; "
            f"expected {EXPECTED}"
        )

    if pool["image_id"].duplicated().any():
        raise RuntimeError("STOP: duplicate image IDs after filtering.")

    index = _index_images()
    paths = []
    missing = []
    for image_id, isic_id in zip(pool["image_id"], pool["isic_id"].astype(str)):
        path = _resolve(isic_id, index) or _resolve(image_id, index)
        if path is None:
            missing.append(image_id)
        paths.append(path)
    if missing:
        raise RuntimeError(f"STOP: {len(missing)} clean images have no file, e.g. {missing[:10]}")
    pool["image_path"] = paths
    if pool["image_path"].duplicated().any():
        dups = pool.loc[pool["image_path"].duplicated(keep=False), ["image_id", "image_path"]]
        raise RuntimeError(f"STOP: duplicate image_path values:\n{dups.head(20)}")

    size_map: dict[int, list[str]] = {}
    for path in pool["image_path"]:
        size_map.setdefault(Path(path).stat().st_size, []).append(path)
    content_dups = []
    for size, group in size_map.items():
        if len(group) < 2:
            continue
        hashes: dict[str, list[str]] = {}
        for path in group:
            digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            hashes.setdefault(digest, []).append(path)
        for digest, same in hashes.items():
            if len(same) > 1:
                content_dups.append({"sha256": digest, "size": size, "paths": same})
    if content_dups:
        raise RuntimeError(f"STOP: duplicate image file contents detected: {content_dups[:5]}")

    pool = assign_splits(pool, seed=SEED)
    cols = [
        "image_id",
        "image_path",
        "label",
        "diagnosis",
        "benign_malignant",
        "patient_id",
        "lesion_id",
        "source",
        "split",
        "group_id",
        "group_level",
        "isic_id",
    ]
    pool = pool[cols].sort_values(["split", "label", "image_id"]).reset_index(drop=True)

    old = {name: pd.read_csv(OLD_DIR / f"{name}.csv") for name in ("manifest", "train", "val", "test")}
    old_ids = {name: set(df["image_id"].map(normalize_isic_id)) for name, df in old.items()}
    old_patients = {name: _idset(df["patient_id"]) for name, df in old.items()}
    clean_ids = set(pool["image_id"])
    overlap = {
        "old_manifest": len(clean_ids & old_ids["manifest"]),
        "old_train": len(clean_ids & old_ids["train"]),
        "old_val": len(clean_ids & old_ids["val"]),
        "old_test": len(clean_ids & old_ids["test"]),
    }

    old_all = pd.concat(
        [old["train"].assign(old_split="train"), old["val"].assign(old_split="val"), old["test"].assign(old_split="test")],
        ignore_index=True,
    )
    old_all["image_id"] = old_all["image_id"].map(normalize_isic_id)
    old_all["patient_id"] = old_all["patient_id"].map(_clean)
    shared_patients = sorted(_idset(pool["patient_id"]) & _idset(old_all["patient_id"]))
    patient_audit = []
    for pid in shared_patients:
        new_rows = pool[pool["patient_id"] == pid]
        old_rows = old_all[old_all["patient_id"] == pid]
        patient_audit.append(
            {
                "patient_id": pid,
                "dicm_clean_image_count": int(len(new_rows)),
                "old_experiment_image_count": int(len(old_rows)),
                "old_split_membership": sorted(old_rows["old_split"].unique().tolist()),
                "new_split_membership": sorted(new_rows["split"].unique().tolist()),
                "old_sources": sorted(old_rows.get("source_dataset", pd.Series(dtype=str)).astype(str).unique().tolist()),
            }
        )

    def _multi(df: pd.DataFrame, col: str) -> list[str]:
        filled = df[col].map(_clean)
        use = df.loc[filled != ""].copy()
        use[col] = filled.loc[use.index]
        counts = use.groupby(col)["split"].nunique()
        return sorted(counts[counts > 1].index.tolist())

    leak_patients = _multi(pool, "patient_id")
    leak_lesions = _multi(pool, "lesion_id")
    leak_images = _multi(pool, "image_id")
    leak_paths = _multi(pool, "image_path")
    if leak_patients or leak_lesions or leak_images or leak_paths:
        raise RuntimeError(
            "STOP: leakage inside the new split: "
            f"patients={len(leak_patients)} lesions={len(leak_lesions)} "
            f"images={len(leak_images)} paths={len(leak_paths)}"
        )

    train = pool[pool["split"] == "train"].copy()
    val = pool[pool["split"] == "val"].copy()
    test = pool[pool["split"] == "test"].copy()
    stats = {name: _split_stats(df) for name, df in (("all", pool), ("train", train), ("val", val), ("test", test))}

    train.to_csv(OUT_DIR / "train.csv", index=False)
    val.to_csv(OUT_DIR / "val.csv", index=False)
    test.to_csv(OUT_DIR / "test.csv", index=False)
    pool.to_csv(OUT_DIR / "manifest.csv", index=False)

    split_detail = {
        "seed": SEED,
        "grouping": "patient_id, else lesion_id, else image_id",
        "target": {"train": 0.70, "val": 0.15, "test": 0.15},
        "splits": stats,
        "group_level_counts": {k: int(v) for k, v in pool["group_level"].value_counts().items()},
    }
    leakage = {
        "new_split_patient_overlap": leak_patients,
        "new_split_lesion_overlap": leak_lesions,
        "new_split_image_overlap": leak_images,
        "new_split_path_overlap": leak_paths,
        "overlap_with_old_experiment": overlap,
        "shared_patient_count": len(shared_patients),
        "shared_patients": patient_audit,
        "old_experiment_untouched": True,
    }
    dataset_report = {
        "source_dataset": "ISIC-DICM-17K official train+valid metadata",
        "melanoma_definition": "diagnosis==melanoma AND benign_malignant in {malignant, indeterminate/malignant}",
        "benign_definition": "diagnosis in pre-specified benign list AND benign_malignant==benign",
        "official_class_used": False,
        "total_images": int(len(pool)),
        "melanoma": n_mel,
        "benign": n_ben,
        "excluded_official_records": int(excluded_n - len(pool)),
        "unique_nonempty_patients": _nunique(pool["patient_id"]),
        "unique_lesions": _nunique(pool["lesion_id"]),
        "missing_patient_ids": int((pool["patient_id"] == "").sum()),
        "missing_lesion_ids": int((pool["lesion_id"] == "").sum()),
        "grouping_strategy": split_detail["grouping"],
        "random_seed": SEED,
        "overlap_with_old_16529_experiment": overlap,
        "splits": stats,
        "leakage_audit": {
            "patient_overlap_between_new_splits": 0,
            "lesion_overlap_between_new_splits": 0,
            "image_overlap_between_new_splits": 0,
        },
        "training_started": False,
    }
    _json_dump(OUT_DIR / "split_detail.json", split_detail)
    _json_dump(OUT_DIR / "leakage_report.json", leakage)
    _json_dump(OUT_DIR / "dataset_report.json", dataset_report)

    readme = [
        "CLEAN DICM-ONLY MELANOMA VS BENIGN DATASET",
        "New directory. The frozen 16,529-image melanoma_benign experiment was not modified.",
        "",
        "Source: ISIC-DICM-17K official train-set-metadata.csv + valid-set-metadata.csv",
        "Official class was not used as the label.",
        "",
        "Melanoma: diagnosis==melanoma AND benign_malignant in {malignant, indeterminate/malignant}",
        "Benign: diagnosis in the design-doc benign list AND benign_malignant==benign",
        "",
        f"Total images: {len(pool)}",
        f"Melanoma: {n_mel}",
        f"Benign: {n_ben}",
        f"Excluded official records: {excluded_n - len(pool)}",
        f"Unique non-empty patients: {_nunique(pool['patient_id'])}",
        f"Unique lesions: {_nunique(pool['lesion_id'])}",
        f"Missing patient_id: {int((pool['patient_id']=='').sum())}",
        f"Missing lesion_id: {int((pool['lesion_id']=='').sum())}",
        "",
        "Grouping: patient_id, else lesion_id, else image_id. seed=42",
        f"Train: {stats['train']}",
        f"Val: {stats['val']}",
        f"Test: {stats['test']}",
        "",
        "Overlap with OLD 16,529-image experiment (old files not changed):",
        f"  manifest={overlap['old_manifest']} train={overlap['old_train']} val={overlap['old_val']} test={overlap['old_test']}",
        f"Shared non-empty patient IDs with old experiment: {len(shared_patients)}",
        "Full per-patient audit is in leakage_report.json.",
        "",
        "New-split leakage: 0 patient / 0 lesion / 0 image overlaps.",
        "TRAINING STARTED: NO",
    ]
    (OUT_DIR / "CLEAN_DATASET_README.txt").write_text("\n".join(readme) + "\n", encoding="utf-8")

    summary = [
        "CLEAN DATASET CREATED",
        "",
        f"Total: {len(pool)}",
        f"Melanoma: {n_mel}",
        f"Benign: {n_ben}",
        "",
        "Train:",
        f"  total: {stats['train']['total']}",
        f"  melanoma: {stats['train']['melanoma']}",
        f"  benign: {stats['train']['benign']}",
        "",
        "Validation:",
        f"  total: {stats['val']['total']}",
        f"  melanoma: {stats['val']['melanoma']}",
        f"  benign: {stats['val']['benign']}",
        "",
        "Test:",
        f"  total: {stats['test']['total']}",
        f"  melanoma: {stats['test']['melanoma']}",
        f"  benign: {stats['test']['benign']}",
        "",
        "Grouping:",
        f"  patient: {int((pool['group_level']=='patient').sum())}",
        f"  lesion fallback: {int((pool['group_level']=='lesion').sum())}",
        f"  image fallback: {int((pool['group_level']=='image').sum())}",
        "",
        "Leakage:",
        "  patient overlap between new splits: 0",
        "  lesion overlap between new splits: 0",
        "  image overlap between new splits: 0",
        "",
        "Overlap with OLD experiment:",
        f"  total: {overlap['old_manifest']}",
        f"  old train: {overlap['old_train']}",
        f"  old validation: {overlap['old_val']}",
        f"  old test: {overlap['old_test']}",
        "",
        f"Shared patients with old experiment: {len(shared_patients)}",
        "",
        "Files created:",
        *[f"  {OUT_DIR / name}" for name in REQUIRED_FILES],
        "",
        "TRAINING STARTED: NO",
    ]
    text = "\n".join(summary) + "\n"
    (OUT_DIR / "CLEAN_DATASET_PREPARATION_REPORT.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
