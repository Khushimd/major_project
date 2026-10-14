"""
Construct the final melanoma-vs-benign research dataset.

DICM melanoma + DICM clean benign + ISIC 2020 strict named benign.
Does not add ISIC 2020 melanoma. Does not train. Does not modify
production models, the app, or previous experiment manifests/splits.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import config

OUT_DIR = config.PROCESSED_DATA_DIR / "final_melanoma_benign"
DICM_DIR = config.PROCESSED_DATA_DIR / "dicm_clean_melanoma_benign"
ISIC_CSV = config.RAW_DATA_DIR / "train.csv"
ISIC_IMG = config.RAW_DATA_DIR / "train"
ISIC_ZIP = Path.home() / "Downloads" / "ISIC_2020_Training_JPEG.zip"
DICM_TRAIN_META = config.ISIC_DICM_LABEL_DIR / "train-set-metadata.csv"
DICM_VALID_META = config.ISIC_DICM_LABEL_DIR / "valid-set-metadata.csv"
ARCHIVE_META = config.BASE_DIR / "metadata.csv"
SEED = 42
EXPECTED_DICM = {"melanoma": 6787, "benign": 4586, "total": 11373}
EXPECTED_ISIC_STRICT = 5417
EXPECTED_OVERLAP = {
    "dicm_melanoma_vs_isic_benign": 0,
    "dicm_benign_vs_isic_benign": 532,
    "dicm_melanoma_vs_dicm_benign": 0,
}
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
REQUIRED = [
    "manifest.csv",
    "train.csv",
    "val.csv",
    "test.csv",
    "dataset_report.json",
    "overlap_report.json",
    "patient_overlap_report.json",
    "source_bias_report.json",
    "split_detail.json",
    "leakage_report.json",
    "CLEAN_FINAL_DATASET_README.txt",
    "FINAL_DATASET_AUDIT_REPORT.md",
]


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
    return "" if text.lower() in {"nan", "none", "<na>", "null", "na"} else text


def _idset(series: pd.Series) -> set[str]:
    return {_clean(v) for v in series if _clean(v)}


def _nunique(series: pd.Series) -> int:
    return int(series.map(_clean).replace("", np.nan).nunique())


def _json_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _counts(series: pd.Series) -> dict:
    filled = series.map(lambda v: _clean(v) or "<MISSING>")
    return {str(k): int(v) for k, v in filled.value_counts().items()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dicm() -> pd.DataFrame:
    df = pd.read_csv(DICM_DIR / "manifest.csv")
    df["image_id"] = df["image_id"].map(normalize_isic_id)
    df["label"] = df["label"].astype(int)
    df["diagnosis"] = df["diagnosis"].map(_clean)
    df["benign_malignant"] = df["benign_malignant"].map(_clean).str.lower()
    df["patient_id"] = df["patient_id"].map(_clean)
    df["lesion_id"] = df["lesion_id"].map(_clean)
    n, m, b = len(df), int((df["label"] == 1).sum()), int((df["label"] == 0).sum())
    if (n, m, b) != (EXPECTED_DICM["total"], EXPECTED_DICM["melanoma"], EXPECTED_DICM["benign"]):
        raise RuntimeError(f"STOP: clean DICM counts changed: got {(n, m, b)}")
    missing = [p for p in df["image_path"].astype(str) if not Path(p).is_file()]
    if missing:
        raise RuntimeError(f"STOP: {len(missing)} DICM images missing, e.g. {missing[:5]}")
    return df


def load_isic_strict_benign() -> pd.DataFrame:
    meta = pd.read_csv(ISIC_CSV)
    dx = meta["diagnosis"].map(_clean)
    bm = meta["benign_malignant"].map(_clean).str.lower()
    keep = meta.loc[dx.isin(BENIGN_DX) & (bm == "benign")].copy()
    if len(keep) != EXPECTED_ISIC_STRICT:
        raise RuntimeError(f"STOP: ISIC strict benign count {len(keep)} != {EXPECTED_ISIC_STRICT}")
    if (dx.loc[keep.index] == "unknown").any() or (dx.loc[keep.index] == "atypical melanocytic proliferation").any():
        raise RuntimeError("STOP: ISIC strict benign pool contains unknown or AMP rows.")
    if (keep["target"] == 1).any() if "target" in keep.columns else False:
        raise RuntimeError("STOP: ISIC strict benign pool contains target=1.")
    keep["image_id"] = keep["image_name"].map(normalize_isic_id)
    keep["diagnosis"] = dx.loc[keep.index]
    keep["benign_malignant"] = "benign"
    keep["patient_id"] = keep["patient_id"].map(_clean)
    keep["lesion_id"] = ""
    keep["label"] = 0
    return keep.reset_index(drop=True)


def extract_missing_isic(image_ids: list[str]) -> dict:
    ISIC_IMG.mkdir(parents=True, exist_ok=True)
    needed = []
    already = 0
    for image_id in image_ids:
        dest = ISIC_IMG / f"{image_id}.jpg"
        if dest.is_file():
            already += 1
        else:
            needed.append(image_id)
    extracted = 0
    if needed:
        if not ISIC_ZIP.is_file():
            raise FileNotFoundError(f"Need {len(needed)} ISIC JPEGs but zip not found: {ISIC_ZIP}")
        wanted = {f"{i}.jpg" for i in needed}
        print(f"Extracting {len(wanted)} missing ISIC 2020 strict-benign JPEGs from zip...", flush=True)
        with zipfile.ZipFile(ISIC_ZIP) as zf:
            for info in zf.infolist():
                name = Path(info.filename).name
                if name not in wanted:
                    continue
                dest = ISIC_IMG / name
                if dest.is_file() and dest.stat().st_size > 0:
                    extracted += 1
                    continue
                with zf.open(info) as src, dest.open("wb") as fh:
                    fh.write(src.read())
                extracted += 1
                if extracted % 250 == 0:
                    print(f"  extracted {extracted}/{len(wanted)}", flush=True)
    still_missing = [i for i in image_ids if not (ISIC_IMG / f"{i}.jpg").is_file()]
    return {"already": already, "extracted": extracted, "needed": len(needed), "still_missing": still_missing}


def overlap_audit(dicm: pd.DataFrame, isic_ben: pd.DataFrame) -> dict:
    dicm_mel = set(dicm.loc[dicm["label"] == 1, "image_id"])
    dicm_ben = set(dicm.loc[dicm["label"] == 0, "image_id"])
    isic_ids = set(isic_ben["image_id"])
    got = {
        "dicm_melanoma_vs_isic_benign": len(dicm_mel & isic_ids),
        "dicm_benign_vs_isic_benign": len(dicm_ben & isic_ids),
        "dicm_melanoma_vs_dicm_benign": len(dicm_mel & dicm_ben),
    }
    if got != EXPECTED_OVERLAP:
        raise RuntimeError(f"STOP: overlap disagrees with previous audit. expected {EXPECTED_OVERLAP}, got {got}")
    dicm_names = {Path(p).name.lower() for p in dicm["image_path"].astype(str)}
    isic_names = {f"{i}.jpg".lower() for i in isic_ids}
    return {
        "image_id_overlap": got,
        "filename_overlap_dicm_vs_isic_benign": len(dicm_names & isic_names),
        "matches_previous_audit": True,
        "dicm_melanoma_ids": len(dicm_mel),
        "dicm_benign_ids": len(dicm_ben),
        "isic_strict_benign_ids": len(isic_ids),
        "shared_ids_dicm_benign_isic_benign": sorted(dicm_ben & isic_ids)[:50],
        "shared_n": got["dicm_benign_vs_isic_benign"],
    }


def patient_overlap_audit(dicm: pd.DataFrame, isic_ben: pd.DataFrame) -> dict:
    parts = {
        "dicm_melanoma": dicm[dicm["label"] == 1],
        "dicm_benign": dicm[dicm["label"] == 0],
        "isic_benign": isic_ben,
    }
    ids = {k: _idset(v["patient_id"]) for k, v in parts.items()}
    pairs = {}
    for a, b in (("dicm_melanoma", "dicm_benign"), ("dicm_melanoma", "isic_benign"), ("dicm_benign", "isic_benign")):
        shared = ids[a] & ids[b]
        n_a = int(parts[a]["patient_id"].map(_clean).isin(shared).sum())
        n_b = int(parts[b]["patient_id"].map(_clean).isin(shared).sum())
        pairs[f"{a}_vs_{b}"] = {"shared_patients": len(shared), "images_in_left": n_a, "images_in_right": n_b}
    all_three = ids["dicm_melanoma"] & ids["dicm_benign"] & ids["isic_benign"]
    missing = {k: int((v["patient_id"].map(_clean) == "").sum()) for k, v in parts.items()}
    return {
        "unique_patients": {k: len(v) for k, v in ids.items()},
        "pairwise": pairs,
        "patients_in_all_three_groups": len(all_three),
        "missing_patient_id": missing,
        "note": "Shared patients are kept. Group-aware splitting assigns every image of a patient to one split.",
    }


def jpeg_size(path: str) -> tuple[int, int] | None:
    try:
        from PIL import Image

        with Image.open(path) as img:
            return int(img.size[0]), int(img.size[1])
    except Exception:
        return None


def source_bias_report(final: pd.DataFrame, dicm_extra: pd.DataFrame | None, archive: pd.DataFrame | None) -> dict:
    groups = {
        "dicm_melanoma": final[(final["source_dataset"] == "ISIC-DICM-17K") & (final["label"] == 1)],
        "dicm_benign": final[(final["source_dataset"] == "ISIC-DICM-17K") & (final["label"] == 0)],
        "isic_benign": final[final["source_dataset"] == "ISIC-2020"],
    }
    report = {"potential_source_domain_confounding_signals": True, "causal_claim": False}
    for name, g in groups.items():
        dims = []
        sample = g.sample(n=min(120, len(g)), random_state=SEED) if len(g) else g
        for p in sample["image_path"].astype(str):
            wh = jpeg_size(p)
            if wh:
                dims.append({"w": wh[0], "h": wh[1]})
        dim_counter = Counter((d["w"], d["h"]) for d in dims)
        report[name] = {
            "n": int(len(g)),
            "diagnosis": _counts(g["diagnosis"]),
            "sex": _counts(g["sex"]) if "sex" in g.columns else {},
            "anatom_site": _counts(g["anatom_site"]) if "anatom_site" in g.columns else {},
            "age_approx_describe": g["age_approx"].astype(float).describe().to_dict() if "age_approx" in g.columns else {},
            "year": _counts(g["year"]) if "year" in g.columns else {},
            "image_type": _counts(g["image_type"]) if "image_type" in g.columns else {},
            "dermoscopic_type": _counts(g["dermoscopic_type"]) if "dermoscopic_type" in g.columns else {},
            "diagnosis_confirm_type": _counts(g["diagnosis_confirm_type"]) if "diagnosis_confirm_type" in g.columns else {},
            "attribution": _counts(g["attribution"]) if "attribution" in g.columns else {},
            "sampled_dimension_counts": {f"{w}x{h}": n for (w, h), n in dim_counter.most_common(12)},
            "sampled_n_for_dimensions": len(dims),
        }
    return report


def assign_splits(pool: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pool = pool.copy()
    keys, levels = [], []
    for row in pool.itertuples(index=False):
        pid, lid, iid = _clean(row.patient_id), _clean(row.lesion_id), _clean(row.image_id)
        if pid:
            keys.append(f"patient:{pid}")
            levels.append("patient")
        elif lid:
            keys.append(f"lesion:{lid}")
            levels.append("lesion")
        else:
            keys.append(f"image:{iid}")
            levels.append("image")
    pool["group_id"] = keys
    pool["group_level"] = levels
    groups = []
    for gid, g in pool.groupby("group_id"):
        groups.append({"group_id": gid, "n": int(len(g)), "stratum": "melanoma_group" if int((g["label"] == 1).sum()) else "benign_only"})
    gdf = pd.DataFrame(groups)
    assigned = {}
    for _, sub in gdf.groupby("stratum"):
        order = np.array(sub.index.to_numpy(), copy=True)
        rng.shuffle(order)
        sub = sub.loc[order]
        target = {k: frac * int(sub["n"].sum()) for k, frac in (("train", 0.70), ("val", 0.15), ("test", 0.15))}
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


def _multi(df: pd.DataFrame, col: str) -> list[str]:
    filled = df[col].map(_clean)
    use = df.loc[filled != ""].copy()
    use[col] = filled.loc[use.index]
    counts = use.groupby(col)["split"].nunique()
    return sorted(counts[counts > 1].index.tolist())


def hash_duplicates(df: pd.DataFrame) -> list[dict]:
    size_map: dict[int, list[str]] = defaultdict(list)
    for path in df["image_path"].astype(str):
        size_map[Path(path).stat().st_size].append(path)
    dups = []
    for size, group in size_map.items():
        if len(set(group)) < 2:
            continue
        hashes: dict[str, list[str]] = defaultdict(list)
        for path in set(group):
            hashes[_sha256(Path(path))].append(path)
        for digest, same in hashes.items():
            if len(same) > 1:
                dups.append({"sha256": digest, "size": size, "paths": same})
    return dups


def main() -> None:
    print("TRAINING PERFORMED: NO", flush=True)
    if OUT_DIR.exists():
        existing = [n for n in REQUIRED if (OUT_DIR / n).exists()]
        if existing:
            raise RuntimeError(f"Refusing to overwrite existing files: {existing}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dicm = load_dicm()
    isic = load_isic_strict_benign()
    print(f"Verified DICM melanoma={int((dicm.label==1).sum())} benign={int((dicm.label==0).sum())}", flush=True)
    print(f"Verified ISIC strict benign metadata={len(isic)}", flush=True)

    overlap = overlap_audit(dicm, isic)
    print("Image overlap matches previous audit.", overlap["image_id_overlap"], flush=True)
    patient_ov = patient_overlap_audit(dicm, isic)

    extract_stats = extract_missing_isic(isic["image_id"].tolist())
    if extract_stats["still_missing"]:
        raise RuntimeError(f"STOP: {len(extract_stats['still_missing'])} ISIC strict benign images still missing, e.g. {extract_stats['still_missing'][:10]}")
    isic = isic.copy()
    isic["image_path"] = [str((ISIC_IMG / f"{i}.jpg").resolve()) for i in isic["image_id"]]
    if not all(Path(p).is_file() for p in isic["image_path"]):
        raise RuntimeError("STOP: ISIC image_path check failed after extraction.")

    # Keep DICM rows; add ISIC benign only when image_id is new.
    dicm_ids = set(dicm["image_id"])
    isic_unique = isic.loc[~isic["image_id"].isin(dicm_ids)].copy()
    dropped_overlap = int(len(isic) - len(isic_unique))
    if dropped_overlap != EXPECTED_OVERLAP["dicm_benign_vs_isic_benign"]:
        extra = set(isic.loc[isic["image_id"].isin(dicm_ids), "image_id"]) - set(dicm.loc[dicm["label"] == 0, "image_id"])
        if extra:
            raise RuntimeError(f"STOP: ISIC benign overlaps DICM non-benign IDs: {sorted(extra)[:20]}")
        if dropped_overlap != EXPECTED_OVERLAP["dicm_benign_vs_isic_benign"]:
            raise RuntimeError(f"STOP: dropped ISIC overlap {dropped_overlap} != expected 532")

    official = pd.concat([pd.read_csv(DICM_TRAIN_META), pd.read_csv(DICM_VALID_META)], ignore_index=True)
    official["image_id"] = official["isic_id"].map(normalize_isic_id)
    extra_cols = [c for c in ("sex", "age_approx", "anatom_site_general", "year", "image_type", "dermoscopic_type", "diagnosis_confirm_type") if c in official.columns]
    official_small = official[["image_id"] + extra_cols].drop_duplicates("image_id")
    archive = None
    if ARCHIVE_META.is_file():
        archive = pd.read_csv(ARCHIVE_META, low_memory=False)
        if "isic_id" in archive.columns:
            archive["image_id"] = archive["isic_id"].map(normalize_isic_id)

    dicm_part = dicm.merge(official_small, on="image_id", how="left")
    dicm_part["source_dataset"] = "ISIC-DICM-17K"
    dicm_part["class_name"] = np.where(dicm_part["label"] == 1, "Melanoma", "Benign")
    dicm_part["anatom_site"] = dicm_part.get("anatom_site_general", pd.Series([""] * len(dicm_part))).map(_clean)
    if archive is not None and "attribution" in archive.columns:
        attr = archive.drop_duplicates("image_id")[["image_id", "attribution"]]
        dicm_part = dicm_part.merge(attr, on="image_id", how="left")
    else:
        dicm_part["attribution"] = ""

    isic_part = isic_unique.copy()
    isic_part["source_dataset"] = "ISIC-2020"
    isic_part["class_name"] = "Benign"
    isic_part["anatom_site"] = isic_part.get("anatom_site_general_challenge", pd.Series([""] * len(isic_part))).map(_clean)
    isic_part["sex"] = isic_part["sex"].map(_clean) if "sex" in isic_part.columns else ""
    isic_part["year"] = ""
    isic_part["image_type"] = ""
    isic_part["dermoscopic_type"] = ""
    isic_part["diagnosis_confirm_type"] = ""
    isic_part["attribution"] = ""
    isic_part["age_approx"] = pd.to_numeric(isic_part.get("age_approx"), errors="coerce")

    cols = [
        "image_id",
        "image_path",
        "label",
        "class_name",
        "source_dataset",
        "diagnosis",
        "benign_malignant",
        "patient_id",
        "lesion_id",
        "sex",
        "age_approx",
        "anatom_site",
        "year",
        "image_type",
        "dermoscopic_type",
        "diagnosis_confirm_type",
        "attribution",
    ]
    for frame in (dicm_part, isic_part):
        for c in cols:
            if c not in frame.columns:
                frame[c] = ""
    pool = pd.concat([dicm_part[cols], isic_part[cols]], ignore_index=True)
    if pool["image_id"].duplicated().any():
        raise RuntimeError("STOP: duplicate image_id in combined pool.")
    if pool["image_path"].duplicated().any():
        raise RuntimeError("STOP: duplicate image_path in combined pool.")
    both_labels = pool.groupby("image_id")["label"].nunique()
    if int((both_labels > 1).sum()):
        raise RuntimeError("STOP: image_id appears under both labels.")

    print("Hashing same-size files for content duplicates...", flush=True)
    content_dups = hash_duplicates(pool)
    dropped_hash_ids = []
    content_dup_groups = content_dups
    if content_dups:
        keep_paths = set(pool["image_path"].astype(str))
        for item in content_dups:
            ordered = sorted(item["paths"])
            dropped_hash_ids.extend(Path(p).stem for p in ordered[1:])
            for p in ordered[1:]:
                keep_paths.discard(p)
        before = len(pool)
        pool = pool[pool["image_path"].astype(str).isin(keep_paths)].copy()
        print(f"Dropped {before - len(pool)} byte-identical extra files: {dropped_hash_ids}", flush=True)
        leftover = hash_duplicates(pool)
        if leftover:
            raise RuntimeError(f"STOP: content duplicates remain after dropping extras: {leftover[:3]}")
    n_mel = int((pool["label"] == 1).sum())
    n_ben = int((pool["label"] == 0).sum())
    if n_mel != 6787:
        raise RuntimeError(f"STOP: final melanoma {n_mel} != 6787")

    pool = assign_splits(pool, seed=SEED)
    leak_p = _multi(pool, "patient_id")
    leak_l = _multi(pool, "lesion_id")
    leak_i = _multi(pool, "image_id")
    leak_path = _multi(pool, "image_path")
    if leak_p or leak_l or leak_i or leak_path:
        raise RuntimeError(
            f"STOP: leakage in new splits patients={len(leak_p)} lesions={len(leak_l)} images={len(leak_i)} paths={len(leak_path)}"
        )
    missing_files = [p for p in pool["image_path"].astype(str) if not Path(p).is_file()]
    if missing_files:
        raise RuntimeError(f"STOP: missing files after split: {len(missing_files)}")

    train = pool[pool["split"] == "train"].copy()
    val = pool[pool["split"] == "val"].copy()
    test = pool[pool["split"] == "test"].copy()
    pool = pool.sort_values(["split", "label", "image_id"]).reset_index(drop=True)
    train.to_csv(OUT_DIR / "train.csv", index=False)
    val.to_csv(OUT_DIR / "val.csv", index=False)
    test.to_csv(OUT_DIR / "test.csv", index=False)
    pool.to_csv(OUT_DIR / "manifest.csv", index=False)

    def split_stats(df):
        src = df.groupby("source_dataset")["label"].value_counts().to_dict()
        return {
            "total": int(len(df)),
            "melanoma": int((df["label"] == 1).sum()),
            "benign": int((df["label"] == 0).sum()),
            "patients": _nunique(df["patient_id"]),
            "lesions": _nunique(df["lesion_id"]),
            "missing_patient_id": int((df["patient_id"].map(_clean) == "").sum()),
            "missing_lesion_id": int((df["lesion_id"].map(_clean) == "").sum()),
            "group_levels": {k: int(v) for k, v in df["group_level"].value_counts().items()},
            "by_source": {str(k): int(v) for k, v in src.items()},
        }

    stats = {name: split_stats(df) for name, df in (("all", pool), ("train", train), ("val", val), ("test", test))}
    ratio = n_ben / n_mel if n_mel else None
    bias = source_bias_report(pool, official_small, archive)

    dataset_report = {
        "training_performed": False,
        "sources": ["ISIC-DICM-17K clean melanoma", "ISIC-DICM-17K clean benign", "ISIC 2020 strict named benign"],
        "isic_2020_melanoma_added": False,
        "seed": SEED,
        "initial_expected": {"dicm_melanoma": 6787, "dicm_benign": 4586, "isic_strict_benign": 5417, "total": 16790},
        "isic_benign_image_id_overlap_dropped": dropped_overlap,
        "byte_identical_extra_files_dropped": dropped_hash_ids,
        "byte_identical_groups_found": content_dup_groups,
        "final": {"melanoma": n_mel, "benign": n_ben, "total": int(len(pool)), "ratio_benign_per_melanoma": ratio},
        "extract_isic": extract_stats,
        "splits": stats,
    }
    leakage = {
        "patient_overlap_between_new_splits": leak_p,
        "lesion_overlap_between_new_splits": leak_l,
        "image_overlap_between_new_splits": leak_i,
        "path_overlap_between_new_splits": leak_path,
        "content_duplicate_files_remaining": [],
        "content_duplicates_dropped_before_split": dropped_hash_ids,
        "passed": True,
    }
    _json_dump(OUT_DIR / "dataset_report.json", dataset_report)
    _json_dump(OUT_DIR / "overlap_report.json", {**overlap, "dropped_isic_benign_duplicate_ids": dropped_overlap, "isic_unique_added": int(len(isic_unique)), "byte_identical_extra_files_dropped": dropped_hash_ids})
    _json_dump(OUT_DIR / "patient_overlap_report.json", patient_ov)
    _json_dump(OUT_DIR / "source_bias_report.json", bias)
    _json_dump(OUT_DIR / "split_detail.json", {"seed": SEED, "grouping": "patient_id, else lesion_id, else image_id", "target": {"train": 0.70, "val": 0.15, "test": 0.15}, "splits": stats})
    _json_dump(OUT_DIR / "leakage_report.json", leakage)

    readme = [
        "FINAL MELANOMA VS BENIGN RESEARCH DATASET",
        "No model training was performed during this dataset-construction stage.",
        "Production models, the application, the old frozen experiment, and the clean DICM directory were not modified.",
        "",
        "Melanoma: clean DICM diagnosis==melanoma AND benign_malignant in {malignant, indeterminate/malignant}. 6787 images.",
        "Benign: clean DICM named-benign (4586) + ISIC 2020 named-benign (5417 metadata) minus 532 duplicate image IDs already in DICM benign.",
        "ISIC 2020 melanoma was NOT added.",
        "",
        f"Final: melanoma={n_mel} benign={n_ben} total={len(pool)} ratio_benign:melanoma={ratio:.3f}:1",
        f"Train {stats['train']}",
        f"Val {stats['val']}",
        f"Test {stats['test']}",
        "Seed=42. Grouping: patient_id -> lesion_id -> image_id.",
        "Leakage across new splits: 0 patient / 0 lesion / 0 image.",
    ]
    (OUT_DIR / "CLEAN_FINAL_DATASET_README.txt").write_text("\n".join(readme) + "\n", encoding="utf-8")

    md = [
        "# Final melanoma-vs-benign research dataset audit",
        "",
        "No model training was performed during this dataset-construction stage.",
        "",
        "TRAINING PERFORMED: NO",
        "PRODUCTION MODELS MODIFIED: NO",
        "APPLICATION MODIFIED: NO",
        "OLD DATASET MODIFIED: NO",
        "CLEAN DICM DIRECTORY MODIFIED: NO",
        "",
        "## 1. Dataset sources",
        "",
        "- ISIC-DICM-17K clean melanoma from `dataset/processed/dicm_clean_melanoma_benign/`",
        "- ISIC-DICM-17K clean benign from the same manifest",
        "- ISIC 2020 strict named benign from `dataset/raw/train.csv` + JPEGs in `dataset/raw/train/`",
        "- ISIC 2020 melanoma was not added",
        "",
        "## 2. Exact inclusion criteria",
        "",
        "- Melanoma: DICM `diagnosis==melanoma` AND `benign_malignant` in {malignant, indeterminate/malignant}",
        "- DICM benign: diagnosis in the pre-specified named benign list AND `benign_malignant==benign`",
        "- ISIC 2020 benign: named benign diagnosis AND `benign_malignant==benign`",
        "",
        "## 3. Exact exclusion criteria",
        "",
        "- Official DICM class was not used as a label",
        "- ISIC 2020 `diagnosis=unknown` excluded",
        "- ISIC 2020 atypical melanocytic proliferation excluded",
        "- ISIC 2020 melanoma excluded (duplicate records vs DICM melanoma, ~581 IDs)",
        "- ISIC 2020 strict-benign image IDs already present in DICM benign (532) kept as the DICM file only",
        "- BCC/SCC/AK/AIMP/other ambiguous DICM diagnoses remain excluded by the clean DICM definition",
        "",
        "## 4. Final counts by source and class",
        "",
        f"- DICM melanoma: {int(((pool.source_dataset=='ISIC-DICM-17K') & (pool.label==1)).sum())}",
        f"- DICM benign: {int(((pool.source_dataset=='ISIC-DICM-17K') & (pool.label==0)).sum())}",
        f"- ISIC 2020 benign added: {int((pool.source_dataset=='ISIC-2020').sum())}",
        f"- ISIC 2020 benign dropped as image-ID duplicates: {dropped_overlap}",
        f"- ISIC 2020 extra copies dropped as byte-identical files: {dropped_hash_ids}",
        "",
        "## 5. Melanoma:benign ratio",
        "",
        f"{n_mel}:{n_ben}  (1:{ratio:.3f})",
        "Benign images were not downsampled to 1:1.",
        "",
        "## 6. Image overlap results",
        "",
        f"- DICM melanoma vs ISIC benign image IDs: {overlap['image_id_overlap']['dicm_melanoma_vs_isic_benign']} (matches previous audit)",
        f"- DICM benign vs ISIC benign image IDs: {overlap['image_id_overlap']['dicm_benign_vs_isic_benign']} (matches previous audit; ISIC copies dropped)",
        f"- DICM melanoma vs DICM benign image IDs: {overlap['image_id_overlap']['dicm_melanoma_vs_dicm_benign']}",
        f"- Filename overlap DICM vs ISIC benign: {overlap['filename_overlap_dicm_vs_isic_benign']}",
        f"- Same-size same-hash extra copies dropped: {dropped_hash_ids}",
        "- Remaining same-hash duplicate files in the final pool: 0",
        "",
        "## 7. Patient overlap results",
        "",
        json.dumps(patient_ov["pairwise"], indent=2),
        "",
        f"Missing patient_id: {patient_ov['missing_patient_id']}",
        "Shared patients were not deleted. Their images are grouped into one split.",
        "",
        "## 8. Lesion overlap results",
        "",
        "ISIC 2020 has no lesion_id. DICM lesion IDs do not cross the new splits.",
        f"Unique non-empty lesions in the final pool: {_nunique(pool['lesion_id'])}",
        "",
        "## 9. Source-related differences",
        "",
        "See `source_bias_report.json`. These are potential source/domain confounding signals, not proof of model bias.",
        "DICM rows have year/image_type/confirmation metadata; ISIC 2020 challenge CSV does not.",
        "ISIC 2020 JPEGs are typically much larger native photographs than many DICM images.",
        "",
        "## 10. Missing metadata limitations",
        "",
        "- Many DICM rows lack patient_id and/or lesion_id; grouping falls back to lesion then image.",
        "- ISIC 2020 has no lesion_id, confirmation type, year, or attribution in the local CSV.",
        "- Patient overlap can exist even when image IDs do not.",
        "",
        "## 11. Train/validation/test counts",
        "",
        f"- Train: {stats['train']['total']}",
        f"- Validation: {stats['val']['total']}",
        f"- Test: {stats['test']['total']}",
        f"- Seed: {SEED}",
        "",
        "## 12. Class counts in each split",
        "",
        f"- Train melanoma/benign: {stats['train']['melanoma']} / {stats['train']['benign']}",
        f"- Val melanoma/benign: {stats['val']['melanoma']} / {stats['val']['benign']}",
        f"- Test melanoma/benign: {stats['test']['melanoma']} / {stats['test']['benign']}",
        "",
        "## 13. Leakage results",
        "",
        "- patient overlap across splits: 0",
        "- lesion overlap across splits: 0",
        "- image overlap across splits: 0",
        "- duplicate hashes across different paths: 0",
        "- no image under both labels",
        "- all referenced files exist",
        "- no duplicate manifest rows",
        "",
        "## 14. Exact random seed",
        "",
        "42",
        "",
        "## 15. Whether the dataset is safe to proceed to training",
        "",
        "YES — internal leakage checks passed. Training was not started.",
        "A later training run should treat this test split as frozen and should not use it for model selection.",
        "",
        "## 16. Remaining limitations",
        "",
        "- Two-source construction (DICM + ISIC 2020) remains a potential domain-confounding signal.",
        "- Missing DICM patient IDs prevent complete patient grouping for those rows.",
        "- ISIC lesion relationships cannot be audited.",
        "- This dataset is not a clinical validation set.",
        "",
        "No model training was performed during this dataset-construction stage.",
    ]
    (OUT_DIR / "FINAL_DATASET_AUDIT_REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("\nFINAL DATASET CREATED")
    print(f"melanoma: {n_mel}")
    print(f"benign: {n_ben}")
    print(f"total: {len(pool)}")
    print(f"ratio melanoma:benign = 1:{ratio:.3f}")
    print(f"train: {stats['train']['total']} (mel {stats['train']['melanoma']} / ben {stats['train']['benign']})")
    print(f"val: {stats['val']['total']} (mel {stats['val']['melanoma']} / ben {stats['val']['benign']})")
    print(f"test: {stats['test']['total']} (mel {stats['test']['melanoma']} / ben {stats['test']['benign']})")
    print(f"image overlap DICM-mel vs ISIC-ben: {overlap['image_id_overlap']['dicm_melanoma_vs_isic_benign']}")
    print("patient leakage across splits: 0")
    print("lesion leakage across splits: 0")
    print("PASSED ALL CHECKS: YES")
    print("TRAINING PERFORMED: NO")
    for name in REQUIRED:
        print(f"  {OUT_DIR / name}")


if __name__ == "__main__":
    main()
