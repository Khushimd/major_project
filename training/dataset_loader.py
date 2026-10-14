"""
Dataset loading utilities for SIIM-ISIC 2020, HAM10000, and ISIC-DICM-17K.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import config
from training.data_integrity import (
    audit_source_overlap,
    audit_split_leakage,
    save_leakage_report,
)
from utils.logger import setup_logger

logger = setup_logger("dataset")

_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def normalize_isic_id(value: str) -> str:
    """ISIC2019_0025105_mel / ISIC_0025105 / ISIC0025105 -> ISIC_0025105."""
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


def _assign_group_ids(dataframe: pd.DataFrame) -> pd.Series:
    """
    Patient- or lesion-aware grouping key.

    ISIC 2020: patient_id
    HAM10000: lesion_id
    Fallback: image stem so singleton images stay in one split.
    """
    n = len(dataframe)
    idx = dataframe.index
    src = (
        dataframe["source"].astype(str)
        if "source" in dataframe.columns
        else pd.Series(["unknown"] * n, index=idx)
    )
    pid = (
        dataframe["patient_id"].fillna("").astype(str).str.strip()
        if "patient_id" in dataframe.columns
        else pd.Series([""] * n, index=idx)
    )
    lid = (
        dataframe["lesion_id"].fillna("").astype(str).str.strip()
        if "lesion_id" in dataframe.columns
        else pd.Series([""] * n, index=idx)
    )
    iid = (
        dataframe["image_id"].fillna("").astype(str).str.strip()
        if "image_id" in dataframe.columns
        else dataframe["image_path"].map(lambda p: Path(str(p)).stem)
    )
    empty = {"", "nan", "none", "na"}
    group = src.str.cat(iid, sep=":image:")
    lesion_ok = ~lid.str.lower().isin(empty)
    patient_ok = ~pid.str.lower().isin(empty)
    group = pd.Series(
        np.where(lesion_ok, src.str.cat(lid, sep=":lesion:"), group),
        index=dataframe.index,
    )
    group = pd.Series(
        np.where(patient_ok, src.str.cat(pid, sep=":patient:"), group),
        index=dataframe.index,
    )
    return group


class MelanomaDataset:
    """
    Modular dataset loader.

    Swap ACTIVE_DATASET in config (or pass dataset_key) to use HAM10000 later.
    """

    def __init__(
        self,
        dataset_key: str = config.ACTIVE_DATASET,
        img_size: Tuple[int, int] = config.IMG_SIZE,
        seed: int = config.SEED,
    ) -> None:
        if dataset_key not in config.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_key}")
        self.cfg = config.DATASETS[dataset_key]
        self.dataset_key = dataset_key
        self.img_size = img_size
        self.seed = seed
        self.df: Optional[pd.DataFrame] = None
        self.official_splits: Optional[Tuple[pd.DataFrame, pd.DataFrame]] = None

    def _dicm_image_roots(self) -> List[Path]:
        """Only DICM image folders — never SIIM raw/ or HAM10000 parts."""
        configured = Path(self.cfg.get("image_dir") or getattr(config, "ISIC_DICM_IMAGE_DIR", config.BASE_DIR / "images"))
        return [
            configured,
            config.BASE_DIR / "images",
            Path(self.cfg["raw_dir"]),
            getattr(config, "ISIC_DICM_LABEL_DIR", config.DATASET_DIR / "isic_dicm_17k"),
        ]

    @staticmethod
    def _index_image_files(roots: List[Path]) -> Dict[str, str]:
        index: Dict[str, str] = {}
        seen: set[str] = set()
        for root in roots:
            if not root.exists():
                continue
            resolved = str(root.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            files = root.rglob("*") if root.is_dir() else []
            for path in files:
                if not path.is_file() or path.suffix.lower() not in _IMAGE_EXTS:
                    continue
                stored = str(path)
                stem = path.stem
                index.setdefault(stem, stored)
                index.setdefault(stem.lower(), stored)
                index.setdefault(normalize_isic_id(stem), stored)
                parent = path.parent.name.strip().lower().replace(" ", "")
                if parent in {"melanoma", "mel", "malignant"}:
                    index.setdefault(f"mel:{normalize_isic_id(stem)}", stored)
                elif parent in {"non-melanoma", "nonmelanoma", "benign", "nevus"}:
                    index.setdefault(f"ben:{normalize_isic_id(stem)}", stored)
        return index

    @staticmethod
    def _resolve_dicm_image(row: pd.Series, index: Dict[str, str], image_col: str) -> Optional[str]:
        candidates = [
            str(row.get(image_col) or ""),
            str(row.get("isic_id") or ""),
            Path(str(row.get(image_col) or "")).stem,
            normalize_isic_id(row.get("isic_id") or row.get(image_col) or ""),
        ]
        for key in candidates:
            if not key or key in {"nan", "None"}:
                continue
            path = index.get(key) or index.get(Path(key).stem) or index.get(normalize_isic_id(key))
            if path:
                return path
        return None

    def _dicm_label(self, row: pd.Series) -> int:
        if "class" not in row.index or pd.isna(row.get("class")):
            raise ValueError("ISIC-DICM-17K rows must include official class=0/1.")
        label = int(self._to_binary_label(row.get("class"), set()))
        if label not in (0, 1):
            raise ValueError(f"Invalid ISIC-DICM-17K class value: {row.get('class')!r}")
        return label

    def _dataframe_from_dicm_csv(self, csv_file: Path, index: Dict[str, str]) -> pd.DataFrame:
        df = pd.read_csv(csv_file)
        image_col, _ = self._detect_columns(df)
        rows = []
        missing = 0
        for _, row in df.iterrows():
            path = self._resolve_dicm_image(row, index, image_col)
            if path is None:
                missing += 1
                continue
            rows.append(
                {
                    "image_path": path,
                    "label": self._dicm_label(row),
                    "image_id": normalize_isic_id(row.get("isic_id") or row.get(image_col) or path),
                    "patient_id": str(row.get("patient_id") or "").strip(),
                    "lesion_id": str(row.get("lesion_id") or "").strip(),
                    "source": self.dataset_key,
                }
            )
        if not rows:
            raise FileNotFoundError(
                "ISIC-DICM-17K metadata was found but no images matched. "
                "Extract the 6.9 GB ISIC-DICM-17K Complete Bundle into "
                f"{Path(self.cfg['raw_dir'])} (jpg files can stay in subfolders)."
            )
        if missing:
            logger.warning("ISIC-DICM-17K: %d metadata rows had no matching image", missing)
        out = pd.DataFrame(rows)
        out["group_id"] = _assign_group_ids(out)
        return out

    def _enrich_dicm_ids(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Fill blank patient/lesion IDs from archive metadata.csv. Never copy diagnosis."""
        archive = config.BASE_DIR / "metadata.csv"
        if not archive.exists() or frame.empty:
            return frame
        extra = pd.read_csv(archive, usecols=lambda c: c in {"isic_id", "patient_id", "lesion_id"})
        extra["image_id"] = extra["isic_id"].map(normalize_isic_id)
        extra = extra.drop_duplicates(subset=["image_id"], keep="first")
        extra = extra.drop(columns=["isic_id"], errors="ignore")
        merged = frame.merge(extra, on="image_id", how="left", suffixes=("", "_archive"))
        for col in ("patient_id", "lesion_id"):
            archive_col = f"{col}_archive"
            if archive_col not in merged.columns:
                continue
            current = merged[col].fillna("").astype(str).str.strip()
            incoming = merged[archive_col].fillna("").astype(str).str.strip()
            empty = current.str.lower().isin({"", "nan", "none", "na"})
            merged[col] = np.where(empty, incoming, current)
            merged.drop(columns=[archive_col], inplace=True)
        return merged

    def _validate_dicm_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            raise FileNotFoundError("ISIC-DICM-17K produced no labeled rows.")
        labels = set(frame["label"].astype(int).unique())
        if labels - {0, 1}:
            raise ValueError(f"ISIC-DICM-17K labels must be 0/1, found {sorted(labels)}")
        missing_files = [p for p in frame["image_path"].astype(str) if not Path(p).is_file()]
        if missing_files:
            raise FileNotFoundError(
                f"ISIC-DICM-17K missing {len(missing_files)} image files, e.g. {missing_files[0]}"
            )
        if frame["image_id"].duplicated().any():
            dups = int(frame["image_id"].duplicated().sum())
            raise ValueError(f"ISIC-DICM-17K has {dups} duplicate isic_id rows.")
        return frame.reset_index(drop=True)

    def _load_isic_dicm_17k(self) -> pd.DataFrame:
        label_dir = Path(getattr(config, "ISIC_DICM_LABEL_DIR", config.DATASET_DIR / "isic_dicm_17k"))
        train_csv = label_dir / "train-set-metadata.csv"
        valid_csv = label_dir / "valid-set-metadata.csv"
        if not train_csv.exists() or not valid_csv.exists():
            raise FileNotFoundError(
                "Official ISIC-DICM-17K labels not found. Expected "
                f"{train_csv} and {valid_csv}."
            )
        index = self._index_image_files(self._dicm_image_roots())
        if not index:
            raise FileNotFoundError(
                "No ISIC-DICM-17K JPEGs found. Expected files in "
                f"{getattr(config, 'ISIC_DICM_IMAGE_DIR', config.BASE_DIR / 'images')}."
            )
        frame = pd.concat(
            [
                self._dataframe_from_dicm_csv(train_csv, index),
                self._dataframe_from_dicm_csv(valid_csv, index),
            ],
            ignore_index=True,
        )
        frame = self._enrich_dicm_ids(frame)
        empty = {"", "nan", "none", "na"}
        frame["patient_id"] = frame["patient_id"].fillna("").astype(str).str.strip()
        frame["lesion_id"] = frame["lesion_id"].fillna("").astype(str).str.strip()
        frame.loc[frame["patient_id"].str.lower().isin(empty), "patient_id"] = ""
        frame.loc[frame["lesion_id"].str.lower().isin(empty), "lesion_id"] = ""
        frame["group_id"] = _assign_group_ids(frame)
        frame = self._validate_dicm_frame(frame)
        self.official_splits = None
        self.df = frame
        logger.info(
            "Loaded ISIC-DICM-17K official class labels: %d images (%d melanoma, %d non-melanoma) | groups=%d",
            len(frame),
            int((frame["label"] == 1).sum()),
            int((frame["label"] == 0).sum()),
            frame["group_id"].nunique(),
        )
        return self.df

    def load_combined(self, source_keys: List[str]) -> pd.DataFrame:
        """Merge multiple datasets into one dataframe (dedupe by image path)."""
        parts: List[pd.DataFrame] = []
        for key in source_keys:
            sub = MelanomaDataset(dataset_key=key, img_size=self.img_size, seed=self.seed)
            sub.load_metadata()
            chunk = sub.df.copy()
            chunk["source"] = key
            parts.append(chunk)
            logger.info(
                "Combined source %s: %d rows (%d malignant)",
                key,
                len(chunk),
                int(chunk["label"].sum()),
            )
        merged = pd.concat(parts, ignore_index=True)
        merged["image_id"] = merged["image_path"].map(lambda p: Path(str(p)).stem)
        merged["group_id"] = _assign_group_ids(merged)
        before = len(merged)
        merged = merged.drop_duplicates(subset=["image_id"], keep="first")
        if len(merged) < before:
            logger.info(
                "Dropped %d duplicate image IDs across HAM10000 / SIIM-ISIC",
                before - len(merged),
            )
        before_paths = len(merged)
        merged = merged.drop_duplicates(subset=["image_path"], keep="first")
        if len(merged) < before_paths:
            logger.info("Dropped %d duplicate image paths", before_paths - len(merged))
        self.df = merged.reset_index(drop=True)
        overlap = audit_source_overlap(self.df)
        logger.info(
            "Combined total: %d samples (%d malignant, %d benign); shared HAM/ISIC IDs=%s",
            len(self.df),
            int(self.df["label"].sum()),
            int((self.df["label"] == 0).sum()),
            overlap.get("shared_image_ids", 0),
        )
        return self.df

    def _find_csv(self, raw_dir: Path, csv_path: Optional[str]) -> Path:
        """Locate train.csv or ISIC 2020 ground-truth / metadata CSV."""
        if csv_path:
            path = Path(csv_path)
            if path.exists():
                return path
            raise FileNotFoundError(f"CSV not found: {path}")

        candidates = [
            raw_dir / self.cfg["csv_name"],
            config.BASE_DIR / self.cfg.get("csv_name", "HAM10000_metadata.csv"),
            raw_dir / "train.csv",
            raw_dir / "ISIC_2020_Training_GroundTruth.csv",
            raw_dir / "ISIC_2020_Training_GroundTruth_v2.csv",
            raw_dir / "ISIC_2020_Training_GroundTruth.csv.zip",
        ]
        # Also accept any *GroundTruth*.csv in raw_dir
        candidates.extend(sorted(raw_dir.glob("*GroundTruth*.csv")))
        candidates.extend(sorted(raw_dir.glob("*groundtruth*.csv")))
        candidates.extend(sorted(raw_dir.glob("train*.csv")))

        for c in candidates:
            if c.exists() and c.suffix.lower() == ".csv":
                return c

        raise FileNotFoundError(
            f"Labels CSV not found under {raw_dir}\n"
            "Place train.csv OR ISIC Training Ground Truth CSV there, "
            "plus JPEG images in dataset/raw/train/"
        )

    @staticmethod
    def _detect_columns(df: pd.DataFrame) -> Tuple[str, str]:
        """Auto-detect image id and label columns across Kaggle / ISIC formats."""
        cols = {c.lower(): c for c in df.columns}

        image_col = None
        for key in ("image_name", "image", "image_id", "isic_id", "filename"):
            if key in cols:
                image_col = cols[key]
                break

        label_col = None
        for key in ("target", "class", "benign_malignant", "diagnosis", "dx", "melanoma"):
            if key in cols:
                label_col = cols[key]
                break

        if image_col is None or label_col is None:
            raise ValueError(
                f"Could not detect image/label columns in CSV. Found: {list(df.columns)}\n"
                "Expected an image id column and target / benign_malignant / diagnosis."
            )
        return image_col, label_col

    @staticmethod
    def _to_binary_label(raw_label, malignant_labels: set) -> int:
        """Normalize various label formats to 0/1."""
        if malignant_labels:
            return 1 if str(raw_label).lower() in malignant_labels else 0

        text = str(raw_label).strip().lower()
        if text in {"1", "1.0", "malignant", "melanoma", "mel"}:
            return 1
        if text in {"0", "0.0", "benign", "nv", "bkl", "df", "vasc", "akiec", "bcc"}:
            return 0
        try:
            return 1 if float(raw_label) >= 0.5 else 0
        except (TypeError, ValueError):
            return 0

    def load_metadata(self, csv_path: Optional[str] = None) -> pd.DataFrame:
        """Load and normalize metadata CSV to columns: image_path, label."""
        if self.dataset_key == "combined":
            sources = self.cfg.get("sources", ["siim_isic_2020", "ham10000"])
            return self.load_combined(sources)
        if self.dataset_key == "isic_dicm_17k":
            return self._load_isic_dicm_17k()

        raw_dir = Path(self.cfg["raw_dir"])
        csv_file = self._find_csv(raw_dir, csv_path)

        df = pd.read_csv(csv_file)
        image_col, label_col = self._detect_columns(df)
        ext = self.cfg.get("image_ext", ".jpg")
        logger.info("Using labels file: %s (image=%s, label=%s)", csv_file.name, image_col, label_col)

        # Resolve image paths — search common subfolders / zip extract layouts
        search_dirs: List[Path] = []
        if self.dataset_key == "ham10000":
            search_dirs.extend(
                [
                    config.BASE_DIR / "HAM10000_images_part_1",
                    config.BASE_DIR / "HAM10000_images_part_2",
                    raw_dir / "HAM10000_images_part_1",
                    raw_dir / "HAM10000_images_part_2",
                ]
            )
            search_dirs.extend(sorted(config.BASE_DIR.glob("HAM10000_images_part_*")))
            search_dirs.extend(sorted(raw_dir.glob("HAM10000_images_part_*")))
        search_dirs.extend(
            [
                raw_dir / "train",
                raw_dir / "jpeg" / "train",
                raw_dir / "JPEG" / "train",
                raw_dir / "images",
                raw_dir / "Images",
                raw_dir / "ISIC_2020_Training_JPEG",
                raw_dir / "ISIC_2020_Training_JPEG" / "train",
                raw_dir,
            ]
        )
        if self.dataset_key != "ham10000" and raw_dir.exists():
            for child in raw_dir.iterdir():
                if child.is_dir():
                    search_dirs.append(child)

        image_index: dict[str, str] = {}
        seen_dirs = set()
        for d in search_dirs:
            if not d.exists() or not d.is_dir():
                continue
            resolved = str(d.resolve())
            if resolved in seen_dirs:
                continue
            seen_dirs.add(resolved)
            for p in d.iterdir():
                if p.is_file() and p.suffix.lower() in {ext.lower(), ".jpg", ".jpeg", ".png"}:
                    image_index.setdefault(p.stem, str(p))

        def resolve_path(name: str) -> Optional[str]:
            return image_index.get(Path(str(name)).stem)

        paths, labels, image_ids, patient_ids, lesion_ids = [], [], [], [], []
        malignant_labels = set(self.cfg.get("malignant_labels", []))
        colmap = {c.lower(): c for c in df.columns}
        patient_col = colmap.get("patient_id") or colmap.get("patient")
        lesion_col = colmap.get("lesion_id") or colmap.get("lesion")

        for _, row in df.iterrows():
            path = resolve_path(row[image_col])
            if path is None:
                continue
            label = self._to_binary_label(row[label_col], malignant_labels)
            image_id = Path(str(row[image_col])).stem
            patient_id = str(row[patient_col]).strip() if patient_col else ""
            lesion_id = str(row[lesion_col]).strip() if lesion_col else ""
            paths.append(path)
            labels.append(label)
            image_ids.append(image_id)
            patient_ids.append(patient_id)
            lesion_ids.append(lesion_id)

        if not paths:
            raise FileNotFoundError(
                "CSV loaded but no images were found. Extract the JPEG zip into "
                f"{raw_dir}/train/ so image files match names in the CSV."
            )

        self.df = pd.DataFrame(
            {
                "image_path": paths,
                "label": labels,
                "image_id": image_ids,
                "patient_id": patient_ids,
                "lesion_id": lesion_ids,
                "source": self.dataset_key,
            }
        )
        self.df["group_id"] = _assign_group_ids(self.df)
        logger.info(
            "Loaded %s samples (%d malignant, %d benign) from %s | groups=%d",
            len(self.df),
            int(self.df["label"].sum()),
            int((self.df["label"] == 0).sum()),
            self.cfg["name"],
            self.df["group_id"].nunique(),
        )
        return self.df

    def split(
        self,
        test_size: float = config.TEST_SPLIT,
        val_size: float = config.VALIDATION_SPLIT,
        grouped: bool = True,
        save: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Patient/lesion-aware stratified train / validation / test split."""
        if self.df is None:
            self.load_metadata()
        if self.dataset_key == "isic_dicm_17k":
            return self._split_isic_dicm_70_15_15(save=save)
        if self.official_splits is not None:
            return self._split_official_dicm(save=save)
        df = self.df.copy()
        if "image_id" not in df.columns:
            df["image_id"] = df["image_path"].map(lambda p: Path(str(p)).stem)
        if "group_id" not in df.columns:
            df["group_id"] = _assign_group_ids(df)

        if grouped and df["group_id"].nunique() > 3:
            train, val, test = self._grouped_split(df, test_size, val_size)
        else:
            logger.warning("Falling back to image-level stratified split.")
            train, val, test = self._image_level_split(df, test_size, val_size)

        report = audit_split_leakage(train, val, test, check_near_duplicates=False)
        if save:
            self.save_splits(train, val, test)
            save_leakage_report(report)

        logger.info(
            "Split -> train=%d val=%d test=%d | train_mal=%d val_mal=%d test_mal=%d",
            len(train),
            len(val),
            len(test),
            int(train["label"].sum()),
            int(val["label"].sum()),
            int(test["label"].sum()),
        )
        return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

    def _dicm_split_dir(self) -> Path:
        return Path(getattr(config, "ISIC_DICM_SPLIT_DIR", config.PROCESSED_DATA_DIR / "isic_dicm_17k"))

    def load_saved_dicm_splits(self) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        split_dir = self._dicm_split_dir()
        paths = [split_dir / name for name in ("train.csv", "val.csv", "test.csv")]
        if not all(p.exists() for p in paths):
            return None
        train, val, test = (pd.read_csv(p) for p in paths)
        logger.info(
            "Loaded saved ISIC-DICM-17K splits: train=%d val=%d test=%d",
            len(train),
            len(val),
            len(test),
        )
        return train, val, test

    def _split_isic_dicm_70_15_15(
        self, save: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = self.df.copy() if self.df is not None else self.load_metadata().copy()
        if "group_id" not in df.columns:
            df["group_id"] = _assign_group_ids(df)
        train, val, test = self._grouped_split(df, test_size=0.15, val_size=0.15)
        report = audit_split_leakage(train, val, test, check_near_duplicates=False)
        if save:
            split_dir = self._dicm_split_dir()
            self.save_splits(train, val, test, split_dir=split_dir)
            save_leakage_report(report, path=split_dir / "leakage_report.json")
        logger.info(
            "ISIC-DICM-17K 70/15/15 grouped split -> train=%d val=%d test=%d | "
            "train_mel=%d val_mel=%d test_mel=%d",
            len(train),
            len(val),
            len(test),
            int(train["label"].sum()),
            int(val["label"].sum()),
            int(test["label"].sum()),
        )
        return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

    def _split_official_dicm(self, save: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train, valid = self.official_splits
        valid = valid.copy()
        if "group_id" not in valid.columns:
            valid["group_id"] = _assign_group_ids(valid)
        val = test = None
        if valid["group_id"].nunique() > 3:
            groups = valid.groupby("group_id", as_index=False).agg(label=("label", "max"))
            try:
                g_val, g_test = train_test_split(
                    groups,
                    test_size=0.5,
                    stratify=groups["label"] if groups["label"].nunique() > 1 else None,
                    random_state=self.seed,
                )
                val = valid[valid["group_id"].isin(set(g_val["group_id"]))]
                test = valid[valid["group_id"].isin(set(g_test["group_id"]))]
            except ValueError:
                val = test = None
        if val is None or test is None or val.empty or test.empty:
            val, test = train_test_split(
                valid,
                test_size=0.5,
                stratify=valid["label"] if valid["label"].nunique() > 1 else None,
                random_state=self.seed,
            )
        report = audit_split_leakage(train, val, test, check_near_duplicates=False)
        if save:
            self.save_splits(train, val, test)
            save_leakage_report(report)
        logger.info(
            "Official ISIC-DICM-17K train + held-out valid split -> train=%d val=%d test=%d | "
            "train_mal=%d val_mal=%d test_mal=%d",
            len(train),
            len(val),
            len(test),
            int(train["label"].sum()),
            int(val["label"].sum()),
            int(test["label"].sum()),
        )
        return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

    def _image_level_split(
        self,
        df: pd.DataFrame,
        test_size: float,
        val_size: float,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_val, test = train_test_split(
            df, test_size=test_size, stratify=df["label"], random_state=self.seed
        )
        relative_val = val_size / (1.0 - test_size)
        train, val = train_test_split(
            train_val,
            test_size=relative_val,
            stratify=train_val["label"],
            random_state=self.seed,
        )
        return train, val, test

    def _grouped_split(
        self,
        df: pd.DataFrame,
        test_size: float,
        val_size: float,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        groups = (
            df.groupby("group_id", as_index=False)
            .agg(label=("label", "max"), n=("label", "size"))
        )
        stratify = groups["label"] if groups["label"].nunique() > 1 else None
        try:
            g_trainval, g_test = train_test_split(
                groups,
                test_size=test_size,
                stratify=stratify,
                random_state=self.seed,
            )
            relative_val = val_size / (1.0 - test_size)
            g_train, g_val = train_test_split(
                g_trainval,
                test_size=relative_val,
                stratify=g_trainval["label"] if stratify is not None else None,
                random_state=self.seed,
            )
        except ValueError:
            logger.warning("Grouped stratify failed; splitting groups without stratify.")
            g_trainval, g_test = train_test_split(
                groups, test_size=test_size, random_state=self.seed
            )
            relative_val = val_size / (1.0 - test_size)
            g_train, g_val = train_test_split(
                g_trainval, test_size=relative_val, random_state=self.seed
            )

        train = df[df["group_id"].isin(set(g_train["group_id"]))]
        val = df[df["group_id"].isin(set(g_val["group_id"]))]
        test = df[df["group_id"].isin(set(g_test["group_id"]))]
        logger.info(
            "Grouped split using %d groups (train=%d val=%d test=%d)",
            len(groups),
            g_train["group_id"].nunique(),
            g_val["group_id"].nunique(),
            g_test["group_id"].nunique(),
        )
        return train, val, test

    @staticmethod
    def save_splits(
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        split_dir: Optional[Path] = None,
    ) -> Path:
        split_dir = Path(split_dir or getattr(config, "SPLIT_DIR", config.PROCESSED_DATA_DIR / "splits"))
        split_dir.mkdir(parents=True, exist_ok=True)
        train.to_csv(split_dir / "train.csv", index=False)
        val.to_csv(split_dir / "val.csv", index=False)
        test.to_csv(split_dir / "test.csv", index=False)
        logger.info("Saved grouped splits to %s", split_dir)
        return split_dir

    @staticmethod
    def load_splits(split_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        split_dir = Path(split_dir or getattr(config, "SPLIT_DIR", config.PROCESSED_DATA_DIR / "splits"))
        train = pd.read_csv(split_dir / "train.csv")
        val = pd.read_csv(split_dir / "val.csv")
        test = pd.read_csv(split_dir / "test.csv")
        logger.info(
            "Loaded saved splits: train=%d val=%d test=%d",
            len(train),
            len(val),
            len(test),
        )
        return train, val, test

    @staticmethod
    def downsample_majority(
        dataframe: pd.DataFrame,
        ratio: Optional[float] = config.BENIGN_TO_MALIGNANT_RATIO,
    ) -> pd.DataFrame:
        """Keep all melanoma rows; sample benign rows to `ratio` : 1 on train only."""
        if not ratio or ratio <= 0 or "label" not in dataframe.columns:
            return dataframe
        mal = dataframe[dataframe["label"] == 1]
        ben = dataframe[dataframe["label"] == 0]
        if mal.empty or ben.empty:
            return dataframe
        n_ben = min(len(ben), max(1, int(len(mal) * ratio)))
        if n_ben >= len(ben):
            return dataframe
        ben = ben.sample(n=n_ben, random_state=config.SEED)
        balanced = pd.concat([mal, ben], ignore_index=True).sample(
            frac=1.0, random_state=config.SEED
        )
        logger.info(
            "Downsampled majority: %d -> %d rows (%d malignant, %d benign, ratio=%.1f:1)",
            len(dataframe),
            len(balanced),
            len(mal),
            n_ben,
            ratio,
        )
        return balanced.reset_index(drop=True)

    @staticmethod
    def oversample_malignant(
        dataframe: pd.DataFrame,
        factor: int = config.MALIGNANT_OVERSAMPLE_FACTOR,
    ) -> pd.DataFrame:
        """Repeat malignant rows so the model sees more melanoma examples per epoch."""
        if factor <= 1 or "label" not in dataframe.columns:
            return dataframe
        mal = dataframe[dataframe["label"] == 1]
        ben = dataframe[dataframe["label"] == 0]
        if mal.empty:
            return dataframe
        mal_up = pd.concat([mal] * factor, ignore_index=True)
        balanced = pd.concat([ben, mal_up], ignore_index=True).sample(
            frac=1.0, random_state=config.SEED
        )
        logger.info(
            "Oversampled train: %d -> %d rows (%d malignant after x%d)",
            len(dataframe),
            len(balanced),
            len(mal_up),
            factor,
        )
        return balanced.reset_index(drop=True)

    def compute_class_weights(self, labels: np.ndarray) -> Dict[int, float]:
        """Compute balanced class weights for imbalanced training."""
        classes = np.unique(labels)
        weights = compute_class_weight("balanced", classes=classes, y=labels)
        mapping = {int(c): float(w) for c, w in zip(classes, weights)}
        logger.info("Class weights: %s", mapping)
        return mapping

    @staticmethod
    def add_balanced_sample_weights(dataframe: pd.DataFrame) -> pd.DataFrame:
        """sklearn 'balanced' weights: n / (2 * n_class). Train split only."""
        out = dataframe.copy()
        y = out["label"].astype(int).values
        n = max(len(y), 1)
        n_pos = max(int((y == 1).sum()), 1)
        n_neg = max(int((y == 0).sum()), 1)
        w_pos = n / (2.0 * n_pos)
        w_neg = n / (2.0 * n_neg)
        out["sample_weight"] = np.where(out["label"].astype(int) == 1, w_pos, w_neg).astype(np.float32)
        logger.info("Sample weights: malignant=%.3f benign=%.3f", w_pos, w_neg)
        return out

    def make_tf_dataset(
        self,
        dataframe: pd.DataFrame,
        batch_size: Optional[int] = None,
        shuffle: bool = True,
        augment: bool = False,
        inference_preprocess: Optional[bool] = None,
    ) -> tf.data.Dataset:
        """Build a performant tf.data pipeline from a dataframe."""
        if augment and shuffle is False:
            logger.warning("Augmentation requested on a non-shuffled split; refusing.")
            augment = False

        batch_size = int(batch_size or config.BATCH_SIZE)
        paths = dataframe["image_path"].astype(str).values
        labels = dataframe["label"].values.astype(np.float32)
        if "sample_weight" in dataframe.columns:
            weights = dataframe["sample_weight"].values.astype(np.float32)
        else:
            weights = np.ones(len(dataframe), dtype=np.float32)
        ds = tf.data.Dataset.from_tensor_slices((paths, labels, weights))

        if shuffle:
            buf = min(len(dataframe), max(2000, batch_size * 128))
            ds = ds.shuffle(buffer_size=buf, seed=self.seed, reshuffle_each_iteration=True)

        match_inference = (
            self.dataset_key == "isic_dicm_17k"
            if inference_preprocess is None
            else bool(inference_preprocess)
        )
        img_h, img_w = int(self.img_size[0]), int(self.img_size[1])

        def _decode_resize(path, label, weight):
            img = tf.io.read_file(path)
            img = tf.image.decode_image(img, channels=3, expand_animations=False)
            img.set_shape([None, None, 3])
            img = tf.image.resize(img, self.img_size)
            img = tf.cast(img, tf.float32) / 255.0
            return img, label, weight

        def _clinical_numpy(path_bytes):
            import cv2
            from preprocessing.image_preprocess import ImagePreprocessor

            path = path_bytes.numpy().decode("utf-8") if hasattr(path_bytes, "numpy") else str(path_bytes)
            raw = cv2.imread(path)
            if raw is None:
                raise FileNotFoundError(path)
            rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            processed = ImagePreprocessor(
                target_size=self.img_size, apply_hair_removal=True
            ).process_for_model(rgb)
            return processed.astype(np.float32)

        def _clinical_load(path, label, weight):
            img = tf.py_function(_clinical_numpy, [path], tf.float32)
            img.set_shape([img_h, img_w, 3])
            return img, label, weight

        if match_inference:
            logger.info(
                "ISIC-DICM-17K preprocess matches inference: hair removal + denoise + resize 224 + /255 (no CLAHE)."
            )
            ds = ds.map(_clinical_load, num_parallel_calls=tf.data.AUTOTUNE)
        else:
            ds = ds.map(_decode_resize, num_parallel_calls=tf.data.AUTOTUNE)

        if augment:
            from preprocessing.augmentation import (
                get_dicm_train_augmentation_fn,
                get_tf_data_augmentation_fn,
            )

            logger.info("Training augmentation on (val/test stay unaugmented).")
            aug_fn = (
                get_dicm_train_augmentation_fn()
                if self.dataset_key == "isic_dicm_17k"
                else get_tf_data_augmentation_fn()
            )

            def _augment(image, label, weight):
                image, label = aug_fn(image, label)
                return image, label, weight

            ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds


def create_synthetic_demo_dataset(
    n_samples: int = 200,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Create a small synthetic RGB dataset for smoke-testing the pipeline
    when SIIM-ISIC images are not yet downloaded.
    """
    import cv2

    out = Path(output_dir or config.PROCESSED_DATA_DIR / "synthetic")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    rng = np.random.default_rng(config.SEED)

    for i in range(n_samples):
        label = 1 if i % 2 == 0 else 0
        img = rng.integers(40, 200, size=(*config.IMG_SIZE, 3), dtype=np.uint8)
        # Draw a blob (lesion-like)
        center = (rng.integers(60, 164), rng.integers(60, 164))
        radius = int(rng.integers(20, 50))
        color = (30, 20, 10) if label == 1 else (180, 140, 120)
        cv2.circle(img, center, radius, color, -1)
        path = out / f"synth_{i:04d}_l{label}.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        rows.append({"image_path": str(path), "label": label})

    df = pd.DataFrame(rows)
    df.to_csv(out / "metadata.csv", index=False)
    logger.info("Synthetic demo dataset created: %d images at %s", n_samples, out)
    return df
