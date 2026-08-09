"""
Dataset loading utilities for SIIM-ISIC 2020 (primary) and HAM10000 (extensible).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import config
from utils.logger import setup_logger

logger = setup_logger("dataset")


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

    def _find_csv(self, raw_dir: Path, csv_path: Optional[str]) -> Path:
        """Locate train.csv or ISIC 2020 ground-truth / metadata CSV."""
        if csv_path:
            path = Path(csv_path)
            if path.exists():
                return path
            raise FileNotFoundError(f"CSV not found: {path}")

        candidates = [
            raw_dir / self.cfg["csv_name"],
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
        for key in ("target", "benign_malignant", "diagnosis", "melanoma"):
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
        raw_dir = Path(self.cfg["raw_dir"])
        csv_file = self._find_csv(raw_dir, csv_path)

        df = pd.read_csv(csv_file)
        image_col, label_col = self._detect_columns(df)
        ext = self.cfg.get("image_ext", ".jpg")
        logger.info("Using labels file: %s (image=%s, label=%s)", csv_file.name, image_col, label_col)

        # Resolve image paths — search common subfolders / zip extract layouts
        search_dirs = [
            raw_dir,
            raw_dir / "train",
            raw_dir / "jpeg" / "train",
            raw_dir / "JPEG" / "train",
            raw_dir / "images",
            raw_dir / "ISIC_2020_Training_JPEG",
            raw_dir / "ISIC_2020_Training_JPEG" / "train",
        ]
        # Include one level of subfolders that contain many jpgs
        for child in raw_dir.iterdir() if raw_dir.exists() else []:
            if child.is_dir():
                search_dirs.append(child)

        def resolve_path(name: str) -> Optional[str]:
            stem = Path(str(name)).stem
            candidates = [stem, f"{stem}{ext}", f"{stem}.png", f"{stem}.jpeg", f"{stem}.JPG"]
            for d in search_dirs:
                for c in candidates:
                    p = d / c
                    if p.exists():
                        return str(p)
            return None

        paths, labels = [], []
        malignant_labels = set(self.cfg.get("malignant_labels", []))

        for _, row in df.iterrows():
            path = resolve_path(row[image_col])
            if path is None:
                continue
            label = self._to_binary_label(row[label_col], malignant_labels)
            paths.append(path)
            labels.append(label)

        if not paths:
            raise FileNotFoundError(
                "CSV loaded but no images were found. Extract the JPEG zip into "
                f"{raw_dir}/train/ so image files match names in the CSV."
            )

        self.df = pd.DataFrame({"image_path": paths, "label": labels})
        logger.info(
            "Loaded %s samples (%d malignant, %d benign) from %s",
            len(self.df),
            int(self.df["label"].sum()),
            int((self.df["label"] == 0).sum()),
            self.cfg["name"],
        )
        return self.df

    def split(
        self,
        test_size: float = config.TEST_SPLIT,
        val_size: float = config.VALIDATION_SPLIT,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Stratified train / validation / test split."""
        if self.df is None:
            self.load_metadata()

        train_val, test = train_test_split(
            self.df,
            test_size=test_size,
            stratify=self.df["label"],
            random_state=self.seed,
        )
        relative_val = val_size / (1.0 - test_size)
        train, val = train_test_split(
            train_val,
            test_size=relative_val,
            stratify=train_val["label"],
            random_state=self.seed,
        )
        logger.info(
            "Split → train=%d val=%d test=%d",
            len(train), len(val), len(test),
        )
        return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

    def compute_class_weights(self, labels: np.ndarray) -> Dict[int, float]:
        """Compute balanced class weights for imbalanced training."""
        classes = np.unique(labels)
        weights = compute_class_weight("balanced", classes=classes, y=labels)
        mapping = {int(c): float(w) for c, w in zip(classes, weights)}
        logger.info("Class weights: %s", mapping)
        return mapping

    def make_tf_dataset(
        self,
        dataframe: pd.DataFrame,
        batch_size: int = config.BATCH_SIZE,
        shuffle: bool = True,
        augment: bool = False,
    ) -> tf.data.Dataset:
        """Build a performant tf.data pipeline from a dataframe."""
        paths = dataframe["image_path"].values
        labels = dataframe["label"].values.astype(np.float32)
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))

        if shuffle:
            ds = ds.shuffle(buffer_size=min(len(dataframe), 1000), seed=self.seed)

        def _load(path, label):
            img = tf.io.read_file(path)
            img = tf.image.decode_image(img, channels=3, expand_animations=False)
            img.set_shape([None, None, 3])
            img = tf.image.resize(img, self.img_size)
            img = tf.cast(img, tf.float32) / 255.0
            return img, label

        ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

        if augment:
            from preprocessing.augmentation import get_tf_data_augmentation_fn
            ds = ds.map(get_tf_data_augmentation_fn(), num_parallel_calls=tf.data.AUTOTUNE)

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
