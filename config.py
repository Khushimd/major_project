"""
Configuration module for Melanoma Clinical Decision Support System.
Centralizes all hyperparameters, paths, and application settings.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent
DATASET_DIR: Path = BASE_DIR / "dataset"
RAW_DATA_DIR: Path = DATASET_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATASET_DIR / "processed"
HAM10000_DIR: Path = BASE_DIR  # HAM10000_metadata.csv + image parts at project root
MODELS_DIR: Path = BASE_DIR / "models" / "saved"
CHECKPOINTS_DIR: Path = BASE_DIR / "models" / "checkpoints"
UPLOAD_DIR: Path = BASE_DIR / "static" / "uploads"
RESULTS_DIR: Path = BASE_DIR / "static" / "results"
REPORTS_DIR: Path = BASE_DIR / "reports" / "generated"
DATABASE_PATH: Path = BASE_DIR / "database" / "melanoma_cdss.db"
LOGS_DIR: Path = BASE_DIR / "logs"

# Ensure critical directories exist
for _d in (
    MODELS_DIR,
    CHECKPOINTS_DIR,
    UPLOAD_DIR,
    RESULTS_DIR,
    REPORTS_DIR,
    LOGS_DIR,
    PROCESSED_DATA_DIR,
    PROCESSED_DATA_DIR / "splits",
    PROCESSED_DATA_DIR / "isic_dicm_17k",
    MODELS_DIR / "experiments",
):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Image Preprocessing
# ---------------------------------------------------------------------------
IMG_SIZE: Tuple[int, int] = (224, 224)
IMG_CHANNELS: int = 3
NORMALIZE_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
NORMALIZE_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# Training Hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE: int = 32
EPOCHS: int = 50
FINE_TUNE_EPOCHS: int = 20
LEARNING_RATE: float = 1e-3
FINE_TUNE_LR: float = 1e-5
WEIGHT_DECAY: float = 1e-4
DROPOUT_RATE: float = 0.4
L2_REG: float = 1e-4
PATIENCE_EARLY_STOP: int = 8
PATIENCE_REDUCE_LR: int = 4
REDUCE_LR_FACTOR: float = 0.5
MIN_LR: float = 1e-7
VALIDATION_SPLIT: float = 0.15
TEST_SPLIT: float = 0.15
SEED: int = 42
USE_FOCAL_LOSS: bool = False  # BCE + oversampling is more stable on this imbalance
FOCAL_LOSS_GAMMA: float = 2.0
FOCAL_LOSS_ALPHA: float = 0.75  # if focal is enabled, weight the melanoma class higher
FINE_TUNE_UNFREEZE_LAYERS: int = 20
# Downsample majority class on the train split only (None = use every benign image)
BENIGN_TO_MALIGNANT_RATIO: float | None = None
USE_MIXED_PRECISION: bool = True
USE_TTA: bool = False  # Keep False for deterministic confidence scores
TTA_STEPS: int = 5
# Oversample malignant rows during training. Experiments override this.
# Default 1 = no oversampling; 4x was overfitting (train AUC 0.97 vs val 0.70).
MALIGNANT_OVERSAMPLE_FACTOR: int = 1
# If True, apply extra melanoma-only photometric / hair / cutout augs.
# Default False: same realistic dermoscopy aug for all training images.
MELANOMA_STRONG_AUG: bool = False
# Train-set class balancing: "weighted_bce" | "focal" | "oversample" | "none"
BALANCE_MODE: str = "weighted_bce"
# Decision thresholds scanned on VALIDATION only (never on test).
THRESHOLD_CANDIDATES: List[float] = [
    0.10, 0.15, 0.20, 0.22, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
]
SPLIT_DIR: Path = PROCESSED_DATA_DIR / "splits"
ISIC_DICM_IMAGE_DIR: Path = BASE_DIR / "images"
ISIC_DICM_LABEL_DIR: Path = DATASET_DIR / "isic_dicm_17k"
ISIC_DICM_SPLIT_DIR: Path = PROCESSED_DATA_DIR / "isic_dicm_17k"
EXPERIMENTS_DIR: Path = MODELS_DIR / "experiments"
ADAPT_BATCH_SIZE: int = 16
ADAPT_EPOCHS: int = 8

# Hybrid = EfficientNetB0 + MobileNetV2 combined for better predictions
# Match ensemble_meta.json (0.85 / 0.15). Threshold 0.45 hid almost all
# malignant scores — evaluation optimal_threshold is ~0.06 for this model.
ENSEMBLE_WEIGHTS: Tuple[float, float] = (0.85, 0.15)
HYBRID_METHOD: str = "soft_voting"
DECISION_THRESHOLD: float = 0.22
PREDICTION_CACHE_VERSION: str = "v13-combined-0.22"

# Class labels
CLASS_NAMES: List[str] = ["Benign", "Malignant"]
NUM_CLASSES: int = 2

# ---------------------------------------------------------------------------
# Model Paths
# ---------------------------------------------------------------------------
EFFICIENTNET_PATH: Path = MODELS_DIR / "efficientnetb0_melanoma.keras"
MOBILENET_PATH: Path = MODELS_DIR / "mobilenetv2_melanoma.keras"
ENSEMBLE_META_PATH: Path = MODELS_DIR / "ensemble_meta.json"
BEST_MODEL_META_PATH: Path = MODELS_DIR / "best_model_meta.json"
METRICS_PATH: Path = MODELS_DIR / "evaluation_metrics.json"
INFERENCE_CONFIG_PATH: Path = MODELS_DIR / "inference_config.json"
UNET_PATH: Path = MODELS_DIR / "unet_segmentation.keras"

# ---------------------------------------------------------------------------
# Flask Application
# ---------------------------------------------------------------------------
FLASK_SECRET_KEY: str = os.environ.get(
    "FLASK_SECRET_KEY", "melanoma-cdss-dev-key-change-in-production"
)
FLASK_HOST: str = "0.0.0.0"
FLASK_PORT: int = 5000
FLASK_DEBUG: bool = os.environ.get("FLASK_DEBUG", "1") == "1"
MAX_CONTENT_LENGTH: int = 16 * 1024 * 1024  # 16 MB
ALLOWED_EXTENSIONS: set = {"png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"}

# ---------------------------------------------------------------------------
# Medical Disclaimer
# ---------------------------------------------------------------------------
MEDICAL_DISCLAIMER: str = (
    "This system is intended as a clinical decision support tool only "
    "and should not replace professional medical diagnosis."
)

# ---------------------------------------------------------------------------
# Image Quality Thresholds
# ---------------------------------------------------------------------------
BLUR_THRESHOLD: float = 100.0  # Laplacian variance below => blurry
MIN_RESOLUTION: Tuple[int, int] = (64, 64)
BRIGHTNESS_LOW: float = 30.0
BRIGHTNESS_HIGH: float = 225.0

# ---------------------------------------------------------------------------
# Dataset Configuration (extensible for HAM10000)
# ---------------------------------------------------------------------------
DATASETS: Dict[str, Dict] = {
    "siim_isic_2020": {
        "name": "SIIM-ISIC 2020",
        "raw_dir": str(RAW_DATA_DIR),
        "csv_name": "train.csv",
        "image_col": "image_name",
        "label_col": "target",
        "image_ext": ".jpg",
    },
    "ham10000": {
        "name": "HAM10000",
        "raw_dir": str(HAM10000_DIR),
        "csv_name": "HAM10000_metadata.csv",
        "image_col": "image_id",
        "label_col": "dx",
        "malignant_labels": ["mel"],
        "image_ext": ".jpg",
    },
    "combined": {
        "name": "SIIM-ISIC 2020 + HAM10000",
        "sources": ["siim_isic_2020", "ham10000"],
    },
    "isic_dicm_17k": {
        "name": "ISIC-DICM-17K",
        "raw_dir": str(DATASET_DIR / "isic_dicm_17k"),
        "image_dir": str(BASE_DIR / "images"),
        "csv_name": "train-set-metadata.csv",
        "image_col": "isic_id",
        "label_col": "class",
        "image_ext": ".jpg",
        "official_train_csv": "train-set-metadata.csv",
        "official_valid_csv": "valid-set-metadata.csv",
    },
    "custom": {
        "name": "Custom",
        "raw_dir": str(DATASET_DIR / "custom"),
        "csv_name": "labels.csv",
        "image_col": "image_name",
        "label_col": "target",
        "image_ext": ".jpg",
    },
}

ACTIVE_DATASET: str = "combined"

# ---------------------------------------------------------------------------
# Inference cache (invalidated after retraining)
# ---------------------------------------------------------------------------
def get_prediction_cache_version() -> str:
    """Read cache version from inference config after training, else config default."""
    path = INFERENCE_CONFIG_PATH
    if path.exists():
        try:
            import json

            data = json.loads(path.read_text(encoding="utf-8-sig"))
            version = data.get("cache_version")
            if version:
                return str(version)
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    return PREDICTION_CACHE_VERSION

# ---------------------------------------------------------------------------
# Risk Level Mapping (AJCC)
# ---------------------------------------------------------------------------
RISK_LEVELS: Dict[str, str] = {
    "Stage 0": "Low",
    "Stage IA": "Low-Moderate",
    "Stage IB": "Moderate",
    "Stage IIA": "Moderate-High",
    "Stage IIB": "High",
    "Stage IIC": "High",
    "Stage III": "Very High",
    "Stage IV": "Critical",
}
