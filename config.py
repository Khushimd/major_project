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
HAM10000_DIR: Path = DATASET_DIR / "ham10000"
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
USE_FOCAL_LOSS: bool = True
FOCAL_LOSS_GAMMA: float = 2.0
FOCAL_LOSS_ALPHA: float = 0.25
USE_MIXED_PRECISION: bool = True
USE_TTA: bool = False  # Keep False for deterministic confidence scores
TTA_STEPS: int = 5

# Hybrid = EfficientNetB0 + MobileNetV2 combined for better predictions
# soft_voting: weighted probability average
# majority_voting: highest-frequency class vote (tie → soft vote)
ENSEMBLE_WEIGHTS: Tuple[float, float] = (0.85, 0.15)
HYBRID_METHOD: str = "soft_voting"  # or "majority_voting"
# Lower than 0.5 — dataset is ~98% benign so models are calibrated low
DECISION_THRESHOLD: float = 0.30

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
}

ACTIVE_DATASET: str = "siim_isic_2020"

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
