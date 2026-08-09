"""
Training callbacks factory.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import tensorflow as tf
from tensorflow import keras

import config


def get_callbacks(
    model_name: str,
    checkpoint_dir: Path = config.CHECKPOINTS_DIR,
) -> List[keras.callbacks.Callback]:
    """
    Build EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, and optional TensorBoard callbacks.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"{model_name}_best.keras"
    log_dir = config.LOGS_DIR / "tensorboard" / model_name

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=config.PATIENCE_EARLY_STOP,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.REDUCE_LR_FACTOR,
            patience=config.PATIENCE_REDUCE_LR,
            min_lr=config.MIN_LR,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            str(config.LOGS_DIR / f"{model_name}_history.csv"),
            append=False,
        ),
    ]

    # TensorBoard is optional — skip if the package is not installed
    try:
        import tensorboard  # noqa: F401

        callbacks.append(
            keras.callbacks.TensorBoard(log_dir=str(log_dir), histogram_freq=0)
        )
    except ImportError:
        pass

    return callbacks
