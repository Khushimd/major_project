"""
Data augmentation utilities for melanoma lesion images.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf

import config


def get_keras_augmentation() -> tf.keras.Sequential:
    """
    Return a Keras Sequential augmentation pipeline for training.

    Includes rotation, flips, brightness, contrast, zoom, and translation.
    """
    layers_list = [
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
        tf.keras.layers.RandomContrast(0.2),
    ]
    # RandomBrightness availability varies across TF / Keras versions
    try:
        layers_list.append(tf.keras.layers.RandomBrightness(0.2))
    except Exception:
        pass

    return tf.keras.Sequential(layers_list, name="augmentation")



def get_tf_data_augmentation_fn():
    """Return a callable suitable for tf.data map for on-the-fly augmentation."""
    augmenter = get_keras_augmentation()

    def _augment(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        image = augmenter(image, training=True)
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image, label

    return _augment


def apply_tta(
    model: tf.keras.Model,
    image_batch: np.ndarray,
    steps: int = config.TTA_STEPS,
) -> np.ndarray:
    """
    Test-Time Augmentation: average predictions over augmented views.

    Args:
        model: Trained Keras model.
        image_batch: Array of shape (1, H, W, 3) or (N, H, W, 3).
        steps: Number of augmented forward passes.

    Returns:
        Averaged probability predictions.
    """
    preds = [model.predict(image_batch, verbose=0)]

    for _ in range(steps - 1):
        aug = image_batch.copy()
        # Random horizontal flip
        if np.random.rand() > 0.5:
            aug = np.flip(aug, axis=2)
        # Random vertical flip
        if np.random.rand() > 0.5:
            aug = np.flip(aug, axis=1)
        # Slight brightness jitter
        factor = 1.0 + np.random.uniform(-0.1, 0.1)
        aug = np.clip(aug * factor, 0.0, 1.0)
        preds.append(model.predict(aug, verbose=0))

    return np.mean(preds, axis=0)


AUGMENTATION_CONFIG: Dict[str, Any] = {
    "rotation_range": 20,
    "width_shift_range": 0.1,
    "height_shift_range": 0.1,
    "shear_range": 0.1,
    "zoom_range": 0.15,
    "horizontal_flip": True,
    "vertical_flip": True,
    "brightness_range": [0.8, 1.2],
    "fill_mode": "reflect",
}


def get_image_data_generator(
    rescale: Optional[float] = None,
) -> "tf.keras.preprocessing.image.ImageDataGenerator":
    """Legacy ImageDataGenerator for compatibility with older training scripts."""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    params = dict(AUGMENTATION_CONFIG)
    if rescale is not None:
        params["rescale"] = rescale
    return ImageDataGenerator(**params)
