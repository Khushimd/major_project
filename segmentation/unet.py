"""
Optional U-Net lesion segmentation module (bonus feature).
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import config
from utils.logger import setup_logger

logger = setup_logger("segmentation")


def conv_block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    x = layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"{name}_c1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"{name}_c2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    return x


def build_unet(
    input_shape: Tuple[int, int, int] = (*config.IMG_SIZE, 3),
    num_classes: int = 1,
) -> keras.Model:
    """
    Lightweight U-Net for binary lesion segmentation.
    """
    inputs = keras.Input(shape=input_shape, name="unet_input")

    # Encoder
    c1 = conv_block(inputs, 32, "e1")
    p1 = layers.MaxPooling2D(2, name="p1")(c1)
    c2 = conv_block(p1, 64, "e2")
    p2 = layers.MaxPooling2D(2, name="p2")(c2)
    c3 = conv_block(p2, 128, "e3")
    p3 = layers.MaxPooling2D(2, name="p3")(c3)

    # Bottleneck
    b = conv_block(p3, 256, "bottleneck")

    # Decoder
    u3 = layers.UpSampling2D(2, name="u3")(b)
    u3 = layers.Concatenate(name="cat3")([u3, c3])
    d3 = conv_block(u3, 128, "d3")

    u2 = layers.UpSampling2D(2, name="u2")(d3)
    u2 = layers.Concatenate(name="cat2")([u2, c2])
    d2 = conv_block(u2, 64, "d2")

    u1 = layers.UpSampling2D(2, name="u1")(d2)
    u1 = layers.Concatenate(name="cat1")([u1, c1])
    d1 = conv_block(u1, 32, "d1")

    outputs = layers.Conv2D(
        num_classes, 1, activation="sigmoid", name="segmentation_mask"
    )(d1)
    model = keras.Model(inputs, outputs, name="UNet_LesionSegmentation")
    logger.info("Built U-Net | params=%s", f"{model.count_params():,}")
    return model


class LesionSegmenter:
    """
    Wrapper that uses a trained U-Net when available, otherwise falls back
    to classical OpenCV thresholding for a coarse lesion mask.
    """

    def __init__(self, model_path=config.UNET_PATH) -> None:
        self.model: Optional[keras.Model] = None
        path = config.UNET_PATH if model_path is None else model_path
        if path.exists():
            self.model = keras.models.load_model(str(path), compile=False)
            logger.info("Loaded U-Net from %s", path)
        else:
            logger.info("U-Net weights not found; using classical segmentation fallback.")

    def segment(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Return binary mask (H, W) uint8 {0, 255}.
        """
        if self.model is not None:
            return self._unet_segment(image_rgb)
        return self._classical_segment(image_rgb)

    def _unet_segment(self, image_rgb: np.ndarray) -> np.ndarray:
        h, w = image_rgb.shape[:2]
        resized = cv2.resize(image_rgb, config.IMG_SIZE)
        batch = np.expand_dims(resized.astype(np.float32) / 255.0, 0)
        pred = self.model.predict(batch, verbose=0)[0, :, :, 0]
        mask = (pred >= 0.5).astype(np.uint8) * 255
        return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def _classical_segment(image_rgb: np.ndarray) -> np.ndarray:
        """Otsu thresholding on grayscale + morphological cleanup."""
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def overlay_mask(
        self,
        image_rgb: np.ndarray,
        mask: Optional[np.ndarray] = None,
        color: Tuple[int, int, int] = (0, 255, 180),
        alpha: float = 0.35,
    ) -> np.ndarray:
        """Overlay segmentation mask contour on the original image."""
        if mask is None:
            mask = self.segment(image_rgb)
        overlay = image_rgb.copy()
        if overlay.dtype != np.uint8:
            overlay = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)
        colored = overlay.copy()
        colored[mask > 0] = (
            (1 - alpha) * colored[mask > 0] + alpha * np.array(color)
        ).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(colored, contours, -1, (255, 220, 0), 2)
        return colored
