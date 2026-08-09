"""
Grad-CAM explainability for melanoma predictions.

Rules:
- Benign  → no heatmap; message "No malignant regions detected."
- Malignant → generate Grad-CAM overlay highlighting lesion regions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import tensorflow as tf

import config
from models.architecture import get_last_conv_layer_name
from utils.logger import setup_logger

logger = setup_logger("gradcam")


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for binary classifiers.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        layer_name: Optional[str] = None,
    ) -> None:
        self.model = model
        self.layer_name = layer_name or self._resolve_layer_name()
        self.grad_model = self._build_grad_model()

    def _resolve_layer_name(self) -> str:
        try:
            name = get_last_conv_layer_name(self.model)
            logger.info("Grad-CAM target layer: %s", name)
            return name
        except ValueError:
            # Common EfficientNet / MobileNet last conv names
            for candidate in (
                "top_conv",
                "Conv_1",
                "block7a_project_conv",
                "out_relu",
            ):
                try:
                    self.model.get_layer(candidate)
                    return candidate
                except ValueError:
                    continue
            raise

    def _find_layer(self, name: str) -> tf.keras.layers.Layer:
        """Find a layer by name, searching nested models."""
        try:
            return self.model.get_layer(name)
        except ValueError:
            pass
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.Model):
                try:
                    return layer.get_layer(name)
                except ValueError:
                    continue
        raise ValueError(f"Layer '{name}' not found in model.")

    def _build_grad_model(self) -> tf.keras.Model:
        conv_layer = self._find_layer(self.layer_name)
        # Use the underlying input tensor to avoid structure mismatch warnings
        model_input = self.model.input
        return tf.keras.Model(
            inputs=model_input,
            outputs=[conv_layer.output, self.model.output],
        )

    def compute_heatmap(
        self,
        image_batch: np.ndarray,
        eps: float = 1e-8,
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for the malignant class.

        Args:
            image_batch: Preprocessed batch (1, H, W, 3).

        Returns:
            Heatmap as float32 array in [0, 1], shape (H_conv, W_conv).
        """
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(image_batch)
            # Binary classifier: maximize malignant probability
            loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + eps)
        return heatmap.numpy()

    def overlay_heatmap(
        self,
        original_rgb: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.58,
        colormap: int = cv2.COLORMAP_TURBO,
    ) -> np.ndarray:
        """
        Overlay heatmap on the original image and draw lesion contours.
        """
        h, w = original_rgb.shape[:2]
        heat_resized = cv2.resize(heatmap, (w, h))
        # Boost contrast so hot spots read clearly in the UI
        heat_norm = np.clip(heat_resized, 0, 1)
        heat_norm = np.power(heat_norm, 0.75)
        heat_uint8 = np.uint8(255 * heat_norm)
        heat_color = cv2.applyColorMap(heat_uint8, colormap)
        heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

        base = original_rgb.copy()
        if base.dtype != np.uint8:
            base = (np.clip(base, 0, 1) * 255).astype(np.uint8)

        overlay = cv2.addWeighted(base, 1 - alpha, heat_color, alpha, 0)

        # Highlight lesion boundaries from high-activation regions
        _, mask = cv2.threshold(heat_uint8, 120, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)

        return overlay

    def generate(
        self,
        image_batch: np.ndarray,
        original_rgb: np.ndarray,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full Grad-CAM pipeline: heatmap + overlay, optionally save.

        Returns:
            (heatmap, overlay_rgb)
        """
        heatmap = self.compute_heatmap(image_batch)
        overlay = self.overlay_heatmap(original_rgb, heatmap)

        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(path), bgr)
            logger.info("Grad-CAM saved to %s", path)

        return heatmap, overlay


def generate_gradcam_if_malignant(
    prediction: str,
    model: tf.keras.Model,
    image_batch: np.ndarray,
    original_rgb: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
) -> dict:
    """
    Conditional Grad-CAM generation based on prediction rules.

    Returns:
        Dict with keys: generated (bool), message, heatmap_path, overlay (optional)
    """
    if prediction.lower() == "benign":
        return {
            "generated": False,
            "message": "No malignant regions detected.",
            "heatmap_path": None,
            "overlay": None,
        }

    cam = GradCAM(model)
    _, overlay = cam.generate(image_batch, original_rgb, save_path=save_path)
    return {
        "generated": True,
        "message": "Malignant regions highlighted via Grad-CAM.",
        "heatmap_path": str(save_path) if save_path else None,
        "overlay": overlay,
    }
