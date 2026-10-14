"""
Grad-CAM explainability for melanoma predictions.

Benign → no heatmap. Malignant → attention overlay on the lesion.
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
    """Gradient-weighted Class Activation Mapping for binary classifiers."""

    def __init__(
        self,
        model: tf.keras.Model,
        layer_name: Optional[str] = None,
    ) -> None:
        self.model = model
        self.layer_name = layer_name or self._resolve_layer_name()
        self._nested_backbone: Optional[tf.keras.Model] = None
        self.grad_model = self._build_grad_model()

    def _resolve_layer_name(self) -> str:
        try:
            name = get_last_conv_layer_name(self.model)
            logger.info("Grad-CAM target layer: %s", name)
            return name
        except ValueError:
            for candidate in (
                "block6a_expand_conv",
                "block5c_project_conv",
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

    def _find_backbone(self) -> Optional[tf.keras.Model]:
        for layer in self.model.layers:
            if not isinstance(layer, tf.keras.Model):
                continue
            try:
                layer.get_layer(self.layer_name)
                return layer
            except ValueError:
                continue
        return None

    def _rebuild_nested_grad_model(self, backbone: tf.keras.Model) -> tf.keras.Model:
        conv = backbone.get_layer(self.layer_name)
        inner = tf.keras.Model(
            inputs=backbone.inputs,
            outputs=[conv.output, backbone.output],
            name="gradcam_inner",
        )
        shape = self.model.input_shape
        if isinstance(shape, (list, tuple)) and shape and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        x_in = tf.keras.Input(shape=shape[1:], name="gradcam_input")
        z = x_in
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            if layer is backbone or layer.name == backbone.name:
                break
            z = layer(z)
        conv_out, feat = inner(z)
        y = feat
        seen = False
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            if layer is backbone or layer.name == backbone.name:
                seen = True
                continue
            if not seen:
                continue
            y = layer(y)
        logger.info("Grad-CAM rebuilt through nested backbone %s", backbone.name)
        return tf.keras.Model(x_in, [conv_out, y], name="gradcam_model")

    def _build_grad_model(self) -> tf.keras.Model:
        conv_layer = self._find_layer(self.layer_name)
        inp = self.model.inputs[0] if getattr(self.model, "inputs", None) else self.model.input
        out = self.model.outputs[0] if getattr(self.model, "outputs", None) else self.model.output
        try:
            return tf.keras.Model(inputs=inp, outputs=[conv_layer.output, out])
        except ValueError:
            backbone = self._find_backbone()
            if backbone is None:
                raise
            return self._rebuild_nested_grad_model(backbone)

    def _heatmap_from_maps(
        self, conv_outputs: tf.Tensor, grads: Optional[tf.Tensor], eps: float
    ) -> np.ndarray:
        conv_map = conv_outputs[0] if len(getattr(conv_outputs, "shape", ())) == 4 else conv_outputs
        activation_map = tf.reduce_mean(tf.nn.relu(conv_map), axis=-1)
        use_activations = grads is None
        if not use_activations:
            use_activations = float(tf.reduce_max(tf.abs(grads))) < 1e-10
        if use_activations:
            logger.warning("Grad-CAM gradients missing; using conv activations.")
            heatmap = activation_map
        else:
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            heatmap = conv_map @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.nn.relu(heatmap)
            if float(tf.reduce_max(heatmap)) < 1e-10:
                heatmap = activation_map
        heatmap = heatmap / (tf.reduce_max(heatmap) + eps)
        return heatmap.numpy()

    def _compute_heatmap_nested(self, x: tf.Tensor, eps: float) -> np.ndarray:
        backbone = self._nested_backbone
        with tf.GradientTape() as tape:
            z = x
            for layer in self.model.layers:
                if isinstance(layer, tf.keras.layers.InputLayer):
                    continue
                if layer is backbone or layer.name == backbone.name:
                    break
                z = layer(z, training=False)
            conv_outputs, features = self.grad_model(z, training=False)
            if isinstance(features, (list, tuple)):
                features = features[0]
            y = features
            seen_backbone = False
            for layer in self.model.layers:
                if isinstance(layer, tf.keras.layers.InputLayer):
                    continue
                if layer is backbone or layer.name == backbone.name:
                    seen_backbone = True
                    continue
                if not seen_backbone:
                    continue
                y = layer(y, training=False)
            pred = tf.reshape(y, (tf.shape(y)[0], -1))
            loss = pred[:, 0]
        grads = tape.gradient(loss, conv_outputs)
        return self._heatmap_from_maps(conv_outputs, grads, eps)

    def compute_heatmap(
        self,
        image_batch: np.ndarray,
        eps: float = 1e-8,
    ) -> np.ndarray:
        """Compute Grad-CAM heatmap for the malignant class in [0, 1]."""
        x = tf.convert_to_tensor(image_batch, dtype=tf.float32)
        if len(x.shape) == 3:
            x = tf.expand_dims(x, 0)
        if self._nested_backbone is not None:
            return self._compute_heatmap_nested(x, eps)
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(x, training=False)
            if isinstance(predictions, (list, tuple)):
                predictions = predictions[0]
            tape.watch(conv_outputs)
            pred = tf.reshape(predictions, (tf.shape(predictions)[0], -1))
            loss = pred[:, 0]

        grads = tape.gradient(loss, conv_outputs)
        return self._heatmap_from_maps(conv_outputs, grads, eps)

    OVERLAY_STYLE_VERSION = "v4vis"

    @staticmethod
    def _circular_roi(h: int, w: int, scale: float = 0.48) -> np.ndarray:
        yy, xx = np.ogrid[:h, :w]
        cy, cx = h / 2.0, w / 2.0
        radius = min(h, w) * scale
        return ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius**2

    @classmethod
    def _imaging_field(cls, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape[:2]
        b = max(8, min(h, w) // 12)
        corners = np.concatenate(
            [
                gray[:b, :b].ravel(),
                gray[:b, w - b :].ravel(),
                gray[h - b :, :b].ravel(),
                gray[h - b :, w - b :].ravel(),
            ]
        )
        if float(np.mean(corners)) >= 40:
            return np.ones((h, w), dtype=bool)

        blur = cv2.GaussianBlur(gray, (21, 21), 0)
        _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw)
        if n_labels <= 1:
            return cls._circular_roi(h, w, 0.48)
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        field_u8 = np.where(labels == best, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(field_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fill = np.zeros((h, w), dtype=np.uint8)
        if contours:
            cv2.drawContours(fill, contours, -1, 255, thickness=-1)
        field = fill > 0
        if float(field.mean()) < 0.18:
            return cls._circular_roi(h, w, 0.48)
        return field

    @staticmethod
    def _skin_sample_mask(field: np.ndarray) -> np.ndarray:
        h, w = field.shape
        field_u8 = field.astype(np.uint8) * 255
        k = max(9, int(0.16 * min(h, w)))
        if k % 2 == 0:
            k += 1
        inner = cv2.erode(field_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        annulus = (field_u8 > 0) & (inner == 0)
        if int(annulus.sum()) < max(200, int(0.02 * max(int(field.sum()), 1))):
            border = max(8, int(0.08 * min(h, w)))
            ring = np.zeros((h, w), dtype=bool)
            ring[:border, :] = True
            ring[h - border :, :] = True
            ring[:, :border] = True
            ring[:, w - border :] = True
            annulus = ring & field
        return annulus

    @staticmethod
    def _pick_component(mask: np.ndarray, field: np.ndarray, gray: np.ndarray) -> np.ndarray:
        h, w = mask.shape
        field_area = max(int(field.sum()), 1)
        min_area = max(40, int(0.004 * field_area))
        max_area = int(0.72 * field_area)
        cy, cx = h / 2.0, w / 2.0
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        best_i, best_score = 0, -1.0
        for i in range(1, n_labels):
            area = float(stats[i, cv2.CC_STAT_AREA])
            if area < min_area or area > max_area:
                continue
            dist_c = float(np.hypot(centroids[i][0] - cx, centroids[i][1] - cy))
            if dist_c > 0.70 * min(h, w):
                continue
            comp = labels == i
            darkness = 1.0 - float(gray[comp].mean()) / 255.0
            score = area * (0.40 + darkness) / (1.0 + dist_c / (0.28 * min(h, w)))
            if score > best_score:
                best_score = score
                best_i = i
        if best_i == 0:
            return np.zeros((h, w), dtype=np.uint8)
        picked = np.where(labels == best_i, 255, 0).astype(np.uint8)
        ys, xs = np.where(picked > 0)
        if len(xs) == 0:
            return picked
        my, mx = float(ys.mean()), float(xs.mean())
        link = max(28.0, 0.28 * min(h, w))
        for i in range(1, n_labels):
            if i == best_i:
                continue
            area = float(stats[i, cv2.CC_STAT_AREA])
            if area < min_area or area > max_area:
                continue
            dist = float(np.hypot(centroids[i][0] - mx, centroids[i][1] - my))
            if dist > link:
                continue
            comp = labels == i
            darkness = 1.0 - float(gray[comp].mean()) / 255.0
            if darkness < 0.12:
                continue
            picked = np.where(comp, 255, picked)
        return picked

    @classmethod
    def _lesion_mask(cls, original_rgb: np.ndarray) -> np.ndarray:
        h, w = original_rgb.shape[:2]
        img = original_rgb.copy()
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
        field = cls._imaging_field(gray)
        skin_px = cls._skin_sample_mask(field)
        if int(skin_px.sum()) < 50:
            skin_px = field

        skin = np.array(
            [
                float(np.median(lab[:, :, 0][skin_px])),
                float(np.median(lab[:, :, 1][skin_px])),
                float(np.median(lab[:, :, 2][skin_px])),
            ],
            dtype=np.float32,
        )
        dist = np.linalg.norm(lab - skin, axis=2)
        dark = np.clip((skin[0] - lab[:, :, 0]) / (float(skin[0]) + 1e-6), 0.0, 1.0)
        dist_scale = float(np.percentile(dist[field], 95)) if np.any(field) else 1.0
        dist_n = np.clip(dist / (dist_scale + 1e-6), 0.0, 1.5)
        score = (0.72 * dark + 0.28 * dist_n).astype(np.float32)
        score = np.where(field, score, 0.0)

        field_vals = score[field]
        field_area = max(int(field.sum()), 1)
        mask = np.zeros((h, w), dtype=np.uint8)
        if field_vals.size:
            for pct in (55, 62, 68, 74, 80, 84, 88, 92):
                thr = max(0.12, float(np.percentile(field_vals, pct)))
                cand = ((score >= thr) & (dark >= 0.05) & field).astype(np.uint8) * 255
                cand = cv2.morphologyEx(
                    cand, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                )
                cand = cv2.morphologyEx(
                    cand, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
                )
                picked = cls._pick_component(cand, field, gray)
                frac = float((picked > 0).sum()) / float(field_area)
                if 0.008 <= frac <= 0.62:
                    mask = picked
                    break
                if frac > 0 and frac <= 0.62 and (picked > 0).sum() > (mask > 0).sum():
                    mask = picked

        if mask.sum() == 0:
            inv = np.where(field, 255 - gray, 0).astype(np.uint8)
            blur = cv2.GaussianBlur(inv, (7, 7), 0)
            _, cand = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cand = np.where(field, cand, 0).astype(np.uint8)
            cand = cv2.morphologyEx(
                cand, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            )
            cand = cv2.morphologyEx(
                cand, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            )
            mask = cls._pick_component(cand, field, gray)

        if mask.sum() == 0:
            return mask

        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        )
        mask = cv2.GaussianBlur(mask, (9, 9), 0)
        _, mask = cv2.threshold(mask, 80, 255, cv2.THRESH_BINARY)
        if field_vals.size:
            loose_thr = max(0.10, float(np.percentile(field_vals, 52)))
            loose = ((score >= loose_thr) & field).astype(np.uint8) * 255
            grown = cv2.dilate(
                mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)), iterations=2
            )
            mask = cv2.bitwise_or(mask, cv2.bitwise_and(loose, grown))
        filled = np.zeros_like(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(filled, contours, -1, 255, thickness=-1)
            mask = filled
        mask = cv2.dilate(
            mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=2
        )
        return mask

    @staticmethod
    def _lesion_suspicious_map(
        original_rgb: np.ndarray, lesion_mask: np.ndarray
    ) -> np.ndarray:
        lab = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        L = lab[:, :, 0]
        chroma = np.linalg.norm(lab[:, :, 1:] - 128.0, axis=2)
        mask = lesion_mask > 0
        if not np.any(mask):
            return np.zeros(L.shape, dtype=np.float32)

        L_les = L[mask]
        dark = (float(L_les.max()) - L) / (float(L_les.max() - L_les.min()) + 1e-6)
        dark = np.clip(dark, 0, 1)
        chroma_n = chroma / (float(chroma[mask].max()) + 1e-6)
        blur = cv2.GaussianBlur(L, (15, 15), 0)
        contrast = np.abs(L - blur)
        contrast = contrast / (float(contrast[mask].max()) + 1e-6)

        sus = (0.55 * dark + 0.25 * chroma_n + 0.20 * contrast).astype(np.float32)
        sus = np.where(mask, sus, 0.0)
        if sus.max() > 0:
            sus /= sus.max()
        return sus

    @classmethod
    def _attention_map(
        cls,
        heatmap: np.ndarray,
        original_rgb: np.ndarray,
        lesion_mask: np.ndarray,
    ) -> np.ndarray:
        h, w = original_rgb.shape[:2]
        cam = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
        cam = np.clip(cam.astype(np.float32), 0, 1)
        cam = cv2.GaussianBlur(cam, (11, 11), 0)

        mask = (lesion_mask > 0).astype(np.float32)
        if mask.sum() < 1:
            gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
            dark = 1.0 - gray / 255.0
            yy, xx = np.ogrid[:h, :w]
            dist = np.sqrt((yy - h / 2.0) ** 2 + (xx - w / 2.0) ** 2)
            center = 1.0 - dist / (dist.max() + 1e-6)
            attention = 0.75 * dark + 0.25 * center
            thr = float(np.percentile(attention, 82))
            attention = np.clip((attention - thr) / (1.0 - thr + 1e-6), 0, 1)
            return np.power(attention, 1.15).astype(np.float32)

        soft = cv2.GaussianBlur(mask, (15, 15), 0)
        if soft.max() > 0:
            soft /= soft.max()

        sus = cls._lesion_suspicious_map(original_rgb, lesion_mask)
        cam_in = cam * mask
        cam_inside = float(cam_in[lesion_mask > 0].mean()) if np.any(lesion_mask > 0) else 0.0

        if cam_inside < 0.12:
            lesion_heat = 0.15 * cam_in + 0.85 * sus
        else:
            lesion_heat = 0.40 * cam_in + 0.60 * sus
        if lesion_heat.max() > 0:
            lesion_heat /= lesion_heat.max()
        lesion_heat = np.power(np.clip(lesion_heat, 0, 1), 1.10)
        attention = np.where(mask > 0, 0.58 + 0.42 * lesion_heat, 0.0)
        attention = soft * attention

        if attention.max() > 0:
            attention /= attention.max()
        return np.clip(attention, 0, 1).astype(np.float32)

    @classmethod
    def overlay_heatmap(
        cls,
        original_rgb: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.82,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """Red heatmap on the lesion, cool recolor on surrounding skin."""
        del alpha
        base = original_rgb.copy()
        if base.dtype != np.uint8:
            base = (np.clip(base, 0, 1) * 255).astype(np.uint8)

        lesion_mask = cls._lesion_mask(base)
        attention = cls._attention_map(heatmap, base, lesion_mask)
        mask_f = (lesion_mask > 0).astype(np.float32)
        soft = cv2.GaussianBlur(mask_f, (21, 21), 0)
        if soft.max() > 0:
            soft /= soft.max()

        heat_uint8 = np.uint8(255 * np.clip(attention, 0, 1))
        heat_color = cv2.applyColorMap(heat_uint8, colormap)
        heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
        heat_hsv = cv2.cvtColor(heat_color, cv2.COLOR_RGB2HSV).astype(np.float32)
        heat_hsv[:, :, 1] = np.clip(heat_hsv[:, :, 1] * 1.35, 0, 255)
        heat_hsv[:, :, 2] = np.clip(heat_hsv[:, :, 2] * 1.10, 0, 255)
        heat_color = cv2.cvtColor(heat_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        gray = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)
        gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(np.float32)
        cool = np.array([32, 68, 128], dtype=np.float32)
        background = np.clip(0.20 * gray3 + 0.80 * cool, 0, 255)

        inside = 0.10 * base.astype(np.float32) + 0.90 * heat_color.astype(np.float32)
        soft3 = np.repeat(soft[..., np.newaxis], 3, axis=2)
        overlay = np.clip(background * (1.0 - soft3) + inside * soft3, 0, 255).astype(np.uint8)

        contours, _ = cv2.findContours(
            lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        min_area = max(80, int(0.004 * base.shape[0] * base.shape[1]))
        contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        if contours:
            thickness = max(3, int(0.007 * min(base.shape[0], base.shape[1])))
            cv2.drawContours(overlay, contours, -1, (20, 20, 20), thickness + 3)
            cv2.drawContours(overlay, contours, -1, (255, 255, 255), thickness)
            cv2.drawContours(overlay, contours, -1, (255, 220, 90), max(1, thickness // 2))

        return overlay

    def generate(
        self,
        image_batch: np.ndarray,
        original_rgb: np.ndarray,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        try:
            heatmap = self.compute_heatmap(image_batch)
        except Exception as exc:
            logger.exception("Grad-CAM heatmap failed (%s); drawing overlay.", exc)
            h, w = original_rgb.shape[:2]
            heatmap = np.zeros((max(h // 8, 7), max(w // 8, 7)), dtype=np.float32)
        overlay = self.overlay_heatmap(original_rgb, heatmap)

        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(path), bgr)
            stamp = Path(str(path) + f".{self.OVERLAY_STYLE_VERSION}")
            stamp.write_text(self.OVERLAY_STYLE_VERSION, encoding="utf-8")
            logger.info("Grad-CAM saved to %s", path)

        return heatmap, overlay


def generate_gradcam_if_malignant(
    prediction: str,
    model: tf.keras.Model,
    image_batch: np.ndarray,
    original_rgb: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
) -> dict:
    if str(prediction or "").lower() == "benign":
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
