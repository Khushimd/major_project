"""Lesion-focused Grad-CAM overlay (v4 visibility)."""

from __future__ import annotations

import numpy as np
import cv2

from gradcam.gradcam import GradCAM


def _synthetic_lesion() -> tuple[np.ndarray, tuple[int, int]]:
    img = np.full((256, 256, 3), (210, 168, 148), dtype=np.uint8)
    cv2.ellipse(img, (118, 132), (48, 40), 18, 0, 360, (72, 38, 24), -1)
    cv2.ellipse(img, (110, 126), (20, 16), 18, 0, 360, (28, 12, 8), -1)
    return img, (118, 132)


def test_lesion_mask_covers_pigment_not_whole_image():
    img, (cx, cy) = _synthetic_lesion()
    mask = GradCAM._lesion_mask(img)
    assert mask[cy, cx] > 0
    assert mask[12, 12] == 0
    frac = float((mask > 0).mean())
    assert 0.02 <= frac <= 0.28


def test_overlay_outlines_lesion():
    img, (cx, cy) = _synthetic_lesion()
    overlay = GradCAM.overlay_heatmap(img, np.zeros((14, 14), dtype=np.float32))
    assert overlay.shape == img.shape
    white = np.all(overlay >= 240, axis=2)
    ys, xs = np.where(white)
    assert len(xs) > 20
    assert np.hypot(xs.mean() - cx, ys.mean() - cy) < 55
    assert int((white[:8, :]).sum() + (white[:, :8]).sum()) < 5


def test_overlay_recolors_surrounding_skin():
    img, (cx, cy) = _synthetic_lesion()
    overlay = GradCAM.overlay_heatmap(img, np.zeros((14, 14), dtype=np.float32))
    corner = overlay[12, 12].astype(np.float32)
    original = img[12, 12].astype(np.float32)
    assert float(np.mean(np.abs(corner - original))) > 25
    assert corner[2] > corner[0]
    assert overlay[cy, cx, 0] > overlay[12, 12, 0]
