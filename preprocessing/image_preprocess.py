"""
Image preprocessing pipeline for dermoscopic lesion images.

Includes hair removal, denoising, CLAHE, normalization, resizing,
and image quality assessment (blur / brightness checks).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

import config
from utils.logger import setup_logger

logger = setup_logger("preprocessing")


@dataclass
class QualityReport:
    """Result of image quality assessment."""

    is_acceptable: bool
    blur_score: float
    brightness: float
    resolution: Tuple[int, int]
    warnings: list


class ImagePreprocessor:
    """
    Dermoscopic image preprocessor.

    Pipeline:
        load → hair removal → denoise → CLAHE → resize → normalize
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = config.IMG_SIZE,
        apply_clahe: bool = True,
        apply_hair_removal: bool = True,
    ) -> None:
        self.target_size = target_size
        self.apply_clahe = apply_clahe
        self.apply_hair_removal = apply_hair_removal

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process(
        self,
        image: np.ndarray,
        return_uint8: bool = False,
    ) -> np.ndarray:
        """
        Run the full preprocessing pipeline.

        Args:
            image: BGR or RGB uint8 image.
            return_uint8: If True, return uint8 [0,255]; else float32 [0,1].

        Returns:
            Preprocessed image array of shape (H, W, 3).
        """
        if image is None or image.size == 0:
            raise ValueError("Empty or invalid image provided for preprocessing.")

        img = image.copy()
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if self.apply_hair_removal:
            img = self.remove_hair(img)

        img = self.denoise(img)

        if self.apply_clahe:
            img = self.apply_clahe_contrast(img)

        img = self.resize(img, self.target_size)
        img = self.normalize(img, to_unit=True)

        if return_uint8:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        return img

    def process_file(
        self,
        path: str,
        return_uint8: bool = False,
    ) -> np.ndarray:
        """Load an image from disk and preprocess it."""
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        # Convert BGR → RGB for model consistency
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.process(image, return_uint8=return_uint8)

    def process_for_model(self, image: np.ndarray) -> np.ndarray:
        """
        Inference preprocessing for the CNN.

        Applies DullRazor-style hair removal + light denoising, then
        resize and /255. CLAHE is skipped to avoid domain shift.
        """
        if image is None or image.size == 0:
            raise ValueError("Empty or invalid image provided for model input.")
        img = image.copy()
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if self.apply_hair_removal:
            img = self.remove_hair(img)
        img = self.denoise(img)
        img = self.resize(img, self.target_size)
        return self.normalize(img, to_unit=True)

    def prepare_for_model(self, image: np.ndarray) -> np.ndarray:
        """
        Prepare a batch tensor for Keras inference.

        Returns:
            Array of shape (1, H, W, 3), float32 in [0, 1].
        """
        processed = self.process_for_model(image)
        return np.expand_dims(processed.astype(np.float32), axis=0)

    def lesion_focused_view(self, image: np.ndarray) -> np.ndarray:
        """
        Hair-cleaned lesion crop with mild CLAHE.

        Used as a second inference view when hair occlusion is significant so
        the CNN sees the pigment network rather than hair strokes.
        """
        if image is None or image.size == 0:
            raise ValueError("Empty or invalid image provided for lesion view.")
        img = image.copy()
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        # Aggressive hair wipe dedicated to this view (NS inpaint).
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blackhat = np.zeros_like(gray, dtype=np.uint8)
        for ksize in (5, 9, 13, 17, 21, 25, 31):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            blackhat = np.maximum(
                blackhat, cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
            )
        thr = max(8, int(np.percentile(blackhat, 85)))
        _, hmask = cv2.threshold(blackhat, thr, 255, cv2.THRESH_BINARY)
        hmask = cv2.dilate(
            hmask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1
        )
        cleaned = cv2.inpaint(img, hmask, inpaintRadius=7, flags=cv2.INPAINT_NS)

        lab = cv2.cvtColor(cleaned, cv2.COLOR_RGB2LAB)
        _, lesion = cv2.threshold(
            lab[:, :, 0], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        lesion = cv2.morphologyEx(
            lesion, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)
        )
        ys, xs = np.where(lesion > 0)
        if len(xs) > 50:
            pad = max(
                15,
                int(0.12 * max(int(xs.max() - xs.min()), int(ys.max() - ys.min()))),
            )
            h, w = cleaned.shape[:2]
            y0, y1 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + pad)
            x0, x1 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + pad)
            crop = cleaned[y0:y1, x0:x1]
        else:
            crop = cleaned

        crop = self.apply_clahe_contrast(crop)
        # Do not denoise after CLAHE — it washes out pigment texture the CNN needs.
        crop = self.resize(crop, self.target_size)
        return self.normalize(crop, to_unit=True)

    def prepare_multiview_batches(
        self, image: np.ndarray
    ) -> Tuple[list, float]:
        """
        Build one or more model inputs for occlusion-robust inference.

        Always includes the standard hair-removed full frame. When hair
        coverage is material, also includes a lesion-focused CLAHE crop.
        Caller should take the max malignant score across views.
        """
        _, hair_ratio = self.hair_artifact_mask(image)
        views = [self.process_for_model(image)]
        if hair_ratio >= 0.05:
            views.append(self.lesion_focused_view(image))
            logger.info(
                "Hair-aware multi-view enabled (hair_ratio=%.3f, views=%d)",
                hair_ratio,
                len(views),
            )
        return views, hair_ratio

    # ------------------------------------------------------------------
    # Individual steps
    # ------------------------------------------------------------------
    @staticmethod
    def hair_artifact_mask(image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Detect dark linear hair artifacts; return (mask uint8, coverage ratio)."""
        if image is None or image.size == 0:
            return np.zeros((1, 1), dtype=np.uint8), 0.0

        img = image.copy()
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if len(img.shape) == 2:
            gray = img
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Wide multi-scale black-hat: catches fine vellus and thick terminal hairs.
        blackhat = np.zeros_like(gray, dtype=np.uint8)
        for ksize in (5, 9, 13, 17, 21, 25, 31):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
            blackhat = np.maximum(blackhat, bh)

        thr = max(8, int(np.percentile(blackhat, 85)))
        _, mask = cv2.threshold(blackhat, thr, 255, cv2.THRESH_BINARY)
        # Drop speckles, then thicken shafts for inpainting coverage.
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        )
        mask = cv2.dilate(
            mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1
        )
        hair_ratio = float(mask.mean()) / 255.0
        return mask, hair_ratio

    @staticmethod
    def remove_hair(image: np.ndarray) -> np.ndarray:
        """
        DullRazor-style hair removal for dermoscopic images.

        Multi-scale black-hat detects thin and thick hairs; Telea for light
        hair and Navier-Stokes inpainting for heavy occlusion.
        """
        if image is None or image.size == 0:
            return image

        img = image.copy()
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        mask, hair_ratio = ImagePreprocessor.hair_artifact_mask(img)
        if hair_ratio < 0.002:
            return img

        # Heavier hair → larger radius + NS inpaint (preserves structure better).
        if hair_ratio >= 0.10:
            radius, flags = 7, cv2.INPAINT_NS
        elif hair_ratio >= 0.05:
            radius, flags = 5, cv2.INPAINT_NS
        else:
            radius, flags = 3, cv2.INPAINT_TELEA

        return cv2.inpaint(img, mask, inpaintRadius=radius, flags=flags)

    @staticmethod
    def denoise(image: np.ndarray) -> np.ndarray:
        """Median filter + light Gaussian blur for noise removal."""
        median = cv2.medianBlur(image, 3)
        blurred = cv2.GaussianBlur(median, (3, 3), 0)
        return blurred

    @staticmethod
    def apply_clahe_contrast(image: np.ndarray) -> np.ndarray:
        """Apply CLAHE on the L-channel in LAB color space."""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_ch = clahe.apply(l_ch)
        merged = cv2.merge([l_ch, a_ch, b_ch])
        return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

    @staticmethod
    def resize(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Resize with area interpolation for downscaling."""
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

    @staticmethod
    def normalize(image: np.ndarray, to_unit: bool = True) -> np.ndarray:
        """Normalize pixel values to [0, 1] float32."""
        img = image.astype(np.float32)
        if to_unit:
            img = img / 255.0
        return img

    # ------------------------------------------------------------------
    # Quality checks (bonus)
    # ------------------------------------------------------------------
    @staticmethod
    def check_quality(image: np.ndarray) -> QualityReport:
        """
        Assess image quality: blur (Laplacian variance), brightness, resolution.
        """
        warnings = []
        if image is None or image.size == 0:
            return QualityReport(False, 0.0, 0.0, (0, 0), ["Invalid image"])

        h, w = image.shape[:2]
        gray = (
            cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            if len(image.shape) == 3
            else image
        )

        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(np.mean(gray))

        if (w, h) < config.MIN_RESOLUTION and (h, w) < config.MIN_RESOLUTION:
            warnings.append(
                f"Low resolution ({w}x{h}). Minimum recommended "
                f"{config.MIN_RESOLUTION[0]}x{config.MIN_RESOLUTION[1]}."
            )

        if blur_score < config.BLUR_THRESHOLD:
            warnings.append(
                f"Image appears blurry (score={blur_score:.1f}). "
                "Consider retaking with better focus."
            )

        if brightness < config.BRIGHTNESS_LOW:
            warnings.append("Image is too dark. Improve lighting.")
        elif brightness > config.BRIGHTNESS_HIGH:
            warnings.append("Image is overexposed. Reduce lighting.")

        acceptable = len(warnings) == 0 or blur_score >= config.BLUR_THRESHOLD * 0.5
        return QualityReport(
            is_acceptable=acceptable,
            blur_score=blur_score,
            brightness=brightness,
            resolution=(w, h),
            warnings=warnings,
        )


def load_and_preprocess(
    path: str,
    target_size: Tuple[int, int] = config.IMG_SIZE,
) -> Tuple[np.ndarray, np.ndarray, QualityReport]:
    """
    Convenience function: load image, check quality, preprocess.

    Returns:
        (original_rgb, model_input_float, quality_report)

    model_input_float includes hair removal + light denoising, then resize + /255.
    """
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    quality = ImagePreprocessor.check_quality(rgb)
    preprocessor = ImagePreprocessor(target_size=target_size, apply_hair_removal=True)
    processed = preprocessor.process_for_model(rgb)
    return rgb, processed, quality
