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
        Inference preprocessing matching training:
        RGB resize → float32 / 255.0  (no CLAHE / hair removal).

        Heavy cosmetic preprocessing shifts the distribution and collapses
        malignant probabilities toward benign.
        """
        if image is None or image.size == 0:
            raise ValueError("Empty or invalid image provided for model input.")
        img = image.copy()
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
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

    # ------------------------------------------------------------------
    # Individual steps
    # ------------------------------------------------------------------
    @staticmethod
    def remove_hair(image: np.ndarray) -> np.ndarray:
        """
        Remove hair artifacts using black-hat morphological filtering
        followed by inpainting (common dermoscopic technique).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        cleaned = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        return cleaned

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

    model_input_float matches training preprocessing (resize + /255).
    """
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    quality = ImagePreprocessor.check_quality(rgb)
    preprocessor = ImagePreprocessor(target_size=target_size)
    processed = preprocessor.process_for_model(rgb)
    return rgb, processed, quality
