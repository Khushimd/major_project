"""
Image quality checker module (bonus feature).
"""

from __future__ import annotations

from preprocessing.image_preprocess import ImagePreprocessor, QualityReport

__all__ = ["ImagePreprocessor", "QualityReport", "check_image_quality"]


def check_image_quality(image) -> QualityReport:
    """Proxy to ImagePreprocessor.check_quality."""
    return ImagePreprocessor.check_quality(image)
