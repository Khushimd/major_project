"""
Data augmentation utilities for melanoma lesion images.

Training uses class-aware augmentation: every train image gets dermoscopy-safe
geometric/color jitter. Melanoma images (label=1) get a stronger set so the
rare class has more visual variety per epoch.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf

import config


def add_synthetic_hair_numpy(
    image: np.ndarray,
    max_hairs: int = 18,
    probability: float = 0.45,
) -> np.ndarray:
    """
    Overlay thin dark strokes that mimic dermoscopic hair artifacts.

    Used during training so the model learns lesion features despite hair.
    Image is float32 in [0, 1] or uint8 [0, 255].
    """
    if np.random.rand() > probability:
        return image

    img = image.copy()
    was_float = np.issubdtype(img.dtype, np.floating)
    if was_float:
        canvas = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    else:
        canvas = img.astype(np.uint8)

    h, w = canvas.shape[:2]
    n_hairs = int(np.random.randint(4, max_hairs + 1))
    for _ in range(n_hairs):
        x1 = int(np.random.randint(0, w))
        y1 = int(np.random.randint(0, h))
        length = int(np.random.randint(int(0.15 * min(h, w)), int(0.55 * min(h, w))))
        angle = np.random.uniform(0, 2 * np.pi)
        x2 = int(np.clip(x1 + length * np.cos(angle), 0, w - 1))
        y2 = int(np.clip(y1 + length * np.sin(angle), 0, h - 1))
        thickness = int(np.random.choice([1, 1, 2, 2, 3]))
        color = (
            int(np.random.randint(5, 45)),
            int(np.random.randint(5, 40)),
            int(np.random.randint(5, 40)),
        )
        cv2.line(canvas, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
        if np.random.rand() > 0.4:
            mx = int((x1 + x2) / 2 + np.random.randint(-12, 13))
            my = int((y1 + y2) / 2 + np.random.randint(-12, 13))
            pts = np.array([[x1, y1], [mx, my], [x2, y2]], dtype=np.int32)
            cv2.polylines(canvas, [pts], False, color, thickness, lineType=cv2.LINE_AA)

    if was_float:
        return (canvas.astype(np.float32) / 255.0).clip(0.0, 1.0)
    return canvas


def get_keras_augmentation(strong: bool = False) -> tf.keras.Sequential:
    """
    Realistic dermoscopy geometric + photometric pipeline.

    Rotations stay small so lesion shape is not destroyed. `strong` only
    slightly increases zoom / brightness; it does not use 360-degree spins.
    """
    if strong:
        layers_list = [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom((-0.15, 0.15)),
            tf.keras.layers.RandomTranslation(0.10, 0.10),
            tf.keras.layers.RandomContrast(0.20),
        ]
        brightness = 0.18
    else:
        layers_list = [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.06),
            tf.keras.layers.RandomZoom((-0.12, 0.12)),
            tf.keras.layers.RandomTranslation(0.08, 0.08),
            tf.keras.layers.RandomContrast(0.15),
        ]
        brightness = 0.12
    try:
        layers_list.append(
            tf.keras.layers.RandomBrightness(brightness, value_range=(0.0, 1.0))
        )
    except TypeError:
        try:
            layers_list.append(tf.keras.layers.RandomBrightness(brightness))
        except Exception:
            pass
    except Exception:
        pass

    name = "melanoma_augmentation" if strong else "augmentation"
    return tf.keras.Sequential(layers_list, name=name)


def _random_gaussian_noise(image: tf.Tensor, stddev: float = 0.025) -> tf.Tensor:
    noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=stddev, dtype=image.dtype)
    return image + noise


def _maybe_downup_blur(image: tf.Tensor, probability: float = 0.25) -> tf.Tensor:
    """Cheap defocus-like blur via downsample / upsample."""

    def _blur() -> tf.Tensor:
        shape = tf.shape(image)
        small_h = tf.maximum(tf.cast(shape[0] // 2, tf.int32), 16)
        small_w = tf.maximum(tf.cast(shape[1] // 2, tf.int32), 16)
        small = tf.image.resize(image, [small_h, small_w], method="bilinear")
        return tf.image.resize(small, [shape[0], shape[1]], method="bilinear")

    return tf.cond(tf.random.uniform([]) < probability, _blur, lambda: image)


def _random_erasing(image: tf.Tensor, probability: float = 0.30) -> tf.Tensor:
    """
    Cutout-style occlusion (small patches only) so hair/ruler artifacts
    do not dominate, without covering the whole lesion.
    """

    def _erase() -> tf.Tensor:
        h = tf.shape(image)[0]
        w = tf.shape(image)[1]
        yy, xx = tf.meshgrid(tf.range(h), tf.range(w), indexing="ij")
        rh = tf.maximum(1, h // tf.random.uniform([], 8, 16, dtype=tf.int32))
        rw = tf.maximum(1, w // tf.random.uniform([], 8, 16, dtype=tf.int32))
        y = tf.random.uniform([], 0, tf.maximum(1, h - rh), dtype=tf.int32)
        x = tf.random.uniform([], 0, tf.maximum(1, w - rw), dtype=tf.int32)
        mask = (yy >= y) & (yy < y + rh) & (xx >= x) & (xx < x + rw)
        mask = tf.cast(mask[..., tf.newaxis], image.dtype)
        fill = tf.random.uniform([], 0.05, 0.35, dtype=image.dtype)
        return image * (1.0 - mask) + fill * mask

    return tf.cond(tf.random.uniform([]) < probability, _erase, lambda: image)


def _photometric_jitter(image: tf.Tensor, strong: bool) -> tf.Tensor:
    if strong:
        image = tf.image.random_saturation(image, 0.65, 1.40)
        image = tf.image.random_hue(image, 0.04)
        image = tf.image.random_contrast(image, 0.75, 1.30)
    else:
        image = tf.image.random_saturation(image, 0.85, 1.15)
        image = tf.image.random_hue(image, 0.02)
    return image


def _make_hair_fn(probability: float):
    def _hair_aug(image: tf.Tensor) -> tf.Tensor:
        def _py(img: np.ndarray) -> np.ndarray:
            return add_synthetic_hair_numpy(
                img.astype(np.float32), probability=probability
            )

        aug = tf.numpy_function(_py, [image], tf.float32)
        aug.set_shape(image.shape)
        return aug

    return _hair_aug


def _maybe_apply(image: tf.Tensor, probability: float, transform):
    return tf.cond(tf.random.uniform([]) < probability, lambda: transform(image), lambda: image)


def _targeted_low_contrast_photometric(image: tf.Tensor) -> tf.Tensor:
    """
    Extra train-only photometric jitter for the targeted-augmentation experiment.

    Conservative ranges only. No aggressive hue shift. Not used on val/test.
    Ops run in float32 and clip to [0, 1] so gamma/contrast cannot emit NaNs
    (tf.image.adjust_gamma uses log and is NaN at 0).
    """
    orig_dtype = image.dtype
    image = tf.cast(image, tf.float32)
    image = tf.clip_by_value(image, 0.0, 1.0)

    def _brightness(img: tf.Tensor) -> tf.Tensor:
        factor = tf.random.uniform([], 0.85, 1.15, dtype=tf.float32)
        return tf.clip_by_value(img * factor, 0.0, 1.0)

    def _contrast(img: tf.Tensor) -> tf.Tensor:
        factor = tf.random.uniform([], 0.75, 1.25, dtype=tf.float32)
        return tf.clip_by_value(tf.image.adjust_contrast(img, factor), 0.0, 1.0)

    def _gamma(img: tf.Tensor) -> tf.Tensor:
        gamma = tf.random.uniform([], 0.85, 1.15, dtype=tf.float32)
        safe = tf.clip_by_value(img, 1e-4, 1.0)
        return tf.clip_by_value(tf.pow(safe, gamma), 0.0, 1.0)

    def _saturation(img: tf.Tensor) -> tf.Tensor:
        factor = tf.random.uniform([], 0.90, 1.10, dtype=tf.float32)
        return tf.clip_by_value(tf.image.adjust_saturation(img, factor), 0.0, 1.0)

    image = _maybe_apply(image, 0.30, _brightness)
    image = _maybe_apply(image, 0.40, _contrast)
    image = _maybe_apply(image, 0.20, _gamma)
    image = _maybe_apply(image, 0.15, _saturation)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return tf.cast(image, orig_dtype)


def get_tf_data_augmentation_fn():
    """
    tf.data map fn applied to TRAINING images only.

    The same realistic dermoscopy transforms are used for melanoma and
    benign images so the model cannot treat synthetic artifacts as a
    class cue. Random erasing / 360-degree rotation are not used.
    """
    geom = get_keras_augmentation(strong=False)
    hair_fn = _make_hair_fn(0.15)

    def _extras(image: tf.Tensor) -> tf.Tensor:
        image = _photometric_jitter(image, strong=False)
        image = _random_gaussian_noise(image, stddev=0.01)
        image = _maybe_downup_blur(image, probability=0.10)
        image = hair_fn(image)
        return tf.clip_by_value(image, 0.0, 1.0)

    def _augment(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        image = geom(image, training=True)
        image = _extras(image)
        return image, label

    return _augment


def get_targeted_low_contrast_train_augmentation_fn():
    """
    Baseline train augmentation plus conservative extra photometric jitter.

    Used only by the targeted-augmentation experiment. Validation and test
    must not call this. Does not oversample validation FNs.
    """
    baseline = get_tf_data_augmentation_fn()

    def _augment(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        image, label = baseline(image, label)
        image = _targeted_low_contrast_photometric(image)
        return tf.clip_by_value(image, 0.0, 1.0), label

    return _augment


def get_mild_contrast_train_augmentation_fn():
    """
    Baseline train augmentation plus ONE extra mild contrast jitter.

    No extra brightness, gamma, or saturation. Used only by the mild-contrast
    experiment. Validation and test must not call this.
    """
    baseline = get_tf_data_augmentation_fn()

    def _contrast(img: tf.Tensor) -> tf.Tensor:
        orig_dtype = img.dtype
        x = tf.clip_by_value(tf.cast(img, tf.float32), 0.0, 1.0)
        factor = tf.random.uniform([], 0.85, 1.15, dtype=tf.float32)
        x = tf.clip_by_value(tf.image.adjust_contrast(x, factor), 0.0, 1.0)
        return tf.cast(x, orig_dtype)

    def _augment(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        image, label = baseline(image, label)
        image = _maybe_apply(image, 0.25, _contrast)
        return tf.clip_by_value(image, 0.0, 1.0), label

    return _augment


def get_dicm_train_augmentation_fn():
    """
    ISIC-DICM-17K train-only aug: reuse the existing mild dermoscopy
    geometric/photometric layers. No synthetic hair, noise, or blur.
    Val/test must not call this.
    """
    geom = get_keras_augmentation(strong=False)

    def _augment(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        image = geom(image, training=True)
        return tf.clip_by_value(image, 0.0, 1.0), label

    return _augment


def apply_tta(
    model: tf.keras.Model,
    image_batch: np.ndarray,
    steps: int = config.TTA_STEPS,
) -> np.ndarray:
    """
    Test-Time Augmentation: average predictions over augmented views.
    """
    preds = [model.predict(image_batch, verbose=0)]

    for _ in range(steps - 1):
        aug = image_batch.copy()
        if np.random.rand() > 0.5:
            aug = np.flip(aug, axis=2)
        if np.random.rand() > 0.5:
            aug = np.flip(aug, axis=1)
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
