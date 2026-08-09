"""
CNN model architectures for melanoma classification.

Implements EfficientNetB0 and MobileNetV2 transfer-learning models
with optional fine-tuning heads.
"""

from __future__ import annotations

from typing import Optional, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

import config
from utils.logger import setup_logger

logger = setup_logger("models")


def _classification_head(
    x: tf.Tensor,
    dropout_rate: float = config.DROPOUT_RATE,
    l2_reg: float = config.L2_REG,
    name_prefix: str = "head",
) -> tf.Tensor:
    """Shared binary classification head."""
    x = layers.GlobalAveragePooling2D(name=f"{name_prefix}_gap")(x)
    x = layers.Dropout(dropout_rate, name=f"{name_prefix}_dropout1")(x)
    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name=f"{name_prefix}_fc1",
    )(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn")(x)
    x = layers.Dropout(dropout_rate / 2, name=f"{name_prefix}_dropout2")(x)
    outputs = layers.Dense(
        1,
        activation="sigmoid",
        name="malignant_prob",
    )(x)
    return outputs


def build_efficientnetb0(
    input_shape: Tuple[int, int, int] = (*config.IMG_SIZE, config.IMG_CHANNELS),
    trainable_base: bool = False,
    dropout_rate: float = config.DROPOUT_RATE,
) -> keras.Model:
    """
    Build EfficientNetB0 transfer-learning model for binary melanoma detection.

    Args:
        input_shape: Input tensor shape.
        trainable_base: If True, unfreeze backbone for fine-tuning.
        dropout_rate: Dropout rate in classification head.

    Returns:
        Compiled-ready Keras Model (not yet compiled).
    """
    inputs = keras.Input(shape=input_shape, name="input_image")
    # EfficientNet expects [0, 255] — scale inside if inputs are [0,1]
    x = layers.Rescaling(255.0, name="rescale_255")(inputs)
    x = keras.applications.efficientnet.preprocess_input(x)

    base = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
    )
    base.trainable = trainable_base
    base._name = "efficientnetb0_backbone"

    outputs = _classification_head(base.output, dropout_rate=dropout_rate, name_prefix="eff")
    model = keras.Model(inputs=inputs, outputs=outputs, name="EfficientNetB0_Melanoma")
    logger.info(
        "Built EfficientNetB0 | trainable_base=%s | params=%s",
        trainable_base,
        f"{model.count_params():,}",
    )
    return model


def build_mobilenetv2(
    input_shape: Tuple[int, int, int] = (*config.IMG_SIZE, config.IMG_CHANNELS),
    trainable_base: bool = False,
    dropout_rate: float = config.DROPOUT_RATE,
) -> keras.Model:
    """
    Build MobileNetV2 transfer-learning model for binary melanoma detection.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")
    x = layers.Rescaling(255.0, name="rescale_255")(inputs)
    x = keras.applications.mobilenet_v2.preprocess_input(x)

    base = keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    base.trainable = trainable_base
    # Apply backbone to preprocessed tensor
    x = base(x)
    outputs = _classification_head(x, dropout_rate=dropout_rate, name_prefix="mob")
    model = keras.Model(inputs=inputs, outputs=outputs, name="MobileNetV2_Melanoma")
    logger.info(
        "Built MobileNetV2 | trainable_base=%s | params=%s",
        trainable_base,
        f"{model.count_params():,}",
    )
    return model


def unfreeze_for_finetuning(
    model: keras.Model,
    n_unfreeze_layers: int = 30,
) -> keras.Model:
    """
    Unfreeze the top N layers of the backbone for fine-tuning.

    Args:
        model: Existing transfer-learning model.
        n_unfreeze_layers: Number of top backbone layers to unfreeze.
    """
    backbone = None
    for layer in model.layers:
        if "backbone" in layer.name or isinstance(
            layer, (keras.Model,)
        ) and layer.name.endswith("backbone"):
            backbone = layer
            break

    # Fallback: find nested application model
    if backbone is None:
        for layer in model.layers:
            if isinstance(layer, keras.Model) and len(layer.layers) > 10:
                backbone = layer
                break

    if backbone is None:
        logger.warning("Could not locate backbone; unfreezing entire model.")
        model.trainable = True
        return model

    backbone.trainable = True
    freeze_until = max(0, len(backbone.layers) - n_unfreeze_layers)
    for i, layer in enumerate(backbone.layers):
        layer.trainable = i >= freeze_until

    trainable = sum(1 for l in backbone.layers if l.trainable)
    logger.info("Fine-tuning: unfroze %d / %d backbone layers", trainable, len(backbone.layers))
    return model


def get_last_conv_layer_name(model: keras.Model) -> str:
    """
    Heuristically find the last convolutional layer name for Grad-CAM.
    """
    conv_names = []
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            for sub in layer.layers:
                if isinstance(
                    sub,
                    (
                        layers.Conv2D,
                        layers.DepthwiseConv2D,
                        layers.SeparableConv2D,
                    ),
                ):
                    conv_names.append(sub.name)
        elif isinstance(
            layer,
            (layers.Conv2D, layers.DepthwiseConv2D, layers.SeparableConv2D),
        ):
            conv_names.append(layer.name)

    if not conv_names:
        # Search all nested layers
        for layer in model.layers:
            try:
                for sub in layer.submodules:
                    if isinstance(
                        sub,
                        (layers.Conv2D, layers.DepthwiseConv2D),
                    ):
                        conv_names.append(sub.name)
            except Exception:
                continue

    if not conv_names:
        raise ValueError("No convolutional layer found for Grad-CAM.")
    return conv_names[-1]


def compile_model(
    model: keras.Model,
    learning_rate: float = config.LEARNING_RATE,
    use_focal_loss: bool = config.USE_FOCAL_LOSS,
    class_weight_ratio: Optional[float] = None,
) -> keras.Model:
    """
    Compile model with AdamW optimizer and BCE / Focal loss.
    """
    try:
        optimizer = keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=config.WEIGHT_DECAY,
        )
    except AttributeError:
        # Fallback for older TF versions without AdamW
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        logger.warning("AdamW unavailable; falling back to Adam.")

    if use_focal_loss:
        loss = FocalLoss(
            gamma=config.FOCAL_LOSS_GAMMA,
            alpha=config.FOCAL_LOSS_ALPHA,
        )
    else:
        loss = keras.losses.BinaryCrossentropy()

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model


class FocalLoss(keras.losses.Loss):
    """
    Binary Focal Loss for handling class imbalance.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        name: str = "focal_loss",
    ) -> None:
        super().__init__(name=name)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        focal = alpha_t * tf.pow(1.0 - p_t, self.gamma) * bce
        return tf.reduce_mean(focal)

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({"gamma": self.gamma, "alpha": self.alpha})
        return cfg
