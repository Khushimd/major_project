"""
Hybrid model combining EfficientNetB0 and MobileNetV2 predictions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

import config
from models.architecture import FocalLoss
from utils.logger import setup_logger

logger = setup_logger("ensemble")


def combine_hybrid_probs(
    p_eff: np.ndarray,
    p_mob: np.ndarray,
    weights: Tuple[float, float] = config.ENSEMBLE_WEIGHTS,
    method: str = config.HYBRID_METHOD,
    threshold: float = config.DECISION_THRESHOLD,
) -> np.ndarray:
    """
    Combine EfficientNet and MobileNet malignant probabilities.

    soft_voting: weighted average
    majority_voting: class with highest vote frequency (tie broken by soft vote)
    """
    p_eff = np.asarray(p_eff, dtype=np.float64).ravel()
    p_mob = np.asarray(p_mob, dtype=np.float64).ravel()
    w_eff, w_mob = weights
    total_w = w_eff + w_mob
    soft = (w_eff * p_eff + w_mob * p_mob) / total_w

    method = (method or "soft_voting").lower()
    if method != "majority_voting":
        return soft

    votes_eff = (p_eff >= threshold).astype(np.int32)
    votes_mob = (p_mob >= threshold).astype(np.int32)
    vote_sum = votes_eff + votes_mob  # 0, 1, or 2

    out = soft.copy()
    # Both vote malignant → take average of their malignant probs
    both_mal = vote_sum == 2
    out[both_mal] = (p_eff[both_mal] + p_mob[both_mal]) / 2.0
    # Both vote benign → average benign confidence as malignant prob
    both_ben = vote_sum == 0
    out[both_ben] = (p_eff[both_ben] + p_mob[both_ben]) / 2.0
    # Tie (disagree) → soft vote already in `out`
    return out


class EnsemblePredictor:
    """
    Hybrid predictor: EfficientNetB0 + MobileNetV2.

    Always uses both models when available for better predictions.
    """

    def __init__(
        self,
        efficientnet: Optional[tf.keras.Model] = None,
        mobilenet: Optional[tf.keras.Model] = None,
        weights: Tuple[float, float] = config.ENSEMBLE_WEIGHTS,
        use_tta: bool = config.USE_TTA,
        method: str = config.HYBRID_METHOD,
    ) -> None:
        self.efficientnet = efficientnet
        self.mobilenet = mobilenet
        self.weights = weights
        self.use_tta = use_tta
        self.method = method
        self._custom_objects = {"FocalLoss": FocalLoss}

    def load(
        self,
        efficientnet_path: Path = config.EFFICIENTNET_PATH,
        mobilenet_path: Path = config.MOBILENET_PATH,
    ) -> "EnsemblePredictor":
        """Load saved Keras models from disk."""
        if efficientnet_path.exists():
            self.efficientnet = tf.keras.models.load_model(
                str(efficientnet_path),
                custom_objects=self._custom_objects,
                compile=False,
            )
            logger.info("Loaded EfficientNet from %s", efficientnet_path)
        else:
            logger.warning("EfficientNet weights not found at %s", efficientnet_path)

        if mobilenet_path.exists():
            self.mobilenet = tf.keras.models.load_model(
                str(mobilenet_path),
                custom_objects=self._custom_objects,
                compile=False,
            )
            logger.info("Loaded MobileNetV2 from %s", mobilenet_path)
        else:
            logger.warning("MobileNetV2 weights not found at %s", mobilenet_path)

        return self

    def _predict_single(
        self,
        model: tf.keras.Model,
        batch: np.ndarray,
    ) -> float:
        """Run inference, optionally with TTA."""
        if self.use_tta:
            from preprocessing.augmentation import apply_tta

            preds = apply_tta(model, batch, steps=config.TTA_STEPS)
        else:
            preds = model.predict(batch, verbose=0)
        return float(np.squeeze(preds))

    def predict(
        self,
        image_batch: np.ndarray,
        return_individual: bool = True,
    ) -> Dict[str, Union[float, str, Dict]]:
        """
        Hybrid prediction for a single preprocessed image batch (1, H, W, 3).

        Returns dict with:
            malignant_prob, prediction, confidence, individual scores, model_used
        """
        if self.efficientnet is None and self.mobilenet is None:
            raise RuntimeError("No models loaded for hybrid prediction.")

        scores: Dict[str, float] = {}
        w_eff, w_mob = self.weights

        if self.efficientnet is not None:
            scores["efficientnet"] = self._predict_single(self.efficientnet, image_batch)
        if self.mobilenet is not None:
            scores["mobilenet"] = self._predict_single(self.mobilenet, image_batch)

        if "efficientnet" in scores and "mobilenet" in scores:
            malignant_prob = float(
                combine_hybrid_probs(
                    np.array([scores["efficientnet"]]),
                    np.array([scores["mobilenet"]]),
                    weights=self.weights,
                    method=self.method,
                )[0]
            )
            thr = config.DECISION_THRESHOLD
            vote_eff = int(scores["efficientnet"] >= thr)
            vote_mob = int(scores["mobilenet"] >= thr)
            if self.method == "majority_voting":
                model_used = (
                    f"Hybrid majority vote "
                    f"(Eff={vote_eff}, Mob={vote_mob}; thr={thr}; "
                    f"weights {w_eff}/{w_mob} on ties)"
                )
            else:
                model_used = (
                    f"Hybrid soft vote (Eff={w_eff}, Mob={w_mob}, thr={thr})"
                )
        elif "efficientnet" in scores:
            malignant_prob = scores["efficientnet"]
            model_used = "EfficientNetB0 (MobileNet unavailable)"
        else:
            malignant_prob = scores["mobilenet"]
            model_used = "MobileNetV2 (EfficientNet unavailable)"

        thr = config.DECISION_THRESHOLD
        label = "Malignant" if malignant_prob >= thr else "Benign"
        confidence = malignant_prob if label == "Malignant" else (1.0 - malignant_prob)

        result: Dict = {
            "malignant_prob": round(malignant_prob, 6),
            "benign_prob": round(1.0 - malignant_prob, 6),
            "prediction": label,
            "confidence": round(confidence, 6),
            "model_used": model_used,
            "decision_threshold": thr,
            "hybrid_method": self.method if len(scores) == 2 else "single",
        }
        if return_individual:
            result["individual"] = {k: round(v, 6) for k, v in scores.items()}
            if len(scores) == 2:
                result["votes"] = {
                    "efficientnet": (
                        "Malignant" if scores["efficientnet"] >= thr else "Benign"
                    ),
                    "mobilenet": (
                        "Malignant" if scores["mobilenet"] >= thr else "Benign"
                    ),
                }
        return result

    @staticmethod
    def _load_best_model_meta() -> Optional[Dict]:
        path = config.BEST_MODEL_META_PATH
        if path.exists():
            with open(path, "r", encoding="utf-8-sig") as fh:
                return json.load(fh)
        return None

    @staticmethod
    def save_ensemble_meta(
        weights: Tuple[float, float] = config.ENSEMBLE_WEIGHTS,
        metrics: Optional[Dict] = None,
        method: str = config.HYBRID_METHOD,
    ) -> None:
        """Persist hybrid configuration."""
        meta = {
            "weights": {"efficientnet": weights[0], "mobilenet": weights[1]},
            "method": method,
            "type": "hybrid",
            "description": "Combination of EfficientNetB0 and MobileNetV2",
            "metrics": metrics or {},
        }
        with open(config.ENSEMBLE_META_PATH, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        logger.info("Saved hybrid meta to %s", config.ENSEMBLE_META_PATH)


def select_best_model(metrics: Dict[str, Dict[str, float]]) -> str:
    """
    Automatically choose the best model based on F1 then ROC-AUC.

    Args:
        metrics: Mapping of model_name -> {accuracy, precision, recall, f1, auc}

    Returns:
        Best model key: 'efficientnet' | 'mobilenet' | 'ensemble'
    """
    def score(m: Dict[str, float]) -> Tuple[float, float]:
        return (m.get("f1", 0.0), m.get("auc", 0.0))

    best = max(metrics.keys(), key=lambda k: score(metrics[k]))
    meta = {
        "best_model": best,
        "selection_criteria": ["f1", "auc"],
        "metrics": metrics,
    }
    with open(config.BEST_MODEL_META_PATH, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    logger.info("Best model selected: %s", best)
    return best
