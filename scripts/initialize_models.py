"""
Initialize and optionally smoke-train models so the web app runs end-to-end.

Usage:
    python scripts/initialize_models.py              # save ImageNet-backed models
    python scripts/initialize_models.py --quick-train # brief synthetic training
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf

import config
from models.architecture import (
    build_efficientnetb0,
    build_mobilenetv2,
    compile_model,
)
from models.ensemble import EnsemblePredictor, select_best_model
from training.dataset_loader import create_synthetic_demo_dataset
from utils.logger import setup_logger

logger = setup_logger("initialize_models")


def save_base_models() -> None:
    """Build transfer-learning models and persist .keras files."""
    logger.info("Building EfficientNetB0…")
    eff = build_efficientnetb0(trainable_base=False)
    eff = compile_model(eff)
    eff.save(str(config.EFFICIENTNET_PATH))
    logger.info("Saved %s", config.EFFICIENTNET_PATH)

    logger.info("Building MobileNetV2…")
    mob = build_mobilenetv2(trainable_base=False)
    mob = compile_model(mob)
    mob.save(str(config.MOBILENET_PATH))
    logger.info("Saved %s", config.MOBILENET_PATH)

    EnsemblePredictor.save_ensemble_meta(config.ENSEMBLE_WEIGHTS)


def quick_train(epochs: int = 2, batch_size: int = 16) -> dict:
    """Short synthetic training so heads adapt slightly for demo predictions."""
    from sklearn.model_selection import train_test_split
    from training.dataset_loader import MelanomaDataset
    from training.evaluate import evaluate_predictions

    df = create_synthetic_demo_dataset(n_samples=160)
    train_df, test_df = train_test_split(
        df, test_size=0.25, stratify=df["label"], random_state=config.SEED
    )
    loader = MelanomaDataset()
    train_ds = loader.make_tf_dataset(train_df, batch_size=batch_size, shuffle=True, augment=True)
    test_ds = loader.make_tf_dataset(test_df, batch_size=batch_size, shuffle=False, augment=False)
    cw = loader.compute_class_weights(train_df["label"].values)

    results = {}
    for name, builder, path in [
        ("efficientnet", build_efficientnetb0, config.EFFICIENTNET_PATH),
        ("mobilenet", build_mobilenetv2, config.MOBILENET_PATH),
    ]:
        logger.info("Quick-training %s for %d epochs…", name, epochs)
        model = builder(trainable_base=False)
        model = compile_model(model, learning_rate=1e-3)
        model.fit(train_ds, epochs=epochs, class_weight=cw, verbose=1)
        model.save(str(path))

        y_true, y_prob = [], []
        for bx, by in test_ds:
            y_prob.append(model.predict(bx, verbose=0).ravel())
            y_true.append(np.asarray(by).ravel())
        y_true = np.concatenate(y_true)
        y_prob = np.concatenate(y_prob)
        results[name] = {
            "y_true": y_true,
            "y_prob": y_prob,
            "metrics": evaluate_predictions(y_true, y_prob),
        }

    w_e, w_m = config.ENSEMBLE_WEIGHTS
    ens_prob = (w_e * results["efficientnet"]["y_prob"] + w_m * results["mobilenet"]["y_prob"]) / (
        w_e + w_m
    )
    ens_metrics = evaluate_predictions(results["efficientnet"]["y_true"], ens_prob)

    all_metrics = {
        "efficientnet": results["efficientnet"]["metrics"],
        "mobilenet": results["mobilenet"]["metrics"],
        "ensemble": ens_metrics,
    }
    best = select_best_model(all_metrics)
    EnsemblePredictor.save_ensemble_meta(config.ENSEMBLE_WEIGHTS, ens_metrics)

    payload = {"best_model": best, "metrics": all_metrics, "note": "synthetic quick-train demo"}
    with open(config.METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Wrote metrics → %s", config.METRICS_PATH)
    return payload


def write_placeholder_metrics() -> None:
    """
    Write illustrative metrics structure when full training has not been run.
    Replace by running: python -m training.train
    """
    payload = {
        "best_model": "ensemble",
        "metrics": {
            "efficientnet": {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "auc": 0.0,
            },
            "mobilenet": {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "auc": 0.0,
            },
            "ensemble": {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "auc": 0.0,
            },
        },
        "note": "Placeholder — run training on SIIM-ISIC 2020 to populate real metrics.",
    }
    with open(config.METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick-train", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(config.SEED)

    if args.quick_train:
        quick_train(epochs=args.epochs)
    else:
        save_base_models()
        if not config.METRICS_PATH.exists():
            write_placeholder_metrics()

    logger.info("Model initialization complete.")


if __name__ == "__main__":
    main()
