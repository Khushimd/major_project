"""
Test-time augmentation on the frozen 83.4% melanoma-vs-benign baseline.

Does not retrain. Does not write production weights, inference_config,
or the frozen baseline experiment folder. Test labels are used only
for a one-shot report after TTA averaging.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import tensorflow as tf

import config
from models.ensemble import combine_hybrid_probs
from training.evaluate import evaluate_predictions
from utils.logger import setup_logger

logger = setup_logger("baseline_tta")

DATA_DIR = config.PROCESSED_DATA_DIR / "final_melanoma_benign"
BASELINE_ROOT = config.BASE_DIR / "models" / "experiments" / "final_melanoma_benign_baseline"
CACHE_DIR = BASELINE_ROOT / "preprocessed"
OUT_DIR = config.BASE_DIR / "models" / "experiments" / "final_melanoma_benign_baseline_tta"
EFF_PATH = BASELINE_ROOT / "efficientnetb0_final_baseline.keras"
MOB_PATH = BASELINE_ROOT / "mobilenetv2_final_baseline.keras"
EXPECTED = {"test": (2437, 1041, 1396)}
ENSEMBLE_WEIGHTS = (0.7, 0.3)
THRESHOLD = 0.50
SEED = 42
BATCH = 8

# Dermoscopy-safe TTA. Cutout, synthetic hair, large hue, and random
# erasing are omitted because they can hide pigment and raise false negatives.
TTA_VIEWS = (
    ("identity", None),
    ("hflip", "hflip"),
    ("vflip", "vflip"),
    ("rot90", "rot90"),
    ("bright_0.90", "bright_down"),
    ("bright_1.10", "bright_up"),
)


def _json_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(value):
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return str(value)

    path.write_text(json.dumps(payload, indent=2, default=_default), encoding="utf-8")


def verify_test() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "test.csv")
    got = (len(df), int((df["label"] == 1).sum()), int((df["label"] == 0).sum()))
    if got != EXPECTED["test"]:
        raise RuntimeError(f"STOP: frozen test counts {got} != {EXPECTED['test']}")
    return df


def attach_cache(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    paths = []
    missing = []
    for image_id in out["image_id"].astype(str):
        cached = CACHE_DIR / f"{image_id}.jpg"
        if not cached.is_file():
            missing.append(image_id)
        paths.append(str(cached))
    if missing:
        raise FileNotFoundError(f"{len(missing)} cached images missing, e.g. {missing[:5]}")
    out["cached_path"] = paths
    return out


def _predict_dataset(model: tf.keras.Model, ds: tf.data.Dataset) -> tuple:
    y_prob, y_true = [], []
    for batch_x, batch_y in ds:
        preds = model.predict(batch_x, verbose=0)
        y_prob.append(np.asarray(preds).ravel())
        y_true.append(np.asarray(batch_y).ravel())
    return np.concatenate(y_true), np.concatenate(y_prob)


def make_view_ds(df: pd.DataFrame, view: str | None) -> tf.data.Dataset:
    paths = df["cached_path"].astype(str).values
    labels = df["label"].values.astype(np.float32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _decode(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, (224, 224))
        img = tf.cast(img, tf.float32) / 255.0
        if view == "hflip":
            img = tf.image.flip_left_right(img)
        elif view == "vflip":
            img = tf.image.flip_up_down(img)
        elif view == "rot90":
            img = tf.image.rot90(img, k=1)
        elif view == "bright_down":
            img = tf.clip_by_value(img * 0.90, 0.0, 1.0)
        elif view == "bright_up":
            img = tf.clip_by_value(img * 1.10, 0.0, 1.0)
        return img, label

    return ds.map(_decode, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH).prefetch(
        tf.data.AUTOTUNE
    )


def predict_views(model: tf.keras.Model, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    stacked = []
    y_ref = None
    for name, view in TTA_VIEWS:
        logger.info("Predicting view %s", name)
        y_true, y_prob = _predict_dataset(model, make_view_ds(df, view))
        if y_ref is None:
            y_ref = y_true
        elif not np.array_equal(y_ref, y_true):
            raise RuntimeError(f"Label order changed on TTA view {name}")
        stacked.append(y_prob.astype(np.float64))
    return y_ref, np.stack(stacked, axis=0)


def main() -> None:
    t0 = time.time()
    tf.keras.utils.set_random_seed(SEED)
    if not EFF_PATH.is_file() or not MOB_PATH.is_file():
        raise FileNotFoundError("Frozen baseline Keras weights are missing.")
    if not CACHE_DIR.is_dir():
        raise FileNotFoundError(f"Preprocessed cache missing: {CACHE_DIR}")

    test_df = attach_cache(verify_test())
    n_test, n_mel, n_ben = EXPECTED["test"]
    if len(test_df) != n_test:
        raise RuntimeError("Unexpected test size")

    logger.info("Loading frozen baseline models")
    eff = tf.keras.models.load_model(str(EFF_PATH), compile=False)
    mob = tf.keras.models.load_model(str(MOB_PATH), compile=False)

    y_true, eff_views = predict_views(eff, test_df)
    _, mob_views = predict_views(mob, test_df)

    p_eff_mean = eff_views.mean(axis=0)
    p_mob_mean = mob_views.mean(axis=0)
    p_eff_max = eff_views.max(axis=0)
    p_mob_max = mob_views.max(axis=0)

    ens_mean = combine_hybrid_probs(
        p_eff_mean, p_mob_mean, weights=ENSEMBLE_WEIGHTS, method="soft_voting"
    )
    ens_max = combine_hybrid_probs(
        p_eff_max, p_mob_max, weights=ENSEMBLE_WEIGHTS, method="soft_voting"
    )

    baseline = {
        "threshold": 0.5,
        "accuracy": 0.8342224045958145,
        "precision": 0.8007554296506137,
        "recall": 0.8146013448607109,
        "sensitivity": 0.8146013448607109,
        "specificity": 0.8488538681948424,
        "f1": 0.8076190476190476,
        "auc": 0.9126790142826079,
        "pr_auc": 0.8884757664753524,
        "tp": 848,
        "fp": 211,
        "tn": 1185,
        "fn": 193,
        "n_positive": n_mel,
        "n_negative": n_ben,
    }
    mean_metrics = evaluate_predictions(y_true, ens_mean, threshold=THRESHOLD)
    max_metrics = evaluate_predictions(y_true, ens_max, threshold=THRESHOLD)

    payload = {
        "purpose": "Reduce false negatives on the frozen baseline test set using TTA only.",
        "not_training": True,
        "production_unchanged": True,
        "test_used_for_tuning": False,
        "dataset": str(DATA_DIR),
        "n_test": n_test,
        "n_melanoma": n_mel,
        "n_benign": n_ben,
        "ensemble_weights": {"efficientnet": ENSEMBLE_WEIGHTS[0], "mobilenet": ENSEMBLE_WEIGHTS[1]},
        "threshold": THRESHOLD,
        "views": [name for name, _ in TTA_VIEWS],
        "suitable_for_fn_reduction": [
            "horizontal flip",
            "vertical flip",
            "90-degree rotate",
            "mild brightness down/up",
        ],
        "not_used": [
            "random erasing / cutout",
            "synthetic hair",
            "large hue shift",
            "360-degree random rotation",
        ],
        "baseline_no_tta": baseline,
        "tta_mean": mean_metrics,
        "tta_max_sensitivity": max_metrics,
        "fn_delta_mean": int(mean_metrics["fn"]) - int(baseline["fn"]),
        "fn_delta_max": int(max_metrics["fn"]) - int(baseline["fn"]),
        "elapsed_seconds": time.time() - t0,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _json_dump(OUT_DIR / "tta_test_metrics.json", payload)
    pd.DataFrame(
        {
            "image_id": test_df["image_id"].astype(str).values,
            "y_true": y_true,
            "p_ens_mean": ens_mean,
            "p_ens_max": ens_max,
        }
    ).to_csv(OUT_DIR / "tta_test_predictions.csv", index=False)

    print("=== Frozen baseline test (no TTA) ===")
    print(
        f"acc={baseline['accuracy']:.4f} rec={baseline['recall']:.4f} "
        f"spec={baseline['specificity']:.4f} prec={baseline['precision']:.4f} "
        f"f1={baseline['f1']:.4f} FN={baseline['fn']} FP={baseline['fp']}"
    )
    print("=== TTA mean (primary) ===")
    print(
        f"acc={mean_metrics['accuracy']:.4f} rec={mean_metrics['recall']:.4f} "
        f"spec={mean_metrics['specificity']:.4f} prec={mean_metrics['precision']:.4f} "
        f"f1={mean_metrics['f1']:.4f} FN={mean_metrics['fn']} FP={mean_metrics['fp']} "
        f"auc={mean_metrics['auc']:.4f}"
    )
    print("=== TTA max (sensitivity-oriented, more FP) ===")
    print(
        f"acc={max_metrics['accuracy']:.4f} rec={max_metrics['recall']:.4f} "
        f"spec={max_metrics['specificity']:.4f} prec={max_metrics['precision']:.4f} "
        f"f1={max_metrics['f1']:.4f} FN={max_metrics['fn']} FP={max_metrics['fp']}"
    )
    print(f"Wrote {OUT_DIR / 'tta_test_metrics.json'}")


if __name__ == "__main__":
    main()
