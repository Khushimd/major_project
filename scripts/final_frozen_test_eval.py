"""
Final frozen test evaluation of Experiment 2-C / Variant A.

No training, no threshold search, no ensemble-weight search.
Writes only under models/evaluation/final_test/.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import precision_recall_curve, roc_curve

import config
from models.architecture import FocalLoss
from models.ensemble import combine_hybrid_probs
from preprocessing.image_preprocess import ImagePreprocessor
from training.evaluate import (
    evaluate_predictions,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
)
from training.train import _predict_dataset
from utils.logger import setup_logger

logger = setup_logger("final_test")

SPLIT_DIR = config.PROCESSED_DATA_DIR / "melanoma_benign"
CACHE_DIR = SPLIT_DIR / "preprocessed"
TEST_CSV = SPLIT_DIR / "test.csv"
EFF_PATH = (
    config.BASE_DIR
    / "models"
    / "experiments"
    / "internal_optimization"
    / "exp2_loss_comparison"
    / "exp2c_bce_smooth"
    / "efficientnetb0_bce_smooth.keras"
)
MOB_PATH = (
    config.BASE_DIR
    / "models"
    / "experiments"
    / "internal_optimization"
    / "exp2_loss_comparison"
    / "exp2c_bce_smooth"
    / "mobilenetv2_bce_smooth.keras"
)
OUT_DIR = config.BASE_DIR / "models" / "evaluation" / "final_test"
WEIGHTS = (0.75, 0.25)
THRESHOLD = 0.30
EXPECTED = {"n": 2473, "melanoma": 1324, "benign": 1149}


def _json_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_jsonable), encoding="utf-8")


def _jsonable(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _score_dist(values: np.ndarray) -> dict:
    values = np.asarray(values).ravel()
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "n": int(values.size),
    }


def _use_cache_or_process(df: pd.DataFrame) -> pd.DataFrame:
    prep = ImagePreprocessor(target_size=(224, 224), apply_clahe=False, apply_hair_removal=True)
    out = df.copy()
    paths = []
    rebuilt = 0
    for image_id, src in zip(out["image_id"].astype(str), out["image_path"].astype(str)):
        cached = CACHE_DIR / f"{image_id}.jpg"
        if cached.is_file():
            paths.append(str(cached))
            continue
        import cv2

        bgr = cv2.imread(src, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(src)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        arr = prep.process_for_model(rgb)
        jpeg = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        cached.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(cached), cv2.cvtColor(jpeg, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        paths.append(str(cached))
        rebuilt += 1
    if rebuilt:
        logger.info("Rebuilt %d missing cache files with process_for_model.", rebuilt)
    out["image_path"] = paths
    return out


def _make_ds(df: pd.DataFrame) -> tf.data.Dataset:
    paths = df["image_path"].astype(str).values
    labels = df["label"].values.astype(np.float32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _decode(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, (224, 224))
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    return ds.map(_decode, num_parallel_calls=tf.data.AUTOTUNE).batch(8).prefetch(tf.data.AUTOTUNE)


def main() -> None:
    if not EFF_PATH.is_file() or not MOB_PATH.is_file():
        raise FileNotFoundError("Experiment 2-C model files were not found.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(TEST_CSV)
    n = len(test)
    n_mel = int((test["label"] == 1).sum())
    n_ben = int((test["label"] == 0).sum())
    if n != EXPECTED["n"] or n_mel != EXPECTED["melanoma"] or n_ben != EXPECTED["benign"]:
        raise RuntimeError(f"Frozen test CSV counts changed: n={n} mel={n_mel} ben={n_ben}")
    if test["image_id"].duplicated().any():
        raise RuntimeError("Duplicate image_id values in test.csv")
    test = _use_cache_or_process(test)
    ds = _make_ds(test)

    eff_sha = _sha256(EFF_PATH)
    mob_sha = _sha256(MOB_PATH)
    logger.info("Loading EfficientNet from %s sha256=%s", EFF_PATH, eff_sha)
    logger.info("Loading MobileNet from %s sha256=%s", MOB_PATH, mob_sha)
    eff = tf.keras.models.load_model(str(EFF_PATH), custom_objects={"FocalLoss": FocalLoss}, compile=False)
    y_true, p_eff = _predict_dataset(eff, ds)
    del eff
    tf.keras.backend.clear_session()
    mob = tf.keras.models.load_model(str(MOB_PATH), custom_objects={"FocalLoss": FocalLoss}, compile=False)
    y_true_m, p_mob = _predict_dataset(mob, ds)
    del mob
    tf.keras.backend.clear_session()
    if not np.array_equal(y_true, y_true_m):
        raise RuntimeError("Label order mismatch between EfficientNet and MobileNet inference.")
    if len(y_true) != EXPECTED["n"]:
        raise RuntimeError("Missing predictions.")

    p_ens = combine_hybrid_probs(p_eff, p_mob, weights=WEIGHTS, method="soft_voting")
    y_pred = (np.asarray(p_ens).ravel() >= THRESHOLD).astype(int)
    metrics = evaluate_predictions(y_true, p_ens, threshold=THRESHOLD)
    if metrics["tp"] + metrics["fn"] != EXPECTED["melanoma"]:
        raise RuntimeError("TP+FN does not equal actual melanoma count.")
    if metrics["tn"] + metrics["fp"] != EXPECTED["benign"]:
        raise RuntimeError("TN+FP does not equal actual benign count.")
    if metrics["tp"] + metrics["tn"] + metrics["fp"] + metrics["fn"] != EXPECTED["n"]:
        raise RuntimeError("Confusion counts do not sum to 2473.")

    pred_df = pd.DataFrame(
        {
            "image_id": test["image_id"].astype(str).values,
            "true_label": y_true.astype(int),
            "efficientnet_probability": np.asarray(p_eff).ravel(),
            "mobilenet_probability": np.asarray(p_mob).ravel(),
            "ensemble_probability": np.asarray(p_ens).ravel(),
            "predicted_label": y_pred,
        }
    )
    pred_df["correct_or_incorrect"] = np.where(pred_df["true_label"] == pred_df["predicted_label"], "correct", "incorrect")
    pred_df.to_csv(OUT_DIR / "final_test_predictions.csv", index=False)
    for name, mask in (
        ("true_positives", (pred_df["true_label"] == 1) & (pred_df["predicted_label"] == 1)),
        ("true_negatives", (pred_df["true_label"] == 0) & (pred_df["predicted_label"] == 0)),
        ("false_positives", (pred_df["true_label"] == 0) & (pred_df["predicted_label"] == 1)),
        ("false_negatives", (pred_df["true_label"] == 1) & (pred_df["predicted_label"] == 0)),
    ):
        pred_df.loc[mask].to_csv(OUT_DIR / f"{name}.csv", index=False)

    fpr, tpr, roc_thr = roc_curve(y_true, p_ens)
    prec, rec, pr_thr = precision_recall_curve(y_true, p_ens)
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thr}).to_csv(OUT_DIR / "roc_curve_data.csv", index=False)
    pr_thr_aligned = np.append(pr_thr, np.nan)
    pd.DataFrame({"precision": prec, "recall": rec, "threshold": pr_thr_aligned}).to_csv(OUT_DIR / "pr_curve_data.csv", index=False)
    plot_roc_curve(y_true, p_ens, OUT_DIR / "roc_curve.png", model_name="Exp2-C ensemble")
    plot_pr_curve(y_true, p_ens, OUT_DIR / "precision_recall_curve.png", model_name="Exp2-C ensemble")
    plot_confusion_matrix(y_true, y_pred, OUT_DIR / "confusion_matrix.png", labels=["Benign", "Melanoma"])

    dist = {
        "overall": _score_dist(p_ens),
        "actual_melanoma": _score_dist(p_ens[y_true == 1]),
        "actual_benign": _score_dist(p_ens[y_true == 0]),
    }
    _json_dump(OUT_DIR / "probability_distribution.json", dist)

    report = {
        "title": "FINAL FROZEN TEST EVALUATION",
        "dataset": "Melanoma vs Benign",
        "test_images": EXPECTED["n"],
        "actual_melanoma": EXPECTED["melanoma"],
        "actual_benign": EXPECTED["benign"],
        "model": "EfficientNetB0 + MobileNetV2",
        "training_loss": "BCE + label smoothing 0.05",
        "ensemble_weights": {"efficientnetb0": WEIGHTS[0], "mobilenetv2": WEIGHTS[1]},
        "threshold": THRESHOLD,
        "preprocessing": "process_for_model: hair removal + light denoise + resize 224 + /255; no CLAHE; no lesion crop; ImageNet preprocess inside the saved models",
        "model_files": {
            "efficientnetb0": {"path": str(EFF_PATH), "sha256": eff_sha},
            "mobilenetv2": {"path": str(MOB_PATH), "sha256": mob_sha},
        },
        "metrics": metrics,
        "predicted_melanoma": int((y_pred == 1).sum()),
        "predicted_benign": int((y_pred == 0).sum()),
        "probability_distribution": dist,
        "integrity": {
            "test_count": n == EXPECTED["n"],
            "label_counts": n_mel == EXPECTED["melanoma"] and n_ben == EXPECTED["benign"],
            "prediction_count": len(pred_df) == EXPECTED["n"],
            "missing_predictions": 0,
            "duplicate_predictions": int(pred_df["image_id"].duplicated().sum()),
            "threshold": THRESHOLD,
            "ensemble_weights": list(WEIGHTS),
            "test_leakage": False,
            "training_performed": False,
            "threshold_tuning": False,
            "ensemble_weight_tuning": False,
            "ph2_used": False,
            "derm7pt_used": False,
        },
        "production_or_live_app_modified": False,
        "experiments_1_2_3_modified": False,
        "frozen_test_split_modified": False,
    }
    _json_dump(OUT_DIR / "FINAL_TEST_EVALUATION.json", report)

    lines = [
        "------------------------------------------",
        "FINAL FROZEN TEST EVALUATION",
        "------------------------------------------",
        "",
        "Dataset:",
        "    Melanoma vs Benign",
        "",
        "Test images:",
        f"    {EXPECTED['n']}",
        "",
        "Actual melanoma:",
        f"    {EXPECTED['melanoma']}",
        "",
        "Actual benign:",
        f"    {EXPECTED['benign']}",
        "",
        "Model:",
        "    EfficientNetB0 + MobileNetV2",
        "",
        "Training loss:",
        "    BCE + label smoothing 0.05",
        "",
        "Ensemble:",
        "    0.75 EfficientNetB0",
        "    0.25 MobileNetV2",
        "",
        "Threshold:",
        "    0.30",
        "",
        "Preprocessing:",
        "    Exact Experiment 2-C / Variant A preprocessing",
        "",
        "Model files loaded:",
        f"    {EFF_PATH}",
        f"    sha256={eff_sha}",
        f"    {MOB_PATH}",
        f"    sha256={mob_sha}",
        "",
        "------------------------------------------",
        "RESULTS",
        "------------------------------------------",
        "",
        f"Accuracy: {metrics['accuracy']:.4f}",
        f"Sensitivity: {metrics['sensitivity']:.4f}",
        f"Specificity: {metrics['specificity']:.4f}",
        f"Precision: {metrics['precision']:.4f}",
        f"F1: {metrics['f1']:.4f}",
        f"ROC-AUC: {metrics['auc']:.4f}",
        f"PR-AUC: {metrics['pr_auc']:.4f}",
        "",
        f"TP: {metrics['tp']}",
        f"TN: {metrics['tn']}",
        f"FP: {metrics['fp']}",
        f"FN: {metrics['fn']}",
        "",
        "------------------------------------------",
        "CONFUSION MATRIX",
        "------------------------------------------",
        "",
        "                 Predicted Benign    Predicted Melanoma",
        f"Actual Benign    {metrics['tn']:<18} {metrics['fp']}",
        f"Actual Melanoma  {metrics['fn']:<18} {metrics['tp']}",
        "",
        "Melanoma: TP={tp} FN={fn} sensitivity={sens:.4f}".format(
            tp=metrics["tp"], fn=metrics["fn"], sens=metrics["sensitivity"]
        ),
        "Benign:   TN={tn} FP={fp} specificity={spec:.4f}".format(
            tn=metrics["tn"], fp=metrics["fp"], spec=metrics["specificity"]
        ),
        "",
        "------------------------------------------",
        "PREDICTION DISTRIBUTION",
        "------------------------------------------",
        "",
        f"Predicted melanoma: {int((y_pred == 1).sum())}",
        f"Predicted benign: {int((y_pred == 0).sum())}",
        "",
        "Probability statistics:",
        f"    overall min/max/mean/median = {dist['overall']['min']:.4f}/{dist['overall']['max']:.4f}/{dist['overall']['mean']:.4f}/{dist['overall']['median']:.4f}",
        f"    actual melanoma min/max/mean/median = {dist['actual_melanoma']['min']:.4f}/{dist['actual_melanoma']['max']:.4f}/{dist['actual_melanoma']['mean']:.4f}/{dist['actual_melanoma']['median']:.4f}",
        f"    actual benign min/max/mean/median = {dist['actual_benign']['min']:.4f}/{dist['actual_benign']['max']:.4f}/{dist['actual_benign']['mean']:.4f}/{dist['actual_benign']['median']:.4f}",
        "",
        "------------------------------------------",
        "INTEGRITY CHECKS",
        "------------------------------------------",
        "",
        f"Test count: PASS ({n})",
        f"Label counts: PASS (melanoma={n_mel}, benign={n_ben})",
        f"Prediction count: PASS ({len(pred_df)})",
        "Missing predictions: 0",
        f"Duplicate predictions: {int(pred_df['image_id'].duplicated().sum())}",
        "Threshold: 0.30 (frozen, not tuned)",
        "Ensemble weights: 0.75 / 0.25 (frozen, not tuned)",
        "Test leakage: none",
        "Training performed: no",
        "PH2 used: no",
        "",
        "------------------------------------------",
        "FILES",
        "------------------------------------------",
        "",
        str(OUT_DIR / "FINAL_TEST_EVALUATION.txt"),
        str(OUT_DIR / "FINAL_TEST_EVALUATION.json"),
        str(OUT_DIR / "final_test_predictions.csv"),
        str(OUT_DIR / "confusion_matrix.png"),
        str(OUT_DIR / "roc_curve.png"),
        str(OUT_DIR / "precision_recall_curve.png"),
        str(OUT_DIR / "probability_distribution.json"),
        str(OUT_DIR / "roc_curve_data.csv"),
        str(OUT_DIR / "pr_curve_data.csv"),
        str(OUT_DIR / "true_positives.csv"),
        str(OUT_DIR / "true_negatives.csv"),
        str(OUT_DIR / "false_positives.csv"),
        str(OUT_DIR / "false_negatives.csv"),
        "",
        "------------------------------------------",
        "FINAL STATUS",
        "------------------------------------------",
        "",
        "Final frozen test evaluation completed.",
    ]
    (OUT_DIR / "FINAL_TEST_EVALUATION.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    logger.info("Final frozen test evaluation completed. No further experiment started.")


if __name__ == "__main__":
    main()
