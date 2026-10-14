"""
ONE baseline training experiment on the frozen final_melanoma_benign dataset.

Writes only under models/experiments/final_melanoma_benign_baseline/.
Does not modify production models, the app, old experiments, or frozen CSVs.
Test is used once after validation-only ensemble and threshold selection.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import precision_recall_curve, roc_curve
from tensorflow import keras

import config
from models.architecture import build_efficientnetb0, build_mobilenetv2, unfreeze_for_finetuning
from models.ensemble import combine_hybrid_probs
from preprocessing.augmentation import get_tf_data_augmentation_fn
from training.evaluate import (
    best_epoch_from_history,
    evaluate_predictions,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
    plot_training_curves,
    save_threshold_table,
    threshold_table,
)
from training.train import _predict_dataset, enable_mixed_precision
from utils.logger import setup_logger

logger = setup_logger("final_baseline")

DATA_DIR = config.PROCESSED_DATA_DIR / "final_melanoma_benign"
EXP_ROOT = config.BASE_DIR / "models" / "experiments" / "final_melanoma_benign_baseline"
CACHE_DIR = EXP_ROOT / "preprocessed"
PROTECTED_ROOTS = [
    (config.MODELS_DIR).resolve(),
    (config.PROCESSED_DATA_DIR / "melanoma_benign").resolve(),
    (config.PROCESSED_DATA_DIR / "dicm_clean_melanoma_benign").resolve(),
    (config.BASE_DIR / "models" / "experiments" / "internal_optimization").resolve(),
    (config.BASE_DIR / "models" / "experiments" / "clean_dicm_melanoma_benign").resolve(),
]
PROTECTED_FILES = {
    config.EFFICIENTNET_PATH.resolve(),
    config.MOBILENET_PATH.resolve(),
    config.INFERENCE_CONFIG_PATH.resolve(),
}
EXPECTED = {
    "train": (11379, 4703, 6676),
    "val": (2438, 1043, 1395),
    "test": (2437, 1041, 1396),
}
STAGES = (
    {"name": "head", "lr": 1e-3, "epochs": 25, "unfreeze": 0, "batch": 16},
    {"name": "partial", "lr": 1e-4, "epochs": 12, "unfreeze": 20, "batch": 8},
    {"name": "final", "lr": 1e-5, "epochs": 12, "unfreeze": 20, "batch": 8},
)
LABEL_SMOOTHING = 0.05
SEED = 42
WEIGHT_GRID = [round(x, 1) for x in np.arange(0.0, 1.01, 0.1)]
THRESHOLD_GRID = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def _jsonable(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _json_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_jsonable), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _refuse_protected(path: Path) -> None:
    resolved = path.resolve()
    if resolved in PROTECTED_FILES:
        raise RuntimeError(f"Refusing to write protected file: {path}")
    if EXP_ROOT.resolve() in resolved.parents or resolved == EXP_ROOT.resolve():
        return
    for root in PROTECTED_ROOTS:
        if resolved == root or root in resolved.parents:
            raise RuntimeError(f"Refusing to write protected path: {path}")


def snapshot_protected() -> dict:
    out = {}
    for path in sorted(PROTECTED_FILES):
        if path.is_file():
            out[str(path)] = {"sha256": _sha256(path), "size": path.stat().st_size}
    for csv_name in ("train.csv", "val.csv", "test.csv", "manifest.csv"):
        for folder in (
            DATA_DIR,
            config.PROCESSED_DATA_DIR / "melanoma_benign",
            config.PROCESSED_DATA_DIR / "dicm_clean_melanoma_benign",
        ):
            path = folder / csv_name
            if path.is_file():
                out[str(path.resolve())] = {"sha256": _sha256(path), "size": path.stat().st_size}
    return out


def assert_unchanged(before: dict) -> None:
    after = snapshot_protected()
    for key, meta in before.items():
        if after.get(key) != meta:
            raise RuntimeError(f"Protected file changed: {key}")


def verify_frozen() -> dict:
    frames = {}
    for name in ("train", "val", "test"):
        df = pd.read_csv(DATA_DIR / f"{name}.csv")
        got = (len(df), int((df["label"] == 1).sum()), int((df["label"] == 0).sum()))
        if got != EXPECTED[name]:
            raise RuntimeError(f"STOP: frozen {name} counts {got} != {EXPECTED[name]}")
        frames[name] = df
    ids = {k: set(v["image_id"].astype(str)) for k, v in frames.items()}
    if ids["train"] & ids["val"] or ids["train"] & ids["test"] or ids["val"] & ids["test"]:
        raise RuntimeError("STOP: image overlap in frozen splits")
    return frames


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


def make_ds(df: pd.DataFrame, *, shuffle: bool, augment: bool, batch: int) -> tf.data.Dataset:
    if augment and not shuffle:
        augment = False
    paths = df["cached_path"].astype(str).values
    labels = df["label"].values.astype(np.float32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        buf = min(len(df), max(2000, batch * 64))
        ds = ds.shuffle(buffer_size=buf, seed=SEED, reshuffle_each_iteration=True)

    def _decode(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, (224, 224))
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    ds = ds.map(_decode, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        aug_fn = get_tf_data_augmentation_fn()
        ds = ds.map(lambda image, label: aug_fn(image, label), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def local_callbacks(model_name: str, ckpt_dir: Path, log_dir: Path):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=config.PATIENCE_EARLY_STOP,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_pr_auc",
            mode="max",
            factor=config.REDUCE_LR_FACTOR,
            patience=config.PATIENCE_REDUCE_LR,
            min_lr=config.MIN_LR,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_dir / f"{model_name}_best.keras"),
            monitor="val_pr_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(log_dir / f"{model_name}_history.csv"), append=False),
    ]


def compile_model(model: keras.Model, learning_rate: float) -> keras.Model:
    try:
        optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=config.WEIGHT_DECAY)
    except AttributeError:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.AUC(curve="PR", name="pr_auc"),
        ],
    )
    return model


def train_one(builder, name: str, train_df, val_df, save_path: Path) -> dict:
    _refuse_protected(save_path)
    ckpt_dir = EXP_ROOT / "checkpoints" / name
    curve_dir = EXP_ROOT / "training_curves"
    log_dir = EXP_ROOT / "logs"
    logger.info("==== Training %s ====", name)
    t0 = time.time()
    hist: dict[str, list] = {}
    model = builder(trainable_base=False)
    stage_epochs = []
    for stage in STAGES:
        if stage["unfreeze"]:
            model = unfreeze_for_finetuning(model, n_unfreeze_layers=stage["unfreeze"])
        model = compile_model(model, stage["lr"])
        train_ds = make_ds(train_df, shuffle=True, augment=True, batch=stage["batch"])
        val_ds = make_ds(val_df, shuffle=False, augment=False, batch=stage["batch"])
        logger.info("Stage %s lr=%s batch=%s", stage["name"], stage["lr"], stage["batch"])
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=stage["epochs"],
            callbacks=local_callbacks(f"{name}_{stage['name']}", ckpt_dir, log_dir),
            verbose=1,
        )
        for key, values in history.history.items():
            hist.setdefault(key, []).extend(list(values))
        stage_epochs.append({"stage": stage["name"], "epochs_run": len(history.history.get("loss", []))})
        ckpt = ckpt_dir / f"{name}_{stage['name']}_best.keras"
        if ckpt.exists():
            model = tf.keras.models.load_model(str(ckpt), compile=False)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    plot_training_curves(hist, curve_dir, name)
    _json_dump(EXP_ROOT / f"training_history_{name.replace('efficientnetb0', 'efficientnet').replace('mobilenetv2', 'mobilenet')}.json", hist)
    val_ds = make_ds(val_df, shuffle=False, augment=False, batch=8)
    y_true, y_prob = _predict_dataset(model, val_ds)
    del model
    tf.keras.backend.clear_session()
    return {
        "path": str(save_path),
        "history": hist,
        "best_epoch_val_pr_auc": best_epoch_from_history(hist, "val_pr_auc"),
        "stage_epochs": stage_epochs,
        "training_seconds": time.time() - t0,
        "y_val_true": y_true,
        "y_val_prob": y_prob,
    }


def select_threshold(y_true, y_prob) -> dict:
    rows = threshold_table(y_true, y_prob, thresholds=THRESHOLD_GRID)
    chosen = max(rows, key=lambda r: (r["f1"], r["auc"], -abs(r["threshold"] - 0.5)))
    return {
        "threshold": float(chosen["threshold"]),
        "rule": "max validation F1 on the predefined 0.20-0.70 grid; ties by ROC-AUC then closeness to 0.50",
        "selected": chosen,
        "table": rows,
    }


def search_ensemble(y_true, p_eff, p_mob) -> dict:
    rows = []
    blends = {}
    for w_eff in WEIGHT_GRID:
        w_mob = round(1.0 - w_eff, 1)
        ens = combine_hybrid_probs(p_eff, p_mob, weights=(w_eff, w_mob), method="soft_voting")
        at_half = evaluate_predictions(y_true, ens, threshold=0.5)
        row = {
            "efficientnet_weight": w_eff,
            "mobilenet_weight": w_mob,
            "roc_auc": float(at_half["auc"]),
            "pr_auc": float(at_half["pr_auc"]),
            "f1_at_0.5": at_half["f1"],
            "accuracy_at_0.5": at_half["accuracy"],
        }
        rows.append(row)
        blends[(w_eff, w_mob)] = ens
    best_w = max(rows, key=lambda r: (r["roc_auc"], r["f1_at_0.5"]))
    ens = blends[(best_w["efficientnet_weight"], best_w["mobilenet_weight"])]
    picked = select_threshold(y_true, ens)
    operating = evaluate_predictions(y_true, ens, threshold=picked["threshold"])
    return {
        "weight_candidates": rows,
        "selected_weights": {
            "efficientnet_weight": best_w["efficientnet_weight"],
            "mobilenet_weight": best_w["mobilenet_weight"],
            "selection_rule": "max validation ROC-AUC, then validation F1 at 0.50",
            "roc_auc": best_w["roc_auc"],
            "f1_at_0.5": best_w["f1_at_0.5"],
        },
        "val_ensemble_prob": ens,
        "threshold_search": picked,
        "operating": operating,
    }


def score_dist(values, labels=None, cls=None) -> dict:
    arr = np.asarray(values, dtype=float).ravel()
    if labels is not None and cls is not None:
        arr = arr[np.asarray(labels).ravel() == cls]
    if arr.size == 0:
        return {}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def write_report(payload: dict) -> None:
    ev = payload["test"]["ensemble"]
    vv = payload["validation"]["ensemble_operating"]
    lines = [
        "# Final melanoma-vs-benign baseline experiment",
        "",
        "This is a research/decision-support experiment. It is **not clinically validated**,",
        "not a clinically calibrated probability, not diagnostic, and not deployment-ready.",
        "Reported scores are AI prediction scores / probability estimates.",
        "",
        "## 1. Objective",
        "",
        "Train one controlled EfficientNetB0 + MobileNetV2 baseline on the frozen",
        "`final_melanoma_benign` dataset, select ensemble weights and threshold on",
        "validation only, then evaluate the frozen test set once.",
        "",
        "## 2. Dataset description",
        "",
        "Frozen dataset at `dataset/processed/final_melanoma_benign/`.",
        "Melanoma = clean DICM melanoma. Benign = clean DICM benign + ISIC 2020 named benign.",
        "ISIC 2020 melanoma was not added. No images were added or removed for this run.",
        "",
        "## 3. Dataset source breakdown",
        "",
        "See the frozen `dataset_report.json` in the dataset directory.",
        "DICM melanoma 6787, DICM benign 4586, ISIC 2020 benign 4881.",
        "",
        "## 4. Class distribution",
        "",
        "Total 16254; melanoma 6787 (41.8%); benign 9467 (58.2%); ratio 1:1.395.",
        "",
        "## 5. Frozen train/validation/test split",
        "",
        "Train 11379 (4703 / 6676). Validation 2438 (1043 / 1395). Test 2437 (1041 / 1396).",
        "Grouping: patient → lesion → image. Seed 42. Splits were not recreated.",
        "",
        "## 6. Leakage audit summary",
        "",
        "Previous construction report: patient/lesion/image overlap across splits = 0.",
        "This training run did not modify splits.",
        "",
        "## 7. Preprocessing",
        "",
        "BGR→RGB; downscale max side 1024 if larger (required for ~30MP ISIC 2020 images);",
        "DullRazor hair removal; light denoise; resize 224; /255; ImageNet preprocess inside the model.",
        "No CLAHE. No lesion crop. Same cache used for train, validation, and test.",
        "",
        "## 8. Augmentation",
        "",
        "Train only via `get_tf_data_augmentation_fn()`: flip, small rotation/zoom/translation,",
        "brightness/contrast, mild saturation/hue, light noise, mild blur, occasional synthetic hair.",
        "Validation and test were not augmented.",
        "",
        "## 9. Architecture",
        "",
        "EfficientNetB0 and MobileNetV2, ImageNet init, shared head:",
        "GAP → Dropout → Dense(256) → BN → Dropout → Dense(1, sigmoid) named `malignant_prob`.",
        "",
        "## 10. Training strategy",
        "",
        "Three-stage fine-tune: frozen head 1e-3 / partial unfreeze 1e-4 / final 1e-5.",
        "Early stopping and checkpoints on validation PR-AUC only. No class weights. No oversampling.",
        "",
        "## 11. Loss function",
        "",
        "BinaryCrossentropy(label_smoothing=0.05). Focal loss was not used.",
        "",
        "## 12. Random seeds",
        "",
        f"Python/NumPy/TensorFlow/dataset seed = {SEED}.",
        "",
        "## 13. EfficientNet validation results",
        "",
        json.dumps(payload["validation"]["efficientnet"], indent=2),
        "",
        "## 14. MobileNet validation results",
        "",
        json.dumps(payload["validation"]["mobilenet"], indent=2),
        "",
        "## 15. Ensemble validation results",
        "",
        json.dumps(vv, indent=2),
        "",
        "## 16. Selected ensemble weights",
        "",
        f"EfficientNet {payload['ensemble']['efficientnet_weight']:.2f} / MobileNet {payload['ensemble']['mobilenet_weight']:.2f}",
        f"Rule: {payload['ensemble']['selection_rule']}",
        "",
        "## 17. Selected threshold",
        "",
        f"{payload['threshold']['threshold']:.2f} ({payload['threshold']['rule']})",
        "",
        "## 18. Final frozen test results",
        "",
        "Evaluated once after weights, ensemble, and threshold were frozen.",
        json.dumps(ev, indent=2),
        "",
        "## 19. Confusion matrix",
        "",
        f"TP {ev['tp']} / TN {ev['tn']} / FP {ev['fp']} / FN {ev['fn']}",
        "",
        "## 20. ROC-AUC",
        "",
        f"Frozen test ROC-AUC {ev['auc']:.4f}",
        "",
        "## 21. PR-AUC",
        "",
        f"Frozen test PR-AUC {ev['pr_auc']:.4f}",
        "",
        "## 22. Training time",
        "",
        f"EfficientNet {payload['timing']['efficientnet_seconds']:.1f}s; MobileNet {payload['timing']['mobilenet_seconds']:.1f}s; total {payload['timing']['total_seconds']:.1f}s.",
        "",
        "## 23. Hardware/environment",
        "",
        json.dumps(payload["environment"], indent=2),
        "",
        "## 24. Limitations",
        "",
        "- Two-source dataset; source confounding is possible.",
        "- Missing DICM patient IDs remain.",
        "- No probability calibration.",
        "- Not clinically validated.",
        "- The previous ~92% experiment used a different (noisier) label definition and is not a like-for-like comparison.",
        "",
        "## 25. Exact model file paths",
        "",
        payload["efficientnet_path"],
        payload["mobilenet_path"],
        "",
        "## 26. Reproducibility information",
        "",
        "Script: `scripts/train_final_melanoma_benign_baseline.py`. Config: `training_config.json`.",
        "Production models, old 92% experiment, clean DICM experiment, and frozen CSVs were not modified.",
        "Test set was not used for tuning. No post-test tuning was performed.",
        "",
        "VALIDATION RESULTS are in sections 13-17. FINAL FROZEN TEST RESULTS are in sections 18-21.",
    ]
    (EXP_ROOT / "FINAL_BASELINE_EXPERIMENT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    EXP_ROOT.mkdir(parents=True, exist_ok=True)
    (EXP_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    before = snapshot_protected()
    _json_dump(EXP_ROOT / "production_snapshot_before.json", before)
    frames = verify_frozen()
    print(
        "\n".join(
            [
                "DATASET: dataset/processed/final_melanoma_benign/",
                "total 16254 melanoma 6787 benign 9467",
                "SPLITS: train 11379 val 2438 test 2437",
                "MODELS: EfficientNetB0 + MobileNetV2",
                "LOSS: BCE + label smoothing 0.05",
                "CLASS WEIGHTS: none",
                "PRODUCTION MODELS MODIFIED: NO",
                "OLD 92% EXPERIMENT MODIFIED: NO",
                "CLEAN DICM EXPERIMENT MODIFIED: NO",
                "FROZEN DATASET MODIFIED: NO",
                "TEST USED FOR TUNING: NO",
                "Safety checks passed. Training may begin.",
                "",
            ]
        ),
        flush=True,
    )
    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "seed": SEED,
        "gpu": [d.name for d in tf.config.list_physical_devices("GPU")],
    }
    _json_dump(
        EXP_ROOT / "training_config.json",
        {
            "dataset": str(DATA_DIR),
            "seed": SEED,
            "label_smoothing": LABEL_SMOOTHING,
            "class_weight": None,
            "oversampling": False,
            "stages": STAGES,
            "weight_grid": WEIGHT_GRID,
            "threshold_grid": THRESHOLD_GRID,
            "ensemble_rule": "max validation ROC-AUC then F1 at 0.50",
            "threshold_rule": "max validation F1 on 0.20-0.70 grid",
            "test_used_for_tuning": False,
            "preprocessing": "hair removal + denoise + 224 + /255; max-side 1024 before hair; no CLAHE; no crop",
            "augmentation": "get_tf_data_augmentation_fn train only",
            "environment": env,
        },
    )
    enable_mixed_precision()
    tf.keras.utils.set_random_seed(SEED)
    np.random.seed(SEED)

    train_df = attach_cache(frames["train"])
    val_df = attach_cache(frames["val"])
    t_all = time.time()
    eff = train_one(build_efficientnetb0, "efficientnetb0", train_df, val_df, EXP_ROOT / "efficientnetb0_final_baseline.keras")
    _json_dump(EXP_ROOT / "logs" / "status.json", {"stage": "efficientnet_done", "seconds": eff["training_seconds"]})
    assert_unchanged(before)
    mob = train_one(build_mobilenetv2, "mobilenetv2", train_df, val_df, EXP_ROOT / "mobilenetv2_final_baseline.keras")
    assert_unchanged(before)

    y_val = np.asarray(eff["y_val_true"]).astype(int)
    search = search_ensemble(y_val, eff["y_val_prob"], mob["y_val_prob"])
    w = search["selected_weights"]
    thr = search["threshold_search"]
    val_eff = evaluate_predictions(y_val, eff["y_val_prob"], threshold=thr["threshold"])
    val_mob = evaluate_predictions(y_val, mob["y_val_prob"], threshold=thr["threshold"])
    pd.DataFrame(
        {
            "image_id": val_df["image_id"].astype(str).values,
            "label": y_val,
            "efficientnet_prob": np.asarray(eff["y_val_prob"]).ravel(),
            "mobilenet_prob": np.asarray(mob["y_val_prob"]).ravel(),
            "ensemble_prob": np.asarray(search["val_ensemble_prob"]).ravel(),
        }
    ).to_csv(EXP_ROOT / "validation_predictions.csv", index=False)
    pd.DataFrame(search["weight_candidates"]).to_csv(EXP_ROOT / "ensemble_weight_search.csv", index=False)
    save_threshold_table(thr["table"], EXP_ROOT / "val_threshold_table.csv")
    _json_dump(EXP_ROOT / "logs" / "status.json", {"stage": "validation_selection_frozen"})

    test_df = attach_cache(frames["test"])
    eff_m = tf.keras.models.load_model(str(EXP_ROOT / "efficientnetb0_final_baseline.keras"), compile=False)
    test_ds = make_ds(test_df, shuffle=False, augment=False, batch=8)
    y_test, p_eff = _predict_dataset(eff_m, test_ds)
    del eff_m
    tf.keras.backend.clear_session()
    mob_m = tf.keras.models.load_model(str(EXP_ROOT / "mobilenetv2_final_baseline.keras"), compile=False)
    y_test2, p_mob = _predict_dataset(mob_m, test_ds)
    del mob_m
    tf.keras.backend.clear_session()
    if not np.array_equal(np.asarray(y_test), np.asarray(y_test2)):
        raise RuntimeError("Test label order mismatch.")
    ens = combine_hybrid_probs(p_eff, p_mob, weights=(w["efficientnet_weight"], w["mobilenet_weight"]), method="soft_voting")
    threshold = float(thr["threshold"])
    test_ens = evaluate_predictions(y_test, ens, threshold=threshold)
    test_eff = evaluate_predictions(y_test, p_eff, threshold=threshold)
    test_mob = evaluate_predictions(y_test, p_mob, threshold=threshold)
    y_pred = (np.asarray(ens) >= threshold).astype(int)
    plot_confusion_matrix(y_test, y_pred, EXP_ROOT / "test_cm.png", labels=["Benign", "Melanoma"])
    plot_roc_curve(y_test, ens, EXP_ROOT / "test_roc.png", "Final baseline ensemble")
    plot_pr_curve(y_test, ens, EXP_ROOT / "test_pr.png", "Final baseline ensemble")
    fpr, tpr, tthr = roc_curve(y_test, ens)
    prec, rec, pthr = precision_recall_curve(y_test, ens)
    _json_dump(EXP_ROOT / "test_roc_curve.json", {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": tthr.tolist()})
    _json_dump(EXP_ROOT / "test_pr_curve.json", {"precision": prec.tolist(), "recall": rec.tolist(), "thresholds": pthr.tolist()})
    pd.DataFrame(
        {
            "image_id": test_df["image_id"].astype(str).values,
            "label": np.asarray(y_test).astype(int),
            "efficientnet_prob": np.asarray(p_eff).ravel(),
            "mobilenet_prob": np.asarray(p_mob).ravel(),
            "ensemble_prob": np.asarray(ens).ravel(),
            "predicted": y_pred,
        }
    ).to_csv(EXP_ROOT / "test_predictions.csv", index=False)

    payload = {
        "efficientnet_path": str(EXP_ROOT / "efficientnetb0_final_baseline.keras"),
        "mobilenet_path": str(EXP_ROOT / "mobilenetv2_final_baseline.keras"),
        "ensemble": w,
        "threshold": {"threshold": threshold, "rule": thr["rule"]},
        "validation": {
            "efficientnet": val_eff,
            "mobilenet": val_mob,
            "ensemble_operating": search["operating"],
        },
        "test": {"efficientnet": test_eff, "mobilenet": test_mob, "ensemble": test_ens},
        "timing": {
            "efficientnet_seconds": eff["training_seconds"],
            "mobilenet_seconds": mob["training_seconds"],
            "total_seconds": time.time() - t_all,
        },
        "efficientnet_best_epoch": eff["best_epoch_val_pr_auc"],
        "mobilenet_best_epoch": mob["best_epoch_val_pr_auc"],
        "score_distributions": {
            "test_all": score_dist(ens),
            "test_melanoma": score_dist(ens, y_test, 1),
            "test_benign": score_dist(ens, y_test, 0),
            "predicted_counts": {"melanoma": int(y_pred.sum()), "benign": int((y_pred == 0).sum())},
            "true_counts": {"melanoma": int(np.asarray(y_test).sum()), "benign": int((np.asarray(y_test) == 0).sum())},
        },
        "environment": env,
        "test_used_for_tuning": False,
        "post_test_tuning": False,
    }
    _json_dump(EXP_ROOT / "training_summary.json", payload)
    _json_dump(
        EXP_ROOT / "training_history_efficientnet.json",
        {"history": eff["history"], "best_epoch_val_pr_auc": eff["best_epoch_val_pr_auc"], "stage_epochs": eff["stage_epochs"]},
    )
    _json_dump(
        EXP_ROOT / "training_history_mobilenet.json",
        {"history": mob["history"], "best_epoch_val_pr_auc": mob["best_epoch_val_pr_auc"], "stage_epochs": mob["stage_epochs"]},
    )
    write_report(payload)
    assert_unchanged(before)
    _json_dump(EXP_ROOT / "production_snapshot_after.json", snapshot_protected())
    _json_dump(EXP_ROOT / "logs" / "status.json", {"stage": "complete"})
    print("\nBASELINE EXPERIMENT COMPLETE")
    print(f"Ensemble weights {w['efficientnet_weight']:.2f}/{w['mobilenet_weight']:.2f} threshold {threshold:.2f}")
    print(
        f"TEST acc={test_ens['accuracy']:.4f} sens={test_ens['sensitivity']:.4f} spec={test_ens['specificity']:.4f} "
        f"prec={test_ens['precision']:.4f} f1={test_ens['f1']:.4f} auc={test_ens['auc']:.4f} pr={test_ens['pr_auc']:.4f}"
    )
    print(f"CM TP={test_ens['tp']} TN={test_ens['tn']} FP={test_ens['fp']} FN={test_ens['fn']}")
    print("PRODUCTION UNCHANGED: YES")
    print("TEST USED FOR TUNING: NO")
    print("POST-TEST TUNING: NO")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        _json_dump(EXP_ROOT / "logs" / "status.json", {"stage": "failed", "error": traceback.format_exc()})
        raise
