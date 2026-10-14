"""
ONE class-weighted experiment on the frozen final_melanoma_benign dataset.

The only intended change vs the frozen 83.4% baseline is moderate class weighting
from the TRAINING SET ONLY. Architecture, preprocessing, augmentation, optimizer,
stages, seeds, ensemble-weight search, and threshold search are identical.

Writes only under models/experiments/final_melanoma_benign_class_weighted/.
Does not modify production, Exp2-C, Derm7pt external, clean DICM, the 83.4%
baseline, or frozen CSVs. Test is used once after validation-only selection.
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

logger = setup_logger("final_class_weighted")

DATA_DIR = config.PROCESSED_DATA_DIR / "final_melanoma_benign"
EXP_ROOT = config.BASE_DIR / "models" / "experiments" / "final_melanoma_benign_class_weighted"
BASELINE_ROOT = config.BASE_DIR / "models" / "experiments" / "final_melanoma_benign_baseline"
CACHE_DIR = BASELINE_ROOT / "preprocessed"
PROTECTED_ROOTS = [
    (config.MODELS_DIR).resolve(),
    (config.PROCESSED_DATA_DIR / "melanoma_benign").resolve(),
    (config.PROCESSED_DATA_DIR / "dicm_clean_melanoma_benign").resolve(),
    (config.PROCESSED_DATA_DIR / "final_melanoma_benign").resolve(),
    (config.BASE_DIR / "models" / "experiments" / "internal_optimization").resolve(),
    (config.BASE_DIR / "models" / "experiments" / "clean_dicm_melanoma_benign").resolve(),
    BASELINE_ROOT.resolve(),
    (config.BASE_DIR / "models" / "experiments" / "old_model_derm7pt_external").resolve(),
    (config.BASE_DIR / "models" / "evaluation").resolve(),
]
PROTECTED_FILES = {
    config.EFFICIENTNET_PATH.resolve(),
    config.MOBILENET_PATH.resolve(),
    config.INFERENCE_CONFIG_PATH.resolve(),
    (BASELINE_ROOT / "efficientnetb0_final_baseline.keras").resolve(),
    (BASELINE_ROOT / "mobilenetv2_final_baseline.keras").resolve(),
    (BASELINE_ROOT / "training_config.json").resolve(),
    (BASELINE_ROOT / "training_summary.json").resolve(),
    (BASELINE_ROOT / "FINAL_BASELINE_EXPERIMENT_REPORT.md").resolve(),
    (
        config.BASE_DIR
        / "models"
        / "experiments"
        / "internal_optimization"
        / "exp2_loss_comparison"
        / "exp2c_bce_smooth"
        / "efficientnetb0_bce_smooth.keras"
    ).resolve(),
    (
        config.BASE_DIR
        / "models"
        / "experiments"
        / "internal_optimization"
        / "exp2_loss_comparison"
        / "exp2c_bce_smooth"
        / "mobilenetv2_bce_smooth.keras"
    ).resolve(),
    (config.BASE_DIR / "models" / "evaluation" / "final_test" / "FINAL_TEST_EVALUATION.json").resolve(),
    (
        config.BASE_DIR
        / "models"
        / "experiments"
        / "old_model_derm7pt_external"
        / "external_metrics.json"
    ).resolve(),
    (config.BASE_DIR / "models" / "experiments" / "old_model_derm7pt_external" / "run_config.json").resolve(),
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
QUOTED_BASELINE_TEST = {
    "accuracy": 0.834,
    "precision": 0.801,
    "sensitivity": 0.815,
    "specificity": 0.849,
    "f1": 0.808,
    "auc": 0.913,
    "pr_auc": 0.888,
    "tp": 848,
    "fp": 211,
    "tn": 1185,
    "fn": 193,
    "efficientnet_weight": 0.70,
    "mobilenet_weight": 0.30,
    "threshold": 0.50,
}


def _jsonable(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _json_dump(path: Path, payload) -> None:
    _refuse_protected(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_jsonable), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    _refuse_protected(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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
    extra = []
    for csv_name in ("train.csv", "val.csv", "test.csv", "manifest.csv"):
        for folder in (
            DATA_DIR,
            config.PROCESSED_DATA_DIR / "melanoma_benign",
            config.PROCESSED_DATA_DIR / "dicm_clean_melanoma_benign",
        ):
            extra.append(folder / csv_name)
    for path in sorted(set(PROTECTED_FILES) | {p.resolve() for p in extra if p.is_file()}):
        if path.is_file():
            out[str(path)] = {"sha256": _sha256(path), "size": path.stat().st_size}
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


def compute_class_weights(train_df: pd.DataFrame) -> dict:
    y = train_df["label"].astype(int).to_numpy()
    n = int(len(y))
    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())
    if n0 == 0 or n1 == 0:
        raise RuntimeError("STOP: training set is missing a class")
    w0 = n / (2.0 * n0)
    w1 = n / (2.0 * n1)
    return {
        "formula": "weight(class) = N / (number_of_classes * class_count)",
        "source": "training set only",
        "N": n,
        "n_classes": 2,
        "benign_count": n0,
        "melanoma_count": n1,
        "benign_weight": float(w0),
        "melanoma_weight": float(w1),
        "keras_class_weight": {"0": float(w0), "1": float(w1)},
        "oversampling": False,
        "undersampling": False,
        "validation_or_test_used": False,
    }


def keras_class_weight(cw: dict) -> dict[int, float]:
    return {0: float(cw["benign_weight"]), 1: float(cw["melanoma_weight"])}


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
    _refuse_protected(ckpt_dir / f"{model_name}_best.keras")
    _refuse_protected(log_dir / f"{model_name}_history.csv")
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


def train_one(builder, name: str, train_df, val_df, save_path: Path, class_weight: dict) -> dict:
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
        logger.info(
            "Stage %s lr=%s batch=%s class_weight=%s",
            stage["name"],
            stage["lr"],
            stage["batch"],
            class_weight,
        )
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=stage["epochs"],
            callbacks=local_callbacks(f"{name}_{stage['name']}", ckpt_dir, log_dir),
            class_weight=class_weight,
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
    hist_name = name.replace("efficientnetb0", "efficientnet").replace("mobilenetv2", "mobilenet")
    _json_dump(EXP_ROOT / f"training_history_{hist_name}.json", hist)
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


def load_frozen_baseline_precision() -> dict:
    path = BASELINE_ROOT / "training_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    ens = payload["test"]["ensemble"]
    return {
        "source": str(path),
        "rerun": False,
        "test_ensemble": ens,
        "efficientnet_weight": payload["ensemble"]["efficientnet_weight"],
        "mobilenet_weight": payload["ensemble"]["mobilenet_weight"],
        "threshold": payload["threshold"]["threshold"],
        "test_efficientnet": payload["test"]["efficientnet"],
        "test_mobilenet": payload["test"]["mobilenet"],
    }


def interpret(new_ens: dict, frozen: dict) -> dict:
    keys = ("accuracy", "sensitivity", "specificity", "precision", "f1", "auc", "pr_auc")
    old = frozen["test_ensemble"]
    deltas = {k: float(new_ens[k]) - float(old[k]) for k in keys}
    improved = [k for k, d in deltas.items() if d > 1e-6]
    worsened = [k for k, d in deltas.items() if d < -1e-6]
    sens_up = deltas["sensitivity"] > 1e-6
    spec_ok = new_ens["specificity"] >= (float(old["specificity"]) - 0.02)
    ranking_up = deltas["auc"] > 1e-6 or deltas["pr_auc"] > 1e-6
    f1_up = deltas["f1"] > 1e-6
    if sens_up and spec_ok and (f1_up or ranking_up) and not (deltas["specificity"] < -0.02 and deltas["f1"] < 0):
        verdict = "evidence of improvement on the internal frozen test set"
        detail = (
            "Sensitivity increased while specificity stayed within 2 points of the baseline, "
            "with F1 and/or ranking metrics also improved."
        )
    elif (f1_up or ranking_up) and (worsened and improved):
        verdict = "trade-off, not a clear improvement"
        detail = "Some metrics improved and others worsened; class weighting shifted the operating point."
    elif worsened and not improved:
        verdict = "no evidence of improvement"
        detail = "All headline metrics were worse than or equal to the frozen baseline."
    elif improved and not worsened:
        verdict = "evidence of improvement on the internal frozen test set"
        detail = "Headline metrics improved versus the frozen baseline."
    else:
        verdict = "trade-off, not a clear improvement"
        detail = "Changes were mixed; do not select this model by accuracy alone."
    return {
        "verdict": verdict,
        "detail": detail,
        "deltas": deltas,
        "improved": improved,
        "worsened": worsened,
        "sensitivity_up": sens_up,
        "specificity_within_2pts": spec_ok,
        "f1_up": f1_up,
        "ranking_up": ranking_up,
        "note": (
            "Internal frozen-test comparison only. Not clinically validated. "
            "Scores are AI probability estimates, not calibrated clinical probabilities. "
            "Derm7pt was not used."
        ),
    }


def fmt_row(name: str, base_v, new_v) -> str:
    return f"{name:<14} {float(base_v):<14.4f} {float(new_v):<14.4f} {float(new_v) - float(base_v):+.4f}"


def write_reports(payload: dict) -> None:
    ev = payload["test"]["ensemble"]
    vv = payload["validation"]["ensemble_operating"]
    cw = payload["class_weights"]
    frozen = payload["frozen_baseline"]
    old = frozen["test_ensemble"]
    interp = payload["interpretation"]
    quoted = QUOTED_BASELINE_TEST
    lines = [
        "# Final melanoma-vs-benign class-weighted experiment",
        "",
        "This is a research/decision-support experiment. It is **not clinically validated**,",
        "not a clinically calibrated probability, not diagnostic, and not deployment-ready.",
        "Reported scores are AI prediction scores / probability estimates.",
        "",
        "## 1. Objective",
        "",
        "Test whether moderate class weighting improves the frozen 83.4% EfficientNetB0 +",
        "MobileNetV2 baseline on `final_melanoma_benign`, keeping every other setting identical.",
        "",
        "## 2. Dataset description",
        "",
        "Frozen dataset at `dataset/processed/final_melanoma_benign/`.",
        "Melanoma = clean DICM melanoma. Benign = clean DICM benign + ISIC 2020 named benign.",
        "No images were added, removed, or reshuffled.",
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
        "BGR→RGB; downscale max side 1024 if larger; DullRazor hair removal; light denoise;",
        "resize 224; /255; ImageNet preprocess inside the model. No CLAHE. No lesion crop.",
        "Read-only reuse of the frozen baseline JPEG cache (identical files; no writes).",
        "",
        "## 8. Augmentation",
        "",
        "Train only via `get_tf_data_augmentation_fn()`. Validation and test were not augmented.",
        "Augmentation is identical to the frozen 83.4% baseline.",
        "",
        "## 9. Architecture",
        "",
        "EfficientNetB0 and MobileNetV2, ImageNet init, shared head:",
        "GAP → Dropout → Dense(256) → BN → Dropout → Dense(1, sigmoid) named `malignant_prob`.",
        "",
        "## 10. Training strategy",
        "",
        "Three-stage fine-tune: frozen head 1e-3 / partial unfreeze 1e-4 / final 1e-5.",
        "Early stopping and checkpoints on validation PR-AUC only. No oversampling.",
        "The only change vs baseline is class weighting.",
        "",
        "## 11. Loss function and class weights",
        "",
        "BinaryCrossentropy(label_smoothing=0.05).",
        f"Formula: `{cw['formula']}`. Source: {cw['source']}.",
        f"N={cw['N']}, benign={cw['benign_count']}, melanoma={cw['melanoma_count']}.",
        f"benign_weight = {cw['N']} / (2 × {cw['benign_count']}) = {cw['benign_weight']:.10f}",
        f"melanoma_weight = {cw['N']} / (2 × {cw['melanoma_count']}) = {cw['melanoma_weight']:.10f}",
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
        "",
        "### A. EfficientNetB0",
        json.dumps(payload["test"]["efficientnet"], indent=2),
        "",
        "### B. MobileNetV2",
        json.dumps(payload["test"]["mobilenet"], indent=2),
        "",
        "### C. Final ensemble",
        json.dumps(ev, indent=2),
        "",
        "## 19. Confusion matrix (ensemble test)",
        "",
        f"TP {ev['tp']} / TN {ev['tn']} / FP {ev['fp']} / FN {ev['fn']}",
        "",
        "## 20. ROC-AUC",
        "",
        f"Frozen test ensemble ROC-AUC {ev['auc']:.4f}",
        "",
        "## 21. PR-AUC",
        "",
        f"Frozen test ensemble PR-AUC {ev['pr_auc']:.4f}",
        "",
        "## 22. Training time",
        "",
        f"EfficientNet {payload['timing']['efficientnet_seconds']:.1f}s; MobileNet {payload['timing']['mobilenet_seconds']:.1f}s; total {payload['timing']['total_seconds']:.1f}s.",
        f"Best epoch (val PR-AUC): EfficientNet {payload['efficientnet_best_epoch']}; MobileNet {payload['mobilenet_best_epoch']}.",
        "",
        "## 23. Hardware/environment",
        "",
        json.dumps(payload["environment"], indent=2),
        "",
        "## 24. Comparison with frozen 83.4% baseline",
        "",
        "Baseline numbers were read from the frozen experiment files and the quoted reference.",
        "The baseline was not rerun.",
        "",
        "Quoted frozen reference (as specified):",
        f"Accuracy {quoted['accuracy']}, Precision {quoted['precision']}, Sensitivity {quoted['sensitivity']},",
        f"Specificity {quoted['specificity']}, F1 {quoted['f1']}, ROC-AUC {quoted['auc']}, PR-AUC {quoted['pr_auc']}.",
        f"CM TP={quoted['tp']} FP={quoted['fp']} TN={quoted['tn']} FN={quoted['fn']}.",
        f"Ensemble 0.70/0.30, threshold 0.50.",
        "",
        "Full-precision comparison against frozen `training_summary.json`:",
        "",
        "Metric         Baseline       Class-weighted  Delta",
        fmt_row("Accuracy", old["accuracy"], ev["accuracy"]),
        fmt_row("Sensitivity", old["sensitivity"], ev["sensitivity"]),
        fmt_row("Specificity", old["specificity"], ev["specificity"]),
        fmt_row("Precision", old["precision"], ev["precision"]),
        fmt_row("F1", old["f1"], ev["f1"]),
        fmt_row("ROC-AUC", old["auc"], ev["auc"]),
        fmt_row("PR-AUC", old["pr_auc"], ev["pr_auc"]),
        "",
        f"Baseline CM: TP {old['tp']} FP {old['fp']} TN {old['tn']} FN {old['fn']}",
        f"New CM:      TP {ev['tp']} FP {ev['fp']} TN {ev['tn']} FN {ev['fn']}",
        "",
        "## 25. Interpretation",
        "",
        f"**{interp['verdict']}**",
        "",
        interp["detail"],
        "",
        interp["note"],
        "",
        "## 26. Limitations",
        "",
        "- Two-source dataset; source confounding is possible.",
        "- Missing DICM patient IDs remain.",
        "- No probability calibration.",
        "- Not clinically validated.",
        "- Internal test comparison only; Derm7pt was not used for tuning or evaluation.",
        "",
        "## 27. Exact model file paths",
        "",
        payload["efficientnet_path"],
        payload["mobilenet_path"],
        "",
        "## 28. Reproducibility information",
        "",
        "Script: `scripts/train_final_melanoma_benign_class_weighted.py`.",
        "Config: `training_config.json`. Class weights: `class_weights.json`.",
        "Production models, Exp2-C, old 92.28% result, Derm7pt external, clean DICM,",
        "the 83.4% baseline, and frozen CSVs were not modified.",
        "Test set was not used for tuning. No post-test tuning was performed.",
        "",
        "VALIDATION RESULTS are in sections 13-17. FINAL FROZEN TEST RESULTS are in sections 18-21.",
    ]
    _write_text(EXP_ROOT / "CLASS_WEIGHTED_EXPERIMENT_REPORT.md", "\n".join(lines) + "\n")

    comparison = [
        "# Comparison: frozen 83.4% baseline vs class-weighted experiment",
        "",
        "Baseline was not rerun. Quoted reference plus full-precision frozen summary.",
        "",
        "| Metric | Baseline (quoted) | Baseline (frozen JSON) | Class-weighted | Delta vs frozen JSON |",
        "|---|---:|---:|---:|---:|",
        f"| Accuracy | {quoted['accuracy']:.3f} | {old['accuracy']:.4f} | {ev['accuracy']:.4f} | {ev['accuracy']-old['accuracy']:+.4f} |",
        f"| Sensitivity | {quoted['sensitivity']:.3f} | {old['sensitivity']:.4f} | {ev['sensitivity']:.4f} | {ev['sensitivity']-old['sensitivity']:+.4f} |",
        f"| Specificity | {quoted['specificity']:.3f} | {old['specificity']:.4f} | {ev['specificity']:.4f} | {ev['specificity']-old['specificity']:+.4f} |",
        f"| Precision | {quoted['precision']:.3f} | {old['precision']:.4f} | {ev['precision']:.4f} | {ev['precision']-old['precision']:+.4f} |",
        f"| F1 | {quoted['f1']:.3f} | {old['f1']:.4f} | {ev['f1']:.4f} | {ev['f1']-old['f1']:+.4f} |",
        f"| ROC-AUC | {quoted['auc']:.3f} | {old['auc']:.4f} | {ev['auc']:.4f} | {ev['auc']-old['auc']:+.4f} |",
        f"| PR-AUC | {quoted['pr_auc']:.3f} | {old['pr_auc']:.4f} | {ev['pr_auc']:.4f} | {ev['pr_auc']-old['pr_auc']:+.4f} |",
        "",
        f"Baseline ensemble 0.70/0.30 threshold 0.50. New ensemble "
        f"{payload['ensemble']['efficientnet_weight']:.2f}/{payload['ensemble']['mobilenet_weight']:.2f} "
        f"threshold {payload['threshold']['threshold']:.2f}.",
        "",
        f"Baseline CM TP={quoted['tp']} FP={quoted['fp']} TN={quoted['tn']} FN={quoted['fn']}.",
        f"New CM TP={ev['tp']} FP={ev['fp']} TN={ev['tn']} FN={ev['fn']}.",
        "",
        f"Verdict: {interp['verdict']}",
        "",
        interp["detail"],
        "",
        "Not clinically validated. Scores are AI probability estimates.",
        "",
    ]
    _write_text(EXP_ROOT / "comparison_report.md", "\n".join(comparison))

    readme = [
        "# final_melanoma_benign_class_weighted",
        "",
        "One controlled experiment: same frozen `final_melanoma_benign` dataset and the same",
        "83.4% baseline settings, with balanced class weights from the training split only.",
        "",
        f"- benign_weight = {cw['benign_weight']:.10f}",
        f"- melanoma_weight = {cw['melanoma_weight']:.10f}",
        "",
        "Start here:",
        "- `CLASS_WEIGHTED_EXPERIMENT_REPORT.md`",
        "- `comparison_report.md`",
        "- `training_summary.json`",
        "- `class_weights.json`",
        "",
        "This directory is the only write target. Frozen baseline, production, Exp2-C,",
        "Derm7pt external, clean DICM, and dataset CSVs were not modified.",
        "",
        "Not clinically validated. `malignant_prob` is a model score, not a calibrated clinical probability.",
        "",
    ]
    _write_text(EXP_ROOT / "README.md", "\n".join(readme))


def write_metrics_tables(payload: dict, y_test, y_pred) -> None:
    ev = payload["test"]["ensemble"]
    rows = []
    for split in ("validation", "test"):
        block = payload[split]
        mapping = {
            "efficientnet": block["efficientnet"],
            "mobilenet": block["mobilenet"],
            "ensemble": block["ensemble_operating"] if split == "validation" else block["ensemble"],
        }
        for model_name, metrics in mapping.items():
            row = {"split": split, "model": model_name}
            row.update(metrics)
            rows.append(row)
    pd.DataFrame(rows).to_csv(EXP_ROOT / "metrics.csv", index=False)
    _json_dump(EXP_ROOT / "metrics.json", {"validation": payload["validation"], "test": payload["test"]})
    cm = pd.DataFrame(
        [[ev["tn"], ev["fp"]], [ev["fn"], ev["tp"]]],
        index=["true_benign", "true_melanoma"],
        columns=["pred_benign", "pred_melanoma"],
    )
    cm.to_csv(EXP_ROOT / "confusion_matrix.csv")
    pd.DataFrame({"y_true": np.asarray(y_test).astype(int), "y_pred": np.asarray(y_pred).astype(int)}).to_csv(
        EXP_ROOT / "test_confusion_pairs.csv", index=False
    )


def main() -> None:
    EXP_ROOT.mkdir(parents=True, exist_ok=True)
    (EXP_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    before = snapshot_protected()
    _json_dump(EXP_ROOT / "production_snapshot_before.json", before)
    frames = verify_frozen()
    class_weights = compute_class_weights(frames["train"])
    _json_dump(EXP_ROOT / "class_weights.json", class_weights)
    cw_keras = keras_class_weight(class_weights)
    print(
        "\n".join(
            [
                "EXPERIMENT: final_melanoma_benign_class_weighted",
                "DATASET: dataset/processed/final_melanoma_benign/",
                "total 16254 melanoma 6787 benign 9467",
                "SPLITS: train 11379 val 2438 test 2437 (FROZEN, not reshuffled)",
                "MODELS: EfficientNetB0 + MobileNetV2 (identical to 83.4% baseline)",
                "LOSS: BCE + label smoothing 0.05",
                f"CLASS WEIGHTS (train only): benign={class_weights['benign_weight']:.10f} melanoma={class_weights['melanoma_weight']:.10f}",
                "OVERSAMPLING: NO",
                "PRODUCTION MODELS MODIFIED: NO",
                "OLD 92% EXPERIMENT MODIFIED: NO",
                "DERM7PT EXTERNAL MODIFIED: NO",
                "CLEAN DICM EXPERIMENT MODIFIED: NO",
                "83.4% BASELINE MODIFIED: NO",
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
            "class_weight": class_weights,
            "oversampling": False,
            "only_intended_change": "class weighting from training set",
            "stages": STAGES,
            "weight_grid": WEIGHT_GRID,
            "threshold_grid": THRESHOLD_GRID,
            "ensemble_rule": "max validation ROC-AUC then F1 at 0.50",
            "threshold_rule": "max validation F1 on 0.20-0.70 grid",
            "test_used_for_tuning": False,
            "preprocessing": "hair removal + denoise + 224 + /255; max-side 1024 before hair; no CLAHE; no crop",
            "preprocess_cache": "read-only reuse of frozen baseline JPEG cache",
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
    eff = train_one(
        build_efficientnetb0,
        "efficientnetb0",
        train_df,
        val_df,
        EXP_ROOT / "efficientnetb0_class_weighted.keras",
        cw_keras,
    )
    _json_dump(EXP_ROOT / "logs" / "status.json", {"stage": "efficientnet_done", "seconds": eff["training_seconds"]})
    assert_unchanged(before)
    mob = train_one(
        build_mobilenetv2,
        "mobilenetv2",
        train_df,
        val_df,
        EXP_ROOT / "mobilenetv2_class_weighted.keras",
        cw_keras,
    )
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
    _json_dump(EXP_ROOT / "ensemble_config.json", w)
    _json_dump(EXP_ROOT / "threshold_selection.json", {"threshold": thr["threshold"], "rule": thr["rule"], "selected": thr["selected"]})
    _json_dump(EXP_ROOT / "logs" / "status.json", {"stage": "validation_selection_frozen", "ensemble": w, "threshold": thr["threshold"]})

    test_df = attach_cache(frames["test"])
    eff_m = tf.keras.models.load_model(str(EXP_ROOT / "efficientnetb0_class_weighted.keras"), compile=False)
    test_ds = make_ds(test_df, shuffle=False, augment=False, batch=8)
    y_test, p_eff = _predict_dataset(eff_m, test_ds)
    del eff_m
    tf.keras.backend.clear_session()
    mob_m = tf.keras.models.load_model(str(EXP_ROOT / "mobilenetv2_class_weighted.keras"), compile=False)
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
    plot_confusion_matrix(y_test, y_pred, EXP_ROOT / "confusion_matrix.png", labels=["Benign", "Melanoma"])
    plot_confusion_matrix(y_test, y_pred, EXP_ROOT / "test_cm.png", labels=["Benign", "Melanoma"])
    plot_roc_curve(y_test, ens, EXP_ROOT / "test_roc.png", "Class-weighted ensemble")
    plot_pr_curve(y_test, ens, EXP_ROOT / "test_pr.png", "Class-weighted ensemble")
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

    frozen = load_frozen_baseline_precision()
    payload = {
        "efficientnet_path": str(EXP_ROOT / "efficientnetb0_class_weighted.keras"),
        "mobilenet_path": str(EXP_ROOT / "mobilenetv2_class_weighted.keras"),
        "class_weights": class_weights,
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
        "frozen_baseline": frozen,
        "quoted_baseline": QUOTED_BASELINE_TEST,
        "test_used_for_tuning": False,
        "post_test_tuning": False,
        "derm7pt_used": False,
    }
    payload["interpretation"] = interpret(test_ens, frozen)
    _json_dump(EXP_ROOT / "training_summary.json", payload)
    _json_dump(
        EXP_ROOT / "training_history_efficientnet.json",
        {"history": eff["history"], "best_epoch_val_pr_auc": eff["best_epoch_val_pr_auc"], "stage_epochs": eff["stage_epochs"]},
    )
    _json_dump(
        EXP_ROOT / "training_history_mobilenet.json",
        {"history": mob["history"], "best_epoch_val_pr_auc": mob["best_epoch_val_pr_auc"], "stage_epochs": mob["stage_epochs"]},
    )
    write_metrics_tables(payload, y_test, y_pred)
    write_reports(payload)
    assert_unchanged(before)
    _json_dump(EXP_ROOT / "production_snapshot_after.json", snapshot_protected())
    _json_dump(EXP_ROOT / "logs" / "status.json", {"stage": "complete", "interpretation": payload["interpretation"]["verdict"]})
    print("\nCLASS-WEIGHTED EXPERIMENT COMPLETE", flush=True)
    print(
        f"Class weights benign={class_weights['benign_weight']:.10f} melanoma={class_weights['melanoma_weight']:.10f}",
        flush=True,
    )
    print(f"Ensemble weights {w['efficientnet_weight']:.2f}/{w['mobilenet_weight']:.2f} threshold {threshold:.2f}", flush=True)
    print(
        f"EFF TEST acc={test_eff['accuracy']:.4f} sens={test_eff['sensitivity']:.4f} spec={test_eff['specificity']:.4f} "
        f"prec={test_eff['precision']:.4f} f1={test_eff['f1']:.4f} auc={test_eff['auc']:.4f} pr={test_eff['pr_auc']:.4f}",
        flush=True,
    )
    print(
        f"MOB TEST acc={test_mob['accuracy']:.4f} sens={test_mob['sensitivity']:.4f} spec={test_mob['specificity']:.4f} "
        f"prec={test_mob['precision']:.4f} f1={test_mob['f1']:.4f} auc={test_mob['auc']:.4f} pr={test_mob['pr_auc']:.4f}",
        flush=True,
    )
    print(
        f"ENS TEST acc={test_ens['accuracy']:.4f} sens={test_ens['sensitivity']:.4f} spec={test_ens['specificity']:.4f} "
        f"prec={test_ens['precision']:.4f} f1={test_ens['f1']:.4f} auc={test_ens['auc']:.4f} pr={test_ens['pr_auc']:.4f}",
        flush=True,
    )
    print(f"CM TP={test_ens['tp']} TN={test_ens['tn']} FP={test_ens['fp']} FN={test_ens['fn']}", flush=True)
    print(f"INTERPRETATION: {payload['interpretation']['verdict']}", flush=True)
    print("PRODUCTION UNCHANGED: YES", flush=True)
    print("BASELINE UNCHANGED: YES", flush=True)
    print("TEST USED FOR TUNING: NO", flush=True)
    print("POST-TEST TUNING: NO", flush=True)
    print("DERM7PT USED: NO", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        try:
            _json_dump(EXP_ROOT / "logs" / "status.json", {"stage": "failed", "error": traceback.format_exc()})
        except Exception:
            pass
        raise
