"""
ONE targeted low-contrast augmentation experiment.

The only training change vs the frozen 83.42% baseline is extra conservative
photometric jitter on the TRAIN set. Ensemble 0.70/0.30 and threshold 0.50
are fixed. No class weights, oversampling, ensemble search, or threshold search.

Writes only under models/experiments/final_melanoma_benign_targeted_augmentation/.
Test is used once only if validation constraints and a meaningful FN/sensitivity
gain are both met.
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
from preprocessing.augmentation import get_targeted_low_contrast_train_augmentation_fn
from training.evaluate import (
    best_epoch_from_history,
    evaluate_predictions,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
    plot_training_curves,
)
from training.train import _predict_dataset, enable_mixed_precision
from utils.logger import setup_logger

logger = setup_logger("targeted_aug")

DATA_DIR = config.PROCESSED_DATA_DIR / "final_melanoma_benign"
EXP_ROOT = config.BASE_DIR / "models" / "experiments" / "final_melanoma_benign_targeted_augmentation"
BASELINE_ROOT = config.BASE_DIR / "models" / "experiments" / "final_melanoma_benign_baseline"
CACHE_DIR = BASELINE_ROOT / "preprocessed"
PROTECTED_ROOTS = [
    config.MODELS_DIR.resolve(),
    DATA_DIR.resolve(),
    BASELINE_ROOT.resolve(),
    (config.BASE_DIR / "models" / "experiments" / "final_melanoma_benign_class_weighted").resolve(),
    (config.BASE_DIR / "models" / "experiments" / "final_melanoma_benign_moderate_weighting").resolve(),
    (config.BASE_DIR / "models" / "experiments" / "final_melanoma_benign_threshold_optimization").resolve(),
    (config.BASE_DIR / "models" / "experiments" / "final_melanoma_benign_fn_analysis").resolve(),
    (config.BASE_DIR / "models" / "experiments" / "final_melanoma_benign_baseline_tta").resolve(),
]
PROTECTED_FILES = {
    config.EFFICIENTNET_PATH.resolve(),
    config.MOBILENET_PATH.resolve(),
    config.INFERENCE_CONFIG_PATH.resolve(),
    (BASELINE_ROOT / "efficientnetb0_final_baseline.keras").resolve(),
    (BASELINE_ROOT / "mobilenetv2_final_baseline.keras").resolve(),
    (BASELINE_ROOT / "training_summary.json").resolve(),
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
ENSEMBLE_WEIGHTS = (0.70, 0.30)
THRESHOLD = 0.50
CONSTRAINTS = {"specificity": 0.82, "precision": 0.78, "f1": 0.80, "accuracy": 0.82}
BASELINE_VAL = {
    "accuracy": 0.8264971287940935,
    "sensitivity": 0.822627037392138,
    "specificity": 0.8293906810035843,
    "precision": 0.7828467153284672,
    "f1": 0.8022440392706872,
    "auc": 0.911757853173744,
    "pr_auc": 0.8893139033924969,
    "tp": 858,
    "tn": 1157,
    "fp": 238,
    "fn": 185,
}
BASELINE_TEST = {
    "accuracy": 0.8342224045958145,
    "sensitivity": 0.8146013448607109,
    "specificity": 0.8488538681948424,
    "precision": 0.8007554296506137,
    "f1": 0.8076190476190476,
    "auc": 0.9126790142826079,
    "pr_auc": 0.8884757664753524,
    "tp": 848,
    "tn": 1185,
    "fp": 211,
    "fn": 193,
}
AUG_SPEC = {
    "keeps_baseline_augmentation": True,
    "extra_brightness": {"factor": [0.85, 1.15], "probability": 0.30},
    "extra_contrast": {"factor": [0.75, 1.25], "probability": 0.40},
    "extra_gamma": {"factor": [0.85, 1.15], "probability": 0.20},
    "extra_saturation": {"factor": [0.90, 1.10], "probability": 0.15},
    "applied_to": "train only",
    "validation_fn_oversampled": False,
}


def _jsonable(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _refuse(path: Path) -> None:
    resolved = path.resolve()
    if resolved in PROTECTED_FILES:
        raise RuntimeError(f"Refusing to write protected file: {path}")
    if EXP_ROOT.resolve() in resolved.parents or resolved == EXP_ROOT.resolve():
        return
    for root in PROTECTED_ROOTS:
        if resolved == root or root in resolved.parents:
            raise RuntimeError(f"Refusing to write protected path: {path}")


def _dump(path: Path, payload) -> None:
    _refuse(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_jsonable), encoding="utf-8")


def _write(path: Path, text: str) -> None:
    _refuse(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_protected() -> dict:
    extra = [DATA_DIR / name for name in ("train.csv", "val.csv", "test.csv", "manifest.csv")]
    extra.extend(
        [
            BASELINE_ROOT / "efficientnetb0_final_baseline.keras",
            BASELINE_ROOT / "mobilenetv2_final_baseline.keras",
            BASELINE_ROOT / "training_summary.json",
        ]
    )
    out = {}
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
    cache_n = len(list(CACHE_DIR.glob("*.jpg")))
    if cache_n != 16254:
        raise RuntimeError(f"STOP: expected 16254 cached jpgs, found {cache_n}")
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
        aug_fn = get_targeted_low_contrast_train_augmentation_fn()
        ds = ds.map(lambda image, label: aug_fn(image, label), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def local_callbacks(model_name: str, ckpt_dir: Path, log_dir: Path):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    _refuse(ckpt_dir / f"{model_name}_best.keras")
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
    _refuse(save_path)
    ckpt_dir = EXP_ROOT / "checkpoints" / name
    log_dir = EXP_ROOT / "logs"
    logger.info("==== Training %s (targeted low-contrast aug) ====", name)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    np.random.seed(SEED)
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
    plot_training_curves(hist, EXP_ROOT / "training_curves", name)
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


def predict_split(model_path: Path, df: pd.DataFrame):
    model = tf.keras.models.load_model(str(model_path), compile=False)
    ds = make_ds(df, shuffle=False, augment=False, batch=8)
    y_true, y_prob = _predict_dataset(model, ds)
    del model
    tf.keras.backend.clear_session()
    return np.asarray(y_true).astype(int), np.asarray(y_prob).ravel()


def constraint_failures(row: dict) -> list[str]:
    return [f"{k} {row[k]:.4f} < {m:.2f}" for k, m in CONSTRAINTS.items() if float(row[k]) < m - 1e-12]


def meaningful_gain(val_m: dict) -> bool:
    fn_drop = int(BASELINE_VAL["fn"]) - int(val_m["fn"])
    sens_up = float(val_m["sensitivity"]) - float(BASELINE_VAL["sensitivity"])
    return fn_drop >= 5 or sens_up >= 0.01


def write_readme(payload: dict) -> None:
    val_m = payload["validation"]
    test_b = payload["test"]
    sel = payload["selection"]
    lines = [
        "# Targeted low-contrast augmentation experiment",
        "",
        "Research / decision-support only. **Not clinically validated.**",
        "",
        "## Objective",
        "",
        "Test whether extra conservative train-time photometric jitter (brightness, contrast,",
        "mild gamma, mild saturation) reduces false negatives versus the frozen 83.42% baseline",
        "without breaking specificity, precision, F1, or accuracy constraints.",
        "",
        "## Frozen baseline",
        "",
        "Ensemble 0.70/0.30, threshold 0.50. Test acc 83.42%, sens 81.46%, spec 84.89%,",
        "prec 80.08%, F1 0.8076, ROC-AUC 0.9127, PR-AUC 0.8885, FN 193.",
        "",
        "## FN-analysis motivation",
        "",
        "Validation-only analysis: FN images were flatter/brighter than TP melanoma",
        "(contrast ~25 vs ~48; brightness ~155 vs ~137). Most FNs scored 0.30–0.50.",
        "Validation FNs were not oversampled.",
        "",
        "## Exact extra augmentation (train only, on top of baseline aug)",
        "",
        json.dumps(AUG_SPEC, indent=2),
        "",
        "## Training configuration",
        "",
        "Same 3-stage fine-tune, BCE + label smoothing 0.05, seed 42, no class weights,",
        "no oversampling. Ensemble and threshold locked at 0.70/0.30 and 0.50.",
        "",
        "## Validation results",
        "",
        json.dumps(val_m, indent=2),
        "",
        f"Constraints satisfied: {sel['meets_constraints']}",
        f"Meaningful FN/sensitivity gain vs baseline val: {sel['meaningful_gain']}",
        f"Test evaluation performed: {test_b['evaluated_once']}",
        "",
        "## Test results",
        "",
        json.dumps(test_b, indent=2),
        "",
        "## Comparison and recommendation",
        "",
        payload["interpretation"]["detail"],
        "",
        f"Replace frozen baseline: {payload['interpretation']['replace_baseline']}",
        "",
        "NO test images were used for model selection. Nothing else was trained after this run.",
        "",
    ]
    _write(EXP_ROOT / "README.md", "\n".join(lines) + "\n")


def main() -> None:
    EXP_ROOT.mkdir(parents=True, exist_ok=True)
    (EXP_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    before = snapshot_protected()
    _dump(EXP_ROOT / "production_snapshot_before.json", before)
    frames = verify_frozen()
    checklist = {
        "frozen_baseline_untouched": True,
        "dataset_unchanged": True,
        "splits_unchanged": True,
        "validation_fns_not_oversampled": True,
        "no_val_test_augmentation": True,
        "no_threshold_search": True,
        "no_ensemble_search": True,
        "no_class_weight_change": True,
        "no_oversampling": True,
        "no_undersampling": True,
        "ham10000": False,
        "ph2_tuning": False,
        "derm7pt_tuning": False,
        "production_unchanged": True,
        "dashboard_unchanged": True,
        "threshold": THRESHOLD,
        "ensemble": {"efficientnet": 0.70, "mobilenet": 0.30},
    }
    _dump(EXP_ROOT / "safety_checklist.json", checklist)
    print(
        "\n".join(
            [
                "EXPERIMENT: final_melanoma_benign_targeted_augmentation",
                "ONLY CHANGE: extra train-only photometric jitter",
                "ENSEMBLE: 0.70/0.30 FIXED",
                "THRESHOLD: 0.50 FIXED",
                "VAL FN OVERSAMPLED: NO",
                "CLASS WEIGHTS: none",
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
        "seed": SEED,
        "gpu": [d.name for d in tf.config.list_physical_devices("GPU")],
    }
    _dump(
        EXP_ROOT / "training_config.json",
        {
            "dataset": str(DATA_DIR),
            "seed": SEED,
            "label_smoothing": LABEL_SMOOTHING,
            "class_weight": None,
            "oversampling": False,
            "augmentation": AUG_SPEC,
            "ensemble_weights": {"efficientnet": 0.70, "mobilenet": 0.30},
            "threshold": THRESHOLD,
            "stages": STAGES,
            "constraints": CONSTRAINTS,
            "test_used_for_tuning": False,
            "environment": env,
        },
    )
    enable_mixed_precision()
    tf.keras.utils.set_random_seed(SEED)
    np.random.seed(SEED)
    train_df = attach_cache(frames["train"])
    val_df = attach_cache(frames["val"])
    t_all = time.time()
    eff_path = EXP_ROOT / "efficientnetb0_targeted_aug.keras"
    mob_path = EXP_ROOT / "mobilenetv2_targeted_aug.keras"

    if not eff_path.is_file():
        eff = train_one(build_efficientnetb0, "efficientnetb0", train_df, val_df, eff_path)
        _dump(
            EXP_ROOT / "training_history_efficientnet.json",
            {"history": eff["history"], "best_epoch_val_pr_auc": eff["best_epoch_val_pr_auc"], "stage_epochs": eff["stage_epochs"], "training_seconds": eff["training_seconds"]},
        )
        _dump(EXP_ROOT / "logs" / "status.json", {"stage": "efficientnet_done", "seconds": eff["training_seconds"]})
        assert_unchanged(before)
        y_val, p_eff = np.asarray(eff["y_val_true"]).astype(int), np.asarray(eff["y_val_prob"]).ravel()
    else:
        logger.info("Resume EfficientNet from disk")
        y_val, p_eff = predict_split(eff_path, val_df)

    if not mob_path.is_file():
        mob = train_one(build_mobilenetv2, "mobilenetv2", train_df, val_df, mob_path)
        _dump(
            EXP_ROOT / "training_history_mobilenet.json",
            {"history": mob["history"], "best_epoch_val_pr_auc": mob["best_epoch_val_pr_auc"], "stage_epochs": mob["stage_epochs"], "training_seconds": mob["training_seconds"]},
        )
        _dump(EXP_ROOT / "logs" / "status.json", {"stage": "mobilenet_done", "seconds": mob["training_seconds"]})
        assert_unchanged(before)
        y_val2, p_mob = np.asarray(mob["y_val_true"]).astype(int), np.asarray(mob["y_val_prob"]).ravel()
        if not np.array_equal(y_val, y_val2):
            raise RuntimeError("Validation label order mismatch")
    else:
        logger.info("Resume MobileNet from disk")
        y_val2, p_mob = predict_split(mob_path, val_df)
        if not np.array_equal(y_val, y_val2):
            raise RuntimeError("Validation label order mismatch")

    ens = combine_hybrid_probs(p_eff, p_mob, weights=ENSEMBLE_WEIGHTS, method="soft_voting")
    val_m = evaluate_predictions(y_val, ens, threshold=THRESHOLD)
    failed = constraint_failures(val_m)
    gain = meaningful_gain(val_m)
    qualify = (not failed) and gain
    pred_val = pd.DataFrame(
        {
            "image_id": val_df["image_id"].astype(str).values,
            "label": y_val,
            "efficientnet_prob": p_eff,
            "mobilenet_prob": p_mob,
            "ensemble_prob": ens,
            "predicted": (ens >= THRESHOLD).astype(int),
        }
    )
    _refuse(EXP_ROOT / "validation_predictions.csv")
    pred_val.to_csv(EXP_ROOT / "validation_predictions.csv", index=False)
    _dump(EXP_ROOT / "validation_metrics.json", {**val_m, "constraint_failures": failed, "meets_constraints": not failed, "meaningful_gain": gain})
    y_pred_val = (ens >= THRESHOLD).astype(int)
    plot_confusion_matrix(y_val, y_pred_val, EXP_ROOT / "confusion_matrix.png", labels=["Benign", "Melanoma"])
    plot_roc_curve(y_val, ens, EXP_ROOT / "roc_curve.png", "Targeted-aug ensemble (validation)")
    plot_pr_curve(y_val, ens, EXP_ROOT / "pr_curve.png", "Targeted-aug ensemble (validation)")
    selection = {
        "meets_constraints": not failed,
        "constraint_failures": failed,
        "meaningful_gain": gain,
        "qualifies_for_test": qualify,
        "rule": "Test only if all constraints hold and FN drops by >=5 or sensitivity rises by >=1 pp vs baseline validation.",
    }
    _dump(EXP_ROOT / "selected_configuration.json", selection)
    _dump(EXP_ROOT / "logs" / "status.json", {"stage": "validation_locked", "selection": selection})

    test_block = {
        "evaluated_once": False,
        "post_test_tuning": False,
        "threshold": THRESHOLD,
        "ensemble_weights": {"efficientnet": 0.70, "mobilenet": 0.30},
        "metrics": None,
        "reason": None,
    }
    if qualify:
        test_df = attach_cache(frames["test"])
        y_test, te_eff = predict_split(eff_path, test_df)
        y_test2, te_mob = predict_split(mob_path, test_df)
        if not np.array_equal(y_test, y_test2):
            raise RuntimeError("Test label order mismatch")
        te_ens = combine_hybrid_probs(te_eff, te_mob, weights=ENSEMBLE_WEIGHTS, method="soft_voting")
        test_m = evaluate_predictions(y_test, te_ens, threshold=THRESHOLD)
        y_pred = (te_ens >= THRESHOLD).astype(int)
        plot_confusion_matrix(y_test, y_pred, EXP_ROOT / "test_confusion_matrix.png", labels=["Benign", "Melanoma"])
        fpr, tpr, tthr = roc_curve(y_test, te_ens)
        prec, rec, pthr = precision_recall_curve(y_test, te_ens)
        _dump(EXP_ROOT / "test_roc_curve.json", {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": tthr.tolist()})
        _dump(EXP_ROOT / "test_pr_curve.json", {"precision": prec.tolist(), "recall": rec.tolist(), "thresholds": pthr.tolist()})
        pred_test = pd.DataFrame(
            {
                "image_id": test_df["image_id"].astype(str).values,
                "label": y_test,
                "efficientnet_prob": te_eff,
                "mobilenet_prob": te_mob,
                "ensemble_prob": te_ens,
                "predicted": y_pred,
            }
        )
        _refuse(EXP_ROOT / "test_predictions.csv")
        pred_test.to_csv(EXP_ROOT / "test_predictions.csv", index=False)
        test_block.update({"evaluated_once": True, "metrics": test_m})
        _dump(EXP_ROOT / "test_metrics.json", test_block)
    else:
        test_block["reason"] = (
            "Validation did not both meet constraints and show a meaningful FN/sensitivity gain. Test not evaluated."
        )
        _dump(EXP_ROOT / "test_metrics.json", test_block)

    comparison = [
        {"strategy": "Frozen baseline", "evaluation": "frozen_test", **BASELINE_TEST},
        {"strategy": "Frozen baseline", "evaluation": "validation", **BASELINE_VAL},
        {"strategy": "Targeted low-contrast augmentation", "evaluation": "validation", **val_m},
    ]
    if test_block["metrics"]:
        comparison.append({"strategy": "Targeted low-contrast augmentation", "evaluation": "frozen_test", **test_block["metrics"]})
    pd.DataFrame(comparison).to_csv(EXP_ROOT / "comparison_table.csv", index=False)

    if qualify and test_block["metrics"]:
        tm = test_block["metrics"]
        fn_drop = 193 - int(tm["fn"])
        balanced = (
            fn_drop > 0
            and float(tm["specificity"]) >= 0.82
            and float(tm["precision"]) >= 0.78
            and float(tm["f1"]) >= 0.80
            and float(tm["accuracy"]) >= 0.82
        )
        detail = (
            f"Frozen-test FN {tm['fn']} vs baseline 193 (delta {int(tm['fn'])-193:+d}). "
            + (
                "Balanced internal-test improvement, but do not auto-replace the frozen baseline."
                if balanced
                else "Sensitivity/FN moved, but the frozen-test trade-off is not a clear replacement for 83.42%."
            )
        )
        interpretation = {"replace_baseline": False, "balanced_test_improvement": balanced, "detail": detail}
    elif not failed and not gain:
        interpretation = {
            "replace_baseline": False,
            "balanced_test_improvement": False,
            "detail": "Validation met the metric floors but did not reduce FN / raise sensitivity enough. Baseline retained. Test not used.",
        }
    else:
        interpretation = {
            "replace_baseline": False,
            "balanced_test_improvement": False,
            "detail": (
                "Targeted augmentation did not provide a sufficiently balanced validation improvement "
                f"(failures: {failed or 'constraints ok but no meaningful FN gain'}). Retain the 83.42% baseline."
            ),
        }

    payload = {
        "experiment": "final_melanoma_benign_targeted_augmentation",
        "augmentation": AUG_SPEC,
        "ensemble_weights": {"efficientnet": 0.70, "mobilenet": 0.30},
        "threshold": THRESHOLD,
        "validation": val_m,
        "selection": selection,
        "test": test_block,
        "baseline_validation": BASELINE_VAL,
        "baseline_test": BASELINE_TEST,
        "interpretation": interpretation,
        "timing": {"total_seconds": time.time() - t_all},
        "environment": env,
        "test_used_for_tuning": False,
        "replace_baseline": False,
        "efficientnet_path": str(eff_path),
        "mobilenet_path": str(mob_path),
    }
    _dump(EXP_ROOT / "training_summary.json", payload)
    write_readme(payload)
    assert_unchanged(before)
    _dump(EXP_ROOT / "production_snapshot_after.json", snapshot_protected())
    _dump(EXP_ROOT / "logs" / "status.json", {"stage": "complete", "selection": selection, "interpretation": interpretation})
    print("\nTARGETED AUGMENTATION EXPERIMENT COMPLETE", flush=True)
    print(
        f"VAL acc={val_m['accuracy']:.4f} sens={val_m['sensitivity']:.4f} spec={val_m['specificity']:.4f} "
        f"prec={val_m['precision']:.4f} f1={val_m['f1']:.4f} FN={val_m['fn']}",
        flush=True,
    )
    print(f"CONSTRAINTS: {not failed}  MEANINGFUL GAIN: {gain}  TEST EVAL: {test_block['evaluated_once']}", flush=True)
    print(interpretation["detail"], flush=True)
    print("BASELINE UNCHANGED: YES", flush=True)
    print("PRODUCTION UNCHANGED: YES", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        try:
            _dump(EXP_ROOT / "logs" / "status.json", {"stage": "failed", "error": traceback.format_exc()})
        except Exception:
            pass
        raise
