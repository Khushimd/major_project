"""
ONE moderate melanoma-weighting experiment on the frozen final_melanoma_benign dataset.

The only training change vs the frozen 83.42% baseline is a predefined moderate
class weight. Ensemble stays 0.70/0.30. Threshold stays 0.50. No ensemble search,
no threshold search, no oversampling, no extra datasets.

Writes only under models/experiments/final_melanoma_benign_moderate_weighting/.
Test is used once after validation-only configuration selection.
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
)
from training.train import _predict_dataset, enable_mixed_precision
from utils.logger import setup_logger

logger = setup_logger("moderate_weighting")

DATA_DIR = config.PROCESSED_DATA_DIR / "final_melanoma_benign"
EXP_ROOT = config.BASE_DIR / "models" / "experiments" / "final_melanoma_benign_moderate_weighting"
BASELINE_ROOT = config.BASE_DIR / "models" / "experiments" / "final_melanoma_benign_baseline"
CLASS_WEIGHTED_ROOT = config.BASE_DIR / "models" / "experiments" / "final_melanoma_benign_class_weighted"
THRESHOLD_OPT_ROOT = config.BASE_DIR / "models" / "experiments" / "final_melanoma_benign_threshold_optimization"
CACHE_DIR = BASELINE_ROOT / "preprocessed"
PROTECTED_ROOTS = [
    config.MODELS_DIR.resolve(),
    DATA_DIR.resolve(),
    (config.PROCESSED_DATA_DIR / "melanoma_benign").resolve(),
    (config.PROCESSED_DATA_DIR / "dicm_clean_melanoma_benign").resolve(),
    BASELINE_ROOT.resolve(),
    CLASS_WEIGHTED_ROOT.resolve(),
    THRESHOLD_OPT_ROOT.resolve(),
    (config.BASE_DIR / "models" / "experiments" / "final_melanoma_benign_baseline_tta").resolve(),
    (config.BASE_DIR / "models" / "experiments" / "old_model_derm7pt_external").resolve(),
    (config.BASE_DIR / "models" / "experiments" / "internal_optimization").resolve(),
    (config.BASE_DIR / "models" / "evaluation").resolve(),
]
PROTECTED_FILES = {
    config.EFFICIENTNET_PATH.resolve(),
    config.MOBILENET_PATH.resolve(),
    config.INFERENCE_CONFIG_PATH.resolve(),
    (BASELINE_ROOT / "efficientnetb0_final_baseline.keras").resolve(),
    (BASELINE_ROOT / "mobilenetv2_final_baseline.keras").resolve(),
    (BASELINE_ROOT / "training_summary.json").resolve(),
    (CLASS_WEIGHTED_ROOT / "efficientnetb0_class_weighted.keras").resolve(),
    (CLASS_WEIGHTED_ROOT / "mobilenetv2_class_weighted.keras").resolve(),
    (CLASS_WEIGHTED_ROOT / "training_summary.json").resolve(),
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
CONSTRAINTS = {
    "specificity": 0.82,
    "precision": 0.78,
    "f1": 0.80,
    "accuracy": 0.82,
}
VARIANTS = (
    {"id": "W_1.00", "benign": 1.00, "melanoma": 1.00, "role": "control/reference"},
    {"id": "W_1.05", "benign": 1.00, "melanoma": 1.05, "role": "moderate candidate"},
    {"id": "W_1.10", "benign": 1.00, "melanoma": 1.10, "role": "moderate candidate"},
    {"id": "W_1.15", "benign": 1.00, "melanoma": 1.15, "role": "moderate candidate"},
)
FROZEN_BASELINE_TEST = {
    "name": "Current frozen baseline",
    "mark": "baseline",
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
    "evaluation": "frozen_test",
}
THRESHOLD_OPT_TEST = {
    "name": "Threshold optimization",
    "mark": "previous ablation",
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
    "evaluation": "frozen_test",
}


def _jsonable(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _refuse_protected(path: Path) -> None:
    resolved = path.resolve()
    if resolved in PROTECTED_FILES:
        raise RuntimeError(f"Refusing to write protected file: {path}")
    if EXP_ROOT.resolve() in resolved.parents or resolved == EXP_ROOT.resolve():
        return
    for root in PROTECTED_ROOTS:
        if resolved == root or root in resolved.parents:
            raise RuntimeError(f"Refusing to write protected path: {path}")


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


def snapshot_protected() -> dict:
    extra = []
    for csv_name in ("train.csv", "val.csv", "test.csv", "manifest.csv"):
        extra.append(DATA_DIR / csv_name)
    extra.extend(
        [
            BASELINE_ROOT / "efficientnetb0_final_baseline.keras",
            BASELINE_ROOT / "mobilenetv2_final_baseline.keras",
            BASELINE_ROOT / "training_summary.json",
            CLASS_WEIGHTED_ROOT / "training_summary.json",
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
        aug_fn = get_tf_data_augmentation_fn()
        ds = ds.map(lambda image, label: aug_fn(image, label), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def local_callbacks(model_name: str, ckpt_dir: Path, log_dir: Path):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    _refuse_protected(ckpt_dir / f"{model_name}_best.keras")
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


def variant_dir(variant: dict) -> Path:
    return EXP_ROOT / "variants" / variant["id"]


def train_one(builder, name: str, train_df, val_df, save_path: Path, class_weight: dict, work: Path) -> dict:
    _refuse_protected(save_path)
    ckpt_dir = work / "checkpoints" / name
    curve_dir = work / "training_curves"
    log_dir = work / "logs"
    logger.info("==== Training %s class_weight=%s ====", name, class_weight)
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
    failed = []
    for key, minimum in CONSTRAINTS.items():
        if float(row[key]) < minimum - 1e-12:
            failed.append(f"{key} {row[key]:.4f} < {minimum:.2f}")
    return failed


def load_class_weighted_test() -> dict:
    path = CLASS_WEIGHTED_ROOT / "training_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    ens = payload["test"]["ensemble"]
    return {
        "name": "Previous balanced class weighting",
        "mark": "previous ablation",
        "accuracy": ens["accuracy"],
        "sensitivity": ens["sensitivity"],
        "specificity": ens["specificity"],
        "precision": ens["precision"],
        "f1": ens["f1"],
        "auc": ens["auc"],
        "pr_auc": ens["pr_auc"],
        "tp": ens["tp"],
        "tn": ens["tn"],
        "fp": ens["fp"],
        "fn": ens["fn"],
        "evaluation": "frozen_test",
        "benign_weight": payload["class_weights"]["benign_weight"],
        "melanoma_weight": payload["class_weights"]["melanoma_weight"],
    }


def write_status(payload: dict) -> None:
    _json_dump(EXP_ROOT / "logs" / "status.json", payload)


def write_outputs(payload: dict) -> None:
    val_rows = payload["validation_results"]
    pd.DataFrame(val_rows).to_csv(EXP_ROOT / "validation_results.csv", index=False)
    pd.DataFrame(payload["comparison_rows"]).to_csv(EXP_ROOT / "comparison_table.csv", index=False)
    _json_dump(EXP_ROOT / "training_summary.json", payload)
    _json_dump(EXP_ROOT / "test_metrics.json", payload["test_metrics"])
    selected = payload["selection"]
    cw_prev = payload["previous_class_weighted"]
    test_m = payload["test_metrics"].get("metrics")
    lines = [
        "# Moderate melanoma weighting experiment",
        "",
        "Research / decision-support evaluation only. **Not clinically validated.**",
        "Scores are AI probability estimates. This does not replace the frozen 83.42% baseline",
        "unless the report below explicitly recommends it (it does not do so automatically).",
        "",
        "## Objective",
        "",
        "Ask whether a *small*, predefined melanoma class-weight can reduce false negatives",
        "versus the frozen 83.42% baseline while keeping specificity, precision, F1, and",
        "accuracy in an acceptable band. This is not the previous balanced-weight experiment.",
        "",
        "## Dataset",
        "",
        "Frozen `dataset/processed/final_melanoma_benign/`.",
        "Train 11379 (4703 melanoma / 6676 benign). Val 2438 (1043 / 1395). Test 2437 (1041 / 1396).",
        "Splits, preprocessing cache, and labels were not modified.",
        "",
        "## Weighting configurations",
        "",
        "benign_weight : melanoma_weight, applied only during `model.fit` via `class_weight`.",
        "",
    ]
    for v in VARIANTS:
        lines.append(f"- `{v['id']}`: 1.00 : {v['melanoma']:.2f} ({v['role']})")
    lines += [
        "",
        "No oversampling, undersampling, HAM10000, PH2, or Derm7pt.",
        "",
        "## Fixed operating point",
        "",
        "Ensemble EfficientNet 0.70 / MobileNet 0.30. Decision threshold 0.50.",
        "No ensemble-weight search. No threshold search.",
        "",
        "## Validation selection rule",
        "",
        "Compare the four configurations on validation at 0.50.",
        "Constraints: specificity >= 82%, precision >= 78%, F1 >= 0.80, accuracy >= 82%.",
        "Among feasible: highest sensitivity, then lowest FN, then highest F1, then",
        "highest specificity, then highest precision, then highest accuracy.",
        "Constraints were not relaxed.",
        "",
        json.dumps(selected, indent=2),
        "",
        "## Frozen test results",
        "",
        json.dumps(payload["test_metrics"], indent=2),
        "",
        "## Comparison with baseline and previous ablations",
        "",
        "Candidate rows 1.00–1.15 in `comparison_table.csv` are **validation**.",
        "Baseline, threshold optimization, and balanced class-weighting rows are **frozen test**.",
        "The selected-model row is the single frozen-test evaluation, or the retained baseline.",
        "",
        f"Previous balanced weights: benign={cw_prev['benign_weight']:.6f}, melanoma={cw_prev['melanoma_weight']:.6f}.",
        "",
        "## Did FN decrease?",
        "",
    ]
    if test_m and payload["selection"]["feasible"] and payload["selection"]["selected_id"] != "retained_baseline":
        fn_drop = 193 - int(test_m["fn"])
        lines.append(
            f"Selected `{payload['selection']['selected_id']}` frozen-test FN={test_m['fn']} "
            f"(baseline 193, delta {fn_drop:+d})."
        )
    else:
        lines.append("No new configuration was selected. Frozen-test FN remains 193 at the 83.42% baseline.")
    lines += [
        "",
        "## Should this replace the baseline?",
        "",
        f"{payload['interpretation']['replace_baseline']}",
        "",
        payload["interpretation"]["detail"],
        "",
        "## What was not modified",
        "",
        "- `models/saved/`",
        "- `models/experiments/final_melanoma_benign_baseline/`",
        "- `models/experiments/final_melanoma_benign_class_weighted/`",
        "- Flask app and doctor dashboard",
        "",
        "Script: `scripts/train_final_melanoma_benign_moderate_weighting.py`.",
        "",
    ]
    _write_text(EXP_ROOT / "README.md", "\n".join(lines) + "\n")


def interpret(selection: dict, test_m: dict | None) -> dict:
    if not selection["feasible"] or selection.get("selected_id") == "retained_baseline":
        return {
            "answer": "NO",
            "replace_baseline": False,
            "better_than_0_50_baseline": False,
            "detail": (
                "Moderate weighting did not provide a sufficiently favorable operating "
                "trade-off under the stated constraints. Retain the 83.42% baseline."
            ),
        }
    if test_m is None:
        return {
            "answer": "NO",
            "replace_baseline": False,
            "better_than_0_50_baseline": False,
            "detail": "A validation winner existed but test metrics are missing; retain the baseline.",
        }
    fn_drop = 193 - int(test_m["fn"])
    spec_ok = float(test_m["specificity"]) >= 0.82
    prec_ok = float(test_m["precision"]) >= 0.78
    f1_ok = float(test_m["f1"]) >= 0.80
    acc_ok = float(test_m["accuracy"]) >= 0.82
    aggressive = (float(FROZEN_BASELINE_TEST["specificity"]) - float(test_m["specificity"])) > 0.04
    useful = fn_drop >= 5 and spec_ok and prec_ok and f1_ok and acc_ok and not aggressive
    if useful:
        return {
            "answer": "YES, with a trade-off (operating-point / weighting change, not a new architecture)",
            "replace_baseline": False,
            "better_than_0_50_baseline": True,
            "detail": (
                f"Frozen-test FN fell by {fn_drop} while specificity/precision/F1/accuracy stayed "
                "in the preferred band. This is evidence of a better weighting trade-off on the "
                "internal test set only. Do not auto-replace the frozen baseline or production."
            ),
            "fn_reduction": fn_drop,
        }
    return {
        "answer": "NO",
        "replace_baseline": False,
        "better_than_0_50_baseline": False,
        "detail": (
            "The validation-selected moderate weight did not give a sufficiently favorable "
            "frozen-test trade-off versus the 83.42% baseline. Retain the baseline."
        ),
        "fn_reduction": fn_drop,
    }


def ensure_variant_trained(variant: dict, train_df, val_df, before: dict) -> dict:
    work = variant_dir(variant)
    work.mkdir(parents=True, exist_ok=True)
    cw = {
        "benign_weight": float(variant["benign"]),
        "melanoma_weight": float(variant["melanoma"]),
        "keras_class_weight": {"0": float(variant["benign"]), "1": float(variant["melanoma"])},
        "formula": "predefined moderate ratio; not inverse-frequency balanced weights",
        "oversampling": False,
        "undersampling": False,
    }
    _json_dump(work / "class_weights.json", cw)
    eff_path = work / "efficientnetb0.keras"
    mob_path = work / "mobilenetv2.keras"
    pred_path = work / "validation_predictions.csv"
    keras_cw = {0: float(variant["benign"]), 1: float(variant["melanoma"])}

    if eff_path.is_file() and mob_path.is_file() and pred_path.is_file():
        logger.info("Resume %s: using existing models and validation predictions", variant["id"])
        pred = pd.read_csv(pred_path)
        y_val = pred["label"].to_numpy().astype(int)
        p_ens = pred["ensemble_prob"].to_numpy()
        metrics = evaluate_predictions(y_val, p_ens, threshold=THRESHOLD)
        return {
            "variant": variant,
            "resumed": True,
            "metrics": metrics,
            "efficientnet_path": str(eff_path),
            "mobilenet_path": str(mob_path),
        }

    if not eff_path.is_file():
        eff = train_one(
            build_efficientnetb0,
            "efficientnetb0",
            train_df,
            val_df,
            eff_path,
            keras_cw,
            work,
        )
        _json_dump(
            work / "training_history_efficientnet.json",
            {"history": eff["history"], "best_epoch_val_pr_auc": eff["best_epoch_val_pr_auc"], "stage_epochs": eff["stage_epochs"], "training_seconds": eff["training_seconds"]},
        )
        write_status({"stage": f"{variant['id']}_efficientnet_done", "seconds": eff["training_seconds"]})
        assert_unchanged(before)
        p_eff = np.asarray(eff["y_val_prob"]).ravel()
        y_val = np.asarray(eff["y_val_true"]).astype(int)
    else:
        logger.info("Resume %s EfficientNet from disk", variant["id"])
        y_val, p_eff = predict_split(eff_path, val_df)

    if not mob_path.is_file():
        mob = train_one(
            build_mobilenetv2,
            "mobilenetv2",
            train_df,
            val_df,
            mob_path,
            keras_cw,
            work,
        )
        _json_dump(
            work / "training_history_mobilenet.json",
            {"history": mob["history"], "best_epoch_val_pr_auc": mob["best_epoch_val_pr_auc"], "stage_epochs": mob["stage_epochs"], "training_seconds": mob["training_seconds"]},
        )
        write_status({"stage": f"{variant['id']}_mobilenet_done", "seconds": mob["training_seconds"]})
        assert_unchanged(before)
        p_mob = np.asarray(mob["y_val_prob"]).ravel()
        y_val2 = np.asarray(mob["y_val_true"]).astype(int)
        if not np.array_equal(y_val, y_val2):
            raise RuntimeError(f"{variant['id']} validation label order mismatch")
    else:
        logger.info("Resume %s MobileNet from disk", variant["id"])
        y_val2, p_mob = predict_split(mob_path, val_df)
        if not np.array_equal(y_val, y_val2):
            raise RuntimeError(f"{variant['id']} validation label order mismatch")

    ens = combine_hybrid_probs(p_eff, p_mob, weights=ENSEMBLE_WEIGHTS, method="soft_voting")
    pred = pd.DataFrame(
        {
            "image_id": val_df["image_id"].astype(str).values,
            "label": y_val,
            "efficientnet_prob": p_eff,
            "mobilenet_prob": p_mob,
            "ensemble_prob": ens,
        }
    )
    _refuse_protected(pred_path)
    pred.to_csv(pred_path, index=False)
    metrics = evaluate_predictions(y_val, ens, threshold=THRESHOLD)
    _json_dump(work / "validation_metrics.json", metrics)
    return {
        "variant": variant,
        "resumed": False,
        "metrics": metrics,
        "efficientnet_path": str(eff_path),
        "mobilenet_path": str(mob_path),
    }


def select_configuration(val_rows: list[dict]) -> dict:
    feasible = [r for r in val_rows if not r["constraint_failures"]]
    if not feasible:
        return {
            "feasible": False,
            "n_feasible": 0,
            "selected_id": "retained_baseline",
            "selected": None,
            "rule": (
                "No moderate-weighting configuration met specificity>=0.82, precision>=0.78, "
                "F1>=0.80, and accuracy>=0.82 on validation at threshold 0.50. "
                "Constraints were not relaxed. Retain the frozen 83.42% baseline."
            ),
        }
    chosen = max(
        feasible,
        key=lambda r: (
            float(r["sensitivity"]),
            -int(r["fn"]),
            float(r["f1"]),
            float(r["specificity"]),
            float(r["precision"]),
            float(r["accuracy"]),
        ),
    )
    return {
        "feasible": True,
        "n_feasible": len(feasible),
        "feasible_ids": [r["id"] for r in feasible],
        "selected_id": chosen["id"],
        "selected": chosen,
        "rule": (
            "Among validation configurations at 0.50 meeting all constraints: "
            "highest sensitivity, then lowest FN, then highest F1, then highest "
            "specificity, then highest precision, then highest accuracy."
        ),
    }


def metric_row(name: str, mark: str, evaluation: str, metrics: dict, extra: dict | None = None) -> dict:
    row = {
        "name": name,
        "mark": mark,
        "evaluation": evaluation,
        "accuracy": metrics.get("accuracy"),
        "sensitivity": metrics.get("sensitivity"),
        "specificity": metrics.get("specificity"),
        "precision": metrics.get("precision"),
        "f1": metrics.get("f1"),
        "auc": metrics.get("auc"),
        "pr_auc": metrics.get("pr_auc"),
        "tp": metrics.get("tp"),
        "tn": metrics.get("tn"),
        "fp": metrics.get("fp"),
        "fn": metrics.get("fn"),
    }
    if extra:
        row.update(extra)
    return row


def main() -> None:
    EXP_ROOT.mkdir(parents=True, exist_ok=True)
    (EXP_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    before = snapshot_protected()
    _json_dump(EXP_ROOT / "production_snapshot_before.json", before)
    frames = verify_frozen()
    checklist = {
        "baseline_files_untouched_before_start": True,
        "final_dataset_untouched": True,
        "splits_frozen": True,
        "test_not_used_for_selection": True,
        "test_evaluated_only_after_validation_selection": True,
        "threshold": THRESHOLD,
        "ensemble": {"efficientnet": ENSEMBLE_WEIGHTS[0], "mobilenet": ENSEMBLE_WEIGHTS[1]},
        "oversampling": False,
        "undersampling": False,
        "ham10000": False,
        "ph2_tuning": False,
        "derm7pt_tuning": False,
        "dashboard_changes": False,
        "production_model_changes": False,
        "cache_jpgs": 16254,
    }
    _json_dump(EXP_ROOT / "safety_checklist.json", checklist)
    print(
        "\n".join(
            [
                "EXPERIMENT: final_melanoma_benign_moderate_weighting",
                "DATASET: dataset/processed/final_melanoma_benign/ (FROZEN)",
                "SPLITS: train 11379 val 2438 test 2437",
                "ENSEMBLE: 0.70 / 0.30 (FIXED, no search)",
                "THRESHOLD: 0.50 (FIXED, no search)",
                "CLASS WEIGHTS: 1.00:1.00, 1.00:1.05, 1.00:1.10, 1.00:1.15",
                "OVERSAMPLING: NO",
                "PRODUCTION MODIFIED: NO",
                "BASELINE MODIFIED: NO",
                "CLASS-WEIGHTED EXPERIMENT MODIFIED: NO",
                "DASHBOARD MODIFIED: NO",
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
            "variants": VARIANTS,
            "ensemble_weights": {"efficientnet": 0.70, "mobilenet": 0.30},
            "threshold": THRESHOLD,
            "ensemble_search": False,
            "threshold_search": False,
            "oversampling": False,
            "only_intended_change": "predefined moderate melanoma class weights",
            "stages": STAGES,
            "constraints": CONSTRAINTS,
            "test_used_for_tuning": False,
            "preprocess_cache": "read-only reuse of frozen baseline JPEG cache",
            "environment": env,
        },
    )
    enable_mixed_precision()
    tf.keras.utils.set_random_seed(SEED)
    np.random.seed(SEED)
    train_df = attach_cache(frames["train"])
    val_df = attach_cache(frames["val"])
    t_all = time.time()

    variant_results = []
    val_rows = []
    for variant in VARIANTS:
        result = ensure_variant_trained(variant, train_df, val_df, before)
        metrics = result["metrics"]
        failed = constraint_failures(metrics)
        row = {
            "id": variant["id"],
            "role": variant["role"],
            "benign_weight": variant["benign"],
            "melanoma_weight": variant["melanoma"],
            "threshold": THRESHOLD,
            "efficientnet_weight": ENSEMBLE_WEIGHTS[0],
            "mobilenet_weight": ENSEMBLE_WEIGHTS[1],
            "constraint_failures": failed,
            "meets_constraints": not failed,
            **metrics,
        }
        val_rows.append(row)
        variant_results.append(result)
        pd.DataFrame(val_rows).to_csv(EXP_ROOT / "validation_results.csv", index=False)
        write_status({"stage": f"{variant['id']}_validation_done", "row": row})
        assert_unchanged(before)

    selection = select_configuration(val_rows)
    _json_dump(EXP_ROOT / "selected_configuration.json", selection)
    write_status({"stage": "validation_selection_frozen", "selection": selection})
    logger.info("Validation selection frozen: %s", selection["selected_id"])

    previous_cw = load_class_weighted_test()
    comparison = [
        metric_row(FROZEN_BASELINE_TEST["name"], FROZEN_BASELINE_TEST["mark"], "frozen_test", FROZEN_BASELINE_TEST),
        metric_row(THRESHOLD_OPT_TEST["name"], THRESHOLD_OPT_TEST["mark"], "frozen_test", THRESHOLD_OPT_TEST),
        metric_row(previous_cw["name"], previous_cw["mark"], "frozen_test", previous_cw),
    ]
    for row in val_rows:
        comparison.append(
            metric_row(
                f"Moderate weighting {row['id']} (1.00:{row['melanoma_weight']:.2f})",
                row["role"],
                "validation",
                row,
                {"benign_weight": row["benign_weight"], "melanoma_weight": row["melanoma_weight"]},
            )
        )

    test_block = {
        "evaluated_once": False,
        "post_test_tuning": False,
        "threshold": THRESHOLD,
        "ensemble_weights": {"efficientnet": 0.70, "mobilenet": 0.30},
        "selected_id": selection["selected_id"],
        "metrics": None,
        "reason": None,
    }
    selected_test_row = metric_row(
        "Final selected model (retained frozen baseline)",
        "final selected candidate",
        "frozen_test",
        FROZEN_BASELINE_TEST,
    )

    if selection["feasible"] and selection["selected_id"] != "retained_baseline":
        chosen = next(v for v in VARIANTS if v["id"] == selection["selected_id"])
        work = variant_dir(chosen)
        test_df = attach_cache(frames["test"])
        y_test, p_eff = predict_split(work / "efficientnetb0.keras", test_df)
        y_test2, p_mob = predict_split(work / "mobilenetv2.keras", test_df)
        if not np.array_equal(y_test, y_test2):
            raise RuntimeError("Test label order mismatch.")
        ens = combine_hybrid_probs(p_eff, p_mob, weights=ENSEMBLE_WEIGHTS, method="soft_voting")
        test_m = evaluate_predictions(y_test, ens, threshold=THRESHOLD)
        y_pred = (ens >= THRESHOLD).astype(int)
        plot_confusion_matrix(y_test, y_pred, EXP_ROOT / "confusion_matrix.png", labels=["Benign", "Melanoma"])
        plot_roc_curve(y_test, ens, EXP_ROOT / "test_roc.png", f"{chosen['id']} ensemble")
        plot_pr_curve(y_test, ens, EXP_ROOT / "test_pr.png", f"{chosen['id']} ensemble")
        fpr, tpr, tthr = roc_curve(y_test, ens)
        prec, rec, pthr = precision_recall_curve(y_test, ens)
        _json_dump(EXP_ROOT / "test_roc_curve.json", {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": tthr.tolist()})
        _json_dump(EXP_ROOT / "test_pr_curve.json", {"precision": prec.tolist(), "recall": rec.tolist(), "thresholds": pthr.tolist()})
        pred_test = pd.DataFrame(
            {
                "image_id": test_df["image_id"].astype(str).values,
                "label": y_test,
                "efficientnet_prob": p_eff,
                "mobilenet_prob": p_mob,
                "ensemble_prob": ens,
                "predicted": y_pred,
            }
        )
        _refuse_protected(EXP_ROOT / "test_predictions.csv")
        pred_test.to_csv(EXP_ROOT / "test_predictions.csv", index=False)
        test_block.update({"evaluated_once": True, "metrics": test_m})
        selected_test_row = metric_row(
            f"Final selected model ({chosen['id']})",
            "final selected candidate",
            "frozen_test",
            test_m,
            {"benign_weight": chosen["benign"], "melanoma_weight": chosen["melanoma"]},
        )
    else:
        test_block["reason"] = "No constrained validation winner; test not used to pick a weight. Baseline retained."
        base_pred = pd.read_csv(BASELINE_ROOT / "test_predictions.csv")
        y_test = base_pred["label"].to_numpy().astype(int)
        ens = base_pred["ensemble_prob"].to_numpy()
        y_pred = (ens >= THRESHOLD).astype(int)
        plot_confusion_matrix(y_test, y_pred, EXP_ROOT / "confusion_matrix.png", labels=["Benign", "Melanoma"])
        plot_roc_curve(y_test, ens, EXP_ROOT / "test_roc.png", "Retained frozen baseline")
        plot_pr_curve(y_test, ens, EXP_ROOT / "test_pr.png", "Retained frozen baseline")

    comparison.append(selected_test_row)
    interpretation = interpret(selection, test_block.get("metrics"))
    payload = {
        "experiment": "final_melanoma_benign_moderate_weighting",
        "variants": VARIANTS,
        "ensemble_weights": {"efficientnet": 0.70, "mobilenet": 0.30},
        "threshold": THRESHOLD,
        "constraints": CONSTRAINTS,
        "validation_results": val_rows,
        "selection": selection,
        "test_metrics": test_block,
        "comparison_rows": comparison,
        "previous_class_weighted": previous_cw,
        "interpretation": interpretation,
        "timing": {"total_seconds": time.time() - t_all},
        "environment": env,
        "test_used_for_tuning": False,
        "post_test_tuning": False,
        "replace_baseline": False,
        "variant_paths": [
            {"id": r["variant"]["id"], "efficientnet": r["efficientnet_path"], "mobilenet": r["mobilenet_path"]}
            for r in variant_results
        ],
    }
    write_outputs(payload)
    assert_unchanged(before)
    _json_dump(EXP_ROOT / "production_snapshot_after.json", snapshot_protected())
    write_status({"stage": "complete", "selection": selection["selected_id"], "interpretation": interpretation})
    print("\nMODERATE WEIGHTING EXPERIMENT COMPLETE", flush=True)
    print(f"Selected: {selection['selected_id']}", flush=True)
    print(f"INTERPRETATION: {interpretation['answer']}", flush=True)
    print("BASELINE UNCHANGED: YES", flush=True)
    print("PRODUCTION UNCHANGED: YES", flush=True)
    print("TEST USED FOR TUNING: NO", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        try:
            _json_dump(EXP_ROOT / "logs" / "status.json", {"stage": "failed", "error": traceback.format_exc()})
        except Exception:
            pass
        raise
