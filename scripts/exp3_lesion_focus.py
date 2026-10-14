"""
Experiment 3: lesion-focused preprocessing vs current full-frame preprocessing.

Writes only under models/experiments/internal_optimization/exp3_lesion_focus/.
Does not overwrite Experiments 1–2, production models, frozen CSVs, or the app.
Does not load the frozen test set or PH2.
"""

from __future__ import annotations

import gc
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import average_precision_score, roc_auc_score
from tensorflow import keras

import config
from models.architecture import (
    FocalLoss,
    build_efficientnetb0,
    build_mobilenetv2,
    unfreeze_for_finetuning,
)
from models.ensemble import combine_hybrid_probs
from preprocessing.augmentation import get_tf_data_augmentation_fn
from preprocessing.image_preprocess import ImagePreprocessor
from training.callbacks import get_callbacks
from training.evaluate import evaluate_predictions, plot_training_curves, save_threshold_table, threshold_table
from training.train import _predict_dataset, enable_mixed_precision
from utils.logger import setup_logger

logger = setup_logger("exp3")

SPLIT_DIR = config.PROCESSED_DATA_DIR / "melanoma_benign"
CACHE_A = SPLIT_DIR / "preprocessed"
EXP_ROOT = config.BASE_DIR / "models" / "experiments" / "internal_optimization"
EXP1_DIR = EXP_ROOT / "exp1_three_stage"
EXP2_DIR = EXP_ROOT / "exp2_loss_comparison"
EXP2C_DIR = EXP2_DIR / "exp2c_bce_smooth"
EXP_DIR = EXP_ROOT / "exp3_lesion_focus"
CACHE_B = EXP_DIR / "variant_b_lesion_focus" / "cache"
PROTECTED = {
    config.EFFICIENTNET_PATH.resolve(),
    config.MOBILENET_PATH.resolve(),
    (config.MODELS_DIR / "efficientnetb0_melanoma_benign.keras").resolve(),
    (config.MODELS_DIR / "mobilenetv2_melanoma_benign.keras").resolve(),
    (config.MODELS_DIR / "melanoma_benign" / "efficientnetb0_melanoma_benign.keras").resolve(),
    (config.MODELS_DIR / "melanoma_benign" / "mobilenetv2_melanoma_benign.keras").resolve(),
}
FROZEN = {"train": (11611, 5966, 5645), "val": (2445, 1240, 1205)}
THRESHOLD_GRID = [round(x, 2) for x in np.arange(0.20, 0.81, 0.05)]
WEIGHT_GRID = [round(x, 2) for x in np.arange(0.00, 1.01, 0.05)]
STAGES = (
    {"name": "head", "lr": 1e-3, "epochs": 25, "unfreeze": 0, "batch": 16},
    {"name": "partial", "lr": 1e-4, "epochs": 12, "unfreeze": 20, "batch": 8},
    {"name": "final", "lr": 1e-5, "epochs": 12, "unfreeze": 20, "batch": 8},
)
LABEL_SMOOTHING = 0.05
LOCALIZE_MAX_SIDE = 512
CROP_MARGIN = 0.20
MIN_MASK_FRAC = 0.02
MIN_CROP_FRAC = 0.25
EXP1_REF = {
    "name": "experiment_1",
    "accuracy": 0.9100,
    "sensitivity": 0.9419,
    "specificity": 0.8772,
    "precision": 0.8875,
    "f1": 0.9139,
    "auc": 0.9686,
    "pr_auc": 0.9718,
    "efficientnet_weight": 0.50,
    "mobilenet_weight": 0.50,
    "threshold": 0.30,
}
EXP2A_REF = {
    "name": "experiment_2a_bce",
    "accuracy": 0.9043,
    "sensitivity": 0.8927,
    "specificity": 0.9162,
    "precision": 0.9164,
    "f1": 0.9044,
    "auc": 0.9687,
    "pr_auc": 0.9718,
    "efficientnet_weight": 0.70,
    "mobilenet_weight": 0.30,
    "threshold": 0.30,
}
EXP2C_REF = {
    "name": "experiment_2c_bce_smooth",
    "accuracy": 0.9137,
    "sensitivity": 0.9379,
    "specificity": 0.8888,
    "precision": 0.8967,
    "f1": 0.9168,
    "auc": 0.9718,
    "pr_auc": 0.9749,
    "efficientnet_weight": 0.75,
    "mobilenet_weight": 0.25,
    "threshold": 0.30,
}
BASELINE_REF = {
    "name": "baseline_two_stage",
    "accuracy": 0.8986,
    "sensitivity": 0.9153,
    "specificity": 0.8813,
    "precision": 0.8881,
    "f1": 0.9015,
    "auc": 0.9580,
    "pr_auc": 0.9604,
    "efficientnet_weight": 0.85,
    "mobilenet_weight": 0.15,
    "threshold": 0.50,
}


def _json_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_jsonable), encoding="utf-8")


def _jsonable(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _refuse_protected(path: Path) -> None:
    resolved = path.resolve()
    if resolved in PROTECTED:
        raise RuntimeError(f"Refusing to write protected path: {path}")
    for locked in (EXP1_DIR, EXP2_DIR, config.MODELS_DIR / "melanoma_benign"):
        if locked.resolve() in resolved.parents or resolved == locked.resolve():
            raise RuntimeError(f"Refusing to write under locked experiment/production path: {path}")


def _assert_frozen(name: str, df: pd.DataFrame) -> None:
    n, mel, ben = FROZEN[name]
    got = (len(df), int((df["label"] == 1).sum()), int((df["label"] == 0).sum()))
    if got != (n, mel, ben):
        raise ValueError(f"Frozen {name} CSV changed: expected {(n, mel, ben)}, got {got}")


def ram_snapshot() -> dict:
    try:
        import psutil

        mem = psutil.virtual_memory()
        return {
            "used_gb": round(mem.used / (1024**3), 2),
            "available_gb": round(mem.available / (1024**3), 2),
            "percent": mem.percent,
        }
    except Exception:
        return {}


def make_ds(df: pd.DataFrame, *, shuffle: bool, augment: bool, batch: int) -> tf.data.Dataset:
    if augment and not shuffle:
        augment = False
    paths = df["image_path"].astype(str).values
    labels = df["label"].values.astype(np.float32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        buf = min(len(df), max(2000, batch * 64))
        ds = ds.shuffle(buffer_size=buf, seed=config.SEED, reshuffle_each_iteration=True)

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


def compile_with_loss(model: keras.Model, learning_rate: float) -> keras.Model:
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


def select_clinical_threshold(y_true, y_prob) -> dict:
    rows = threshold_table(y_true, y_prob, thresholds=THRESHOLD_GRID)
    valid = [r for r in rows if r["sensitivity"] >= 0.85 and r["specificity"] >= 0.70]
    if valid:
        chosen = max(valid, key=lambda r: (r["f1"], r["sensitivity"], r["specificity"], -abs(r["threshold"] - 0.5)))
        rule = "sens>=0.85 and spec>=0.70, max F1"
    else:
        valid = [r for r in rows if r["sensitivity"] >= 0.80 and r["specificity"] >= 0.65]
        if valid:
            chosen = max(valid, key=lambda r: (r["f1"], r["sensitivity"], r["specificity"]))
            rule = "sens>=0.80 and spec>=0.65, max F1"
        else:
            chosen = max(rows, key=lambda r: (r["f1"], r["specificity"], r["sensitivity"]))
            rule = "max F1 with specificity tie-break (fallback; avoid all-melanoma)"
    return {"threshold": float(chosen["threshold"]), "rule": rule, "selected": chosen, "table": rows}


def search_ensemble(y_true, p_eff, p_mob) -> dict:
    rows = []
    blends = {}
    for w_eff in WEIGHT_GRID:
        w_mob = round(1.0 - w_eff, 2)
        ens = combine_hybrid_probs(p_eff, p_mob, weights=(w_eff, w_mob), method="soft_voting")
        at_half = evaluate_predictions(y_true, ens, threshold=0.5)
        row = {
            "efficientnet_weight": w_eff,
            "mobilenet_weight": w_mob,
            "roc_auc": float(roc_auc_score(y_true, ens)),
            "pr_auc": float(average_precision_score(y_true, ens)),
            "accuracy_at_0.5": at_half["accuracy"],
            "sensitivity_at_0.5": at_half["sensitivity"],
            "specificity_at_0.5": at_half["specificity"],
            "precision_at_0.5": at_half["precision"],
            "f1_at_0.5": at_half["f1"],
        }
        rows.append(row)
        blends[(w_eff, w_mob)] = ens
    best_w = max(rows, key=lambda r: (r["roc_auc"], r["pr_auc"]))
    ens = blends[(best_w["efficientnet_weight"], best_w["mobilenet_weight"])]
    picked = select_clinical_threshold(y_true, ens)
    operating = evaluate_predictions(y_true, ens, threshold=picked["threshold"])
    return {
        "weight_candidates": rows,
        "selected_weights": {
            "efficientnet_weight": best_w["efficientnet_weight"],
            "mobilenet_weight": best_w["mobilenet_weight"],
            "selection_rule": "max validation ROC-AUC, then PR-AUC",
            "roc_auc": best_w["roc_auc"],
            "pr_auc": best_w["pr_auc"],
        },
        "val_ensemble_prob": ens,
        "threshold_search": picked,
        "operating": operating,
    }


def _downscale_max_side(rgb: np.ndarray, max_side: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return rgb
    return cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def lesion_focused_preprocess(rgb: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Conservative lesion-focused view.

    Reuses existing DullRazor hair removal, light denoise, and the project's
    classical Otsu lesion mask. Does not apply CLAHE. Does not use U-Net
    weights (none are present). Falls back to the full frame if localization
    is unreliable.
    """
    prep = ImagePreprocessor(target_size=(224, 224), apply_clahe=False, apply_hair_removal=True)
    img = rgb.copy()
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = _downscale_max_side(img, LOCALIZE_MAX_SIDE)
    img = prep.remove_hair(img)
    img = prep.denoise(img)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    _, mask = cv2.threshold(lab[:, :, 0], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n_labels, _labels, stats, cents = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    used_full = True
    bbox = None
    crop = img
    h, w = img.shape[:2]
    chosen = None
    if n_labels > 1:
        cx, cy = w / 2.0, h / 2.0
        candidates = []
        for i in range(1, n_labels):
            x, y, bw, bh, area = (int(stats[i, c]) for c in range(5))
            touches = int(x <= 1) + int(y <= 1) + int(x + bw >= w - 2) + int(y + bh >= h - 2)
            frac = area / float(h * w)
            if frac < MIN_MASK_FRAC:
                continue
            if touches >= 3 or (touches >= 2 and frac >= 0.40):
                continue
            dist = float(np.hypot(cents[i][0] - cx, cents[i][1] - cy))
            candidates.append((touches, dist, -area, i))
        if candidates:
            candidates.sort()
            chosen = candidates[0][3]
    if chosen is not None:
        x, y, bw, bh = (int(stats[chosen, c]) for c in range(4))
        pad_x = max(int(bw * CROP_MARGIN), int(0.05 * w))
        pad_y = max(int(bh * CROP_MARGIN), int(0.05 * h))
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(w, x + bw + pad_x)
        y1 = min(h, y + bh + pad_y)
        min_w = int(MIN_CROP_FRAC * w)
        min_h = int(MIN_CROP_FRAC * h)
        if (x1 - x0) < min_w:
            extra = min_w - (x1 - x0)
            x0 = max(0, x0 - extra // 2)
            x1 = min(w, x1 + extra - extra // 2)
        if (y1 - y0) < min_h:
            extra = min_h - (y1 - y0)
            y0 = max(0, y0 - extra // 2)
            y1 = min(h, y1 + extra - extra // 2)
        crop = img[y0:y1, x0:x1]
        bbox = [x0, y0, x1, y1]
        used_full = False
    out = prep.normalize(prep.resize(crop, (224, 224)), to_unit=True)
    info = {
        "fallback_full_frame": used_full,
        "bbox": bbox,
        "crop_h": int(crop.shape[0]),
        "crop_w": int(crop.shape[1]),
        "src_h": int(img.shape[0]),
        "src_w": int(img.shape[1]),
    }
    return out, info


def cache_variant_b(df: pd.DataFrame, split: str) -> pd.DataFrame:
    CACHE_B.mkdir(parents=True, exist_ok=True)
    stats_path = CACHE_B / f"{split}_crop_stats.csv"
    rows = []
    paths = []
    t0 = time.time()
    for i, rec in enumerate(df.itertuples(index=False), start=1):
        dest = CACHE_B / f"{rec.image_id}.jpg"
        if dest.is_file():
            paths.append(str(dest))
            continue
        src = Path(str(rec.image_path))
        bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read original image {src}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        arr, info = lesion_focused_preprocess(rgb)
        jpeg = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(str(dest), cv2.cvtColor(jpeg, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        paths.append(str(dest))
        info["image_id"] = rec.image_id
        info["split"] = split
        rows.append(info)
        if i % 250 == 0 or i == len(df):
            logger.info("Variant B cache %s %d/%d ram=%s", split, i, len(df), ram_snapshot())
            _json_dump(EXP_DIR / "status.json", {"stage": f"cache_b_{split}", "done": i, "total": len(df), "ram": ram_snapshot()})
    out = df.copy()
    out["image_path"] = paths
    if rows:
        extra = pd.DataFrame(rows)
        if stats_path.exists():
            extra = pd.concat([pd.read_csv(stats_path), extra], ignore_index=True)
        extra.to_csv(stats_path, index=False)
    logger.info("Variant B %s cache ready in %.1fs", split, time.time() - t0)
    return out


def use_variant_a_cache(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    missing = []
    paths = []
    for image_id in out["image_id"].astype(str):
        cached = CACHE_A / f"{image_id}.jpg"
        if cached.is_file():
            paths.append(str(cached))
        else:
            missing.append(image_id)
    if missing:
        raise FileNotFoundError(f"{len(missing)} Variant A cache files missing, e.g. {missing[:5]}")
    out["image_path"] = paths
    return out


def train_one(builder, name: str, run_dir: Path, train_df, val_df, save_path: Path) -> dict:
    ckpt_dir = run_dir / "checkpoints"
    curve_dir = run_dir / "training_curves"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    curve_dir.mkdir(parents=True, exist_ok=True)
    logger.info("==== Exp3 training %s in %s ram=%s ====", name, run_dir.name, ram_snapshot())
    t0 = time.time()
    oom = False
    hist: dict[str, list] = {}
    model = builder(trainable_base=False)
    stage_epochs = []
    try:
        for stage in STAGES:
            if stage["unfreeze"]:
                model = unfreeze_for_finetuning(model, n_unfreeze_layers=stage["unfreeze"])
            model = compile_with_loss(model, stage["lr"])
            train_ds = make_ds(train_df, shuffle=True, augment=True, batch=stage["batch"])
            val_ds = make_ds(val_df, shuffle=False, augment=False, batch=stage["batch"])
            logger.info("Stage %s lr=%s batch=%s ram=%s", stage["name"], stage["lr"], stage["batch"], ram_snapshot())
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=stage["epochs"],
                callbacks=get_callbacks(f"exp3_{run_dir.name}_{name}_{stage['name']}", checkpoint_dir=ckpt_dir),
                verbose=1,
            )
            for key, values in history.history.items():
                hist.setdefault(key, []).extend(list(values))
            stage_epochs.append({"stage": stage["name"], "epochs_run": len(history.history.get("loss", []))})
            ckpt = ckpt_dir / f"exp3_{run_dir.name}_{name}_{stage['name']}_best.keras"
            if ckpt.exists():
                model = tf.keras.models.load_model(str(ckpt), custom_objects={"FocalLoss": FocalLoss}, compile=False)
            gc.collect()
    except tf.errors.ResourceExhaustedError:
        oom = True
        logger.exception("OOM during %s %s", run_dir.name, name)
        raise
    except MemoryError:
        oom = True
        raise

    _refuse_protected(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    plot_training_curves(hist, curve_dir, name)
    _json_dump(run_dir / f"{name}_history.json", hist)
    val_ds = make_ds(val_df, shuffle=False, augment=False, batch=8)
    y_true, y_prob = _predict_dataset(model, val_ds)
    del model
    tf.keras.backend.clear_session()
    gc.collect()
    return {
        "path": str(save_path),
        "history": hist,
        "stage_epochs": stage_epochs,
        "training_seconds": time.time() - t0,
        "oom": oom,
        "ram": ram_snapshot(),
        "y_val_true": y_true,
        "y_val_prob": y_prob,
    }


def infer_one(model_path: Path, dest_path: Path, val_df) -> dict:
    _refuse_protected(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.resolve() != model_path.resolve():
        shutil.copy2(model_path, dest_path)
    model = tf.keras.models.load_model(str(dest_path), custom_objects={"FocalLoss": FocalLoss}, compile=False)
    val_ds = make_ds(val_df, shuffle=False, augment=False, batch=8)
    y_true, y_prob = _predict_dataset(model, val_ds)
    del model
    tf.keras.backend.clear_session()
    gc.collect()
    return {
        "path": str(dest_path),
        "history": {},
        "stage_epochs": [],
        "training_seconds": 0.0,
        "oom": False,
        "ram": ram_snapshot(),
        "y_val_true": y_true,
        "y_val_prob": y_prob,
        "reused_from": str(model_path),
    }


def score_distribution(probs: np.ndarray, y: np.ndarray) -> dict:
    def _one(values):
        values = np.asarray(values).ravel()
        return {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "n": int(values.size),
        }

    return {
        "all": _one(probs),
        "melanoma": _one(probs[y == 1]),
        "benign": _one(probs[y == 0]),
    }


def evaluate_variant(key: str, label: str, train_df, val_df, train: bool, sources: dict) -> dict:
    run_dir = EXP_DIR / key
    run_dir.mkdir(parents=True, exist_ok=True)
    marker = run_dir / "run_complete.json"
    if marker.exists() and (run_dir / "validation_predictions.csv").exists():
        logger.info("Skipping %s; already complete.", key)
        return json.loads(marker.read_text(encoding="utf-8"))

    t0 = time.time()
    if train:
        eff = train_one(build_efficientnetb0, "efficientnetb0", run_dir, train_df, val_df, run_dir / "efficientnetb0.keras")
        _json_dump(EXP_DIR / "status.json", {"stage": f"{key}_efficientnet_done", "seconds": eff["training_seconds"], "ram": ram_snapshot()})
        mob = train_one(build_mobilenetv2, "mobilenetv2", run_dir, train_df, val_df, run_dir / "mobilenetv2.keras")
    else:
        eff = infer_one(sources["efficientnetb0"], run_dir / "efficientnetb0.keras", val_df)
        mob = infer_one(sources["mobilenetv2"], run_dir / "mobilenetv2.keras", val_df)

    y_val = np.asarray(eff["y_val_true"]).astype(int)
    search = search_ensemble(y_val, eff["y_val_prob"], mob["y_val_prob"])
    threshold = float(search["threshold_search"]["threshold"])
    ens_prob = np.asarray(search["val_ensemble_prob"]).ravel()
    pred = (ens_prob >= threshold).astype(int)
    val_eff = evaluate_predictions(y_val, eff["y_val_prob"], threshold=threshold)
    val_mob = evaluate_predictions(y_val, mob["y_val_prob"], threshold=threshold)
    val_ens = search["operating"]
    dist = {
        "efficientnet": score_distribution(np.asarray(eff["y_val_prob"]), y_val),
        "mobilenet": score_distribution(np.asarray(mob["y_val_prob"]), y_val),
        "ensemble": score_distribution(ens_prob, y_val),
    }
    pred_df = pd.DataFrame(
        {
            "image_id": val_df["image_id"].astype(str).values,
            "label": y_val,
            "efficientnet_prob": np.asarray(eff["y_val_prob"]).ravel(),
            "mobilenet_prob": np.asarray(mob["y_val_prob"]).ravel(),
            "ensemble_prob": ens_prob,
            "ensemble_pred": pred,
        }
    )
    pred_df.to_csv(run_dir / "validation_predictions.csv", index=False)
    pd.DataFrame(search["weight_candidates"]).to_csv(run_dir / "ensemble_weight_search.csv", index=False)
    save_threshold_table(search["threshold_search"]["table"], run_dir / "val_threshold_table.csv")
    payload = {
        "key": key,
        "label": label,
        "evaluated_on": "validation_only",
        "test_used": False,
        "ph2_used": False,
        "oom": bool(eff.get("oom") or mob.get("oom")),
        "wall_seconds": time.time() - t0,
        "ram": ram_snapshot(),
        "efficientnet": {
            "path": eff["path"],
            "training_seconds": eff["training_seconds"],
            "stage_epochs": eff["stage_epochs"],
            "reused_from": eff.get("reused_from"),
            "validation": val_eff,
        },
        "mobilenet": {
            "path": mob["path"],
            "training_seconds": mob["training_seconds"],
            "stage_epochs": mob["stage_epochs"],
            "reused_from": mob.get("reused_from"),
            "validation": val_mob,
        },
        "ensemble": {
            **search["selected_weights"],
            "threshold": threshold,
            "threshold_rule": search["threshold_search"]["rule"],
            "validation": val_ens,
            "predicted_melanoma": int((pred == 1).sum()),
            "predicted_benign": int((pred == 0).sum()),
            "score_distribution": dist["ensemble"],
        },
        "score_distributions": dist,
    }
    _json_dump(run_dir / "validation_metrics.json", payload)
    _json_dump(marker, payload)
    logger.info(
        "%s ensemble acc=%.4f sens=%.4f spec=%.4f prec=%.4f f1=%.4f auc=%.4f pr=%.4f thr=%.2f w=%.2f/%.2f",
        key,
        val_ens["accuracy"],
        val_ens["sensitivity"],
        val_ens["specificity"],
        val_ens["precision"],
        val_ens["f1"],
        val_ens["auc"],
        val_ens["pr_auc"],
        threshold,
        search["selected_weights"]["efficientnet_weight"],
        search["selected_weights"]["mobilenet_weight"],
    )
    return payload


def write_reports(results: list[dict]) -> None:
    a = next(r for r in results if r["key"] == "variant_a_current")
    b = next(r for r in results if r["key"] == "variant_b_lesion_focus")
    ev_a = a["ensemble"]["validation"]
    ev_b = b["ensemble"]["validation"]
    diff_ab = {k: ev_b[k] - ev_a[k] for k in ("accuracy", "sensitivity", "specificity", "precision", "f1")}
    diff_ab["roc_auc"] = ev_b["auc"] - ev_a["auc"]
    diff_ab["pr_auc"] = ev_b["pr_auc"] - ev_a["pr_auc"]
    diff_e2 = {
        "accuracy": ev_b["accuracy"] - EXP2C_REF["accuracy"],
        "sensitivity": ev_b["sensitivity"] - EXP2C_REF["sensitivity"],
        "specificity": ev_b["specificity"] - EXP2C_REF["specificity"],
        "precision": ev_b["precision"] - EXP2C_REF["precision"],
        "f1": ev_b["f1"] - EXP2C_REF["f1"],
        "roc_auc": ev_b["auc"] - EXP2C_REF["auc"],
        "pr_auc": ev_b["pr_auc"] - EXP2C_REF["pr_auc"],
    }
    b_better = (ev_b["auc"], ev_b["pr_auc"]) > (ev_a["auc"], ev_a["pr_auc"]) and ev_b["specificity"] >= 0.70
    best = b if b_better else a
    ev = best["ensemble"]["validation"]

    rows = []
    for ref in (BASELINE_REF, EXP1_REF, EXP2A_REF, EXP2C_REF):
        rows.append(
            {
                "config": ref["name"],
                "accuracy": ref["accuracy"],
                "sensitivity": ref["sensitivity"],
                "specificity": ref["specificity"],
                "precision": ref["precision"],
                "f1": ref["f1"],
                "roc_auc": ref["auc"],
                "pr_auc": ref["pr_auc"],
                "efficientnet_weight": ref["efficientnet_weight"],
                "mobilenet_weight": ref["mobilenet_weight"],
                "threshold": ref["threshold"],
            }
        )
    for item, name in ((a, "experiment_3_variant_a"), (b, "experiment_3_variant_b")):
        e = item["ensemble"]
        v = e["validation"]
        rows.append(
            {
                "config": name,
                "accuracy": v["accuracy"],
                "sensitivity": v["sensitivity"],
                "specificity": v["specificity"],
                "precision": v["precision"],
                "f1": v["f1"],
                "roc_auc": v["auc"],
                "pr_auc": v["pr_auc"],
                "efficientnet_weight": e["efficientnet_weight"],
                "mobilenet_weight": e["mobilenet_weight"],
                "threshold": e["threshold"],
            }
        )
    pd.DataFrame(rows).to_csv(EXP_DIR / "preprocessing_comparison.csv", index=False)
    pd.DataFrame(rows).to_csv(EXP_DIR / "ensemble_comparison.csv", index=False)
    model_rows = []
    for item in results:
        for model_name in ("efficientnet", "mobilenet", "ensemble"):
            src = item[model_name]["validation"] if model_name != "ensemble" else item["ensemble"]["validation"]
            model_rows.append(
                {
                    "variant": item["key"],
                    "model": model_name,
                    "accuracy": src["accuracy"],
                    "sensitivity": src["sensitivity"],
                    "specificity": src["specificity"],
                    "precision": src["precision"],
                    "f1": src["f1"],
                    "roc_auc": src["auc"],
                    "pr_auc": src["pr_auc"],
                    "threshold": src["threshold"],
                }
            )
    pd.DataFrame(model_rows).to_csv(EXP_DIR / "model_comparison.csv", index=False)
    thr_parts = []
    for item in results:
        table = pd.read_csv(EXP_DIR / item["key"] / "val_threshold_table.csv")
        table.insert(0, "variant", item["key"])
        thr_parts.append(table)
    pd.concat(thr_parts, ignore_index=True).to_csv(EXP_DIR / "threshold_comparison.csv", index=False)

    best_eff = max(results, key=lambda r: (r["efficientnet"]["validation"]["auc"], r["efficientnet"]["validation"]["pr_auc"]))
    best_mob = max(results, key=lambda r: (r["mobilenet"]["validation"]["auc"], r["mobilenet"]["validation"]["pr_auc"]))
    cm = {
        "tp": ev["tp"],
        "fp": ev["fp"],
        "tn": ev["tn"],
        "fn": ev["fn"],
    }
    report = {
        "experiment": "exp3_lesion_focus",
        "evaluated_on": "validation_only",
        "test_used": False,
        "ph2_used": False,
        "variant_b_improved_over_a": b_better,
        "best_efficientnet": {"variant": best_eff["key"], "validation": best_eff["efficientnet"]["validation"]},
        "best_mobilenet": {"variant": best_mob["key"], "validation": best_mob["mobilenet"]["validation"]},
        "best_ensemble": {
            "variant": best["key"],
            "efficientnet_weight": best["ensemble"]["efficientnet_weight"],
            "mobilenet_weight": best["ensemble"]["mobilenet_weight"],
            "threshold": best["ensemble"]["threshold"],
            "threshold_rule": best["ensemble"]["threshold_rule"],
            "validation": ev,
            "confusion_matrix": cm,
            "predicted_melanoma": best["ensemble"]["predicted_melanoma"],
            "predicted_benign": best["ensemble"]["predicted_benign"],
            "score_distribution": best["ensemble"]["score_distribution"],
        },
        "difference_b_minus_a": diff_ab,
        "difference_b_minus_exp2c": diff_e2,
        "runs": results,
        "oom": any(item.get("oom") for item in results),
        "production_or_live_app_modified": False,
    }
    _json_dump(EXP_DIR / "EXP3_VALIDATION_REPORT.json", report)
    dist = best["ensemble"]["score_distribution"]
    lines = [
        "EXPERIMENT 3 — LESION-FOCUSED PREPROCESSING (validation only)",
        "",
        f"1. Variant B improved over Variant A: {'YES' if b_better else 'NO'}",
        f"2. Best EfficientNet: {best_eff['key']} AUC={best_eff['efficientnet']['validation']['auc']:.4f}",
        f"3. Best MobileNet: {best_mob['key']} AUC={best_mob['mobilenet']['validation']['auc']:.4f}",
        f"4. Best ensemble: {best['key']}",
        f"5. Selected ensemble weights: {best['ensemble']['efficientnet_weight']:.2f}/{best['ensemble']['mobilenet_weight']:.2f}",
        f"6. Selected threshold: {best['ensemble']['threshold']:.2f} ({best['ensemble']['threshold_rule']})",
        f"7. Accuracy: {ev['accuracy']:.4f}",
        f"8. Sensitivity: {ev['sensitivity']:.4f}",
        f"9. Specificity: {ev['specificity']:.4f}",
        f"10. Precision: {ev['precision']:.4f}",
        f"11. F1: {ev['f1']:.4f}",
        f"12. ROC-AUC: {ev['auc']:.4f}",
        f"13. PR-AUC: {ev['pr_auc']:.4f}",
        "14. Difference from Experiment 2-C:",
        f"    acc={diff_e2['accuracy']:+.4f} sens={diff_e2['sensitivity']:+.4f} spec={diff_e2['specificity']:+.4f} prec={diff_e2['precision']:+.4f} f1={diff_e2['f1']:+.4f} auc={diff_e2['roc_auc']:+.4f} pr={diff_e2['pr_auc']:+.4f}",
        f"15. Confusion matrix TP={cm['tp']} FP={cm['fp']} TN={cm['tn']} FN={cm['fn']}",
        f"16. Ensemble score dist melanoma mean/median={dist['melanoma']['mean']:.4f}/{dist['melanoma']['median']:.4f} benign={dist['benign']['mean']:.4f}/{dist['benign']['median']:.4f}",
        f"17. Predicted melanoma={best['ensemble']['predicted_melanoma']} benign={best['ensemble']['predicted_benign']}",
        f"18. Training seconds: A_eff={a['efficientnet']['training_seconds']:.1f} A_mob={a['mobilenet']['training_seconds']:.1f} B_eff={b['efficientnet']['training_seconds']:.1f} B_mob={b['mobilenet']['training_seconds']:.1f}",
        f"19. RAM snapshot: {best.get('ram')}",
        f"20. OOM: {'YES' if report['oom'] else 'none'}",
        "21. Files created under models/experiments/internal_optimization/exp3_lesion_focus/",
        "22. Production/live-app files modified: NO",
        "",
        "Variant A reused Experiment 2-C models (read-only control). Variant B was trained.",
        "Test set was not loaded. PH2 was not used. Experiment 4 was not started.",
    ]
    (EXP_DIR / "EXP3_VALIDATION_REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    if config.ACTIVE_DATASET != "combined":
        raise RuntimeError("ACTIVE_DATASET must stay 'combined'.")
    enable_mixed_precision()
    tf.keras.utils.set_random_seed(config.SEED)
    config.USE_FOCAL_LOSS = False
    for folder in (EXP_DIR, EXP_DIR / "config", EXP_DIR / "variant_a_current", EXP_DIR / "variant_b_lesion_focus", CACHE_B):
        folder.mkdir(parents=True, exist_ok=True)
    if not (EXP2C_DIR / "efficientnetb0_bce_smooth.keras").is_file():
        raise FileNotFoundError("Experiment 2-C models are required as Variant A control.")

    train = pd.read_csv(SPLIT_DIR / "train.csv")
    val = pd.read_csv(SPLIT_DIR / "val.csv")
    _assert_frozen("train", train)
    _assert_frozen("val", val)
    logger.info("Loaded frozen train/val only. Test CSV was not read. PH2 was not used. ram=%s", ram_snapshot())

    _json_dump(
        EXP_DIR / "config" / "experiment_config.json",
        {
            "experiment": "exp3_lesion_focus",
            "seed": config.SEED,
            "loss": "BCE + label_smoothing=0.05",
            "input_size": 224,
            "finetune": STAGES,
            "variant_a": {
                "preprocessing": "process_for_model: hair + denoise + 224 + /255; existing cache",
                "models": "reused Experiment 2-C (not retrained)",
            },
            "variant_b": {
                "preprocessing": (
                    "downscale max-side 512; existing hair removal; existing denoise; "
                    "LAB-L Otsu from existing lesion_focused_view; keep interior component "
                    "nearest center; reject vignette-like border blobs; crop with 20% margin; "
                    "expand if crop < 25% of frame; fallback to full frame if localization fails; "
                    "resize 224; /255; no CLAHE; no U-Net weights"
                ),
                "margin": CROP_MARGIN,
                "min_crop_frac": MIN_CROP_FRAC,
                "localize_max_side": LOCALIZE_MAX_SIDE,
            },
            "test_loaded": False,
            "ph2_used": False,
            "overwrites_experiment_1_or_2": False,
        },
    )

    train_a = use_variant_a_cache(train)
    val_a = use_variant_a_cache(val)
    results = []
    try:
        _json_dump(EXP_DIR / "status.json", {"stage": "variant_a", "ram": ram_snapshot()})
        results.append(
            evaluate_variant(
                "variant_a_current",
                "EXP3-A current preprocessing control (Experiment 2-C models)",
                train_a,
                val_a,
                train=False,
                sources={
                    "efficientnetb0": EXP2C_DIR / "efficientnetb0_bce_smooth.keras",
                    "mobilenetv2": EXP2C_DIR / "mobilenetv2_bce_smooth.keras",
                },
            )
        )
        _json_dump(EXP_DIR / "status.json", {"stage": "cache_b", "ram": ram_snapshot()})
        train_b = cache_variant_b(train, "train")
        val_b = cache_variant_b(val, "val")
        _json_dump(EXP_DIR / "status.json", {"stage": "variant_b_train", "ram": ram_snapshot()})
        results.append(
            evaluate_variant(
                "variant_b_lesion_focus",
                "EXP3-B conservative lesion-focused crop",
                train_b,
                val_b,
                train=True,
                sources={},
            )
        )
    except Exception:
        text = traceback.format_exc()
        oom = "ResourceExhausted" in text or "Unable to allocate" in text or "OOM" in text
        _json_dump(EXP_DIR / "status.json", {"stage": "failed", "oom": oom, "error": text, "ram": ram_snapshot()})
        if len(results) == 2:
            write_reports(results)
        raise

    write_reports(results)
    _json_dump(EXP_DIR / "status.json", {"stage": "complete", "oom": False, "ram": ram_snapshot()})
    logger.info("Experiment 3 finished. Stopping as requested. Experiment 4 was not started.")


if __name__ == "__main__":
    main()
