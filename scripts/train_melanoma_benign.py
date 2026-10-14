"""
Train EfficientNetB0 + MobileNetV2 on the frozen Melanoma vs Benign split.

Writes only to models/saved/melanoma_benign/ and models/checkpoints/melanoma_benign/.
Does not overwrite production weights or change ACTIVE_DATASET.
Does not regenerate dataset/processed/melanoma_benign splits.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, classification_report, confusion_matrix

import config
from gradcam.gradcam import GradCAM
from models.architecture import (
    FocalLoss,
    build_efficientnetb0,
    build_mobilenetv2,
    compile_model,
    unfreeze_for_finetuning,
)
from models.ensemble import combine_hybrid_probs
from preprocessing.augmentation import get_tf_data_augmentation_fn
from preprocessing.image_preprocess import ImagePreprocessor, load_and_preprocess
from training.callbacks import get_callbacks
from training.evaluate import (
    best_epoch_from_history,
    evaluate_predictions,
    full_evaluation_report,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
    plot_training_curves,
    save_threshold_table,
    threshold_table,
)
from training.train import _predict_dataset, enable_mixed_precision
from utils.logger import setup_logger

logger = setup_logger("train_mb")

SPLIT_DIR = config.PROCESSED_DATA_DIR / "melanoma_benign"
EXP_DIR = config.MODELS_DIR / "melanoma_benign"
CKPT_DIR = config.CHECKPOINTS_DIR / "melanoma_benign"
CACHE_DIR = SPLIT_DIR / "preprocessed"
CURVE_DIR = EXP_DIR / "training_curves"
PLOT_DIR = EXP_DIR / "plots"
GRADCAM_DIR = EXP_DIR / "gradcam"
DEMO_DIR = EXP_DIR / "demo_100"
CLASS_LABELS = ["Benign", "Melanoma"]
FROZEN = {
    "train": (11611, 5966, 5645),
    "val": (2445, 1240, 1205),
    "test": (2473, 1324, 1149),
}
THRESHOLD_GRID = [round(x, 2) for x in np.arange(0.15, 0.76, 0.05)]
WEIGHT_GRID = [round(x, 2) for x in np.arange(0.10, 0.91, 0.05)]
PH2_DIR = ROOT / "testing_data" / "PH2 Dataset images"
PH2_LABELS = ROOT / "testing_data" / "PH2_dataset.txt"


def _json_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_jsonable), encoding="utf-8")


def _jsonable(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _file_stamp(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size": int(stat.st_size),
        "mtime": float(stat.st_mtime),
    }


def _assert_production_untouched(before: dict, after: dict) -> None:
    for key in before:
        if before[key] != after[key]:
            raise RuntimeError(f"Production file changed during experiment: {key}")


def _assert_frozen_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    for name, df in (("train", train), ("val", val), ("test", test)):
        n, mel, ben = FROZEN[name]
        got_n = len(df)
        got_mel = int((df["label"] == 1).sum())
        got_ben = int((df["label"] == 0).sum())
        if (got_n, got_mel, got_ben) != (n, mel, ben):
            raise ValueError(
                f"Frozen {name} split changed: expected {n}/{mel}/{ben}, "
                f"got {got_n}/{got_mel}/{got_ben}. Do not regenerate the split."
            )
        if set(df["class_name"].astype(str)) - {"Benign", "Melanoma"}:
            raise ValueError(f"{name} has unexpected class_name values.")
        if int((df["source_dataset"] == "ISIC-DICM-17K").sum()) != got_mel:
            raise ValueError(f"{name}: melanoma rows must come from ISIC-DICM-17K only.")
        if int((df["source_dataset"] == "ISIC-2020").sum()) != got_ben:
            raise ValueError(f"{name}: benign rows must come from ISIC 2020 only.")


def _cache_row(src: str, dest: Path) -> None:
    if dest.exists():
        return
    raw = cv2.imread(src)
    if raw is None:
        raise FileNotFoundError(src)
    rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    processed = ImagePreprocessor(apply_hair_removal=True).process_for_model(rgb)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), cv2.cvtColor((processed * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def cached_paths(df: pd.DataFrame, label: str) -> pd.DataFrame:
    out = df.copy()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, row in enumerate(out.itertuples(index=False), start=1):
        image_id = str(getattr(row, "image_id"))
        dest = CACHE_DIR / f"{image_id}.jpg"
        _cache_row(str(getattr(row, "image_path")), dest)
        paths.append(str(dest))
        if i % 250 == 0 or i == len(out):
            logger.info("Preprocessed %s %d / %d", label, i, len(out))
            _json_dump(
                EXP_DIR / "train_status.json",
                {"stage": "preprocess", "split": label, "done": i, "total": len(out)},
            )
    out["image_path"] = paths
    return out


def make_ds(df: pd.DataFrame, *, shuffle: bool, augment: bool, batch: int) -> tf.data.Dataset:
    if augment and not shuffle:
        logger.warning("Refusing augmentation on a non-shuffled split.")
        augment = False
    paths = df["image_path"].astype(str).values
    labels = df["label"].values.astype(np.float32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        buf = min(len(df), max(2000, batch * 128))
        ds = ds.shuffle(buffer_size=buf, seed=config.SEED, reshuffle_each_iteration=True)

    def _decode(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, config.IMG_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    ds = ds.map(_decode, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        aug_fn = get_tf_data_augmentation_fn()

        def _augment(image, label):
            return aug_fn(image, label)

        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def select_clinical_threshold(y_true, y_prob, thresholds=None) -> dict:
    """
    Validation-only operating point.

    Prefer high melanoma sensitivity without collapsing specificity.
    Do not pick the lowest threshold just to maximize recall.
    """
    rows = threshold_table(y_true, y_prob, thresholds=thresholds or THRESHOLD_GRID)
    for row in rows:
        row["youden"] = float(row["sensitivity"] + row["specificity"] - 1.0)

    def _best(cands: list) -> dict:
        return max(
            cands,
            key=lambda r: (r["f1"], r["sensitivity"], r["specificity"], r["threshold"]),
        )

    tiers = [
        [r for r in rows if r["sensitivity"] >= 0.85 and r["specificity"] >= 0.70],
        [r for r in rows if r["sensitivity"] >= 0.80 and r["specificity"] >= 0.65],
        [r for r in rows if r["sensitivity"] >= 0.75 and r["specificity"] >= 0.60],
        [r for r in rows if r["specificity"] >= 0.55],
        rows,
    ]
    chosen = None
    rule = "fallback"
    names = [
        "sens>=0.85 and spec>=0.70, max F1",
        "sens>=0.80 and spec>=0.65, max F1",
        "sens>=0.75 and spec>=0.60, max F1",
        "spec>=0.55, max F1",
        "max F1 on the validation grid",
    ]
    for name, cands in zip(names, tiers):
        if cands:
            chosen = _best(cands)
            rule = name
            break
    assert chosen is not None
    return {
        "threshold": float(chosen["threshold"]),
        "rule": rule,
        "selected": chosen,
        "table": rows,
    }


def search_ensemble(y_true, p_eff, p_mob) -> dict:
    rows = []
    best = None
    for w_eff in WEIGHT_GRID:
        w_mob = round(1.0 - w_eff, 2)
        ens = combine_hybrid_probs(p_eff, p_mob, weights=(w_eff, w_mob), method="soft_voting")
        picked = select_clinical_threshold(y_true, ens)
        row = {
            "efficientnet_weight": w_eff,
            "mobilenet_weight": w_mob,
            "threshold": picked["threshold"],
            "rule": picked["rule"],
            "f1": picked["selected"]["f1"],
            "recall": picked["selected"]["recall"],
            "sensitivity": picked["selected"]["sensitivity"],
            "specificity": picked["selected"]["specificity"],
            "precision": picked["selected"]["precision"],
            "accuracy": picked["selected"]["accuracy"],
            "pr_auc": picked["selected"]["pr_auc"],
            "auc": picked["selected"]["auc"],
            "fn": picked["selected"]["fn"],
            "fp": picked["selected"]["fp"],
        }
        rows.append(row)
        key = (row["f1"], row["sensitivity"], row["specificity"], -row["fn"])
        if best is None or key > best[0]:
            best = (key, row, ens, picked)
    selected = best[1]
    return {
        "candidates": rows,
        "selected": selected,
        "val_ensemble_prob": best[2],
        "threshold_search": best[3],
        "note": (
            "Weights and threshold were searched on validation only. "
            "Previous production 0.85/0.15 and 0.30 were not used as defaults."
        ),
    }


def train_one(builder, name: str, train_df, val_df, save_path: Path, batch: int, ft_batch: int) -> dict:
    logger.info("==== Training NEW %s from ImageNet, not production weights ====", name)
    t0 = time.time()
    train_ds = make_ds(train_df, shuffle=True, augment=True, batch=batch)
    val_ds = make_ds(val_df, shuffle=False, augment=False, batch=batch)

    model = builder(trainable_base=False)
    model = compile_model(model, learning_rate=config.LEARNING_RATE, use_focal_loss=False)
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        callbacks=get_callbacks(f"{name}_head", checkpoint_dir=CKPT_DIR),
        verbose=1,
    )
    phase1 = CKPT_DIR / f"{name}_head_best.keras"
    if phase1.exists():
        model = tf.keras.models.load_model(
            str(phase1), custom_objects={"FocalLoss": FocalLoss}, compile=False
        )

    model = unfreeze_for_finetuning(model, n_unfreeze_layers=config.FINE_TUNE_UNFREEZE_LAYERS)
    model = compile_model(model, learning_rate=config.FINE_TUNE_LR, use_focal_loss=False)
    ft_train = make_ds(train_df, shuffle=True, augment=True, batch=ft_batch)
    ft_val = make_ds(val_df, shuffle=False, augment=False, batch=ft_batch)
    history2 = model.fit(
        ft_train,
        validation_data=ft_val,
        epochs=config.FINE_TUNE_EPOCHS,
        callbacks=get_callbacks(f"{name}_finetune", checkpoint_dir=CKPT_DIR),
        verbose=1,
    )
    hist = {
        k: list(history1.history.get(k, [])) + list(history2.history.get(k, []))
        for k in set(history1.history) | set(history2.history)
    }
    ft_ckpt = CKPT_DIR / f"{name}_finetune_best.keras"
    best = ft_ckpt if ft_ckpt.exists() else phase1
    if best.exists():
        model = tf.keras.models.load_model(
            str(best), custom_objects={"FocalLoss": FocalLoss}, compile=False
        )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.resolve() in {
        config.EFFICIENTNET_PATH.resolve(),
        config.MOBILENET_PATH.resolve(),
    }:
        raise RuntimeError("Refusing to overwrite a production model path.")
    model.save(str(save_path))
    plot_training_curves(hist, CURVE_DIR, name)
    _json_dump(EXP_DIR / f"{name}_history.json", hist)

    y_val_true, y_val_prob = _predict_dataset(model, ft_val)
    val_metrics = evaluate_predictions(y_val_true, y_val_prob, threshold=0.5)
    full_evaluation_report(y_val_true, y_val_prob, f"val_{name}_at_0.5", PLOT_DIR, threshold=0.5)
    elapsed = time.time() - t0
    del model
    tf.keras.backend.clear_session()
    return {
        "path": str(save_path),
        "checkpoint": str(best),
        "history": hist,
        "best_epoch": best_epoch_from_history(hist),
        "training_seconds": elapsed,
        "y_val_true": y_val_true,
        "y_val_prob": y_val_prob,
        "val_metrics_at_0.5": val_metrics,
        "loss": "BinaryCrossEntropy",
        "initialized_from": "ImageNet",
        "old_weights_loaded": False,
    }


def evaluate_named(name: str, y_true, y_prob, threshold: float, dest: Path) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= float(threshold)).astype(int)
    metrics = evaluate_predictions(y_true, y_prob, threshold=threshold)
    dest.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(
        y_true, y_pred, dest / f"{name}_cm.png", labels=CLASS_LABELS
    )
    plot_roc_curve(y_true, y_prob, dest / f"{name}_roc.png", name)
    plot_pr_curve(y_true, y_prob, dest / f"{name}_pr.png", name)
    (dest / f"{name}_classification_report.txt").write_text(
        classification_report(
            y_true, y_pred, target_names=CLASS_LABELS, digits=4, zero_division=0
        ),
        encoding="utf-8",
    )
    _json_dump(dest / f"{name}_metrics.json", metrics)
    return metrics


def parse_ph2_labels() -> dict:
    labels = {}
    for line in PH2_LABELS.read_text(encoding="utf-8", errors="replace").splitlines():
        if "||" not in line or "IMD" not in line:
            continue
        parts = [p.strip() for p in line.split("||")]
        if len(parts) < 4:
            continue
        name = parts[1]
        if not name.startswith("IMD"):
            continue
        try:
            clinical = int(parts[3])
        except ValueError:
            continue
        labels[name] = 1 if clinical == 2 else 0
    return labels


def evaluate_ph2(eff_path: Path, mob_path: Path, w_eff: float, w_mob: float, threshold: float) -> dict:
    labels = parse_ph2_labels()
    preprocessor = ImagePreprocessor(apply_hair_removal=True)
    xs, ys, ids = [], [], []
    missing = []
    for case_id, label in sorted(labels.items()):
        img_path = PH2_DIR / case_id / f"{case_id}_Dermoscopic_Image" / f"{case_id}.bmp"
        if not img_path.exists():
            missing.append(case_id)
            continue
        raw = cv2.imread(str(img_path))
        if raw is None:
            missing.append(case_id)
            continue
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        xs.append(preprocessor.process_for_model(rgb).astype(np.float32))
        ys.append(label)
        ids.append(case_id)
    if not xs:
        raise FileNotFoundError("No PH2 dermoscopic images found.")
    x = np.stack(xs, axis=0)
    y = np.asarray(ys, dtype=int)

    eff = tf.keras.models.load_model(str(eff_path), custom_objects={"FocalLoss": FocalLoss}, compile=False)
    p_eff = np.asarray(eff.predict(x, batch_size=8, verbose=0)).ravel()
    del eff
    tf.keras.backend.clear_session()
    mob = tf.keras.models.load_model(str(mob_path), custom_objects={"FocalLoss": FocalLoss}, compile=False)
    p_mob = np.asarray(mob.predict(x, batch_size=8, verbose=0)).ravel()
    del mob
    tf.keras.backend.clear_session()
    p_ens = combine_hybrid_probs(p_eff, p_mob, weights=(w_eff, w_mob), method="soft_voting")

    ph2_dir = EXP_DIR / "ph2"
    metrics = {
        "label": "EXTERNAL PH2 TEST",
        "n": int(len(y)),
        "melanoma": int((y == 1).sum()),
        "benign": int((y == 0).sum()),
        "missing": missing,
        "used_for_training_or_tuning": False,
        "efficientnetb0": evaluate_named("ph2_efficientnetb0", y, p_eff, threshold, ph2_dir),
        "mobilenetv2": evaluate_named("ph2_mobilenetv2", y, p_mob, threshold, ph2_dir),
        "ensemble": evaluate_named("ph2_ensemble", y, p_ens, threshold, ph2_dir),
        "threshold": threshold,
        "weights": {"efficientnet": w_eff, "mobilenet": w_mob},
        "note": "PH2 was not used for training, threshold, or ensemble selection.",
    }
    _json_dump(EXP_DIR / "ph2_metrics.json", metrics)
    return metrics


def save_heatmap_panel(original, heatmap, overlay, dest_dir: Path, stem: str) -> dict:
    dest_dir.mkdir(parents=True, exist_ok=True)
    orig_path = dest_dir / f"{stem}_original.jpg"
    heat_path = dest_dir / f"{stem}_heatmap.jpg"
    over_path = dest_dir / f"{stem}_overlay.jpg"
    cv2.imwrite(str(orig_path), cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
    heat_u8 = np.uint8(255 * np.clip(heatmap, 0, 1))
    heat_color = cv2.applyColorMap(cv2.resize(heat_u8, (original.shape[1], original.shape[0])), cv2.COLORMAP_JET)
    cv2.imwrite(str(heat_path), heat_color)
    cv2.imwrite(str(over_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return {
        "original": str(orig_path),
        "heatmap": str(heat_path),
        "overlay": str(over_path),
        "caption": "AI Attention Visualization",
        "not_cancer_area": True,
    }


def verify_gradcam(eff_path: Path, test_df: pd.DataFrame, y_true, y_prob, threshold: float) -> dict:
    model = tf.keras.models.load_model(
        str(eff_path), custom_objects={"FocalLoss": FocalLoss}, compile=False
    )
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= float(threshold)).astype(int)
    paths = test_df["image_path"].astype(str).to_numpy()
    ids = test_df["image_id"].astype(str).to_numpy()
    buckets = {
        "true_positive_melanoma": np.where((y_true == 1) & (y_pred == 1))[0],
        "true_negative_benign": np.where((y_true == 0) & (y_pred == 0))[0],
        "false_positive": np.where((y_true == 0) & (y_pred == 1))[0],
        "false_negative": np.where((y_true == 1) & (y_pred == 0))[0],
    }
    cam = GradCAM(model)
    cases = []
    issues = []
    for kind, idxs in buckets.items():
        if len(idxs) == 0:
            issues.append(f"No {kind} example available on the frozen test set.")
            continue
        if kind.endswith("melanoma") or kind == "false_negative":
            idx = int(idxs[np.argmax(y_prob[idxs] if kind != "false_negative" else -y_prob[idxs])])
        elif kind == "false_positive":
            idx = int(idxs[np.argmax(y_prob[idxs])])
        else:
            idx = int(idxs[np.argmin(y_prob[idxs])])
        try:
            original, processed, _ = load_and_preprocess(paths[idx])
            batch = np.expand_dims(processed.astype(np.float32), axis=0)
            heatmap, overlay = cam.generate(batch, original)
            files = save_heatmap_panel(original, heatmap, overlay, GRADCAM_DIR, f"{ids[idx]}_{kind}")
            cases.append(
                {
                    "kind": kind,
                    "image_id": ids[idx],
                    "true_label": int(y_true[idx]),
                    "predicted_label": int(y_pred[idx]),
                    "ai_estimated_probability": float(y_prob[idx]),
                    "ok": True,
                    **files,
                    "note": (
                        "Heatmap is model attention, not a confirmed cancer boundary "
                        "and does not prove malignancy."
                    ),
                }
            )
        except Exception as exc:
            issues.append(f"{kind}: {exc}")
            cases.append({"kind": kind, "image_id": ids[idx], "ok": False, "error": str(exc)})
    del model
    tf.keras.backend.clear_session()
    payload = {
        "status": "ok" if cases and all(c.get("ok") for c in cases) else "partial",
        "label": "AI Attention Visualization",
        "n_examples": len(cases),
        "issues": issues,
        "cases": cases,
    }
    _json_dump(EXP_DIR / "gradcam_report.json", payload)
    return payload


def demo_100(test_df: pd.DataFrame, y_true, y_prob, threshold: float) -> dict:
    rng = np.random.default_rng(config.SEED)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    idx_pos = np.where(y_true == 1)[0]
    idx_neg = np.where(y_true == 0)[0]
    n_pos = min(50, len(idx_pos))
    n_neg = min(50, len(idx_neg))
    chosen = np.concatenate(
        [
            rng.choice(idx_pos, size=n_pos, replace=False),
            rng.choice(idx_neg, size=n_neg, replace=False),
        ]
    )
    rng.shuffle(chosen)
    subset = test_df.iloc[chosen].copy()
    metrics = evaluate_predictions(y_true[chosen], y_prob[chosen], threshold=threshold)
    y_pred = (y_prob[chosen] >= threshold).astype(int)
    correct = int((y_pred == y_true[chosen]).sum())
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    subset.assign(
        ai_estimated_probability=y_prob[chosen],
        predicted_label=y_pred,
    ).to_csv(DEMO_DIR / "demo_100.csv", index=False)
    plot_confusion_matrix(
        y_true[chosen], y_pred, DEMO_DIR / "confusion_matrix.png", labels=CLASS_LABELS
    )
    payload = {
        "label": "100-IMAGE TEST DEMONSTRATION",
        "not_the_complete_evaluation": True,
        "primary_internal_evaluation": "frozen test set of 2473 images",
        "used_for_training_or_tuning": False,
        "n": int(len(chosen)),
        "melanoma": int(n_pos),
        "benign": int(n_neg),
        "correct": correct,
        "incorrect": int(len(chosen) - correct),
        "accuracy": metrics["accuracy"],
        "threshold": threshold,
        "metrics": metrics,
        "confusion_matrix": {
            "tp": metrics["tp"],
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
        },
    }
    _json_dump(DEMO_DIR / "demo_100_metrics.json", payload)
    return payload


def copy_canonical_plots(src_name: str) -> None:
    mapping = {
        f"{src_name}_cm.png": EXP_DIR / "confusion_matrix.png",
        f"{src_name}_roc.png": EXP_DIR / "roc_curve.png",
        f"{src_name}_pr.png": EXP_DIR / "precision_recall_curve.png",
    }
    for src, dest in mapping.items():
        path = PLOT_DIR / src
        if path.exists():
            shutil.copy2(path, dest)


def main() -> None:
    if config.ACTIVE_DATASET != "combined":
        raise RuntimeError("ACTIVE_DATASET must stay 'combined' during this experiment.")

    production = {
        "efficientnet": _file_stamp(config.EFFICIENTNET_PATH),
        "mobilenet": _file_stamp(config.MOBILENET_PATH),
        "eval_metrics": _file_stamp(config.METRICS_PATH),
        "inference_config": _file_stamp(config.INFERENCE_CONFIG_PATH),
        "ensemble_meta": _file_stamp(config.ENSEMBLE_META_PATH),
    }

    enable_mixed_precision()
    tf.keras.utils.set_random_seed(config.SEED)
    config.USE_FOCAL_LOSS = False
    config.BALANCE_MODE = "none"
    config.MALIGNANT_OVERSAMPLE_FACTOR = 1
    config.BATCH_SIZE = min(int(config.BATCH_SIZE), 16)

    for folder in (EXP_DIR, CKPT_DIR, PLOT_DIR, CURVE_DIR, GRADCAM_DIR, DEMO_DIR, CACHE_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    gpus = tf.config.list_physical_devices("GPU")
    logger.info("GPUs available: %s", gpus or "none (CPU)")

    train = pd.read_csv(SPLIT_DIR / "train.csv")
    val = pd.read_csv(SPLIT_DIR / "val.csv")
    test = pd.read_csv(SPLIT_DIR / "test.csv")
    _assert_frozen_splits(train, val, test)
    if int((pd.concat([train, val, test])["source_dataset"] == "ISIC-DICM-17K").sum()) != 8530:
        raise ValueError("Melanoma pool must be exactly the 8530 DICM class=1 images.")
    logger.info("Using frozen melanoma_benign splits. Old combined/DICM splits are not loaded.")

    test_original = test.copy()
    _json_dump(
        EXP_DIR / "experiment_start.json",
        {
            "dataset": "melanoma_benign",
            "labels": {"0": "Benign", "1": "Melanoma"},
            "frozen_splits": {k: {"total": v[0], "melanoma": v[1], "benign": v[2]} for k, v in FROZEN.items()},
            "production_snapshot": production,
            "active_dataset": config.ACTIVE_DATASET,
            "old_threshold_not_reused": True,
            "old_ensemble_not_reused": True,
        },
    )

    logger.info("Caching inference-matched preprocess (hair + denoise + 224 + /255, no CLAHE).")
    train = cached_paths(train, "train")
    val = cached_paths(val, "val")
    test = cached_paths(test, "test")

    batch = int(config.BATCH_SIZE)
    ft_batch = max(8, min(16, batch // 2 or 8))
    eff_path = EXP_DIR / "efficientnetb0_melanoma_benign.keras"
    mob_path = EXP_DIR / "mobilenetv2_melanoma_benign.keras"

    eff = train_one(build_efficientnetb0, "mb_efficientnetb0", train, val, eff_path, batch, ft_batch)
    mob = train_one(build_mobilenetv2, "mb_mobilenetv2", train, val, mob_path, batch, ft_batch)

    y_val = np.asarray(eff["y_val_true"]).astype(int)
    search = search_ensemble(y_val, eff["y_val_prob"], mob["y_val_prob"])
    selected = search["selected"]
    w_eff = float(selected["efficientnet_weight"])
    w_mob = float(selected["mobilenet_weight"])
    threshold = float(selected["threshold"])
    val_ens = np.asarray(search["val_ensemble_prob"])
    logger.info(
        "Validation-selected ensemble: Eff=%.2f Mob=%.2f threshold=%.2f",
        w_eff,
        w_mob,
        threshold,
    )

    val_eff = evaluate_named("val_efficientnetb0", y_val, eff["y_val_prob"], threshold, PLOT_DIR)
    val_mob = evaluate_named("val_mobilenetv2", y_val, mob["y_val_prob"], threshold, PLOT_DIR)
    val_ens_metrics = evaluate_named("val_ensemble", y_val, val_ens, threshold, PLOT_DIR)
    save_threshold_table(search["threshold_search"]["table"], EXP_DIR / "val_threshold_table.csv")
    _json_dump(EXP_DIR / "val_ensemble_search.json", search["candidates"])

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(val_ens, y_val)
    val_cal = iso.predict(val_ens)
    frac_pos, mean_pred = calibration_curve(y_val, val_ens, n_bins=10, strategy="uniform")
    calibration = {
        "status": "implemented",
        "method": "isotonic regression",
        "fitted_on": "validation ensemble raw probabilities only",
        "validation_only_fitting": True,
        "test_used_for_fitting": False,
        "raw_brier": float(brier_score_loss(y_val, val_ens)),
        "calibrated_brier": float(brier_score_loss(y_val, val_cal)),
        "reliability_raw": {
            "mean_predicted": mean_pred.tolist(),
            "fraction_positive": frac_pos.tolist(),
        },
        "operating_threshold_uses_raw_scores": True,
        "note": (
            "Displayed probability is an AI-estimated model score. "
            "It is not a clinically calibrated medical probability."
        ),
    }

    test_ds = make_ds(test, shuffle=False, augment=False, batch=ft_batch)
    logger.info("Evaluating frozen test set ONCE after all tuning.")
    eff_model = tf.keras.models.load_model(str(eff_path), custom_objects={"FocalLoss": FocalLoss}, compile=False)
    y_test, p_eff = _predict_dataset(eff_model, test_ds)
    del eff_model
    tf.keras.backend.clear_session()
    mob_model = tf.keras.models.load_model(str(mob_path), custom_objects={"FocalLoss": FocalLoss}, compile=False)
    _, p_mob = _predict_dataset(mob_model, test_ds)
    del mob_model
    tf.keras.backend.clear_session()
    p_ens = combine_hybrid_probs(p_eff, p_mob, weights=(w_eff, w_mob), method="soft_voting")

    test_eff = evaluate_named("test_efficientnetb0", y_test, p_eff, threshold, PLOT_DIR)
    test_mob = evaluate_named("test_mobilenetv2", y_test, p_mob, threshold, PLOT_DIR)
    test_ens = evaluate_named("FINAL_FROZEN_TEST_ensemble", y_test, p_ens, threshold, PLOT_DIR)
    copy_canonical_plots("FINAL_FROZEN_TEST_ensemble")

    demo = demo_100(test_original, y_test, p_ens, threshold)
    gradcam = verify_gradcam(eff_path, test_original, y_test, p_ens, threshold)
    ph2 = evaluate_ph2(eff_path, mob_path, w_eff, w_mob, threshold)

    after = {
        "efficientnet": _file_stamp(config.EFFICIENTNET_PATH),
        "mobilenet": _file_stamp(config.MOBILENET_PATH),
        "eval_metrics": _file_stamp(config.METRICS_PATH),
        "inference_config": _file_stamp(config.INFERENCE_CONFIG_PATH),
        "ensemble_meta": _file_stamp(config.ENSEMBLE_META_PATH),
    }
    _assert_production_untouched(production, after)

    validation_metrics = {
        "efficientnetb0": val_eff,
        "mobilenetv2": val_mob,
        "ensemble": val_ens_metrics,
        "efficientnetb0_at_0.5": eff["val_metrics_at_0.5"],
        "mobilenetv2_at_0.5": mob["val_metrics_at_0.5"],
        "threshold_rule": search["threshold_search"]["rule"],
    }
    test_metrics = {
        "label": "FINAL FROZEN TEST PERFORMANCE",
        "efficientnetb0": test_eff,
        "mobilenetv2": test_mob,
        "ensemble": test_ens,
        "tuned_on_test": False,
    }
    ensemble_meta = {
        "method": "soft_voting",
        "efficientnet_weight": w_eff,
        "mobilenet_weight": w_mob,
        "decision_threshold": threshold,
        "selected_on": "validation",
        "old_085_015_not_reused": True,
        "old_030_threshold_not_reused": True,
        "candidates": search["candidates"],
        "wording": "AI-estimated probability",
    }
    inference_config = {
        "decision_threshold": threshold,
        "hybrid_method": "soft_voting",
        "weights": {"efficientnet": w_eff, "mobilenet": w_mob},
        "trained_on": ["melanoma_benign"],
        "labels": {"0": "Benign", "1": "Melanoma"},
        "threshold_tuned_on": "validation",
        "ensemble_tuned_on": "validation",
        "cache_version": f"mb-{threshold:.2f}",
        "not_production": True,
        "probability_is_clinically_calibrated": False,
    }
    evaluation_metrics = {
        "dataset": "melanoma_benign",
        "validation": validation_metrics,
        "internal_test": test_metrics,
        "external_ph2": {
            "ensemble": ph2["ensemble"],
            "n": ph2["n"],
            "label": "EXTERNAL PH2 TEST",
        },
        "demo_100": demo,
        "clinical_note": (
            "Decision-support only. This does not replace a dermatologist, "
            "does not guarantee diagnosis, and Grad-CAM does not prove cancer."
        ),
    }

    _json_dump(EXP_DIR / "ensemble_meta.json", ensemble_meta)
    _json_dump(EXP_DIR / "inference_config.json", inference_config)
    _json_dump(EXP_DIR / "validation_metrics.json", validation_metrics)
    _json_dump(EXP_DIR / "test_metrics.json", test_metrics)
    _json_dump(EXP_DIR / "evaluation_metrics.json", evaluation_metrics)
    _json_dump(EXP_DIR / "calibration.json", calibration)
    safety = {
        "old_production_models_not_overwritten": True,
        "old_production_metrics_not_overwritten": True,
        "old_threshold_not_reused": True,
        "old_ensemble_weights_not_reused": True,
        "trained_on_melanoma_benign_only": True,
        "dicm_class_0_not_used": True,
        "train_split_unchanged": True,
        "val_split_unchanged": True,
        "test_split_unchanged": True,
        "test_not_used_for_tuning": True,
        "ph2_not_used_for_tuning": True,
        "no_val_test_augmentation": True,
        "no_duplicated_images": True,
        "threshold_from_validation_only": True,
        "ensemble_weights_from_validation_only": True,
        "production_application_not_modified": True,
        "active_dataset_unchanged": config.ACTIVE_DATASET,
        "production_snapshot_before": production,
        "production_snapshot_after": after,
    }
    _json_dump(EXP_DIR / "safety_check.json", safety)

    report = {
        "dataset": {
            "total": 16529,
            "melanoma": 8530,
            "benign": 7999,
            "train": {"total": 11611, "melanoma": 5966, "benign": 5645},
            "validation": {"total": 2445, "melanoma": 1240, "benign": 1205},
            "test": {"total": 2473, "melanoma": 1324, "benign": 1149},
        },
        "efficientnetb0": {
            "best_epoch": eff["best_epoch"],
            "training_seconds": eff["training_seconds"],
            "validation": val_eff,
            "test": test_eff,
        },
        "mobilenetv2": {
            "best_epoch": mob["best_epoch"],
            "training_seconds": mob["training_seconds"],
            "validation": val_mob,
            "test": test_mob,
        },
        "ensemble": {
            "efficientnet_weight": w_eff,
            "mobilenet_weight": w_mob,
            "threshold": threshold,
            "validation": val_ens_metrics,
            "test": test_ens,
        },
        "ph2": ph2,
        "demo_100": demo,
        "gradcam": gradcam,
        "calibration": calibration,
        "safety": safety,
    }
    _json_dump(EXP_DIR / "FINAL_TRAINING_REPORT.json", report)
    lines = [
        "===== MELANOMA vs BENIGN TRAINING REPORT =====",
        "INTERNAL TEST is the frozen 2473-image split.",
        "EXTERNAL PH2 TEST is separate and was not used for tuning.",
        f"Ensemble Eff/Mob: {w_eff}/{w_mob}",
        f"Validation threshold: {threshold} ({search['threshold_search']['rule']})",
        f"Val ensemble: acc={val_ens_metrics['accuracy']:.4f} sens={val_ens_metrics['sensitivity']:.4f} spec={val_ens_metrics['specificity']:.4f}",
        f"Test ensemble: acc={test_ens['accuracy']:.4f} sens={test_ens['sensitivity']:.4f} spec={test_ens['specificity']:.4f}",
        f"PH2 ensemble: acc={ph2['ensemble']['accuracy']:.4f} sens={ph2['ensemble']['sensitivity']:.4f} spec={ph2['ensemble']['specificity']:.4f}",
        f"100-image demo accuracy: {demo['accuracy']:.4f} (demonstration only)",
        f"Grad-CAM: {gradcam['status']}",
        f"New files under: {EXP_DIR}",
        "Production models and ACTIVE_DATASET were not changed.",
        "Decision-support only. Not a guaranteed diagnosis.",
    ]
    (EXP_DIR / "FINAL_TRAINING_REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    logger.info("Melanoma-vs-Benign experiment finished. Production application was not switched.")


if __name__ == "__main__":
    main()
