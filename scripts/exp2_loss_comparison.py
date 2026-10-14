"""
Experiment 2: loss-function comparison on the Experiment 1 3-stage schedule.

Writes only under models/experiments/internal_optimization/exp2_loss_comparison/.
Does not overwrite Experiment 1, production models, frozen CSVs, or the app.
Does not load the frozen test set or PH2.
"""

from __future__ import annotations

import json
import shutil
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
from training.callbacks import get_callbacks
from training.evaluate import evaluate_predictions, plot_training_curves, save_threshold_table, threshold_table
from training.train import _predict_dataset, enable_mixed_precision
from utils.logger import setup_logger

logger = setup_logger("exp2")

SPLIT_DIR = config.PROCESSED_DATA_DIR / "melanoma_benign"
CACHE_DIR = SPLIT_DIR / "preprocessed"
EXP_ROOT = config.BASE_DIR / "models" / "experiments" / "internal_optimization"
EXP1_DIR = EXP_ROOT / "exp1_three_stage"
EXP_DIR = EXP_ROOT / "exp2_loss_comparison"
PROTECTED = {
    config.EFFICIENTNET_PATH.resolve(),
    config.MOBILENET_PATH.resolve(),
    (config.MODELS_DIR / "efficientnetb0_melanoma_benign.keras").resolve(),
    (config.MODELS_DIR / "mobilenetv2_melanoma_benign.keras").resolve(),
    (config.MODELS_DIR / "melanoma_benign" / "efficientnetb0_melanoma_benign.keras").resolve(),
    (config.MODELS_DIR / "melanoma_benign" / "mobilenetv2_melanoma_benign.keras").resolve(),
    (EXP1_DIR / "efficientnetb0_exp1.keras").resolve(),
    (EXP1_DIR / "mobilenetv2_exp1.keras").resolve(),
}
FROZEN = {"train": (11611, 5966, 5645), "val": (2445, 1240, 1205)}
THRESHOLD_GRID = [round(x, 2) for x in np.arange(0.20, 0.81, 0.05)]
WEIGHT_GRID = [round(x, 2) for x in np.arange(0.00, 1.01, 0.05)]
STAGES = (
    {"name": "head", "lr": 1e-3, "epochs": 25, "unfreeze": 0, "batch": 16},
    {"name": "partial", "lr": 1e-4, "epochs": 12, "unfreeze": 20, "batch": 8},
    {"name": "final", "lr": 1e-5, "epochs": 12, "unfreeze": 20, "batch": 8},
)
FOCAL_ALPHA = 0.50
FOCAL_GAMMA = 2.0
LABEL_SMOOTHING = 0.05
EXP1_REF = {
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
LOSSES = (
    {
        "key": "exp2a_bce",
        "name": "bce",
        "label": "EXP2-A Binary Cross Entropy (Experiment 1 control, not retrained)",
        "train": False,
        "source_models": {
            "efficientnetb0": EXP1_DIR / "efficientnetb0_exp1.keras",
            "mobilenetv2": EXP1_DIR / "mobilenetv2_exp1.keras",
        },
    },
    {
        "key": "exp2b_focal",
        "name": "focal",
        "label": f"EXP2-B Focal Loss (alpha={FOCAL_ALPHA}, gamma={FOCAL_GAMMA})",
        "train": True,
        "source_models": {},
    },
    {
        "key": "exp2c_bce_smooth",
        "name": "bce_smooth",
        "label": f"EXP2-C BCE + label smoothing ({LABEL_SMOOTHING})",
        "train": True,
        "source_models": {},
    },
)


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
    if resolved in PROTECTED or EXP1_DIR.resolve() in resolved.parents:
        raise RuntimeError(f"Refusing to write protected path: {path}")


def _assert_frozen(name: str, df: pd.DataFrame) -> None:
    n, mel, ben = FROZEN[name]
    got = (len(df), int((df["label"] == 1).sum()), int((df["label"] == 0).sum()))
    if got != (n, mel, ben):
        raise ValueError(f"Frozen {name} CSV changed: expected {(n, mel, ben)}, got {got}")


def _use_cache(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    missing = []
    paths = []
    for image_id, src in zip(out["image_id"].astype(str), out["image_path"].astype(str)):
        cached = CACHE_DIR / f"{image_id}.jpg"
        if cached.is_file():
            paths.append(str(cached))
        else:
            missing.append(image_id)
            paths.append(src)
    if missing:
        raise FileNotFoundError(f"{len(missing)} cached images missing, e.g. {missing[:5]}")
    out["image_path"] = paths
    return out


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


def compile_with_loss(model: keras.Model, learning_rate: float, loss_name: str) -> keras.Model:
    try:
        optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=config.WEIGHT_DECAY)
    except AttributeError:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    if loss_name == "focal":
        loss = FocalLoss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA)
    elif loss_name == "bce_smooth":
        loss = keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING)
    else:
        loss = keras.losses.BinaryCrossentropy()
    model.compile(
        optimizer=optimizer,
        loss=loss,
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
    """
    Validation-only operating point.

    Prefer high melanoma sensitivity without collapsing specificity.
    Do not pick a threshold only because it maximizes accuracy.
    """
    rows = threshold_table(y_true, y_prob, thresholds=THRESHOLD_GRID)
    tiers = [
        ("sens>=0.85 and spec>=0.70, max F1", [r for r in rows if r["sensitivity"] >= 0.85 and r["specificity"] >= 0.70]),
        ("sens>=0.80 and spec>=0.65, max F1", [r for r in rows if r["sensitivity"] >= 0.80 and r["specificity"] >= 0.65]),
        ("max F1 on validation grid (fallback)", rows),
    ]
    chosen = None
    rule = ""
    for name, cands in tiers:
        if not cands:
            continue
        chosen = max(cands, key=lambda r: (r["f1"], r["sensitivity"], r["specificity"], -abs(r["threshold"] - 0.5)))
        rule = name
        break
    return {"threshold": float(chosen["threshold"]), "rule": rule, "selected": chosen, "table": rows}


def search_ensemble(y_true, p_eff, p_mob) -> dict:
    """Select weights by ROC-AUC then PR-AUC. Threshold is chosen afterward."""
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


def train_one(builder, name: str, loss_name: str, run_dir: Path, train_df, val_df, save_path: Path) -> dict:
    ckpt_dir = run_dir / "checkpoints"
    curve_dir = run_dir / "training_curves"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    curve_dir.mkdir(parents=True, exist_ok=True)
    logger.info("==== Exp2 %s training %s ====", loss_name, name)
    t0 = time.time()
    oom = False
    hist: dict[str, list] = {}
    model = builder(trainable_base=False)
    stage_epochs = []
    try:
        for stage in STAGES:
            if stage["unfreeze"]:
                model = unfreeze_for_finetuning(model, n_unfreeze_layers=stage["unfreeze"])
            model = compile_with_loss(model, stage["lr"], loss_name)
            train_ds = make_ds(train_df, shuffle=True, augment=True, batch=stage["batch"])
            val_ds = make_ds(val_df, shuffle=False, augment=False, batch=stage["batch"])
            logger.info("Stage %s lr=%s batch=%s loss=%s", stage["name"], stage["lr"], stage["batch"], loss_name)
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=stage["epochs"],
                callbacks=get_callbacks(f"exp2_{loss_name}_{name}_{stage['name']}", checkpoint_dir=ckpt_dir),
                verbose=1,
            )
            for key, values in history.history.items():
                hist.setdefault(key, []).extend(list(values))
            stage_epochs.append({"stage": stage["name"], "epochs_run": len(history.history.get("loss", []))})
            ckpt = ckpt_dir / f"exp2_{loss_name}_{name}_{stage['name']}_best.keras"
            if ckpt.exists():
                model = tf.keras.models.load_model(
                    str(ckpt), custom_objects={"FocalLoss": FocalLoss}, compile=False
                )
    except tf.errors.ResourceExhaustedError:
        oom = True
        logger.exception("OOM during %s %s", loss_name, name)
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
    return {
        "path": str(save_path),
        "history": hist,
        "stage_epochs": stage_epochs,
        "training_seconds": time.time() - t0,
        "oom": oom,
        "y_val_true": y_true,
        "y_val_prob": y_prob,
    }


def infer_one(model_path: Path, dest_path: Path, name: str, val_df) -> dict:
    _refuse_protected(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.resolve() != model_path.resolve():
        shutil.copy2(model_path, dest_path)
    model = tf.keras.models.load_model(str(dest_path), custom_objects={"FocalLoss": FocalLoss}, compile=False)
    val_ds = make_ds(val_df, shuffle=False, augment=False, batch=8)
    y_true, y_prob = _predict_dataset(model, val_ds)
    del model
    tf.keras.backend.clear_session()
    return {
        "path": str(dest_path),
        "history": {},
        "stage_epochs": [],
        "training_seconds": 0.0,
        "oom": False,
        "y_val_true": y_true,
        "y_val_prob": y_prob,
        "reused_from": str(model_path),
    }


def evaluate_run(spec: dict, train_df, val_df, oom_so_far: bool) -> dict:
    run_dir = EXP_DIR / spec["key"]
    run_dir.mkdir(parents=True, exist_ok=True)
    _json_dump(
        run_dir / "configuration.json",
        {
            "key": spec["key"],
            "loss": spec["name"],
            "label": spec["label"],
            "seed": config.SEED,
            "input_size": 224,
            "augmentation": "identical to Experiment 1: get_tf_data_augmentation_fn, train only, no JPEG",
            "finetune": STAGES,
            "focal_alpha": FOCAL_ALPHA if spec["name"] == "focal" else None,
            "focal_gamma": FOCAL_GAMMA if spec["name"] == "focal" else None,
            "label_smoothing": LABEL_SMOOTHING if spec["name"] == "bce_smooth" else 0.0,
            "retrained": spec["train"],
            "test_loaded": False,
            "ph2_used": False,
        },
    )
    marker = run_dir / "run_complete.json"
    if marker.exists() and (run_dir / "validation_predictions.csv").exists():
        logger.info("Skipping %s; already complete.", spec["key"])
        return json.loads(marker.read_text(encoding="utf-8"))

    if spec["train"] and oom_so_far:
        raise RuntimeError(f"Not starting {spec['key']} after a previous OOM.")

    eff_path = run_dir / f"efficientnetb0_{spec['name']}.keras"
    mob_path = run_dir / f"mobilenetv2_{spec['name']}.keras"
    t0 = time.time()
    if spec["train"]:
        eff = train_one(build_efficientnetb0, "efficientnetb0", spec["name"], run_dir, train_df, val_df, eff_path)
        _json_dump(EXP_DIR / "status.json", {"stage": f"{spec['key']}_efficientnet_done", "seconds": eff["training_seconds"]})
        mob = train_one(build_mobilenetv2, "mobilenetv2", spec["name"], run_dir, train_df, val_df, mob_path)
    else:
        eff = infer_one(spec["source_models"]["efficientnetb0"], eff_path, "efficientnetb0", val_df)
        mob = infer_one(spec["source_models"]["mobilenetv2"], mob_path, "mobilenetv2", val_df)

    y_val = np.asarray(eff["y_val_true"]).astype(int)
    search = search_ensemble(y_val, eff["y_val_prob"], mob["y_val_prob"])
    weights = search["selected_weights"]
    threshold = float(search["threshold_search"]["threshold"])
    rule = search["threshold_search"]["rule"]
    val_eff_05 = evaluate_predictions(y_val, eff["y_val_prob"], threshold=0.5)
    val_mob_05 = evaluate_predictions(y_val, mob["y_val_prob"], threshold=0.5)
    val_eff_op = evaluate_predictions(y_val, eff["y_val_prob"], threshold=select_clinical_threshold(y_val, eff["y_val_prob"])["threshold"])
    val_mob_op = evaluate_predictions(y_val, mob["y_val_prob"], threshold=select_clinical_threshold(y_val, mob["y_val_prob"])["threshold"])
    val_ens = search["operating"]

    pred_df = pd.DataFrame(
        {
            "image_id": val_df["image_id"].astype(str).values,
            "label": y_val,
            "efficientnet_prob": np.asarray(eff["y_val_prob"]).ravel(),
            "mobilenet_prob": np.asarray(mob["y_val_prob"]).ravel(),
            "ensemble_prob": np.asarray(search["val_ensemble_prob"]).ravel(),
        }
    )
    pred_df.to_csv(run_dir / "validation_predictions.csv", index=False)
    pd.DataFrame(search["weight_candidates"]).to_csv(run_dir / "ensemble_weight_search.csv", index=False)
    save_threshold_table(search["threshold_search"]["table"], run_dir / "val_threshold_table.csv")

    payload = {
        "key": spec["key"],
        "loss": spec["name"],
        "label": spec["label"],
        "evaluated_on": "validation_only",
        "test_used": False,
        "ph2_used": False,
        "oom": bool(eff.get("oom") or mob.get("oom")),
        "wall_seconds": time.time() - t0,
        "efficientnet": {
            "path": eff["path"],
            "training_seconds": eff["training_seconds"],
            "stage_epochs": eff["stage_epochs"],
            "reused_from": eff.get("reused_from"),
            "validation_at_0.5": val_eff_05,
            "validation_operating": val_eff_op,
        },
        "mobilenet": {
            "path": mob["path"],
            "training_seconds": mob["training_seconds"],
            "stage_epochs": mob["stage_epochs"],
            "reused_from": mob.get("reused_from"),
            "validation_at_0.5": val_mob_05,
            "validation_operating": val_mob_op,
        },
        "ensemble": {
            **weights,
            "threshold": threshold,
            "threshold_rule": rule,
            "validation": val_ens,
        },
        "weight_candidates": search["weight_candidates"],
    }
    _json_dump(run_dir / "validation_metrics.json", payload)
    _json_dump(run_dir / "ensemble_results.json", payload["ensemble"])
    _json_dump(marker, payload)
    logger.info(
        "%s ensemble auc=%.4f pr=%.4f acc=%.4f sens=%.4f spec=%.4f prec=%.4f f1=%.4f thr=%.2f w=%.2f/%.2f",
        spec["key"],
        val_ens["auc"],
        val_ens["pr_auc"],
        val_ens["accuracy"],
        val_ens["sensitivity"],
        val_ens["specificity"],
        val_ens["precision"],
        val_ens["f1"],
        threshold,
        weights["efficientnet_weight"],
        weights["mobilenet_weight"],
    )
    return payload


def _metric_row(loss: str, model: str, metrics: dict) -> dict:
    return {
        "loss": loss,
        "model": model,
        "accuracy": metrics.get("accuracy"),
        "sensitivity": metrics.get("sensitivity"),
        "specificity": metrics.get("specificity"),
        "precision": metrics.get("precision"),
        "f1": metrics.get("f1"),
        "roc_auc": metrics.get("auc"),
        "pr_auc": metrics.get("pr_auc"),
        "threshold": metrics.get("threshold"),
    }


def write_final_reports(results: list[dict]) -> None:
    loss_rows = []
    ens_rows = []
    thr_rows = []
    for item in results:
        loss_rows.append(_metric_row(item["loss"], "efficientnetb0", item["efficientnet"]["validation_operating"]))
        loss_rows.append(_metric_row(item["loss"], "mobilenetv2", item["mobilenet"]["validation_operating"]))
        loss_rows.append(_metric_row(item["loss"], "ensemble", item["ensemble"]["validation"]))
        ens = item["ensemble"]
        ens_rows.append(
            {
                "loss": item["loss"],
                "efficientnet_weight": ens["efficientnet_weight"],
                "mobilenet_weight": ens["mobilenet_weight"],
                "threshold": ens["threshold"],
                "threshold_rule": ens["threshold_rule"],
                "accuracy": ens["validation"]["accuracy"],
                "sensitivity": ens["validation"]["sensitivity"],
                "specificity": ens["validation"]["specificity"],
                "precision": ens["validation"]["precision"],
                "f1": ens["validation"]["f1"],
                "roc_auc": ens["validation"]["auc"],
                "pr_auc": ens["validation"]["pr_auc"],
            }
        )
        table = pd.read_csv(EXP_DIR / item["key"] / "val_threshold_table.csv")
        table.insert(0, "loss", item["loss"])
        thr_rows.append(table)

    pd.DataFrame(loss_rows).to_csv(EXP_DIR / "loss_comparison.csv", index=False)
    pd.DataFrame(ens_rows).to_csv(EXP_DIR / "ensemble_comparison.csv", index=False)
    pd.concat(thr_rows, ignore_index=True).to_csv(EXP_DIR / "threshold_comparison.csv", index=False)

    best_eff = max(results, key=lambda r: (r["efficientnet"]["validation_operating"]["auc"], r["efficientnet"]["validation_operating"]["pr_auc"]))
    best_mob = max(results, key=lambda r: (r["mobilenet"]["validation_operating"]["auc"], r["mobilenet"]["validation_operating"]["pr_auc"]))
    best_ens = max(results, key=lambda r: (r["ensemble"]["validation"]["auc"], r["ensemble"]["validation"]["pr_auc"]))
    ev = best_ens["ensemble"]["validation"]
    diffs = {k: ev[{"auc": "auc", "pr_auc": "pr_auc"}.get(k, k)] - EXP1_REF[k] for k in EXP1_REF if k in ev or k in {"auc", "pr_auc"}}
    diffs = {
        "accuracy": ev["accuracy"] - EXP1_REF["accuracy"],
        "sensitivity": ev["sensitivity"] - EXP1_REF["sensitivity"],
        "specificity": ev["specificity"] - EXP1_REF["specificity"],
        "precision": ev["precision"] - EXP1_REF["precision"],
        "f1": ev["f1"] - EXP1_REF["f1"],
        "roc_auc": ev["auc"] - EXP1_REF["auc"],
        "pr_auc": ev["pr_auc"] - EXP1_REF["pr_auc"],
    }
    meaningful = diffs["roc_auc"] >= 0.003 or diffs["pr_auc"] >= 0.003
    report = {
        "experiment": "exp2_loss_comparison",
        "evaluated_on": "validation_only",
        "test_used": False,
        "ph2_used": False,
        "experiment_1_reference": EXP1_REF,
        "runs": results,
        "best_efficientnet": {
            "loss": best_eff["loss"],
            "validation": best_eff["efficientnet"]["validation_operating"],
        },
        "best_mobilenet": {
            "loss": best_mob["loss"],
            "validation": best_mob["mobilenet"]["validation_operating"],
        },
        "best_ensemble": {
            "loss": best_ens["loss"],
            "efficientnet_weight": best_ens["ensemble"]["efficientnet_weight"],
            "mobilenet_weight": best_ens["ensemble"]["mobilenet_weight"],
            "threshold": best_ens["ensemble"]["threshold"],
            "threshold_rule": best_ens["ensemble"]["threshold_rule"],
            "validation": ev,
        },
        "difference_from_experiment_1": diffs,
        "meaningful_improvement_over_exp1": meaningful,
        "oom": any(item.get("oom") for item in results),
    }
    _json_dump(EXP_DIR / "EXP2_VALIDATION_REPORT.json", report)
    lines = [
        "EXPERIMENT 2 — LOSS FUNCTION COMPARISON (validation only)",
        "",
        "1. Experiment 1 reference ensemble:",
        f"   acc={EXP1_REF['accuracy']:.4f} sens={EXP1_REF['sensitivity']:.4f} spec={EXP1_REF['specificity']:.4f} prec={EXP1_REF['precision']:.4f} f1={EXP1_REF['f1']:.4f} auc={EXP1_REF['auc']:.4f} pr={EXP1_REF['pr_auc']:.4f}",
        "",
    ]
    labels = {"bce": "2. BCE (EXP2-A control)", "focal": "3. Focal Loss (EXP2-B)", "bce_smooth": "4. BCE + label smoothing (EXP2-C)"}
    for item in results:
        e = item["ensemble"]["validation"]
        lines.append(labels[item["loss"]])
        lines.append(
            f"   ensemble acc={e['accuracy']:.4f} sens={e['sensitivity']:.4f} spec={e['specificity']:.4f} prec={e['precision']:.4f} f1={e['f1']:.4f} auc={e['auc']:.4f} pr={e['pr_auc']:.4f}"
        )
        lines.append(
            f"   weights={item['ensemble']['efficientnet_weight']:.2f}/{item['ensemble']['mobilenet_weight']:.2f} thr={item['ensemble']['threshold']:.2f} ({item['ensemble']['threshold_rule']})"
        )
        lines.append(
            f"   Eff AUC={item['efficientnet']['validation_operating']['auc']:.4f} Mob AUC={item['mobilenet']['validation_operating']['auc']:.4f}"
        )
        lines.append("")
    lines.extend(
        [
            f"5. Best EfficientNet: {best_eff['loss']} AUC={best_eff['efficientnet']['validation_operating']['auc']:.4f} PR={best_eff['efficientnet']['validation_operating']['pr_auc']:.4f}",
            f"6. Best MobileNet: {best_mob['loss']} AUC={best_mob['mobilenet']['validation_operating']['auc']:.4f} PR={best_mob['mobilenet']['validation_operating']['pr_auc']:.4f}",
            f"7. Best ensemble: {best_ens['loss']}",
            f"8. Selected ensemble weights: {best_ens['ensemble']['efficientnet_weight']:.2f}/{best_ens['ensemble']['mobilenet_weight']:.2f}",
            f"9. Selected validation threshold: {best_ens['ensemble']['threshold']:.2f} ({best_ens['ensemble']['threshold_rule']})",
            f"10. Accuracy: {ev['accuracy']:.4f}",
            f"11. Sensitivity: {ev['sensitivity']:.4f}",
            f"12. Specificity: {ev['specificity']:.4f}",
            f"13. Precision: {ev['precision']:.4f}",
            f"14. F1: {ev['f1']:.4f}",
            f"15. ROC-AUC: {ev['auc']:.4f}",
            f"16. PR-AUC: {ev['pr_auc']:.4f}",
            "17. Difference from Experiment 1:",
            f"   acc={diffs['accuracy']:+.4f} sens={diffs['sensitivity']:+.4f} spec={diffs['specificity']:+.4f} prec={diffs['precision']:+.4f} f1={diffs['f1']:+.4f} auc={diffs['roc_auc']:+.4f} pr={diffs['pr_auc']:+.4f}",
            f"18. Training time seconds: " + ", ".join(f"{r['loss']}={r['wall_seconds']:.1f}" for r in results),
            f"19. RAM/OOM: {'YES' if report['oom'] else 'none'}",
            "20. Files created under models/experiments/internal_optimization/exp2_loss_comparison/",
            "",
            f"Meaningful improvement over Experiment 1: {'YES' if meaningful else 'NO'}",
            "Test set was not loaded. PH2 was not used.",
            "No model was promoted. Experiment 1 was not overwritten.",
        ]
    )
    (EXP_DIR / "EXP2_VALIDATION_REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    if config.ACTIVE_DATASET != "combined":
        raise RuntimeError("ACTIVE_DATASET must stay 'combined'.")
    enable_mixed_precision()
    tf.keras.utils.set_random_seed(config.SEED)
    config.USE_FOCAL_LOSS = False
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    if not (EXP1_DIR / "efficientnetb0_exp1.keras").is_file() or not (EXP1_DIR / "mobilenetv2_exp1.keras").is_file():
        raise FileNotFoundError("Experiment 1 models are required as the BCE control and were not found.")

    train = pd.read_csv(SPLIT_DIR / "train.csv")
    val = pd.read_csv(SPLIT_DIR / "val.csv")
    _assert_frozen("train", train)
    _assert_frozen("val", val)
    train = _use_cache(train)
    val = _use_cache(val)
    logger.info("Loaded frozen train/val only. Test CSV was not read. PH2 was not used.")

    _json_dump(
        EXP_DIR / "experiment_config.json",
        {
            "experiment": "exp2_loss_comparison",
            "seed": config.SEED,
            "control": "EXP2-A reuses Experiment 1 BCE models (read-only). Not retrained.",
            "focal": {"alpha": FOCAL_ALPHA, "gamma": FOCAL_GAMMA, "chosen_a_priori": True},
            "label_smoothing": LABEL_SMOOTHING,
            "ensemble_weight_rule": "max validation ROC-AUC, then PR-AUC",
            "threshold_rule": "sens>=0.85 and spec>=0.70, max F1; fallbacks documented in code",
            "test_loaded": False,
            "ph2_used": False,
            "overwrites_experiment_1": False,
        },
    )

    results = []
    oom = False
    try:
        for spec in LOSSES:
            _json_dump(EXP_DIR / "status.json", {"stage": spec["key"], "oom": oom})
            results.append(evaluate_run(spec, train, val, oom))
    except Exception:
        text = traceback.format_exc()
        oom = "ResourceExhausted" in text or "Unable to allocate" in text or "OOM" in text
        _json_dump(EXP_DIR / "status.json", {"stage": "failed", "oom": oom, "error": text})
        if results:
            write_final_reports(results)
        raise

    write_final_reports(results)
    _json_dump(EXP_DIR / "status.json", {"stage": "complete", "oom": False})
    logger.info("Experiment 2 finished. Stopping as requested. Experiment 3 was not started.")


if __name__ == "__main__":
    main()
