"""
Experiment 1: 3-stage fine-tuning on frozen melanoma-vs-benign train/val.

Writes only under models/experiments/internal_optimization/exp1_three_stage/.
Does not load the frozen test set, PH2, or production models.
"""

from __future__ import annotations

import json
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

import config
from models.architecture import (
    FocalLoss,
    build_efficientnetb0,
    build_mobilenetv2,
    compile_model,
    unfreeze_for_finetuning,
)
from models.ensemble import combine_hybrid_probs
from preprocessing.augmentation import get_tf_data_augmentation_fn
from training.callbacks import get_callbacks
from training.evaluate import (
    evaluate_predictions,
    plot_training_curves,
    save_threshold_table,
    threshold_table,
)
from training.train import _predict_dataset, enable_mixed_precision
from utils.logger import setup_logger

logger = setup_logger("exp1")

SPLIT_DIR = config.PROCESSED_DATA_DIR / "melanoma_benign"
CACHE_DIR = SPLIT_DIR / "preprocessed"
EXP_ROOT = config.BASE_DIR / "models" / "experiments" / "internal_optimization"
EXP_DIR = EXP_ROOT / "exp1_three_stage"
CKPT_DIR = EXP_DIR / "checkpoints"
CURVE_DIR = EXP_DIR / "training_curves"
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
WEIGHT_GRID = [round(x, 2) for x in np.arange(0.10, 0.91, 0.05)]
STAGES = (
    {"name": "head", "lr": 1e-3, "epochs": 25, "unfreeze": 0, "batch": 16},
    {"name": "partial", "lr": 1e-4, "epochs": 12, "unfreeze": 20, "batch": 8},
    {"name": "final", "lr": 1e-5, "epochs": 12, "unfreeze": 20, "batch": 8},
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
    if path.resolve() in PROTECTED:
        raise RuntimeError(f"Refusing to write protected model path: {path}")


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


def select_clinical_threshold(y_true, y_prob) -> dict:
    rows = threshold_table(y_true, y_prob, thresholds=THRESHOLD_GRID)
    tiers = [
        ("sens>=0.85 and spec>=0.70, max F1", [r for r in rows if r["sensitivity"] >= 0.85 and r["specificity"] >= 0.70]),
        ("sens>=0.80 and spec>=0.65, max F1", [r for r in rows if r["sensitivity"] >= 0.80 and r["specificity"] >= 0.65]),
        ("max F1 on validation grid", rows),
    ]
    chosen = None
    rule = ""
    for name, cands in tiers:
        if not cands:
            continue
        chosen = max(cands, key=lambda r: (r["f1"], r["sensitivity"], r["specificity"], r["threshold"]))
        rule = name
        break
    return {"threshold": float(chosen["threshold"]), "rule": rule, "selected": chosen, "table": rows}


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
            **{k: picked["selected"][k] for k in ("f1", "sensitivity", "specificity", "precision", "accuracy", "pr_auc", "auc")},
        }
        rows.append(row)
        key = (row["f1"], row["sensitivity"], row["specificity"], row["auc"])
        if best is None or key > best[0]:
            best = (key, row, ens, picked)
    return {"candidates": rows, "selected": best[1], "val_ensemble_prob": best[2], "threshold_search": best[3]}


def train_one(builder, name: str, train_df, val_df, save_path: Path) -> dict:
    logger.info("==== Exp1 3-stage training %s ====", name)
    t0 = time.time()
    oom = False
    hist: dict[str, list] = {}
    model = builder(trainable_base=False)
    stage_epochs = []
    try:
        for stage in STAGES:
            if stage["unfreeze"]:
                model = unfreeze_for_finetuning(model, n_unfreeze_layers=stage["unfreeze"])
            model = compile_model(model, learning_rate=stage["lr"], use_focal_loss=False)
            train_ds = make_ds(train_df, shuffle=True, augment=True, batch=stage["batch"])
            val_ds = make_ds(val_df, shuffle=False, augment=False, batch=stage["batch"])
            logger.info("Stage %s lr=%s batch=%s", stage["name"], stage["lr"], stage["batch"])
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=stage["epochs"],
                callbacks=get_callbacks(f"exp1_{name}_{stage['name']}", checkpoint_dir=CKPT_DIR),
                verbose=1,
            )
            for key, values in history.history.items():
                hist.setdefault(key, []).extend(list(values))
            stage_epochs.append({"stage": stage["name"], "epochs_run": len(history.history.get("loss", []))})
            ckpt = CKPT_DIR / f"exp1_{name}_{stage['name']}_best.keras"
            if ckpt.exists():
                model = tf.keras.models.load_model(
                    str(ckpt), custom_objects={"FocalLoss": FocalLoss}, compile=False
                )
    except tf.errors.ResourceExhaustedError as exc:
        oom = True
        logger.exception("OOM during %s: %s", name, exc)
        raise
    except MemoryError:
        oom = True
        raise

    _refuse_protected(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    plot_training_curves(hist, CURVE_DIR, name)
    _json_dump(EXP_DIR / f"{name}_history.json", hist)
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


def main() -> None:
    if config.ACTIVE_DATASET != "combined":
        raise RuntimeError("ACTIVE_DATASET must stay 'combined'.")
    enable_mixed_precision()
    tf.keras.utils.set_random_seed(config.SEED)
    config.USE_FOCAL_LOSS = False
    for folder in (EXP_DIR, CKPT_DIR, CURVE_DIR):
        folder.mkdir(parents=True, exist_ok=True)

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
            "experiment": "exp1_three_stage_finetune",
            "seed": config.SEED,
            "input_size": 224,
            "loss": "BinaryCrossEntropy",
            "augmentation": "current baseline get_tf_data_augmentation_fn, train only",
            "stages": STAGES,
            "splits": {"train": str(SPLIT_DIR / "train.csv"), "val": str(SPLIT_DIR / "val.csv")},
            "test_loaded": False,
            "ph2_used": False,
        },
    )

    eff_path = EXP_DIR / "efficientnetb0_exp1.keras"
    mob_path = EXP_DIR / "mobilenetv2_exp1.keras"
    status = {"stage": "train", "oom": False}
    try:
        eff = train_one(build_efficientnetb0, "efficientnetb0", train, val, eff_path)
        _json_dump(EXP_DIR / "status.json", {"stage": "efficientnet_done", "seconds": eff["training_seconds"]})
        mob = train_one(build_mobilenetv2, "mobilenetv2", train, val, mob_path)
    except Exception:
        status["oom"] = "ResourceExhausted" in traceback.format_exc() or "Unable to allocate" in traceback.format_exc()
        status["error"] = traceback.format_exc()
        _json_dump(EXP_DIR / "status.json", status)
        raise

    y_val = np.asarray(eff["y_val_true"]).astype(int)
    search = search_ensemble(y_val, eff["y_val_prob"], mob["y_val_prob"])
    selected = search["selected"]
    threshold = float(selected["threshold"])
    val_eff = evaluate_predictions(y_val, eff["y_val_prob"], threshold=threshold)
    val_mob = evaluate_predictions(y_val, mob["y_val_prob"], threshold=threshold)
    val_ens = evaluate_predictions(y_val, search["val_ensemble_prob"], threshold=threshold)
    save_threshold_table(search["threshold_search"]["table"], EXP_DIR / "val_threshold_table.csv")

    report = {
        "experiment": "exp1_three_stage_finetune",
        "evaluated_on": "validation_only",
        "test_used": False,
        "ph2_used": False,
        "oom": False,
        "efficientnet": {
            "training_seconds": eff["training_seconds"],
            "stage_epochs": eff["stage_epochs"],
            "validation": val_eff,
        },
        "mobilenet": {
            "training_seconds": mob["training_seconds"],
            "stage_epochs": mob["stage_epochs"],
            "validation": val_mob,
        },
        "ensemble": {
            "efficientnet_weight": selected["efficientnet_weight"],
            "mobilenet_weight": selected["mobilenet_weight"],
            "threshold": threshold,
            "rule": selected["rule"],
            "validation": val_ens,
        },
        "ensemble_candidates": search["candidates"],
    }
    _json_dump(EXP_DIR / "validation_metrics.json", report)
    _json_dump(EXP_DIR / "ensemble_meta.json", report["ensemble"])
    lines = [
        "EXPERIMENT 1 — 3-stage fine-tuning (validation only)",
        f"Eff time_s={eff['training_seconds']:.1f} stages={eff['stage_epochs']}",
        f"Mob time_s={mob['training_seconds']:.1f} stages={mob['stage_epochs']}",
        f"Ensemble w={selected['efficientnet_weight']}/{selected['mobilenet_weight']} thr={threshold}",
        f"Val ensemble acc={val_ens['accuracy']:.4f} sens={val_ens['sensitivity']:.4f} spec={val_ens['specificity']:.4f} f1={val_ens['f1']:.4f} auc={val_ens['auc']:.4f} pr={val_ens['pr_auc']:.4f}",
        "Test set was not loaded. PH2 was not used.",
    ]
    (EXP_DIR / "EXP1_VALIDATION_REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    logger.info("Experiment 1 finished. Stopping as requested.")


if __name__ == "__main__":
    main()
