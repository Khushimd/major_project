"""
Clean DICM melanoma-vs-benign training experiment.

Writes only under models/experiments/clean_dicm_melanoma_benign/.
Does not modify production models, the Flask app, the old 16,529-image
dataset, or Experiments 1-3. Test is used once after validation-only
selection. PH2 is not used.
"""

from __future__ import annotations

import hashlib
import json
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
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

import config
from gradcam.gradcam import GradCAM
from models.architecture import build_efficientnetb0, build_mobilenetv2, unfreeze_for_finetuning
from models.ensemble import combine_hybrid_probs
from preprocessing.augmentation import get_tf_data_augmentation_fn
from preprocessing.image_preprocess import ImagePreprocessor
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

logger = setup_logger("clean_dicm_exp")

DATA_DIR = config.PROCESSED_DATA_DIR / "dicm_clean_melanoma_benign"
OLD_DATA = config.PROCESSED_DATA_DIR / "melanoma_benign"
EXP_ROOT = config.BASE_DIR / "models" / "experiments" / "clean_dicm_melanoma_benign"
CACHE_DIR = EXP_ROOT / "preprocessed"
OLD_EXP = config.BASE_DIR / "models" / "experiments" / "internal_optimization"
PROTECTED = {
    config.EFFICIENTNET_PATH.resolve(),
    config.MOBILENET_PATH.resolve(),
    (config.MODELS_DIR / "efficientnetb0_melanoma_benign.keras").resolve(),
    (config.MODELS_DIR / "mobilenetv2_melanoma_benign.keras").resolve(),
    (config.MODELS_DIR / "melanoma_benign" / "efficientnetb0_melanoma_benign.keras").resolve(),
    (config.MODELS_DIR / "melanoma_benign" / "mobilenetv2_melanoma_benign.keras").resolve(),
    config.INFERENCE_CONFIG_PATH.resolve(),
}
EXPECTED = {
    "manifest": (11373, 6787, 4586),
    "train": (7961, 4747, 3214),
    "val": (1706, 1016, 690),
    "test": (1706, 1024, 682),
}
STAGES = (
    {"name": "head", "lr": 1e-3, "epochs": 25, "unfreeze": 0, "batch": 16},
    {"name": "partial", "lr": 1e-4, "epochs": 12, "unfreeze": 20, "batch": 8},
    {"name": "final", "lr": 1e-5, "epochs": 12, "unfreeze": 20, "batch": 8},
)
LABEL_SMOOTHING = 0.05
SEED = 42
WEIGHT_GRID = [round(x, 1) for x in np.arange(0.0, 1.01, 0.1)]
THRESHOLD_GRID = [round(x, 2) for x in np.arange(0.15, 0.81, 0.05)]
EMPTY = {"", "nan", "none", "na", "null"}
EXPERIMENTS = (
    {
        "key": "experiment_A_bce_smoothing",
        "name": "A",
        "label": "Experiment A: BCE + label smoothing 0.05, no class weighting",
        "use_class_weight": False,
    },
    {
        "key": "experiment_B_weighted_bce_smoothing",
        "name": "B",
        "label": "Experiment B: class-weighted BCE + label smoothing 0.05 (train split only)",
        "use_class_weight": True,
    },
)


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
    if resolved in PROTECTED:
        raise RuntimeError(f"Refusing to write protected path: {path}")
    if OLD_DATA.resolve() in resolved.parents or resolved == OLD_DATA.resolve():
        raise RuntimeError(f"Refusing to write into the old dataset: {path}")
    if OLD_EXP.resolve() in resolved.parents or resolved == OLD_EXP.resolve():
        raise RuntimeError(f"Refusing to write into old experiments: {path}")
    if config.MODELS_DIR.resolve() in resolved.parents and EXP_ROOT.resolve() not in resolved.parents and resolved != EXP_ROOT.resolve():
        if resolved.parent == config.MODELS_DIR.resolve() or "saved" in {p.name for p in resolved.parents}:
            if EXP_ROOT.resolve() not in resolved.parents:
                if "experiments" not in str(resolved).replace("\\", "/"):
                    raise RuntimeError(f"Refusing to write near production models: {path}")


def _clean(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in EMPTY else text


def _idset(series: pd.Series) -> set[str]:
    return {_clean(v) for v in series if _clean(v)}


def _counts(df: pd.DataFrame) -> tuple[int, int, int]:
    return len(df), int((df["label"] == 1).sum()), int((df["label"] == 0).sum())


def _overlap(a: set[str], b: set[str]) -> int:
    return len(a & b)


def snapshot_production() -> dict:
    out = {}
    for path in sorted(PROTECTED):
        if path.is_file():
            out[str(path)] = {"sha256": _sha256(path), "size": path.stat().st_size}
    for name in ("train.csv", "val.csv", "test.csv", "manifest.csv"):
        path = OLD_DATA / name
        if path.is_file():
            out[str(path.resolve())] = {"sha256": _sha256(path), "size": path.stat().st_size}
    return out


def assert_production_unchanged(before: dict) -> None:
    after = snapshot_production()
    for key, meta in before.items():
        if after.get(key) != meta:
            raise RuntimeError(f"Protected file changed during experiment: {key}")


def verify_dataset() -> dict:
    required = [
        "manifest.csv",
        "train.csv",
        "val.csv",
        "test.csv",
        "split_detail.json",
        "leakage_report.json",
        "dataset_report.json",
        "CLEAN_DATASET_README.txt",
        "CLEAN_DATASET_PREPARATION_REPORT.txt",
    ]
    missing = [name for name in required if not (DATA_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Clean dataset files missing: {missing}")

    frames = {name: pd.read_csv(DATA_DIR / f"{name}.csv") for name in ("manifest", "train", "val", "test")}
    mismatches = {}
    for name, df in frames.items():
        got = _counts(df)
        if got != EXPECTED[name]:
            mismatches[name] = {"expected": EXPECTED[name], "got": got}
    if mismatches:
        raise RuntimeError(f"STOP: clean split counts do not match. {mismatches}")

    for name, df in frames.items():
        if set(df["label"].unique()) - {0, 1}:
            raise RuntimeError(f"{name} has labels other than 0/1")
        if "source" in df.columns and set(df["source"].astype(str).unique()) - {"ISIC-DICM-17K"}:
            raise RuntimeError(f"{name} contains a non-DICM source")
        if "diagnosis" in df.columns:
            mel = df.loc[df["label"] == 1, "diagnosis"].map(_clean)
            if set(mel.unique()) - {"melanoma"}:
                raise RuntimeError(f"{name} melanoma rows have unexpected diagnosis")

    leak = {}
    for col in ("image_id", "lesion_id", "patient_id"):
        train_ids = _idset(frames["train"][col])
        val_ids = _idset(frames["val"][col])
        test_ids = _idset(frames["test"][col])
        leak[col] = {
            "train_val": _overlap(train_ids, val_ids),
            "train_test": _overlap(train_ids, test_ids),
            "val_test": _overlap(val_ids, test_ids),
        }
        if any(leak[col].values()):
            raise RuntimeError(f"STOP: leakage on {col}: {leak[col]}")

    return {
        "path": str(DATA_DIR),
        "counts": {name: {"total": n, "melanoma": m, "benign": b} for name, (n, m, b) in EXPECTED.items()},
        "leakage": leak,
        "sources": "ISIC-DICM-17K only",
    }


def print_safety_banner(verified: dict, class_weights: dict | None) -> None:
    lines = [
        "DATASET:",
        f"path {verified['path']}",
        f"total {verified['counts']['manifest']['total']}",
        f"melanoma {verified['counts']['manifest']['melanoma']}",
        f"benign {verified['counts']['manifest']['benign']}",
        "",
        "SPLITS:",
        f"train {verified['counts']['train']}",
        f"validation {verified['counts']['val']}",
        f"test {verified['counts']['test']}",
        "",
        "MODELS:",
        "EfficientNetB0",
        "MobileNetV2",
        "",
        "EXPERIMENTS:",
        "A = BCE + label smoothing",
        "B = class-weighted BCE + label smoothing",
        f"class_weights_from_train_only {class_weights}",
        "",
        "PRODUCTION MODELS MODIFIED:",
        "NO",
        "",
        "OLD DATASET MODIFIED:",
        "NO",
        "",
        "OLD TEST SET USED FOR TUNING:",
        "NO",
        "",
        "PH2 USED:",
        "NO",
        "",
        "NEW TEST SET USED FOR TUNING:",
        "NO (frozen until after validation selection)",
        "",
        "Safety checks passed. Training may begin.",
    ]
    text = "\n".join(lines) + "\n"
    print(text)
    logger.info("Safety preflight passed")
    (EXP_ROOT / "configs" / "SAFETY_PREFLIGHT.txt").write_text(text, encoding="utf-8")


MAX_SIDE_BEFORE_HAIR = 1024


def _downscale_for_hair(rgb):
    h, w = rgb.shape[:2]
    longest = max(h, w)
    if longest <= MAX_SIDE_BEFORE_HAIR:
        return rgb
    scale = MAX_SIDE_BEFORE_HAIR / float(longest)
    return cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _cache_one(payload: tuple[str, str]) -> str:
    src, dest_s = payload
    dest = Path(dest_s)
    if dest.is_file():
        return dest_s
    raw = cv2.imread(src, cv2.IMREAD_COLOR)
    if raw is None:
        raise FileNotFoundError(src)
    rgb = _downscale_for_hair(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
    processed = ImagePreprocessor(
        target_size=(224, 224), apply_clahe=False, apply_hair_removal=True
    ).process_for_model(rgb)
    dest.parent.mkdir(parents=True, exist_ok=True)
    jpeg = np.clip(processed * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(
        str(dest),
        cv2.cvtColor(jpeg, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), 95],
    )
    return dest_s


def cache_split(df: pd.DataFrame, label: str) -> pd.DataFrame:
    out = df.copy()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for row in out.itertuples(index=False):
        dest = CACHE_DIR / f"{getattr(row, 'image_id')}.jpg"
        jobs.append((str(getattr(row, "image_path")), str(dest)))
    already = sum(Path(dest).is_file() for _, dest in jobs)
    logger.info("Using preprocess cache for %s (%d / %d already present).", label, already, len(jobs))
    for i, job in enumerate(jobs, start=1):
        _cache_one(job)
        if i % 250 == 0 or i == len(jobs):
            logger.info("Preprocessed %s %d / %d", label, i, len(out))
            _json_dump(
                EXP_ROOT / "logs" / "status.json",
                {"stage": "preprocess", "split": label, "done": i, "total": len(out)},
            )
    out["cached_path"] = [dest for _, dest in jobs]
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


def train_one(builder, name: str, run_dir: Path, train_df, val_df, save_path: Path, class_weight) -> dict:
    _refuse_protected(save_path)
    ckpt_dir = run_dir / "checkpoints" / name
    curve_dir = run_dir / "training_curves"
    log_dir = EXP_ROOT / "logs" / run_dir.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    curve_dir.mkdir(parents=True, exist_ok=True)
    logger.info("==== Training %s in %s ====", name, run_dir.name)
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
        logger.info("Stage %s lr=%s batch=%s class_weight=%s", stage["name"], stage["lr"], stage["batch"], class_weight)
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=stage["epochs"],
            callbacks=local_callbacks(f"{run_dir.name}_{name}_{stage['name']}", ckpt_dir, log_dir),
            class_weight=class_weight,
            verbose=1,
        )
        for key, values in history.history.items():
            hist.setdefault(key, []).extend(list(values))
        stage_epochs.append({"stage": stage["name"], "epochs_run": len(history.history.get("loss", []))})
        ckpt = ckpt_dir / f"{run_dir.name}_{name}_{stage['name']}_best.keras"
        if ckpt.exists():
            model = tf.keras.models.load_model(str(ckpt), compile=False)
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
        "best_epoch_val_pr_auc": best_epoch_from_history(hist, "val_pr_auc"),
        "stage_epochs": stage_epochs,
        "training_seconds": time.time() - t0,
        "y_val_true": y_true,
        "y_val_prob": y_prob,
    }


def select_threshold(y_true, y_prob) -> dict:
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
    picked = select_threshold(y_true, ens)
    operating = evaluate_predictions(y_true, ens, threshold=picked["threshold"])
    return {
        "weight_candidates": rows,
        "selected_weights": {
            "efficientnet_weight": best_w["efficientnet_weight"],
            "mobilenet_weight": best_w["mobilenet_weight"],
            "selection_rule": "max validation ROC-AUC, then PR-AUC; weights searched at 0.1 steps",
            "roc_auc": best_w["roc_auc"],
            "pr_auc": best_w["pr_auc"],
        },
        "val_ensemble_prob": ens,
        "threshold_search": picked,
        "operating": operating,
    }


def run_experiment(spec: dict, train_df, val_df, class_weight) -> dict:
    run_dir = EXP_ROOT / spec["key"]
    run_dir.mkdir(parents=True, exist_ok=True)
    marker = run_dir / "run_complete.json"
    if marker.exists() and (run_dir / "validation_predictions.csv").exists():
        logger.info("Skipping %s; already complete.", spec["key"])
        return json.loads(marker.read_text(encoding="utf-8"))

    used_weights = class_weight if spec["use_class_weight"] else None
    _json_dump(
        run_dir / "configuration.json",
        {
            "experiment": spec["key"],
            "label": spec["label"],
            "seed": SEED,
            "dataset": str(DATA_DIR),
            "counts": {k: {"total": n, "melanoma": m, "benign": b} for k, (n, m, b) in EXPECTED.items() if k != "manifest"},
            "preprocessing": "BGR->RGB; downscale max side 1024 if larger (required because many DICM images are ~30MP); DullRazor hair removal; light denoise; resize 224; /255; ImageNet preprocess inside model; no CLAHE; no lesion crop",
            "augmentation": "get_tf_data_augmentation_fn: flip, small rotation/zoom/translation, brightness/contrast, mild saturation/hue, light noise, mild blur, occasional synthetic hair; train only",
            "optimizer": "AdamW with staged LR 1e-3 / 1e-4 / 1e-5",
            "loss": "BinaryCrossentropy(label_smoothing=0.05)",
            "class_weight": used_weights,
            "finetune": STAGES,
            "test_used_for_tuning": False,
            "ph2_used": False,
        },
    )
    eff_path = run_dir / "efficientnetb0.keras"
    mob_path = run_dir / "mobilenetv2.keras"
    t0 = time.time()
    eff = train_one(build_efficientnetb0, "efficientnetb0", run_dir, train_df, val_df, eff_path, used_weights)
    _json_dump(EXP_ROOT / "logs" / "status.json", {"stage": f"{spec['key']}_efficientnet_done", "seconds": eff["training_seconds"]})
    mob = train_one(build_mobilenetv2, "mobilenetv2", run_dir, train_df, val_df, mob_path, used_weights)

    y_val = np.asarray(eff["y_val_true"]).astype(int)
    search = search_ensemble(y_val, eff["y_val_prob"], mob["y_val_prob"])
    weights = search["selected_weights"]
    threshold = float(search["threshold_search"]["threshold"])
    rule = search["threshold_search"]["rule"]
    val_eff = evaluate_predictions(y_val, eff["y_val_prob"], threshold=threshold)
    val_mob = evaluate_predictions(y_val, mob["y_val_prob"], threshold=threshold)
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
    pred_df[["image_id", "label"]].to_csv(run_dir / "validation_labels.csv", index=False)
    pd.DataFrame(search["weight_candidates"]).to_csv(run_dir / "ensemble_weight_search.csv", index=False)
    save_threshold_table(search["threshold_search"]["table"], run_dir / "val_threshold_table.csv")
    np.save(run_dir / "validation_ensemble_scores.npy", np.asarray(search["val_ensemble_prob"]).ravel())
    np.save(run_dir / "validation_labels.npy", y_val)

    payload = {
        "key": spec["key"],
        "label": spec["label"],
        "evaluated_on": "validation_only",
        "test_used": False,
        "ph2_used": False,
        "wall_seconds": time.time() - t0,
        "class_weight": used_weights,
        "efficientnet": {
            "path": eff["path"],
            "training_seconds": eff["training_seconds"],
            "best_epoch_val_pr_auc": eff["best_epoch_val_pr_auc"],
            "stage_epochs": eff["stage_epochs"],
            "validation_at_selected_threshold": val_eff,
        },
        "mobilenet": {
            "path": mob["path"],
            "training_seconds": mob["training_seconds"],
            "best_epoch_val_pr_auc": mob["best_epoch_val_pr_auc"],
            "stage_epochs": mob["stage_epochs"],
            "validation_at_selected_threshold": val_mob,
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
    _json_dump(run_dir / "selected_threshold.json", {"threshold": threshold, "rule": rule, "validation": val_ens})
    _json_dump(marker, payload)
    logger.info(
        "%s ensemble auc=%.4f pr=%.4f acc=%.4f sens=%.4f spec=%.4f thr=%.2f w=%.2f/%.2f",
        spec["key"],
        val_ens["auc"],
        val_ens["pr_auc"],
        val_ens["accuracy"],
        val_ens["sensitivity"],
        val_ens["specificity"],
        threshold,
        weights["efficientnet_weight"],
        weights["mobilenet_weight"],
    )
    return payload


def select_winner(results: list[dict]) -> dict:
    winner = max(
        results,
        key=lambda r: (
            r["ensemble"]["validation"]["auc"],
            r["ensemble"]["validation"]["pr_auc"],
            r["ensemble"]["validation"]["f1"],
        ),
    )
    selection = {
        "selected_experiment": winner["key"],
        "selection_rule": "validation ensemble ROC-AUC, then PR-AUC, then F1 at the validation-selected threshold",
        "test_used": False,
        "winner": {
            "key": winner["key"],
            "efficientnet_weight": winner["ensemble"]["efficientnet_weight"],
            "mobilenet_weight": winner["ensemble"]["mobilenet_weight"],
            "threshold": winner["ensemble"]["threshold"],
            "threshold_rule": winner["ensemble"]["threshold_rule"],
            "validation": winner["ensemble"]["validation"],
        },
        "candidates": [
            {
                "key": r["key"],
                "roc_auc": r["ensemble"]["validation"]["auc"],
                "pr_auc": r["ensemble"]["validation"]["pr_auc"],
                "f1": r["ensemble"]["validation"]["f1"],
                "threshold": r["ensemble"]["threshold"],
                "weights": [r["ensemble"]["efficientnet_weight"], r["ensemble"]["mobilenet_weight"]],
            }
            for r in results
        ],
    }
    _json_dump(EXP_ROOT / "evaluation" / "validation_selection.json", selection)
    return selection


def _quiet_metrics(y_true, y_prob, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob).astype(float).ravel()
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

    spec_den = tn + fp
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / spec_den) if spec_den else 0.0,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }


def bootstrap_ci(y_true, y_prob, threshold: float, n_boot: int = 2000) -> dict:
    rng = np.random.default_rng(SEED)
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob).astype(float).ravel()
    n = len(y_true)
    keys = ["accuracy", "sensitivity", "specificity", "precision", "f1", "auc", "pr_auc"]
    store = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        m = _quiet_metrics(yt, y_prob[idx], threshold)
        for k in keys:
            store[k].append(m[k])
    out = {}
    for k, values in store.items():
        arr = np.asarray(values, dtype=float)
        out[k] = {
            "mean": float(arr.mean()),
            "ci95_low": float(np.percentile(arr, 2.5)),
            "ci95_high": float(np.percentile(arr, 97.5)),
        }
    return out


def curve_payloads(y_true, y_prob) -> tuple[dict, dict]:
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    prec, rec, pthr = precision_recall_curve(y_true, y_prob)
    roc = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thr.tolist()}
    pr = {"precision": prec.tolist(), "recall": rec.tolist(), "thresholds": pthr.tolist()}
    return roc, pr


def evaluate_frozen_test(winner: dict, test_df: pd.DataFrame) -> dict:
    eval_dir = EXP_ROOT / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    marker = eval_dir / "frozen_test_metrics.json"
    if marker.exists() and (eval_dir / "test_predictions.csv").exists():
        logger.info("Frozen test evaluation already exists; not rerunning.")
        return json.loads(marker.read_text(encoding="utf-8"))

    run_dir = EXP_ROOT / winner["selected_experiment"]
    eff = tf.keras.models.load_model(str(run_dir / "efficientnetb0.keras"), compile=False)
    test_ds = make_ds(test_df, shuffle=False, augment=False, batch=8)
    y_true, p_eff = _predict_dataset(eff, test_ds)
    del eff
    tf.keras.backend.clear_session()
    mob = tf.keras.models.load_model(str(run_dir / "mobilenetv2.keras"), compile=False)
    y_true2, p_mob = _predict_dataset(mob, test_ds)
    del mob
    tf.keras.backend.clear_session()
    if not np.array_equal(np.asarray(y_true), np.asarray(y_true2)):
        raise RuntimeError("Test label order mismatch between models.")

    w = winner["winner"]
    ens = combine_hybrid_probs(
        p_eff,
        p_mob,
        weights=(w["efficientnet_weight"], w["mobilenet_weight"]),
        method="soft_voting",
    )
    threshold = float(w["threshold"])
    metrics = evaluate_predictions(y_true, ens, threshold=threshold)
    ci = bootstrap_ci(y_true, ens, threshold)
    roc, pr = curve_payloads(y_true, ens)
    y_pred = (np.asarray(ens) >= threshold).astype(int)
    plot_confusion_matrix(y_true, y_pred, eval_dir / "test_cm.png", labels=["Benign", "Melanoma"])
    plot_roc_curve(y_true, ens, eval_dir / "test_roc.png", "Clean DICM ensemble")
    plot_pr_curve(y_true, ens, eval_dir / "test_pr.png", "Clean DICM ensemble")
    _json_dump(eval_dir / "test_roc_curve.json", roc)
    _json_dump(eval_dir / "test_pr_curve.json", pr)
    pred_df = pd.DataFrame(
        {
            "image_id": test_df["image_id"].astype(str).values,
            "label": np.asarray(y_true).astype(int),
            "efficientnet_prob": np.asarray(p_eff).ravel(),
            "mobilenet_prob": np.asarray(p_mob).ravel(),
            "ensemble_prob": np.asarray(ens).ravel(),
            "predicted": y_pred,
        }
    )
    pred_df.to_csv(eval_dir / "test_predictions.csv", index=False)
    pred_df[["image_id", "label"]].to_csv(eval_dir / "test_labels.csv", index=False)
    np.save(eval_dir / "test_ensemble_scores.npy", np.asarray(ens).ravel())
    np.save(eval_dir / "test_labels.npy", np.asarray(y_true).astype(int))
    payload = {
        "evaluated_on": "frozen_clean_dicm_test_once",
        "selected_experiment": winner["selected_experiment"],
        "efficientnet_weight": w["efficientnet_weight"],
        "mobilenet_weight": w["mobilenet_weight"],
        "threshold": threshold,
        "threshold_rule": w["threshold_rule"],
        "metrics": metrics,
        "bootstrap_ci95": ci,
        "note": "These are model scores / AI probability estimates, not clinically calibrated probabilities.",
        "rerun_after_test": False,
    }
    _json_dump(marker, payload)
    _json_dump(eval_dir / "test_confusion.json", {"tp": metrics["tp"], "tn": metrics["tn"], "fp": metrics["fp"], "fn": metrics["fn"]})
    return payload


def qualitative_gradcam(winner: dict, test_df: pd.DataFrame) -> None:
    eval_dir = EXP_ROOT / "evaluation" / "gradcam"
    eval_dir.mkdir(parents=True, exist_ok=True)
    preds = pd.read_csv(EXP_ROOT / "evaluation" / "test_predictions.csv")
    merged = test_df.copy()
    merged["image_id"] = merged["image_id"].astype(str)
    preds["image_id"] = preds["image_id"].astype(str)
    merged = merged.merge(preds, on="image_id", suffixes=("", "_pred"))
    cases = {
        "true_positive": merged[(merged["label"] == 1) & (merged["predicted"] == 1)],
        "true_negative": merged[(merged["label"] == 0) & (merged["predicted"] == 0)],
        "false_positive": merged[(merged["label"] == 0) & (merged["predicted"] == 1)],
        "false_negative": merged[(merged["label"] == 1) & (merged["predicted"] == 0)],
    }
    run_dir = EXP_ROOT / winner["selected_experiment"]
    model = tf.keras.models.load_model(str(run_dir / "efficientnetb0.keras"), compile=False)
    cam = GradCAM(model)
    chosen = {}
    for name, subset in cases.items():
        if subset.empty:
            chosen[name] = None
            continue
        row = subset.sort_values("ensemble_prob", ascending=(name in {"true_negative", "false_negative"})).iloc[0]
        raw = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        cached = cv2.imread(str(row["cached_path"]), cv2.IMREAD_COLOR)
        cached_rgb = cv2.cvtColor(cached, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        batch = np.expand_dims(cached_rgb, 0)
        dest = eval_dir / f"{name}_{row['image_id']}.jpg"
        cam.generate(batch, rgb, save_path=dest)
        chosen[name] = {
            "image_id": str(row["image_id"]),
            "label": int(row["label"]),
            "predicted": int(row["predicted"]),
            "ensemble_score": float(row["ensemble_prob"]),
            "path": str(dest),
        }
    del model
    tf.keras.backend.clear_session()
    _json_dump(
        eval_dir / "gradcam_cases.json",
        {
            "model": "EfficientNetB0 from the validation-selected experiment",
            "interpretation": "Grad-CAM visualizes image regions contributing to the melanoma score. It does not prove that the model is focusing on the correct lesion boundary.",
            "used_for_model_selection": False,
            "cases": chosen,
        },
    )


def write_report(verified: dict, class_weights: dict, results: list[dict], winner: dict, test_eval: dict) -> None:
    ev = test_eval["metrics"]
    ci = test_eval["bootstrap_ci95"]
    lines = [
        "# Clean DICM Melanoma-vs-Benign Experiment",
        "",
        "This is an internal research experiment. It is **not clinically validated**.",
        "Reported scores are **AI probability estimates / model scores**, not clinically calibrated probabilities.",
        "This report does not claim state-of-the-art performance and does not compare against published papers.",
        "",
        "## 1. Dataset definition",
        "",
        "Source: `dataset/processed/dicm_clean_melanoma_benign/`.",
        "Melanoma (label=1): official DICM diagnosis == melanoma AND benign_malignant in {malignant, indeterminate/malignant}.",
        "Benign (label=0): diagnosis in the pre-specified clean benign list AND benign_malignant == benign.",
        "Official DICM class was not used. ISIC 2020 and HAM10000 were not added. Images were not deleted to rebalance.",
        "",
        "## 2. Dataset counts",
        "",
        f"- Total: {verified['counts']['manifest']['total']} (melanoma {verified['counts']['manifest']['melanoma']}, benign {verified['counts']['manifest']['benign']})",
        f"- Train: {verified['counts']['train']}",
        f"- Validation: {verified['counts']['val']}",
        f"- Test: {verified['counts']['test']}",
        "",
        "## 3. Leakage checks",
        "",
        "Non-empty patient IDs, lesion IDs, and image IDs have zero overlap among the new train, validation, and test splits.",
        f"Details: `{verified['leakage']}`.",
        "",
        "## 4. Preprocessing",
        "",
        "Read image, BGR to RGB, downscale to max side 1024 when larger (many DICM images are ~30MP,",
        "which makes full-resolution DullRazor intractable on CPU), then DullRazor-style hair removal,",
        "light denoising, resize 224x224, scale to [0,1].",
        "ImageNet preprocessing is applied inside EfficientNetB0 and MobileNetV2.",
        "CLAHE and aggressive lesion cropping were not used.",
        f"Cached copies were written only under `{CACHE_DIR}`.",
        "",
        "## 5. Augmentation",
        "",
        "Training only, via `get_tf_data_augmentation_fn()`: horizontal/vertical flip, small rotation, small zoom,",
        "small translation, moderate brightness/contrast, mild saturation/hue, light Gaussian noise, mild blur,",
        "and occasional synthetic hair. No extra augmentation was added.",
        "",
        "## 6. Model architecture",
        "",
        "Two independent ImageNet-initialized CNNs with the existing shared classification head:",
        "EfficientNetB0 and MobileNetV2, sigmoid melanoma score output.",
        "New files only; production weights were not loaded or overwritten.",
        "",
        "## 7. Training configuration",
        "",
        f"- Seed: {SEED}",
        "- Optimizer: AdamW, 3-stage schedule (head 1e-3 / partial 1e-4 / final 1e-5)",
        f"- Loss: BinaryCrossentropy(label_smoothing={LABEL_SMOOTHING})",
        f"- Experiment A class weights: none",
        f"- Experiment B class weights from train split only: {class_weights}",
        "- Checkpoint / early stopping monitor: validation PR-AUC",
        "- Test labels were not used for loss, weights, ensemble, or threshold",
        "",
        "## 8. Experiment A results (validation)",
        "",
    ]
    by_key = {r["key"]: r for r in results}
    a = by_key["experiment_A_bce_smoothing"]
    b = by_key["experiment_B_weighted_bce_smoothing"]

    def _blk(item):
        e = item["ensemble"]["validation"]
        return [
            f"- Ensemble weights: EfficientNet {item['ensemble']['efficientnet_weight']:.2f} / MobileNet {item['ensemble']['mobilenet_weight']:.2f}",
            f"- Threshold: {item['ensemble']['threshold']:.2f} ({item['ensemble']['threshold_rule']})",
            f"- Accuracy {e['accuracy']:.4f}; sensitivity {e['sensitivity']:.4f}; specificity {e['specificity']:.4f}",
            f"- Precision {e['precision']:.4f}; F1 {e['f1']:.4f}; ROC-AUC {e['auc']:.4f}; PR-AUC {e['pr_auc']:.4f}",
            f"- EfficientNet val ROC-AUC {item['efficientnet']['validation_at_selected_threshold']['auc']:.4f}",
            f"- MobileNet val ROC-AUC {item['mobilenet']['validation_at_selected_threshold']['auc']:.4f}",
        ]

    lines.extend(_blk(a))
    lines.extend(["", "## 9. Experiment B results (validation)", ""])
    lines.extend(_blk(b))
    lines.extend(
        [
            "",
            "## 10. Validation-based selection",
            "",
            f"Selected experiment: `{winner['selected_experiment']}`",
            f"Rule: {winner['selection_rule']}",
            f"Frozen ensemble weights: EfficientNet {winner['winner']['efficientnet_weight']:.2f} / MobileNet {winner['winner']['mobilenet_weight']:.2f}",
            f"Frozen threshold: {winner['winner']['threshold']:.2f}",
            f"Threshold rule: {winner['winner']['threshold_rule']}",
            "The old 0.75/0.25 weights and 0.30 threshold were not reused automatically.",
            "The test set was not used for this decision.",
            "",
            "## 11. Frozen test results",
            "",
            "One evaluation after architecture, training, checkpoints, ensemble weights, and threshold were frozen.",
            f"- Accuracy: {ev['accuracy']:.4f} (95% bootstrap CI {ci['accuracy']['ci95_low']:.4f}-{ci['accuracy']['ci95_high']:.4f})",
            f"- Sensitivity / recall: {ev['sensitivity']:.4f} ({ci['sensitivity']['ci95_low']:.4f}-{ci['sensitivity']['ci95_high']:.4f})",
            f"- Specificity: {ev['specificity']:.4f} ({ci['specificity']['ci95_low']:.4f}-{ci['specificity']['ci95_high']:.4f})",
            f"- Precision: {ev['precision']:.4f} ({ci['precision']['ci95_low']:.4f}-{ci['precision']['ci95_high']:.4f})",
            f"- F1-score: {ev['f1']:.4f} ({ci['f1']['ci95_low']:.4f}-{ci['f1']['ci95_high']:.4f})",
            f"- ROC-AUC: {ev['auc']:.4f} ({ci['auc']['ci95_low']:.4f}-{ci['auc']['ci95_high']:.4f})",
            f"- PR-AUC: {ev['pr_auc']:.4f} ({ci['pr_auc']['ci95_low']:.4f}-{ci['pr_auc']['ci95_high']:.4f})",
            "",
            "## 12. Confusion matrix",
            "",
            f"- TP {ev['tp']}",
            f"- TN {ev['tn']}",
            f"- FP {ev['fp']}",
            f"- FN {ev['fn']}",
            "",
            "## 13. ROC-AUC and PR-AUC",
            "",
            f"Frozen test ROC-AUC {ev['auc']:.4f}; PR-AUC {ev['pr_auc']:.4f}.",
            "Curve data: `models/experiments/clean_dicm_melanoma_benign/evaluation/test_roc_curve.json` and `test_pr_curve.json`.",
            "",
            "## 14. Limitations",
            "",
            "- Single-source DICM images; hospital / camera confounding may remain.",
            "- Missing patient IDs required lesion- or image-level grouping for some rows.",
            "- No external validation in this run.",
            "- Scores are not claimed to be clinically calibrated probabilities.",
            "- Grad-CAM is qualitative only and does not prove correct lesion localization.",
            "- This model was not deployed and does not replace the production system.",
            "",
            "## 15. Reproducibility",
            "",
            f"- Script: `scripts/train_clean_dicm_melanoma_benign.py`",
            f"- Seed: {SEED}",
            f"- Dataset: `{DATA_DIR}`",
            f"- Output: `{EXP_ROOT}`",
            "- Production models, old dataset CSVs, and Experiments 1-3 were not modified.",
            "- PH2 was not used.",
            "",
        ]
    )
    text = "\n".join(lines) + "\n"
    (EXP_ROOT / "CLEAN_DICM_EXPERIMENT_REPORT.md").write_text(text, encoding="utf-8")
    (EXP_ROOT / "reports" / "CLEAN_DICM_EXPERIMENT_REPORT.md").write_text(text, encoding="utf-8")
    print(text)


def main() -> None:
    EXP_ROOT.mkdir(parents=True, exist_ok=True)
    (EXP_ROOT / "configs").mkdir(parents=True, exist_ok=True)
    (EXP_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    (EXP_ROOT / "evaluation").mkdir(parents=True, exist_ok=True)
    (EXP_ROOT / "reports").mkdir(parents=True, exist_ok=True)

    before = snapshot_production()
    _json_dump(EXP_ROOT / "configs" / "production_snapshot_before.json", before)
    verified = verify_dataset()
    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "val.csv")
    y_train = train["label"].astype(int).to_numpy()
    balanced = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    class_weights = {0: float(balanced[0]), 1: float(balanced[1])}
    _json_dump(
        EXP_ROOT / "configs" / "experiment_config.json",
        {
            "experiment": "clean_dicm_melanoma_benign",
            "seed": SEED,
            "dataset": str(DATA_DIR),
            "reuse": {
                "architecture": "models.architecture build_efficientnetb0 / build_mobilenetv2 / shared head",
                "schedule": "Experiment 2 3-stage finetune",
                "augmentation": "preprocessing.augmentation.get_tf_data_augmentation_fn",
                "preprocess": "ImagePreprocessor.process_for_model after optional max-side-1024 downscale",
                "ensemble_search": "validation-only weighted soft voting",
                "threshold_search": "validation-only sensitivity/specificity tiers then max F1",
            },
            "not_reused": {
                "old_ensemble_weights": "0.75/0.25 not copied",
                "old_threshold": "0.30 not copied",
                "old_dataset": str(OLD_DATA),
                "production_models": [str(p) for p in PROTECTED],
            },
            "class_weights_train_only": class_weights,
            "weight_grid": WEIGHT_GRID,
            "threshold_grid": THRESHOLD_GRID,
            "test_used_for_tuning": False,
            "ph2_used": False,
        },
    )
    print_safety_banner(verified, class_weights)

    enable_mixed_precision()
    tf.keras.utils.set_random_seed(SEED)
    np.random.seed(SEED)

    logger.info("Caching train/validation preprocess only. Test images are not cached yet.")
    train = cache_split(train, "train")
    val = cache_split(val, "val")

    results = []
    try:
        for spec in EXPERIMENTS:
            _json_dump(EXP_ROOT / "logs" / "status.json", {"stage": spec["key"]})
            results.append(run_experiment(spec, train, val, class_weights))
            assert_production_unchanged(before)
    except Exception:
        text = traceback.format_exc()
        _json_dump(EXP_ROOT / "logs" / "status.json", {"stage": "failed", "error": text})
        raise

    winner = select_winner(results)
    _json_dump(EXP_ROOT / "logs" / "status.json", {"stage": "caching_test_after_selection"})
    logger.info("Validation selection frozen. Caching and evaluating the test set once.")
    test = cache_split(pd.read_csv(DATA_DIR / "test.csv"), "test")
    test_eval = evaluate_frozen_test(winner, test)
    qualitative_gradcam(winner, test)
    write_report(verified, class_weights, results, winner, test_eval)
    assert_production_unchanged(before)
    _json_dump(EXP_ROOT / "configs" / "production_snapshot_after.json", snapshot_production())
    _json_dump(EXP_ROOT / "logs" / "status.json", {"stage": "complete"})
    logger.info("Clean DICM experiment finished. Production and old experiments were not modified.")


if __name__ == "__main__":
    main()
