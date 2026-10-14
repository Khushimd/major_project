"""
Train EfficientNetB0 + MobileNetV2 on the frozen ISIC-DICM-17K split.

Writes only to models/saved/isic_dicm_17k/ and models/checkpoints/isic_dicm_17k/.
Does not overwrite production weights or change ACTIVE_DATASET.
"""

from __future__ import annotations

import json
import sys
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
from sklearn.metrics import brier_score_loss

import config
from gradcam.gradcam import GradCAM, generate_gradcam_if_malignant
from models.architecture import (
    FocalLoss,
    build_efficientnetb0,
    build_mobilenetv2,
    compile_model,
    unfreeze_for_finetuning,
)
from models.ensemble import combine_hybrid_probs
from preprocessing.image_preprocess import ImagePreprocessor, load_and_preprocess
from training.callbacks import get_callbacks
from training.dataset_loader import MelanomaDataset
from training.evaluate import (
    best_epoch_from_history,
    evaluate_predictions,
    find_optimal_threshold,
    full_evaluation_report,
    plot_training_curves,
    save_threshold_table,
    threshold_table,
)
from training.train import _predict_dataset, enable_mixed_precision
from utils.logger import setup_logger

logger = setup_logger("train_dicm")

EXP_DIR = config.MODELS_DIR / "isic_dicm_17k"
CKPT_DIR = config.CHECKPOINTS_DIR / "isic_dicm_17k"
CACHE_DIR = config.ISIC_DICM_SPLIT_DIR / "preprocessed"
PLOT_DIR = EXP_DIR / "plots"
WEIGHT_GRID = [round(x, 2) for x in np.arange(0.10, 0.91, 0.05)]


def _json_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_jsonable), encoding="utf-8")


def _jsonable(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _assert_frozen_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    expected = {
        "train": (11859, 5875),
        "val": (2616, 1334),
        "test": (2585, 1321),
    }
    for name, df in (("train", train), ("val", val), ("test", test)):
        n, mel = expected[name]
        got_n, got_mel = len(df), int((df["label"] == 1).sum())
        if (got_n, got_mel) != (n, mel):
            raise ValueError(
                f"Frozen {name} split changed: expected {n}/{mel} melanoma, got {got_n}/{got_mel}. "
                "Do not regenerate dataset/processed/isic_dicm_17k splits."
            )


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


def cached_paths(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, row in enumerate(out.itertuples(index=False), start=1):
        image_id = str(getattr(row, "image_id"))
        dest = CACHE_DIR / f"{image_id}.jpg"
        _cache_row(str(getattr(row, "image_path")), dest)
        paths.append(str(dest))
        if i % 500 == 0:
            logger.info("Preprocessed %d / %d images", i, len(out))
    out["image_path"] = paths
    return out


def make_ds(loader: MelanomaDataset, df: pd.DataFrame, *, shuffle: bool, augment: bool, batch: int):
    return loader.make_tf_dataset(
        df,
        batch_size=batch,
        shuffle=shuffle,
        augment=augment,
        inference_preprocess=False,
    )


def train_one(
    builder,
    name: str,
    train_df,
    val_df,
    loader,
    save_path: Path,
    batch: int,
    ft_batch: int,
) -> dict:
    logger.info("==== Training %s ====", name)
    train_ds = make_ds(loader, train_df, shuffle=True, augment=True, batch=batch)
    val_ds = make_ds(loader, val_df, shuffle=False, augment=False, batch=batch)

    model = builder(trainable_base=False)
    model = compile_model(model, learning_rate=config.LEARNING_RATE, use_focal_loss=False)
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        callbacks=get_callbacks(f"{name}_head", checkpoint_dir=CKPT_DIR),
        verbose=1,
    )
    hist = {k: list(v) for k, v in history1.history.items()}
    phase1 = CKPT_DIR / f"{name}_head_best.keras"
    if phase1.exists():
        model = tf.keras.models.load_model(
            str(phase1), custom_objects={"FocalLoss": FocalLoss}, compile=False
        )

    model = unfreeze_for_finetuning(model, n_unfreeze_layers=config.FINE_TUNE_UNFREEZE_LAYERS)
    model = compile_model(model, learning_rate=config.FINE_TUNE_LR, use_focal_loss=False)
    ft_train = make_ds(loader, train_df, shuffle=True, augment=True, batch=ft_batch)
    ft_val = make_ds(loader, val_df, shuffle=False, augment=False, batch=ft_batch)
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
    model.save(str(save_path))
    plot_training_curves(hist, PLOT_DIR, name)
    _json_dump(EXP_DIR / f"{name}_history.json", hist)

    y_val_true, y_val_prob = _predict_dataset(model, ft_val)
    val_metrics = evaluate_predictions(y_val_true, y_val_prob, threshold=0.5)
    full_evaluation_report(y_val_true, y_val_prob, f"val_{name}", PLOT_DIR, threshold=0.5)
    del model
    tf.keras.backend.clear_session()
    return {
        "path": str(save_path),
        "checkpoint": str(best),
        "history": hist,
        "best_epoch": best_epoch_from_history(hist),
        "y_val_true": y_val_true,
        "y_val_prob": y_val_prob,
        "val_metrics_at_0.5": val_metrics,
    }


def search_ensemble(y_true, p_eff, p_mob) -> dict:
    rows = []
    best = None
    for w_eff in WEIGHT_GRID:
        w_mob = round(1.0 - w_eff, 2)
        ens = combine_hybrid_probs(p_eff, p_mob, weights=(w_eff, w_mob), method="soft_voting")
        thr, picked = find_optimal_threshold(y_true, ens, metric="f1")
        row = {
            "efficientnet_weight": w_eff,
            "mobilenet_weight": w_mob,
            "threshold": thr,
            "f1": picked["f1"],
            "recall": picked["recall"],
            "specificity": picked["specificity"],
            "precision": picked["precision"],
            "accuracy": picked["accuracy"],
            "pr_auc": picked["pr_auc"],
            "auc": picked["auc"],
        }
        rows.append(row)
        key = (row["f1"], row["recall"], row["specificity"])
        if best is None or key > best[0]:
            best = (key, row, ens)
    return {"candidates": rows, "selected": best[1], "val_ensemble_prob": best[2]}


def evaluate_split(name: str, y_true, y_prob, threshold: float) -> dict:
    report = full_evaluation_report(y_true, y_prob, name, PLOT_DIR, threshold=threshold)
    return report["metrics"]


def verify_gradcam(eff_path: Path, test_df: pd.DataFrame, y_prob: np.ndarray) -> dict:
    model = tf.keras.models.load_model(
        str(eff_path), custom_objects={"FocalLoss": FocalLoss}, compile=False
    )
    out_dir = EXP_DIR / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    labels = test_df["label"].astype(int).to_numpy()
    paths = test_df["image_path"].astype(str).to_numpy()
    ids = test_df["image_id"].astype(str).to_numpy()
    picks = []
    mel = np.where(labels == 1)[0]
    ben = np.where(labels == 0)[0]
    if len(mel):
        picks.append(int(mel[np.argmax(y_prob[mel])]))
        picks.append(int(mel[0]))
    if len(ben):
        picks.append(int(ben[np.argmin(y_prob[ben])]))
    cam = GradCAM(model)
    for idx in picks:
        original, processed, _ = load_and_preprocess(paths[idx])
        batch = np.expand_dims(processed.astype(np.float32), axis=0)
        pred = "Malignant" if y_prob[idx] >= 0.5 else "Benign"
        dest = out_dir / f"{ids[idx]}_{pred.lower()}_overlay.jpg"
        try:
            if pred == "Malignant":
                info = generate_gradcam_if_malignant(pred, model, batch, original, save_path=dest)
                ok = bool(info.get("generated")) and dest.exists()
            else:
                cam.generate(batch, original, save_path=dest)
                ok = dest.exists()
            results.append(
                {
                    "image_id": ids[idx],
                    "true_label": int(labels[idx]),
                    "ai_estimated_probability": float(y_prob[idx]),
                    "prediction": pred,
                    "overlay": str(dest) if dest.exists() else None,
                    "ok": ok,
                    "note": "Heatmap is model attention, not a confirmed cancer boundary.",
                }
            )
        except Exception as exc:
            results.append({"image_id": ids[idx], "ok": False, "error": str(exc)})
    del model
    tf.keras.backend.clear_session()
    return {
        "status": "ok" if results and all(r.get("ok") for r in results) else "partial",
        "cases": results,
    }


def demo_100(test_df: pd.DataFrame, y_true, y_prob, threshold: float) -> dict:
    rng = np.random.default_rng(config.SEED)
    y_true = np.asarray(y_true).astype(int)
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
    metrics = evaluate_predictions(y_true[chosen], np.asarray(y_prob)[chosen], threshold=threshold)
    subset_path = EXP_DIR / "demo_100_from_test.csv"
    subset.assign(ai_estimated_probability=np.asarray(y_prob)[chosen]).to_csv(subset_path, index=False)
    payload = {
        "label": "100-image demonstration subset from the independent test set",
        "not_the_complete_evaluation": True,
        "n": int(len(chosen)),
        "melanoma": int(n_pos),
        "non_melanoma": int(n_neg),
        "threshold": threshold,
        "metrics": metrics,
        "csv": str(subset_path),
    }
    _json_dump(EXP_DIR / "demo_100_metrics.json", payload)
    return payload


def main() -> None:
    if config.ACTIVE_DATASET != "combined":
        raise RuntimeError("ACTIVE_DATASET must stay 'combined' during this experiment.")
    for path in (config.EFFICIENTNET_PATH, config.MOBILENET_PATH):
        if not path.exists():
            logger.warning("Production file missing (not overwritten): %s", path)

    enable_mixed_precision()
    tf.keras.utils.set_random_seed(config.SEED)
    config.USE_FOCAL_LOSS = False
    config.BALANCE_MODE = "none"
    config.MALIGNANT_OVERSAMPLE_FACTOR = 1
    config.BATCH_SIZE = min(int(config.BATCH_SIZE), 16)
    config.FINE_TUNE_EPOCHS = min(int(config.FINE_TUNE_EPOCHS), 15)

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    split_dir = config.ISIC_DICM_SPLIT_DIR
    train = pd.read_csv(split_dir / "train.csv")
    val = pd.read_csv(split_dir / "val.csv")
    test = pd.read_csv(split_dir / "test.csv")
    _assert_frozen_splits(train, val, test)
    logger.info("Using frozen DICM splits only. SIIM/HAM are not loaded.")

    test_original = test.copy()
    logger.info("Caching inference-matched preprocess (hair + denoise + 224 + /255).")
    train = cached_paths(train)
    val = cached_paths(val)
    test = cached_paths(test)

    loader = MelanomaDataset(dataset_key="isic_dicm_17k")
    batch = int(config.BATCH_SIZE)
    ft_batch = max(8, min(16, batch // 2 or 8))

    eff_path = EXP_DIR / "efficientnetb0_melanoma.keras"
    mob_path = EXP_DIR / "mobilenetv2_melanoma.keras"
    if str(eff_path.resolve()) == str(config.EFFICIENTNET_PATH.resolve()):
        raise RuntimeError("Refusing to overwrite production EfficientNet.")
    if str(mob_path.resolve()) == str(config.MOBILENET_PATH.resolve()):
        raise RuntimeError("Refusing to overwrite production MobileNet.")

    eff = train_one(build_efficientnetb0, "dicm_efficientnetb0", train, val, loader, eff_path, batch, ft_batch)
    mob = train_one(build_mobilenetv2, "dicm_mobilenetv2", train, val, loader, mob_path, batch, ft_batch)

    y_val = np.asarray(eff["y_val_true"]).astype(int)
    search = search_ensemble(y_val, eff["y_val_prob"], mob["y_val_prob"])
    selected = search["selected"]
    w_eff = float(selected["efficientnet_weight"])
    w_mob = float(selected["mobilenet_weight"])
    threshold = float(selected["threshold"])
    val_ens = np.asarray(search["val_ensemble_prob"])
    logger.info("Frozen ensemble: Eff=%.2f Mob=%.2f threshold=%.2f (validation only)", w_eff, w_mob, threshold)

    val_eff = evaluate_split("val_efficientnetb0", y_val, eff["y_val_prob"], threshold)
    val_mob = evaluate_split("val_mobilenetv2", y_val, mob["y_val_prob"], threshold)
    val_ens_metrics = evaluate_split("val_ensemble", y_val, val_ens, threshold)
    rows = threshold_table(y_val, val_ens)
    save_threshold_table(rows, EXP_DIR / "val_threshold_table.csv")

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(val_ens, y_val)
    val_cal = iso.predict(val_ens)
    frac_pos, mean_pred = calibration_curve(y_val, val_ens, n_bins=10, strategy="uniform")
    calibration = {
        "method": "isotonic",
        "fitted_on": "validation_ensemble_raw_probabilities",
        "raw_brier": float(brier_score_loss(y_val, val_ens)),
        "calibrated_brier": float(brier_score_loss(y_val, val_cal)),
        "reliability_raw": {"mean_predicted": mean_pred.tolist(), "fraction_positive": frac_pos.tolist()},
        "note": "AI-estimated probability is not medical certainty. Threshold stays on raw scores.",
    }

    test_ds = make_ds(loader, test, shuffle=False, augment=False, batch=ft_batch)
    eff_model = tf.keras.models.load_model(str(eff_path), custom_objects={"FocalLoss": FocalLoss}, compile=False)
    y_test, p_eff = _predict_dataset(eff_model, test_ds)
    del eff_model
    tf.keras.backend.clear_session()
    mob_model = tf.keras.models.load_model(str(mob_path), custom_objects={"FocalLoss": FocalLoss}, compile=False)
    _, p_mob = _predict_dataset(mob_model, test_ds)
    del mob_model
    tf.keras.backend.clear_session()
    p_ens = combine_hybrid_probs(p_eff, p_mob, weights=(w_eff, w_mob), method="soft_voting")
    p_ens_cal = iso.predict(p_ens)

    test_eff = evaluate_split("test_efficientnetb0", y_test, p_eff, threshold)
    test_mob = evaluate_split("test_mobilenetv2", y_test, p_mob, threshold)
    test_ens = evaluate_split("test_ensemble", y_test, p_ens, threshold)
    test_ens_cal = evaluate_predictions(y_test, p_ens_cal, threshold=threshold)

    demo = demo_100(test_original, y_test, p_ens, threshold)
    gradcam = verify_gradcam(eff_path, test_original, p_ens)

    experiment = {
        "dataset": "isic_dicm_17k",
        "active_dataset_unchanged": config.ACTIVE_DATASET,
        "production_models_untouched": [
            str(config.EFFICIENTNET_PATH),
            str(config.MOBILENET_PATH),
        ],
        "splits": {
            "train": str(split_dir / "train.csv"),
            "val": str(split_dir / "val.csv"),
            "test": str(split_dir / "test.csv"),
        },
        "preprocessing": "process_for_model: hair removal + denoise + resize 224 + /255; ImageNet preprocess inside models",
        "augmentation": "train only: flip, small rotation/zoom/translation, moderate brightness/contrast; no oversampling; BCE",
        "ensemble": {
            "method": "soft_voting",
            "efficientnet_weight": w_eff,
            "mobilenet_weight": w_mob,
            "decision_threshold": threshold,
            "selected_on": "validation",
            "candidates": search["candidates"],
            "wording": "AI-estimated probability",
        },
        "validation": {
            "efficientnetb0": val_eff,
            "mobilenetv2": val_mob,
            "ensemble": val_ens_metrics,
            "efficientnetb0_at_0.5": eff["val_metrics_at_0.5"],
            "mobilenetv2_at_0.5": mob["val_metrics_at_0.5"],
        },
        "test": {
            "efficientnetb0": test_eff,
            "mobilenetv2": test_mob,
            "ensemble": test_ens,
            "ensemble_calibrated_investigation": test_ens_cal,
        },
        "calibration": calibration,
        "demo_100": demo,
        "gradcam": gradcam,
        "paths": {
            "efficientnet": str(eff_path),
            "mobilenet": str(mob_path),
            "plots": str(PLOT_DIR),
            "experiment_dir": str(EXP_DIR),
        },
        "clinical_note": (
            "Decision-support only. The doctor remains responsible for diagnosis. "
            "Grad-CAM is attention, not a confirmed cancer boundary. "
            "Do not treat probability as certainty."
        ),
    }
    _json_dump(EXP_DIR / "ensemble_config.json", experiment["ensemble"])
    _json_dump(EXP_DIR / "inference_config.json", {
        "decision_threshold": threshold,
        "hybrid_method": "soft_voting",
        "weights": {"efficientnet": w_eff, "mobilenet": w_mob},
        "trained_on": ["isic_dicm_17k"],
        "threshold_tuned_on": "validation",
        "cache_version": f"dicm-{threshold:.2f}",
        "not_production": True,
    })
    _json_dump(EXP_DIR / "evaluation_report.json", experiment)

    lines = [
        "ISIC-DICM-17K training evaluation (not production)",
        f"Ensemble weights Eff/Mob: {w_eff}/{w_mob}",
        f"Validation threshold: {threshold}",
        "",
        "Validation EfficientNetB0: " + json.dumps(val_eff, default=_jsonable),
        "Validation MobileNetV2: " + json.dumps(val_mob, default=_jsonable),
        "Validation Ensemble: " + json.dumps(val_ens_metrics, default=_jsonable),
        "",
        "Test EfficientNetB0: " + json.dumps(test_eff, default=_jsonable),
        "Test MobileNetV2: " + json.dumps(test_mob, default=_jsonable),
        "Test Ensemble: " + json.dumps(test_ens, default=_jsonable),
        "",
        f"Grad-CAM: {gradcam['status']}",
        f"100-image demo accuracy: {demo['metrics']['accuracy']}",
        f"New files under: {EXP_DIR}",
        "ACTIVE_DATASET unchanged. Production .keras files unchanged.",
    ]
    (EXP_DIR / "EVALUATION_REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    logger.info("DICM experiment finished. Production application was not switched.")


if __name__ == "__main__":
    main()
