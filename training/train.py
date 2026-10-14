"""
Main training script for EfficientNetB0, MobileNetV2, and Ensemble evaluation.

Usage:
    python -m training.train --dataset siim_isic_2020
    python -m training.train --dataset siim_isic_2020 --resume   # reuse EfficientNet, train MobileNet
    python -m training.train --dataset siim_isic_2020 --continue-efficientnet  # resume EfficientNet fine-tune
    python -m training.train --dataset isic_dicm_17k
    python -m training.train --dataset combined --adapt  # SIIM + HAM10000 fine-tune
    python -m training.train --dataset ham10000 --adapt  # adapt SIIM weights to HAM10000
    python -m training.train --synthetic
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from models.architecture import (
    FocalLoss,
    build_efficientnetb0,
    build_mobilenetv2,
    compile_model,
    unfreeze_for_finetuning,
)
from models.ensemble import EnsemblePredictor, combine_hybrid_probs, select_best_model
from training.callbacks import get_callbacks
from training.dataset_loader import MelanomaDataset, create_synthetic_demo_dataset
from training.evaluate import (
    best_epoch_from_history,
    evaluate_predictions,
    find_optimal_threshold,
    full_evaluation_report,
    plot_training_curves,
    save_threshold_table,
    threshold_table,
    plot_threshold_pr,
)
from utils.logger import setup_logger

logger = setup_logger("train")


def enable_mixed_precision() -> None:
    """Enable mixed precision if a GPU is available and configured."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus and config.USE_MIXED_PRECISION:
        try:
            from tensorflow.keras import mixed_precision

            mixed_precision.set_global_policy("mixed_float16")
            logger.info("Mixed precision enabled (GPU detected: %d).", len(gpus))
        except Exception as exc:
            logger.warning("Could not enable mixed precision: %s", exc)
    else:
        logger.info("Training on CPU / mixed precision disabled.")


def _predict_dataset(model: tf.keras.Model, ds: tf.data.Dataset) -> tuple:
    y_prob, y_true = [], []
    for batch in ds:
        if isinstance(batch, (list, tuple)):
            batch_x, batch_y = batch[0], batch[1]
        else:
            batch_x, batch_y = batch
        preds = model.predict(batch_x, verbose=0)
        y_prob.append(np.asarray(preds).ravel())
        y_true.append(np.asarray(batch_y).ravel())
    return np.concatenate(y_true), np.concatenate(y_prob)


def _make_finetune_datasets(train_df, val_df, loader, batch_size: int):
    """Smaller batches for fine-tuning to avoid RAM crashes."""
    train_ds = loader.make_tf_dataset(
        train_df, batch_size=batch_size, shuffle=True, augment=True
    )
    val_ds = loader.make_tf_dataset(
        val_df, batch_size=batch_size, shuffle=False, augment=False
    )
    return train_ds, val_ds


def apply_balance_mode(train_df: pd.DataFrame, mode: str | None = None) -> pd.DataFrame:
    """Apply exactly one train-set balancing strategy. Val/test must not use this."""
    mode = (mode or getattr(config, "BALANCE_MODE", "weighted_bce") or "none").lower()
    if mode in {"weighted_bce", "weighted", "class_weight"}:
        config.USE_FOCAL_LOSS = False
        return MelanomaDataset.add_balanced_sample_weights(train_df)
    if mode in {"focal", "focal_loss"}:
        config.USE_FOCAL_LOSS = True
        out = train_df.copy()
        out["sample_weight"] = 1.0
        logger.info("Balance mode=focal (no oversampling, no class weights)")
        return out
    if mode in {"oversample", "oversample_x2"}:
        config.USE_FOCAL_LOSS = False
        factor = int(getattr(config, "MALIGNANT_OVERSAMPLE_FACTOR", 2) or 2)
        if factor <= 1:
            factor = 2
        out = MelanomaDataset.oversample_malignant(train_df, factor=factor)
        out = out.copy()
        out["sample_weight"] = 1.0
        logger.info("Balance mode=oversample x%d (no class weights, BCE)", factor)
        return out
    config.USE_FOCAL_LOSS = False
    out = train_df.copy()
    out["sample_weight"] = 1.0
    logger.info("Balance mode=none")
    return out


def _val_pr_auc(y_true, y_prob) -> float:
    from sklearn.metrics import average_precision_score

    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob).astype(float).ravel()
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(average_precision_score(y_true, y_prob))


def _load_best_by_val_prauc(candidates: list, val_ds) -> tuple:
    """Load the checkpoint with the highest validation PR-AUC."""
    scored = []
    for path in candidates:
        path = Path(path)
        if not path.exists():
            continue
        model = tf.keras.models.load_model(
            str(path), custom_objects={"FocalLoss": FocalLoss}, compile=False
        )
        y_true, y_prob = _predict_dataset(model, val_ds)
        score = _val_pr_auc(y_true, y_prob)
        logger.info("Checkpoint %s val PR-AUC=%.4f", path.name, score)
        scored.append((score, path))
        del model
        tf.keras.backend.clear_session()
    if not scored:
        return None, None
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_path = scored[0]
    logger.info("Keeping %s (val PR-AUC=%.4f)", best_path.name, best_score)
    model = tf.keras.models.load_model(
        str(best_path), custom_objects={"FocalLoss": FocalLoss}, compile=False
    )
    return best_path, model


def train_model(
    builder,
    name: str,
    train_ds,
    val_ds,
    test_ds,
    class_weight,
    save_path: Path,
    skip_finetune: bool = False,
    train_df=None,
    val_df=None,
    loader=None,
    evaluate_test: bool = False,
) -> dict:
    """Two-phase training: frozen backbone, then memory-safe fine-tuning."""
    logger.info("=" * 60)
    logger.info("Training %s", name)
    logger.info("=" * 60)

    model = builder(trainable_base=False)
    model = compile_model(model, learning_rate=config.LEARNING_RATE)
    callbacks = get_callbacks(name)

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )
    hist = {k: list(history1.history.get(k, [])) for k in history1.history.keys()}
    phase1_ckpt = config.CHECKPOINTS_DIR / f"{name}_best.keras"
    if phase1_ckpt.exists():
        logger.info("Restoring best phase-1 checkpoint: %s", phase1_ckpt)
        model = tf.keras.models.load_model(
            str(phase1_ckpt), custom_objects={"FocalLoss": FocalLoss}, compile=False
        )
    chosen_ckpt = phase1_ckpt if phase1_ckpt.exists() else None

    if not skip_finetune and config.FINE_TUNE_EPOCHS > 0:
        ft_batch = max(8, min(16, config.BATCH_SIZE // 2))
        n_unfreeze = config.FINE_TUNE_UNFREEZE_LAYERS
        logger.info(
            "Fine-tuning %s with batch=%d, unfreeze_top=%d layers (memory-safe)",
            name,
            ft_batch,
            n_unfreeze,
        )
        if phase1_ckpt.exists():
            logger.info("Loading phase-1 checkpoint: %s", phase1_ckpt)
            model = tf.keras.models.load_model(
                str(phase1_ckpt), custom_objects={"FocalLoss": FocalLoss}, compile=False
            )

        model = unfreeze_for_finetuning(model, n_unfreeze_layers=n_unfreeze)
        model = compile_model(model, learning_rate=config.FINE_TUNE_LR)

        ft_train, ft_val = train_ds, val_ds
        if train_df is not None and loader is not None:
            ft_train, ft_val = _make_finetune_datasets(train_df, val_df, loader, ft_batch)

        history2 = model.fit(
            ft_train,
            validation_data=ft_val,
            epochs=config.FINE_TUNE_EPOCHS,
            class_weight=class_weight,
            callbacks=get_callbacks(f"{name}_finetune"),
            verbose=1,
        )
        hist = {
            k: list(history1.history.get(k, [])) + list(history2.history.get(k, []))
            for k in set(history1.history.keys()) | set(history2.history.keys())
        }

        ft_ckpt = config.CHECKPOINTS_DIR / f"{name}_finetune_best.keras"
        best_path, loaded = _load_best_by_val_prauc(
            [phase1_ckpt, ft_ckpt], val_ds if val_df is None else ft_val
        )
        if loaded is not None:
            model = loaded
            chosen_ckpt = best_path

    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    logger.info("Saved %s -> %s (source=%s)", name, save_path, chosen_ckpt)

    plot_dir = config.MODELS_DIR / "plots"
    plot_training_curves(hist, plot_dir, name)

    y_val_true, y_val_prob = _predict_dataset(model, val_ds)
    val_metrics = evaluate_predictions(y_val_true, y_val_prob, threshold=0.5)

    y_true = y_prob = None
    eval_metrics = dict(val_metrics)
    if evaluate_test and test_ds is not None:
        y_true, y_prob = _predict_dataset(model, test_ds)
        eval_out = full_evaluation_report(y_true, y_prob, name, plot_dir)
        eval_metrics = eval_out["metrics"]

    del model
    tf.keras.backend.clear_session()

    return {
        "metrics": eval_metrics,
        "val_metrics": val_metrics,
        "y_true": y_true,
        "y_prob": y_prob,
        "y_val_true": y_val_true,
        "y_val_prob": y_val_prob,
        "history": hist,
        "best_epoch": best_epoch_from_history(hist),
        "checkpoint": str(chosen_ckpt) if chosen_ckpt else str(save_path),
    }


def _load_model_for_eval(path: Path) -> tf.keras.Model:
    return tf.keras.models.load_model(
        str(path), custom_objects={"FocalLoss": FocalLoss}, compile=False
    )


def continue_efficientnet_finetune(
    train_df,
    val_df,
    test_df,
    loader,
    class_weight,
) -> None:
    """
    Continue EfficientNet fine-tuning from the last checkpoint (memory-safe),
    then rebuild ensemble with existing MobileNet if available.
    """
    src = config.CHECKPOINTS_DIR / "efficientnetb0_finetune_best.keras"
    alt = config.CHECKPOINTS_DIR / "efficientnetb0_best.keras"
    if not src.exists():
        src = alt
    if not src.exists():
        raise FileNotFoundError("No EfficientNet checkpoint found to continue from.")

    logger.info("Continuing EfficientNet fine-tune from %s", src)
    ft_batch = 8
    train_ds, val_ds = _make_finetune_datasets(train_df, val_df, loader, ft_batch)
    test_ds = loader.make_tf_dataset(test_df, batch_size=ft_batch, shuffle=False, augment=False)

    model = tf.keras.models.load_model(
        str(src), custom_objects={"FocalLoss": FocalLoss}, compile=False
    )
    model = unfreeze_for_finetuning(model, n_unfreeze_layers=config.FINE_TUNE_UNFREEZE_LAYERS)
    model = compile_model(model, learning_rate=config.FINE_TUNE_LR)

    epochs = min(config.FINE_TUNE_EPOCHS, 10)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=get_callbacks("efficientnetb0_finetune"),
        verbose=1,
    )

    ft_ckpt = config.CHECKPOINTS_DIR / "efficientnetb0_finetune_best.keras"
    if ft_ckpt.exists():
        model = tf.keras.models.load_model(
            str(ft_ckpt), custom_objects={"FocalLoss": FocalLoss}, compile=False
        )
    model.save(str(config.EFFICIENTNET_PATH))
    logger.info("Saved continued EfficientNet -> %s", config.EFFICIENTNET_PATH)

    y_true_e, y_prob_e = _predict_dataset(model, test_ds)
    eff_eval = full_evaluation_report(
        y_true_e, y_prob_e, "efficientnetb0", config.MODELS_DIR / "plots"
    )
    del model
    tf.keras.backend.clear_session()

    # Rebuild ensemble if MobileNet already exists
    if config.MOBILENET_PATH.exists():
        mob = _load_model_for_eval(config.MOBILENET_PATH)
        _, y_prob_m = _predict_dataset(mob, test_ds)
        del mob
        tf.keras.backend.clear_session()
        ens_prob = combine_hybrid_probs(y_prob_e, y_prob_m)
        ens_eval = full_evaluation_report(
            y_true_e, ens_prob, "hybrid", config.MODELS_DIR / "plots"
        )
        from training.evaluate import evaluate_predictions

        mob_metrics = evaluate_predictions(y_true_e, y_prob_m)
        all_metrics = {
            "efficientnet": eff_eval["metrics"],
            "mobilenet": mob_metrics,
            "ensemble": ens_eval["metrics"],
        }
        best = select_best_model(all_metrics)
        EnsemblePredictor.save_ensemble_meta(config.ENSEMBLE_WEIGHTS, ens_eval["metrics"])
        with open(config.METRICS_PATH, "w", encoding="utf-8") as fh:
            json.dump({"best_model": best, "metrics": all_metrics}, fh, indent=2)
        logger.info("Updated ensemble after EfficientNet continue. Best: %s", best)
    else:
        with open(config.METRICS_PATH, "w", encoding="utf-8") as fh:
            json.dump(
                {"best_model": "efficientnet", "metrics": {"efficientnet": eff_eval["metrics"]}},
                fh,
                indent=2,
            )
        logger.info("EfficientNet continue complete (MobileNet not trained yet).")


def adapt_models_to_dataset(
    train_df,
    val_df,
    test_df,
    loader,
    class_weight,
    dataset_key: str,
    adapt_epochs: int = config.ADAPT_EPOCHS,
    skip_mobilenet: bool = False,
    resume_adapt_ckpt: bool = False,
    n_unfreeze_layers: int = config.FINE_TUNE_UNFREEZE_LAYERS,
) -> None:
    """
    Fine-tune existing weights on SIIM-ISIC + HAM10000 (or another dataset).

    Uses malignant oversampling and a tuned threshold for better generalization.
    """
    from training.evaluate import evaluate_predictions, find_optimal_threshold

    if not config.EFFICIENTNET_PATH.exists():
        raise FileNotFoundError(
            f"No saved EfficientNet at {config.EFFICIENTNET_PATH}. "
            "Train on SIIM-ISIC first: python -m training.train --dataset siim_isic_2020"
        )

    logger.info(
        "Adapting models to %s (%d train / %d val / %d test)",
        dataset_key,
        len(train_df),
        len(val_df),
        len(test_df),
    )
    ft_batch = config.ADAPT_BATCH_SIZE
    test_ds = loader.make_tf_dataset(
        test_df, batch_size=ft_batch, shuffle=False, augment=False
    )

    def _adapt_one(name: str, save_path: Path) -> tuple:
        if not save_path.exists():
            raise FileNotFoundError(f"Missing weights: {save_path}")
        train_ds, val_ds = _make_finetune_datasets(
            train_df, val_df, loader, ft_batch
        )
        cb_name = f"{name}_adapt"
        ckpt = config.CHECKPOINTS_DIR / f"{cb_name}_best.keras"
        load_path = ckpt if resume_adapt_ckpt and ckpt.exists() else save_path
        if load_path != save_path:
            logger.info("Resuming %s from checkpoint %s", name, load_path)
        model = _load_model_for_eval(load_path)
        model = unfreeze_for_finetuning(model, n_unfreeze_layers=n_unfreeze_layers)
        model = compile_model(model, learning_rate=config.FINE_TUNE_LR)
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=adapt_epochs,
            class_weight=class_weight,
            callbacks=get_callbacks(cb_name),
            verbose=1,
        )
        if ckpt.exists():
            model = _load_model_for_eval(ckpt)
        model.save(str(save_path))
        logger.info("Adapted %s -> %s", name, save_path)
        y_true, y_prob = _predict_dataset(model, test_ds)
        del model
        tf.keras.backend.clear_session()
        return y_true, y_prob

    y_true_e, y_prob_e = _adapt_one("efficientnetb0", config.EFFICIENTNET_PATH)
    eff_eval = full_evaluation_report(
        y_true_e, y_prob_e, f"efficientnetb0_{dataset_key}", config.MODELS_DIR / "plots"
    )

    y_prob_m = np.zeros_like(y_prob_e)
    mob_metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auc": 0.0}
    if config.MOBILENET_PATH.exists() and not skip_mobilenet:
        _, y_prob_m = _adapt_one("mobilenetv2", config.MOBILENET_PATH)
        mob_metrics = evaluate_predictions(y_true_e, y_prob_m)
        full_evaluation_report(
            y_true_e, y_prob_m, f"mobilenetv2_{dataset_key}", config.MODELS_DIR / "plots"
        )
    elif config.MOBILENET_PATH.exists() and skip_mobilenet:
        logger.info("Skipping MobileNet adapt (fast mode); using existing MobileNet weights.")
        mob = _load_model_for_eval(config.MOBILENET_PATH)
        _, y_prob_m = _predict_dataset(mob, test_ds)
        mob_metrics = evaluate_predictions(y_true_e, y_prob_m)
        del mob
        tf.keras.backend.clear_session()

    ens_prob = combine_hybrid_probs(y_prob_e, y_prob_m)
    ens_eval = full_evaluation_report(
        y_true_e, ens_prob, f"hybrid_{dataset_key}", config.MODELS_DIR / "plots"
    )
    opt_thr, thr_metrics = find_optimal_threshold(y_true_e, ens_prob)
    cache_version = f"v5-{dataset_key}-{opt_thr:.2f}"

    inference_cfg = {
        "decision_threshold": round(opt_thr, 4),
        "hybrid_method": "soft_voting",
        "adapted_on": dataset_key,
        "trained_on": ["siim_isic_2020", "ham10000"] if dataset_key == "combined" else [dataset_key],
        "threshold_metrics": thr_metrics,
        "cache_version": cache_version,
    }
    with open(config.INFERENCE_CONFIG_PATH, "w", encoding="utf-8") as fh:
        json.dump(inference_cfg, fh, indent=2)
    logger.info(
        "Saved inference config (threshold=%.2f, cache=%s) -> %s",
        opt_thr,
        cache_version,
        config.INFERENCE_CONFIG_PATH,
    )

    all_metrics = {
        "efficientnet": eff_eval["metrics"],
        "mobilenet": mob_metrics,
        "ensemble": ens_eval["metrics"],
        "ensemble_at_optimal_threshold": thr_metrics,
        "optimal_threshold": opt_thr,
        "evaluated_on": dataset_key,
    }
    best = select_best_model(
        {
            "efficientnet": eff_eval["metrics"],
            "mobilenet": mob_metrics,
            "ensemble": ens_eval["metrics"],
        }
    )
    EnsemblePredictor.save_ensemble_meta(config.ENSEMBLE_WEIGHTS, ens_eval["metrics"])
    with open(config.METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump({"best_model": best, "metrics": all_metrics}, fh, indent=2)

    logger.info(
        "Adaptation complete on %s. Set PREDICTION_CACHE_VERSION in config.py to "
        "invalidate old upload cache (e.g. 'v4-adapt-%s').",
        dataset_key,
        dataset_key,
    )
    logger.info("Metrics:\n%s", json.dumps(all_metrics, indent=2))


def resume_from_checkpoints(
    train_df,
    val_df,
    test_df,
    loader,
    class_weight,
    skip_finetune: bool = False,
) -> None:
    """
    Continue after an OOM crash:
    - Reuse EfficientNet fine-tune checkpoint if present
    - Train MobileNet (memory-safe)
    - Evaluate ensemble
    """
    train_ds = loader.make_tf_dataset(train_df, shuffle=True, augment=True)
    val_ds = loader.make_tf_dataset(val_df, shuffle=False, augment=False)
    test_ds = loader.make_tf_dataset(test_df, shuffle=False, augment=False)

    # --- EfficientNet: use best available checkpoint ---
    ft_ckpt = config.CHECKPOINTS_DIR / "efficientnetb0_finetune_best.keras"
    p1_ckpt = config.CHECKPOINTS_DIR / "efficientnetb0_best.keras"
    eff_src = ft_ckpt if ft_ckpt.exists() else p1_ckpt
    if not eff_src.exists():
        raise FileNotFoundError(
            "No EfficientNet checkpoint found. Run full training first "
            "(without --resume)."
        )

    logger.info("Resume: using EfficientNet weights from %s", eff_src)
    shutil.copy2(eff_src, config.EFFICIENTNET_PATH)
    eff_model = _load_model_for_eval(config.EFFICIENTNET_PATH)
    y_true_e, y_prob_e = _predict_dataset(eff_model, test_ds)
    eff_eval = full_evaluation_report(
        y_true_e, y_prob_e, "efficientnetb0", config.MODELS_DIR / "plots"
    )
    del eff_model
    tf.keras.backend.clear_session()

    # --- MobileNet: train from scratch with safer fine-tune ---
    config.BATCH_SIZE = min(config.BATCH_SIZE, 24)
    mob = train_model(
        build_mobilenetv2,
        "mobilenetv2",
        train_ds,
        val_ds,
        test_ds,
        class_weight,
        config.MOBILENET_PATH,
        skip_finetune=skip_finetune,
        train_df=train_df,
        val_df=val_df,
        loader=loader,
    )

    # --- Hybrid (EfficientNet + MobileNet) ---
    ens_prob = combine_hybrid_probs(y_prob_e, mob["y_prob"])
    ens_eval = full_evaluation_report(
        y_true_e, ens_prob, "hybrid", config.MODELS_DIR / "plots"
    )

    all_metrics = {
        "efficientnet": eff_eval["metrics"],
        "mobilenet": mob["metrics"],
        "ensemble": ens_eval["metrics"],
    }
    best = select_best_model(all_metrics)
    EnsemblePredictor.save_ensemble_meta(config.ENSEMBLE_WEIGHTS, ens_eval["metrics"])
    with open(config.METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump({"best_model": best, "metrics": all_metrics}, fh, indent=2)

    logger.info("Resume training complete. Best model: %s", best)
    logger.info("Metrics summary:\n%s", json.dumps(all_metrics, indent=2))


def prepare_data(
    use_synthetic: bool,
    dataset_key: str,
    max_samples: int | None,
):
    if use_synthetic:
        df = create_synthetic_demo_dataset(n_samples=240)
        from sklearn.model_selection import train_test_split

        train_df, temp = train_test_split(
            df, test_size=0.3, stratify=df["label"], random_state=config.SEED
        )
        val_df, test_df = train_test_split(
            temp, test_size=0.5, stratify=temp["label"], random_state=config.SEED
        )
        loader = MelanomaDataset(dataset_key="siim_isic_2020")
        loader.df = df
        return train_df, val_df, test_df, loader

    loader = MelanomaDataset(dataset_key=dataset_key)
    if dataset_key == "isic_dicm_17k" and not max_samples:
        saved = loader.load_saved_dicm_splits()
        if saved:
            loader.load_metadata()
            return saved[0], saved[1], saved[2], loader
    loader.load_metadata()
    if max_samples and len(loader.df) > max_samples:
        mal = loader.df[loader.df["label"] == 1]
        ben = loader.df[loader.df["label"] == 0]
        n_ben = max(max_samples - len(mal), max_samples // 5)
        ben = ben.sample(n=min(n_ben, len(ben)), random_state=config.SEED)
        loader.df = (
            pd.concat([mal, ben], ignore_index=True)
            .sample(frac=1.0, random_state=config.SEED)
            .reset_index(drop=True)
        )
        logger.info(
            "Subset: %d images (%d malignant)",
            len(loader.df),
            int(loader.df["label"].sum()),
        )
    train_df, val_df, test_df = loader.split()
    return train_df, val_df, test_df, loader


def run_training(
    use_synthetic: bool = False,
    dataset_key: str = config.ACTIVE_DATASET,
    max_samples: int | None = None,
    skip_finetune: bool = False,
    resume: bool = False,
    continue_efficientnet: bool = False,
    adapt: bool = False,
    adapt_epochs: int = 10,
    skip_mobilenet_adapt: bool = False,
    resume_adapt_ckpt: bool = False,
    adapt_unfreeze_layers: int = config.FINE_TUNE_UNFREEZE_LAYERS,
) -> None:
    enable_mixed_precision()
    tf.keras.utils.set_random_seed(config.SEED)

    train_df, val_df, test_df, loader = prepare_data(
        use_synthetic, dataset_key, max_samples
    )
    if not use_synthetic:
        ratio = getattr(config, "BENIGN_TO_MALIGNANT_RATIO", None)
        if ratio:
            train_df = loader.downsample_majority(train_df, ratio=ratio)
        train_df = apply_balance_mode(train_df)
    class_weight = None

    if adapt:
        adapt_models_to_dataset(
            train_df,
            val_df,
            test_df,
            loader,
            class_weight,
            dataset_key,
            adapt_epochs,
            skip_mobilenet=skip_mobilenet_adapt,
            resume_adapt_ckpt=resume_adapt_ckpt,
            n_unfreeze_layers=adapt_unfreeze_layers,
        )
        return

    if continue_efficientnet:
        continue_efficientnet_finetune(
            train_df, val_df, test_df, loader, class_weight
        )
        return

    if resume:
        resume_from_checkpoints(
            train_df, val_df, test_df, loader, class_weight, skip_finetune=skip_finetune
        )
        return

    train_ds = loader.make_tf_dataset(train_df, shuffle=True, augment=True)
    val_ds = loader.make_tf_dataset(val_df, shuffle=False, augment=False)
    test_ds = loader.make_tf_dataset(test_df, shuffle=False, augment=False)

    eff = train_model(
        build_efficientnetb0,
        "efficientnetb0",
        train_ds,
        val_ds,
        test_ds,
        class_weight,
        config.EFFICIENTNET_PATH,
        skip_finetune=skip_finetune,
        train_df=train_df,
        val_df=val_df,
        loader=loader,
        evaluate_test=True,
    )

    mob = train_model(
        build_mobilenetv2,
        "mobilenetv2",
        train_ds,
        val_ds,
        test_ds,
        class_weight,
        config.MOBILENET_PATH,
        skip_finetune=skip_finetune,
        train_df=train_df,
        val_df=val_df,
        loader=loader,
        evaluate_test=True,
    )

    plot_dir = config.MODELS_DIR / "plots"
    ens_val = combine_hybrid_probs(eff["y_val_prob"], mob["y_val_prob"])
    ens_prob = combine_hybrid_probs(eff["y_prob"], mob["y_prob"])

    val_metrics_map = {
        "efficientnet": evaluate_predictions(eff["y_val_true"], eff["y_val_prob"], threshold=0.5),
        "mobilenet": evaluate_predictions(mob["y_val_true"], mob["y_val_prob"], threshold=0.5),
        "ensemble": evaluate_predictions(eff["y_val_true"], ens_val, threshold=0.5),
    }
    best = select_best_model(val_metrics_map)

    opt_thr, val_thr = find_optimal_threshold(eff["y_val_true"], ens_val)
    rows = val_thr.get("threshold_table") or threshold_table(eff["y_val_true"], ens_val)
    save_threshold_table(rows, plot_dir / "val_threshold_table.csv")
    plot_threshold_pr(rows, plot_dir / "val_threshold_pr.png", selected=opt_thr)

    ens_eval = full_evaluation_report(
        eff["y_true"], ens_prob, "hybrid", plot_dir, threshold=opt_thr
    )
    test_at_thr = evaluate_predictions(eff["y_true"], ens_prob, threshold=opt_thr)
    trained_on = (
        ["siim_isic_2020", "ham10000"] if dataset_key == "combined" else [dataset_key]
    )
    cache_version = f"v14-{dataset_key}-{opt_thr:.2f}"
    inference_cfg = {
        "decision_threshold": round(float(opt_thr), 4),
        "hybrid_method": "soft_voting",
        "adapted_on": dataset_key,
        "trained_on": trained_on,
        "threshold_tuned_on": "validation",
        "validation_threshold_metrics": {
            k: v for k, v in val_thr.items() if k != "threshold_table"
        },
        "test_metrics_at_val_threshold": test_at_thr,
        "balance_mode": getattr(config, "BALANCE_MODE", None),
        "cache_version": cache_version,
    }
    with open(config.INFERENCE_CONFIG_PATH, "w", encoding="utf-8") as fh:
        json.dump(inference_cfg, fh, indent=2)

    all_metrics = {
        "efficientnet_val": val_metrics_map["efficientnet"],
        "mobilenet_val": val_metrics_map["mobilenet"],
        "ensemble_val": val_metrics_map["ensemble"],
        "efficientnet_test_0.5": eff["metrics"],
        "mobilenet_test_0.5": mob["metrics"],
        "ensemble_test_at_val_threshold": test_at_thr,
        "optimal_threshold": opt_thr,
        "threshold_tuned_on": "validation",
        "best_model_selected_on": "validation",
        "evaluated_on": dataset_key,
        "trained_on": trained_on,
    }
    EnsemblePredictor.save_ensemble_meta(config.ENSEMBLE_WEIGHTS, ens_eval["metrics"])

    with open(config.METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump({"best_model": best, "metrics": all_metrics}, fh, indent=2)

    logger.info("Training complete. Best model (validation): %s", best)
    logger.info("Metrics summary:\n%s", json.dumps(all_metrics, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Train Melanoma CDSS models")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic demo data")
    parser.add_argument(
        "--dataset",
        default=config.ACTIVE_DATASET,
        choices=list(config.DATASETS.keys()),
        help="Dataset key",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast training: ~5k images, 3 epochs, no fine-tune",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap dataset size (keeps all malignant)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue after crash: reuse EfficientNet checkpoint, train MobileNet",
    )
    parser.add_argument(
        "--continue-efficientnet",
        action="store_true",
        help="Resume EfficientNet fine-tune from checkpoint (after MobileNet finishes)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Train EfficientNet completely (transfer+finetune) then MobileNet — best for accuracy",
    )
    parser.add_argument(
        "--no-finetune",
        action="store_true",
        help="Skip fine-tuning phase (uses less RAM)",
    )
    parser.add_argument(
        "--adapt",
        action="store_true",
        help="Fine-tune existing SIIM-ISIC weights on --dataset (domain adaptation)",
    )
    parser.add_argument(
        "--adapt-epochs",
        type=int,
        default=config.ADAPT_EPOCHS,
        help=f"Epochs for --adapt fine-tuning (default: {config.ADAPT_EPOCHS})",
    )
    parser.add_argument(
        "--fast-adapt",
        action="store_true",
        help="Fast combined adapt: 12k samples, 3 epochs, batch 32, EfficientNet only (~1-2h)",
    )
    parser.add_argument(
        "--full-data",
        action="store_true",
        help="Use every benign image (skip train-set majority downsampling)",
    )
    parser.add_argument(
        "--balance",
        choices=["weighted_bce", "focal", "oversample", "none"],
        default=None,
        help="Train-set balancing strategy (val/test are never resampled)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    skip_finetune = bool(args.no_finetune)
    max_samples = args.max_samples
    resume = bool(args.resume) and not bool(args.full) and not bool(args.continue_efficientnet)
    skip_mobilenet_adapt = False
    resume_adapt_ckpt = False
    adapt_unfreeze_layers = config.FINE_TUNE_UNFREEZE_LAYERS

    if getattr(args, "full_data", False):
        config.BENIGN_TO_MALIGNANT_RATIO = None
        logger.info("FULL DATA: keeping every benign training image")

    if getattr(args, "balance", None):
        config.BALANCE_MODE = args.balance
        logger.info("Balance mode: %s", args.balance)
    elif args.dataset == "isic_dicm_17k":
        config.BALANCE_MODE = "none"
        config.MALIGNANT_OVERSAMPLE_FACTOR = 1
        config.BENIGN_TO_MALIGNANT_RATIO = None
        logger.info("ISIC-DICM-17K is already 50/50; no class rebalancing.")

    if args.fast_adapt:
        args.adapt = True
        args.adapt_epochs = 3
        max_samples = max_samples or 12000
        config.ADAPT_BATCH_SIZE = 32
        config.MALIGNANT_OVERSAMPLE_FACTOR = 3
        config.PATIENCE_EARLY_STOP = 2
        config.PATIENCE_REDUCE_LR = 2
        skip_mobilenet_adapt = True
        resume_adapt_ckpt = True
        adapt_unfreeze_layers = 12
        logger.info(
            "FAST ADAPT: epochs=%d, max_samples=%d, batch=%d, EfficientNet only",
            args.adapt_epochs,
            max_samples,
            config.ADAPT_BATCH_SIZE,
        )

    if args.quick:
        config.EPOCHS = 3
        config.FINE_TUNE_EPOCHS = 0
        config.PATIENCE_EARLY_STOP = 2
        config.PATIENCE_REDUCE_LR = 1
        config.BATCH_SIZE = 48
        max_samples = max_samples or 5000
        skip_finetune = True
        logger.info(
            "QUICK MODE: epochs=%d, max_samples=%d, batch=%d",
            config.EPOCHS,
            max_samples,
            config.BATCH_SIZE,
        )

    if args.epochs:
        config.EPOCHS = args.epochs
        config.FINE_TUNE_EPOCHS = 0 if args.quick else max(1, args.epochs // 3)
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
        config.ADAPT_BATCH_SIZE = args.batch_size

    # Safer defaults on CPU to reduce OOM during fine-tune
    if not args.quick and not args.fast_adapt:
        config.BATCH_SIZE = min(config.BATCH_SIZE, 16)
        config.ADAPT_BATCH_SIZE = min(config.ADAPT_BATCH_SIZE, 16)
        config.FINE_TUNE_EPOCHS = min(config.FINE_TUNE_EPOCHS, 15)
        config.PATIENCE_EARLY_STOP = min(config.PATIENCE_EARLY_STOP, 6)

    if args.full:
        resume = False
        skip_finetune = False
        logger.info(
            "FULL ACCURACY MODE: retrain EfficientNet completely, then MobileNet "
            "(batch=%d, epochs=%d, finetune_epochs=%d)",
            config.BATCH_SIZE,
            config.EPOCHS,
            config.FINE_TUNE_EPOCHS,
        )

    run_training(
        use_synthetic=args.synthetic,
        dataset_key=args.dataset,
        max_samples=max_samples,
        skip_finetune=skip_finetune,
        resume=resume,
        continue_efficientnet=bool(args.continue_efficientnet),
        adapt=bool(args.adapt),
        adapt_epochs=args.adapt_epochs,
        skip_mobilenet_adapt=skip_mobilenet_adapt,
        resume_adapt_ckpt=resume_adapt_ckpt,
        adapt_unfreeze_layers=adapt_unfreeze_layers,
    )
