"""
Main training script for EfficientNetB0, MobileNetV2, and Ensemble evaluation.

Usage:
    python -m training.train --dataset siim_isic_2020
    python -m training.train --dataset siim_isic_2020 --resume   # reuse EfficientNet, train MobileNet
    python -m training.train --dataset siim_isic_2020 --continue-efficientnet  # resume EfficientNet fine-tune
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
from training.evaluate import full_evaluation_report, plot_training_curves
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
    for batch_x, batch_y in ds:
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
) -> dict:
    """Two-phase training: transfer learning then memory-safe fine-tuning."""
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

    # Fine-tuning phase (lower memory: fewer layers + smaller batch)
    if not skip_finetune and config.FINE_TUNE_EPOCHS > 0:
        ft_batch = max(8, min(16, config.BATCH_SIZE // 2))
        n_unfreeze = 12
        logger.info(
            "Fine-tuning %s with batch=%d, unfreeze_top=%d layers (memory-safe)",
            name,
            ft_batch,
            n_unfreeze,
        )
        # Prefer best checkpoint from phase 1 if available
        ckpt = config.CHECKPOINTS_DIR / f"{name}_best.keras"
        if ckpt.exists():
            logger.info("Loading phase-1 checkpoint: %s", ckpt)
            model = tf.keras.models.load_model(
                str(ckpt), custom_objects={"FocalLoss": FocalLoss}, compile=False
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
            for k in history1.history.keys()
        }

        ft_ckpt = config.CHECKPOINTS_DIR / f"{name}_finetune_best.keras"
        if ft_ckpt.exists():
            logger.info("Restoring best fine-tune checkpoint: %s", ft_ckpt)
            model = tf.keras.models.load_model(
                str(ft_ckpt), custom_objects={"FocalLoss": FocalLoss}, compile=False
            )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    logger.info("Saved %s → %s", name, save_path)

    plot_dir = config.MODELS_DIR / "plots"
    plot_training_curves(hist, plot_dir, name)

    y_true, y_prob = _predict_dataset(model, test_ds)
    eval_out = full_evaluation_report(y_true, y_prob, name, plot_dir)

    # Free memory before next model
    del model
    tf.keras.backend.clear_session()

    return {
        "metrics": eval_out["metrics"],
        "y_true": y_true,
        "y_prob": y_prob,
        "history": hist,
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
    model = unfreeze_for_finetuning(model, n_unfreeze_layers=12)
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
    logger.info("Saved continued EfficientNet → %s", config.EFFICIENTNET_PATH)

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
) -> None:
    enable_mixed_precision()
    tf.keras.utils.set_random_seed(config.SEED)

    train_df, val_df, test_df, loader = prepare_data(
        use_synthetic, dataset_key, max_samples
    )
    class_weight = loader.compute_class_weights(train_df["label"].values)

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
    )

    ens_prob = combine_hybrid_probs(eff["y_prob"], mob["y_prob"])
    ens_eval = full_evaluation_report(
        eff["y_true"], ens_prob, "hybrid", config.MODELS_DIR / "plots"
    )

    all_metrics = {
        "efficientnet": eff["metrics"],
        "mobilenet": mob["metrics"],
        "ensemble": ens_eval["metrics"],
    }
    best = select_best_model(all_metrics)
    EnsemblePredictor.save_ensemble_meta(config.ENSEMBLE_WEIGHTS, ens_eval["metrics"])

    with open(config.METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump({"best_model": best, "metrics": all_metrics}, fh, indent=2)

    logger.info("Training complete. Best model: %s", best)
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    skip_finetune = bool(args.no_finetune)
    max_samples = args.max_samples
    resume = bool(args.resume) and not bool(args.full) and not bool(args.continue_efficientnet)

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

    # Safer defaults on CPU to reduce OOM during fine-tune
    if not args.quick:
        config.BATCH_SIZE = min(config.BATCH_SIZE, 16)
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
    )
