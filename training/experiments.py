"""
Controlled melanoma experiments: leakage audit, loss/balancing comparison,
architecture comparison, validation-only thresholding, untouched test eval.

Usage:
    python -m training.experiments --audit-only
    python -m training.experiments --dataset combined
    python -m training.experiments --dataset combined --skip-near-dup
"""

from __future__ import annotations

import argparse
import csv
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
from training.data_integrity import (
    audit_source_overlap,
    audit_split_leakage,
    save_leakage_report,
)
from training.dataset_loader import MelanomaDataset
from training.evaluate import (
    best_epoch_from_history,
    evaluate_predictions,
    find_optimal_threshold,
    full_evaluation_report,
    plot_threshold_pr,
    plot_training_curves,
    save_threshold_table,
    threshold_table,
)
from training.train import (
    _load_best_by_val_prauc,
    _make_finetune_datasets,
    _predict_dataset,
    apply_balance_mode,
    enable_mixed_precision,
    train_model,
)
from utils.logger import setup_logger

logger = setup_logger("experiments")

BALANCE_EXPERIMENTS = [
    {
        "name": "weighted_bce",
        "loss": "weighted_bce",
        "balance_mode": "weighted_bce",
        "oversample_factor": 1,
        "use_focal": False,
    },
    {
        "name": "focal",
        "loss": "focal",
        "balance_mode": "focal",
        "oversample_factor": 1,
        "use_focal": True,
    },
    {
        "name": "oversample_x2",
        "loss": "bce",
        "balance_mode": "oversample",
        "oversample_factor": 2,
        "use_focal": False,
    },
]

TABLE_FIELDS = [
    "Experiment",
    "Loss",
    "Class balancing method",
    "Oversampling factor",
    "Learning rate",
    "Frozen/unfrozen layers",
    "Best epoch",
    "Validation AUC",
    "Validation PR-AUC",
    "Validation recall",
    "Validation precision",
    "Validation F1",
    "Test AUC",
    "Test PR-AUC",
    "Test recall",
    "Test precision",
    "Test F1",
    "Threshold",
]


def _exp_dir(name: str) -> Path:
    path = Path(getattr(config, "EXPERIMENTS_DIR", config.MODELS_DIR / "experiments")) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cpu_caps() -> None:
    if not tf.config.list_physical_devices("GPU"):
        config.BATCH_SIZE = min(int(config.BATCH_SIZE), 16)
        config.ADAPT_BATCH_SIZE = min(int(getattr(config, "ADAPT_BATCH_SIZE", 16)), 16)
        logger.info("CPU caps: batch=%d", config.BATCH_SIZE)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items() if k != "threshold_table"}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def prepare_grouped_split(dataset_key: str, reuse_split: bool) -> tuple:
    split_dir = Path(getattr(config, "SPLIT_DIR", config.PROCESSED_DATA_DIR / "splits"))
    loader = MelanomaDataset(dataset_key=dataset_key)
    if reuse_split and (split_dir / "train.csv").exists():
        train_df, val_df, test_df = MelanomaDataset.load_splits(split_dir)
        loader.df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        logger.info("Reusing saved grouped split from %s", split_dir)
    else:
        loader.load_metadata()
        train_df, val_df, test_df = loader.split(grouped=True, save=True)

    for name, df in ("train", train_df), ("val", val_df), ("test", test_df):
        assert df["label"].nunique() >= 1, f"{name} split is empty"
    overlap = audit_source_overlap(loader.df if loader.df is not None else train_df)
    return train_df, val_df, test_df, loader, overlap


def run_audit(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    source_overlap: dict,
    check_near_duplicates: bool,
) -> dict:
    report = audit_split_leakage(
        train_df,
        val_df,
        test_df,
        check_near_duplicates=check_near_duplicates,
    )
    report["ham_isic_image_id_overlap"] = source_overlap
    report["notes"] = {
        "val_augmented": False,
        "test_augmented": False,
        "val_oversampled": False,
        "test_oversampled": False,
        "split": "patient_id (ISIC) / lesion_id (HAM) grouped, stratified on group melanoma flag",
    }
    save_leakage_report(report)
    return report


def _row_from_metrics(
    spec: dict,
    val_m: dict,
    test_m: dict | None,
    threshold: float | None,
    best_epoch: int,
    frozen_desc: str,
    lr: float,
) -> dict:
    test_m = test_m or {}
    return {
        "Experiment": spec["name"],
        "Loss": spec["loss"],
        "Class balancing method": spec["balance_mode"],
        "Oversampling factor": spec["oversample_factor"],
        "Learning rate": lr,
        "Frozen/unfrozen layers": frozen_desc,
        "Best epoch": best_epoch,
        "Validation AUC": val_m.get("auc"),
        "Validation PR-AUC": val_m.get("pr_auc"),
        "Validation recall": val_m.get("recall"),
        "Validation precision": val_m.get("precision"),
        "Validation F1": val_m.get("f1"),
        "Test AUC": test_m.get("auc"),
        "Test PR-AUC": test_m.get("pr_auc"),
        "Test recall": test_m.get("recall"),
        "Test precision": test_m.get("precision"),
        "Test F1": test_m.get("f1"),
        "Threshold": threshold,
    }


def write_table(rows: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = [{k: row.get(k, "") for k in TABLE_FIELDS} for row in rows]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=TABLE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(cleaned)
    path.with_suffix(".json").write_text(json.dumps(_jsonable(cleaned), indent=2), encoding="utf-8")
    logger.info("Experiment table: %s", path)
    return path


def _configure_spec(spec: dict) -> None:
    config.BALANCE_MODE = spec["balance_mode"]
    config.USE_FOCAL_LOSS = bool(spec["use_focal"])
    config.MALIGNANT_OVERSAMPLE_FACTOR = int(spec["oversample_factor"])
    config.BENIGN_TO_MALIGNANT_RATIO = None


def train_efficientnet_stage1(
    spec: dict,
    train_raw: pd.DataFrame,
    val_df: pd.DataFrame,
    loader: MelanomaDataset,
    skip_finetune: bool,
) -> dict:
    _configure_spec(spec)
    exp_dir = _exp_dir(spec["name"])
    train_df = apply_balance_mode(train_raw.copy(), spec["balance_mode"])
    val_clean = val_df.copy()
    if "sample_weight" in val_clean.columns:
        val_clean = val_clean.drop(columns=["sample_weight"])

    train_ds = loader.make_tf_dataset(train_df, shuffle=True, augment=True)
    val_ds = loader.make_tf_dataset(val_clean, shuffle=False, augment=False)
    save_path = exp_dir / "efficientnetb0.keras"
    result = train_model(
        build_efficientnetb0,
        f"efficientnetb0_{spec['name']}",
        train_ds,
        val_ds,
        None,
        None,
        save_path,
        skip_finetune=skip_finetune,
        train_df=train_df,
        val_df=val_clean,
        loader=loader,
        evaluate_test=False,
    )
    val_m = evaluate_predictions(result["y_val_true"], result["y_val_prob"], threshold=0.5)
    np.save(exp_dir / "eff_val_true.npy", result["y_val_true"])
    np.save(exp_dir / "eff_val_prob.npy", result["y_val_prob"])
    (exp_dir / "efficientnet_val_metrics.json").write_text(
        json.dumps(_jsonable(val_m), indent=2), encoding="utf-8"
    )
    plot_training_curves(result["history"], exp_dir / "plots", "efficientnetb0")
    result["val_metrics"] = val_m
    result["save_path"] = str(save_path)
    result["train_df"] = train_df
    return result


def train_mobilenet(
    spec: dict,
    train_raw: pd.DataFrame,
    val_df: pd.DataFrame,
    loader: MelanomaDataset,
) -> dict:
    _configure_spec(spec)
    exp_dir = _exp_dir(spec["name"])
    train_df = apply_balance_mode(train_raw.copy(), spec["balance_mode"])
    val_clean = val_df.copy()
    if "sample_weight" in val_clean.columns:
        val_clean = val_clean.drop(columns=["sample_weight"])
    train_ds = loader.make_tf_dataset(train_df, shuffle=True, augment=True)
    val_ds = loader.make_tf_dataset(val_clean, shuffle=False, augment=False)
    save_path = exp_dir / "mobilenetv2.keras"
    result = train_model(
        build_mobilenetv2,
        f"mobilenetv2_{spec['name']}",
        train_ds,
        val_ds,
        None,
        None,
        save_path,
        skip_finetune=False,
        train_df=train_df,
        val_df=val_clean,
        loader=loader,
        evaluate_test=False,
    )
    val_m = evaluate_predictions(result["y_val_true"], result["y_val_prob"], threshold=0.5)
    np.save(exp_dir / "mob_val_true.npy", result["y_val_true"])
    np.save(exp_dir / "mob_val_prob.npy", result["y_val_prob"])
    result["val_metrics"] = val_m
    result["save_path"] = str(save_path)
    return result


def evaluate_architecture_on_split(
    y_true,
    p_eff,
    p_mob,
    split_name: str,
    out_dir: Path,
    threshold: float,
) -> dict:
    p_hyb = combine_hybrid_probs(p_eff, p_mob)
    results = {
        "efficientnet": evaluate_predictions(y_true, p_eff, threshold=threshold),
        "mobilenet": evaluate_predictions(y_true, p_mob, threshold=threshold),
        "hybrid": evaluate_predictions(y_true, p_hyb, threshold=threshold),
    }
    for name, prob in (
        ("efficientnet", p_eff),
        ("mobilenet", p_mob),
        ("hybrid", p_hyb),
    ):
        full_evaluation_report(
            y_true, prob, f"{split_name}_{name}", out_dir, threshold=threshold
        )
    return results


def promote_winner(spec: dict, threshold: float, best_arch: str, extra: dict) -> None:
    exp_dir = _exp_dir(spec["name"])
    for src, dst in (
        (exp_dir / "efficientnetb0.keras", config.EFFICIENTNET_PATH),
        (exp_dir / "mobilenetv2.keras", config.MOBILENET_PATH),
    ):
        if src.exists():
            shutil.copy2(src, dst)
            logger.info("Copied %s -> %s", src, dst)

    cache_version = f"v14-{spec['name']}-{threshold:.2f}"
    inference_cfg = {
        "decision_threshold": round(float(threshold), 4),
        "hybrid_method": "soft_voting",
        "trained_on": ["siim_isic_2020", "ham10000"],
        "threshold_tuned_on": "validation",
        "balance_mode": spec["balance_mode"],
        "best_architecture_by_validation": best_arch,
        "cache_version": cache_version,
        **extra,
    }
    config.INFERENCE_CONFIG_PATH.write_text(
        json.dumps(_jsonable(inference_cfg), indent=2), encoding="utf-8"
    )
    config.DECISION_THRESHOLD = float(threshold)
    config.PREDICTION_CACHE_VERSION = cache_version


def _load_completed_compare(val_ds) -> tuple:
    """Reload compare-stage artifacts so evaluation can finish without retraining."""
    compare_results = {}
    table_rows = []
    for spec in BALANCE_EXPERIMENTS:
        exp_dir = _exp_dir(spec["name"])
        metrics_path = exp_dir / "efficientnet_val_metrics.json"
        ckpt = exp_dir / "efficientnetb0.keras"
        if not metrics_path.exists() or not ckpt.exists():
            raise FileNotFoundError(f"Missing compare artifacts for {spec['name']} in {exp_dir}")
        val_m = json.loads(metrics_path.read_text(encoding="utf-8"))
        true_path = exp_dir / "eff_val_true.npy"
        prob_path = exp_dir / "eff_val_prob.npy"
        y_true = np.load(true_path) if true_path.exists() else None
        y_prob = np.load(prob_path) if prob_path.exists() else None
        if y_true is None or y_prob is None:
            logger.info("Recomputing validation predictions for %s", spec["name"])
            model = tf.keras.models.load_model(
                str(ckpt), custom_objects={"FocalLoss": FocalLoss}, compile=False
            )
            y_true, y_prob = _predict_dataset(model, val_ds)
            del model
            tf.keras.backend.clear_session()
        compare_results[spec["name"]] = {
            "val_metrics": val_m,
            "y_val_true": y_true,
            "y_val_prob": y_prob,
            "save_path": str(ckpt),
            "history": {},
            "best_epoch": 0,
        }
        table_rows.append(
            _row_from_metrics(
                spec,
                val_m,
                None,
                None,
                0,
                "stage1 frozen backbone",
                config.LEARNING_RATE,
            )
        )
    return compare_results, table_rows


def _load_mobilenet_val(winner_name: str, val_ds, y_val_true):
    exp_dir = _exp_dir(winner_name)
    mob_path = exp_dir / "mobilenetv2.keras"
    if not mob_path.exists():
        raise FileNotFoundError(f"Missing MobileNet weights: {mob_path}")
    prob_path = exp_dir / "mob_val_prob.npy"
    if prob_path.exists():
        return {"y_val_prob": np.load(prob_path), "save_path": str(mob_path)}
    logger.info("Recomputing MobileNet validation predictions")
    model = tf.keras.models.load_model(
        str(mob_path), custom_objects={"FocalLoss": FocalLoss}, compile=False
    )
    _, y_prob = _predict_dataset(model, val_ds)
    np.save(prob_path, y_prob)
    if y_val_true is not None:
        np.save(exp_dir / "mob_val_true.npy", y_val_true)
    del model
    tf.keras.backend.clear_session()
    return {"y_val_prob": y_prob, "save_path": str(mob_path)}


def run_experiments(
    dataset_key: str,
    audit_only: bool,
    reuse_split: bool,
    check_near_duplicates: bool,
    compare_finetune: bool,
    finalize_only: bool = False,
) -> None:
    enable_mixed_precision()
    _cpu_caps()
    tf.keras.utils.set_random_seed(config.SEED)

    train_raw, val_df, test_df, loader, source_overlap = prepare_grouped_split(
        dataset_key, reuse_split=reuse_split
    )
    report = run_audit(
        train_raw,
        val_df,
        test_df,
        source_overlap,
        check_near_duplicates=check_near_duplicates,
    )
    logger.info("Leakage clean=%s", report.get("clean"))
    if audit_only:
        print(json.dumps(_jsonable(report), indent=2))
        return

    val_ds = loader.make_tf_dataset(val_df, shuffle=False, augment=False)
    test_ds = loader.make_tf_dataset(test_df, shuffle=False, augment=False)

    table_rows = []
    compare_results = {}
    if finalize_only:
        compare_results, table_rows = _load_completed_compare(val_ds)
        logger.info("Finalize-only: reusing trained checkpoints, skipping retraining.")
    else:
        for spec in BALANCE_EXPERIMENTS:
            logger.info("=== COMPARE %s ===", spec["name"])
            result = train_efficientnet_stage1(
                spec,
                train_raw,
                val_df,
                loader,
                skip_finetune=not compare_finetune,
            )
            compare_results[spec["name"]] = result
            table_rows.append(
                _row_from_metrics(
                    spec,
                    result["val_metrics"],
                    None,
                    threshold=None,
                    best_epoch=result.get("best_epoch") or 0,
                    frozen_desc="stage1 frozen backbone" + ("" if not compare_finetune else "+ stage2"),
                    lr=config.LEARNING_RATE,
                )
            )
            write_table(
                table_rows,
                Path(config.EXPERIMENTS_DIR) / "experiment_table.csv",
            )

    winner_name = max(
        compare_results,
        key=lambda n: (
            compare_results[n]["val_metrics"].get("pr_auc", 0.0),
            compare_results[n]["val_metrics"].get("auc", 0.0),
        ),
    )
    winner = next(s for s in BALANCE_EXPERIMENTS if s["name"] == winner_name)
    logger.info(
        "Winner by validation PR-AUC: %s (PR-AUC=%.4f AUC=%.4f recall=%.4f)",
        winner_name,
        compare_results[winner_name]["val_metrics"]["pr_auc"],
        compare_results[winner_name]["val_metrics"]["auc"],
        compare_results[winner_name]["val_metrics"]["recall"],
    )

    # If compare skipped fine-tune, run stage-2 on the winning EfficientNet now.
    if not finalize_only and not compare_finetune:
        logger.info("=== STAGE-2 fine-tune winner EfficientNet (%s) ===", winner_name)
        _configure_spec(winner)
        exp_dir = _exp_dir(winner_name)
        ckpt = Path(compare_results[winner_name]["save_path"])
        model = tf.keras.models.load_model(
            str(ckpt), custom_objects={"FocalLoss": FocalLoss}, compile=False
        )
        train_df = apply_balance_mode(train_raw.copy(), winner["balance_mode"])
        model = unfreeze_for_finetuning(model, n_unfreeze_layers=config.FINE_TUNE_UNFREEZE_LAYERS)
        model = compile_model(model, learning_rate=config.FINE_TUNE_LR)
        ft_train, ft_val = _make_finetune_datasets(train_df, val_df, loader, max(8, min(16, config.BATCH_SIZE // 2)))
        hist2 = model.fit(
            ft_train,
            validation_data=ft_val,
            epochs=config.FINE_TUNE_EPOCHS,
            callbacks=get_callbacks(f"efficientnetb0_{winner_name}_finetune"),
            verbose=1,
        )
        phase1 = config.CHECKPOINTS_DIR / f"efficientnetb0_{winner_name}_best.keras"
        phase2 = config.CHECKPOINTS_DIR / f"efficientnetb0_{winner_name}_finetune_best.keras"
        best_path, loaded = _load_best_by_val_prauc([phase1, phase2, ckpt], val_ds)
        if loaded is not None:
            model = loaded
        model.save(str(ckpt))
        y_val_true, y_val_prob = _predict_dataset(model, val_ds)
        compare_results[winner_name]["y_val_true"] = y_val_true
        compare_results[winner_name]["y_val_prob"] = y_val_prob
        compare_results[winner_name]["val_metrics"] = evaluate_predictions(
            y_val_true, y_val_prob, threshold=0.5
        )
        compare_results[winner_name]["best_epoch"] = best_epoch_from_history(
            {
                k: list(compare_results[winner_name]["history"].get(k, []))
                + list(hist2.history.get(k, []))
                for k in set(compare_results[winner_name]["history"]) | set(hist2.history)
            }
        )
        np.save(exp_dir / "eff_val_true.npy", y_val_true)
        np.save(exp_dir / "eff_val_prob.npy", y_val_prob)
        del model
        tf.keras.backend.clear_session()

    if finalize_only:
        logger.info("=== Reusing trained MobileNet for winner %s ===", winner_name)
        mob = _load_mobilenet_val(winner_name, val_ds, compare_results[winner_name]["y_val_true"])
    else:
        logger.info("=== Train MobileNet with winner config %s ===", winner_name)
        mob = train_mobilenet(winner, train_raw, val_df, loader)

    exp_dir = _exp_dir(winner_name)
    plots = exp_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    y_val = compare_results[winner_name]["y_val_true"]
    p_eff_val = compare_results[winner_name]["y_val_prob"]
    p_mob_val = mob["y_val_prob"]
    p_hyb_val = combine_hybrid_probs(p_eff_val, p_mob_val)

    val_arch = {
        "efficientnet": evaluate_predictions(y_val, p_eff_val, threshold=0.5),
        "mobilenet": evaluate_predictions(y_val, p_mob_val, threshold=0.5),
        "hybrid": evaluate_predictions(y_val, p_hyb_val, threshold=0.5),
    }
    best_arch = select_best_model(
        {
            "efficientnet": val_arch["efficientnet"],
            "mobilenet": val_arch["mobilenet"],
            "ensemble": val_arch["hybrid"],
        }
    )
    arch_key = {"efficientnet": p_eff_val, "mobilenet": p_mob_val, "ensemble": p_hyb_val, "hybrid": p_hyb_val}
    chosen_val_prob = arch_key.get(best_arch, p_hyb_val)
    opt_thr, val_thr = find_optimal_threshold(y_val, chosen_val_prob)
    rows = val_thr.get("threshold_table") or threshold_table(y_val, chosen_val_prob)
    save_threshold_table(rows, plots / "val_threshold_table.csv")
    plot_threshold_pr(rows, plots / "val_threshold_pr.png", selected=opt_thr)

    logger.info("=== Untouched TEST evaluation at val threshold %.2f ===", opt_thr)
    eff_model = tf.keras.models.load_model(
        str(exp_dir / "efficientnetb0.keras"),
        custom_objects={"FocalLoss": FocalLoss},
        compile=False,
    )
    y_test, p_eff_test = _predict_dataset(eff_model, test_ds)
    del eff_model
    tf.keras.backend.clear_session()
    mob_model = tf.keras.models.load_model(
        str(exp_dir / "mobilenetv2.keras"),
        custom_objects={"FocalLoss": FocalLoss},
        compile=False,
    )
    _, p_mob_test = _predict_dataset(mob_model, test_ds)
    del mob_model
    tf.keras.backend.clear_session()

    test_arch = evaluate_architecture_on_split(
        y_test, p_eff_test, p_mob_test, "test", plots, threshold=opt_thr
    )
    val_arch_at_thr = evaluate_architecture_on_split(
        y_val, p_eff_val, p_mob_val, "val", plots, threshold=opt_thr
    )

    test_best = test_arch.get(
        "hybrid" if best_arch in {"ensemble", "hybrid"} else best_arch,
        test_arch["hybrid"],
    )
    val_best = val_arch_at_thr.get(
        "hybrid" if best_arch in {"ensemble", "hybrid"} else best_arch,
        val_arch_at_thr["hybrid"],
    )

    table_rows.append(
        _row_from_metrics(
            {**winner, "name": f"{winner_name}+hybrid"},
            val_best,
            test_best,
            threshold=opt_thr,
            best_epoch=compare_results[winner_name].get("best_epoch") or 0,
            frozen_desc=f"stage1 frozen + stage2 unfreeze top {config.FINE_TUNE_UNFREEZE_LAYERS}",
            lr=config.FINE_TUNE_LR,
        )
    )
    write_table(table_rows, Path(config.EXPERIMENTS_DIR) / "experiment_table.csv")

    summary = {
        "winner_balance": winner,
        "best_architecture_by_validation": best_arch,
        "threshold_selected_on": "validation",
        "threshold": opt_thr,
        "validation_at_threshold": _jsonable(val_best),
        "test_at_val_threshold": _jsonable(test_best),
        "validation_architectures_at_0.5": _jsonable(val_arch),
        "test_architectures_at_val_threshold": _jsonable(test_arch),
        "compare_val_pr_auc": {
            name: compare_results[name]["val_metrics"] for name in compare_results
        },
        "leakage_clean": report.get("clean"),
    }
    (Path(config.EXPERIMENTS_DIR) / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2), encoding="utf-8"
    )
    config.METRICS_PATH.write_text(
        json.dumps({"best_model": best_arch, "metrics": _jsonable(summary)}, indent=2),
        encoding="utf-8",
    )
    EnsemblePredictor.save_ensemble_meta(config.ENSEMBLE_WEIGHTS, test_arch["hybrid"])
    promote_winner(
        winner,
        opt_thr,
        best_arch,
        {
            "validation_metrics": val_best,
            "test_metrics_at_val_threshold": test_best,
        },
    )
    logger.info("Experiments complete. Summary written to models/saved/experiments/summary.json")


def parse_args():
    parser = argparse.ArgumentParser(description="Melanoma controlled experiments")
    parser.add_argument("--dataset", default=config.ACTIVE_DATASET, choices=list(config.DATASETS.keys()))
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--reuse-split", action="store_true")
    parser.add_argument("--skip-near-dup", action="store_true")
    parser.add_argument("--compare-finetune", action="store_true", help="Fine-tune during the 3-way compare (slower)")
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="Skip retraining; evaluate saved checkpoints and write the experiment summary",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--finetune-epochs", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.epochs:
        config.EPOCHS = int(args.epochs)
    if args.finetune_epochs is not None:
        config.FINE_TUNE_EPOCHS = int(args.finetune_epochs)
    else:
        config.FINE_TUNE_EPOCHS = min(int(config.FINE_TUNE_EPOCHS), 8)
    config.EPOCHS = min(int(config.EPOCHS), 12)
    config.PATIENCE_EARLY_STOP = min(int(config.PATIENCE_EARLY_STOP), 4)
    config.PATIENCE_REDUCE_LR = min(int(config.PATIENCE_REDUCE_LR), 2)

    run_experiments(
        dataset_key=args.dataset,
        audit_only=bool(args.audit_only),
        reuse_split=bool(args.reuse_split),
        check_near_duplicates=not bool(args.skip_near_dup),
        compare_finetune=bool(args.compare_finetune),
        finalize_only=bool(getattr(args, "finalize", False)),
    )
