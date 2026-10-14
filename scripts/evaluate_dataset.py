"""
Evaluate saved models on a labeled dataset (no training).

Usage:
    python scripts/evaluate_dataset.py --dataset ham10000
    python scripts/evaluate_dataset.py --dataset custom
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from models.ensemble import EnsemblePredictor, combine_hybrid_probs, resolve_decision_threshold
from training.dataset_loader import MelanomaDataset
from training.evaluate import evaluate_predictions, find_optimal_threshold, full_evaluation_report
from training.train import _predict_dataset
from utils.logger import setup_logger

logger = setup_logger("eval_dataset")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate models on a labeled dataset")
    parser.add_argument(
        "--dataset",
        default="ham10000",
        choices=list(config.DATASETS.keys()),
        help="Dataset key (must have CSV labels + images on disk)",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    loader = MelanomaDataset(dataset_key=args.dataset)
    loader.load_metadata()
    if args.max_samples and len(loader.df) > args.max_samples:
        mal = loader.df[loader.df["label"] == 1]
        ben = loader.df[loader.df["label"] == 0]
        n_ben = max(args.max_samples - len(mal), args.max_samples // 5)
        ben = ben.sample(n=min(n_ben, len(ben)), random_state=config.SEED)
        loader.df = (
            pd.concat([mal, ben], ignore_index=True)
            .sample(frac=1.0, random_state=config.SEED)
            .reset_index(drop=True)
        )

    _, _, test_df = loader.split()
    test_ds = loader.make_tf_dataset(test_df, batch_size=16, shuffle=False, augment=False)

    ensemble = EnsemblePredictor(use_tta=False)
    ensemble.load()
    thr = resolve_decision_threshold()
    logger.info("Using decision threshold=%.2f", thr)

    y_true_list, y_prob_e, y_prob_m = [], [], []
    for batch_x, batch_y in test_ds:
        batch = np.asarray(batch_x)
        for i in range(batch.shape[0]):
            single = np.expand_dims(batch[i], axis=0)
            pred = ensemble.predict(single, return_individual=True)
            y_prob_e.append(pred["individual"]["efficientnet"])
            y_prob_m.append(pred["individual"].get("mobilenet", 0.0))
        y_true_list.append(np.asarray(batch_y).ravel())

    y_true = np.concatenate(y_true_list)
    y_prob_e = np.asarray(y_prob_e)
    y_prob_m = np.asarray(y_prob_m)
    ens_prob = combine_hybrid_probs(y_prob_e, y_prob_m)

    out_dir = config.MODELS_DIR / "plots" / f"eval_{args.dataset}"
    eff_m = full_evaluation_report(y_true, y_prob_e, "efficientnet", out_dir)["metrics"]
    mob_m = evaluate_predictions(y_true, y_prob_m, threshold=thr)
    ens_m = full_evaluation_report(y_true, ens_prob, "hybrid", out_dir)["metrics"]
    opt_thr, opt_m = find_optimal_threshold(y_true, ens_prob)

    summary = {
        "dataset": args.dataset,
        "n_test": int(len(y_true)),
        "decision_threshold": thr,
        "optimal_threshold": opt_thr,
        "efficientnet": eff_m,
        "mobilenet": mob_m,
        "ensemble": ens_m,
        "ensemble_at_optimal_threshold": opt_m,
    }
    out_path = config.MODELS_DIR / f"eval_{args.dataset}.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Saved report → %s", out_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
