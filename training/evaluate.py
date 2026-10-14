"""
Model evaluation utilities: metrics, curves, confusion matrix, thresholds.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import config
from utils.logger import setup_logger

logger = setup_logger("evaluate")

THRESHOLD_CANDIDATES: List[float] = list(
    getattr(
        config,
        "THRESHOLD_CANDIDATES",
        [0.10, 0.15, 0.20, 0.22, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
    )
)


def _as_arrays(y_true, y_prob) -> Tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob).astype(float).ravel()
    return y_true, y_prob


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute classification metrics from probabilities at one threshold."""
    y_true, y_prob = _as_arrays(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = _confusion_counts(y_true, y_pred)
    spec_den = tn + fp
    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / spec_den) if spec_den else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
        "pr_auc": (
            float(average_precision_score(y_true, y_prob))
            if len(np.unique(y_true)) > 1
            else 0.0
        ),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "n_positive": int(y_true.sum()),
        "n_negative": int((y_true == 0).sum()),
    }
    logger.info(
        "thr=%.2f acc=%.4f prec=%.4f rec=%.4f spec=%.4f f1=%.4f auc=%.4f pr_auc=%.4f",
        threshold,
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["specificity"],
        metrics["f1"],
        metrics["auc"],
        metrics["pr_auc"],
    )
    return metrics


def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return int(tn), int(fp), int(fn), int(tp)


def threshold_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[Sequence[float]] = None,
) -> List[Dict[str, float]]:
    thresholds = list(thresholds or THRESHOLD_CANDIDATES)
    return [evaluate_predictions(y_true, y_prob, threshold=float(t)) for t in thresholds]


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
    thresholds: Optional[Sequence[float]] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Choose a decision threshold on the provided labels/scores only.

    Search is restricted to the listed clinical candidates so a very low
    cutoff cannot be chosen just to inflate melanoma recall.
    Ties prefer the higher threshold (fewer false positives).
    """
    rows = threshold_table(y_true, y_prob, thresholds=thresholds)
    best = None
    for row in rows:
        score = float(row.get(metric, 0.0))
        if best is None or score > best[0] + 1e-12 or (
            abs(score - best[0]) <= 1e-12 and row["threshold"] > best[1]["threshold"]
        ):
            best = (score, row)
    assert best is not None
    logger.info(
        "Selected threshold=%.2f on provided split (%s=%.4f)",
        best[1]["threshold"],
        metric,
        best[0],
    )
    selected = {k: v for k, v in best[1].items() if k != "threshold_table"}
    selected["threshold_table"] = [
        {k: v for k, v in row.items() if k != "threshold_table"} for row in rows
    ]
    return float(selected["threshold"]), selected


def save_threshold_table(rows: List[Dict[str, float]], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = []
    for row in rows or []:
        cleaned.append({k: v for k, v in row.items() if k != "threshold_table"})
    if not cleaned:
        return path
    fieldnames = list(cleaned[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(cleaned)
    return path


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path,
    labels: Optional[List[str]] = None,
) -> Path:
    labels = labels or config.CLASS_NAMES
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Path,
    model_name: str = "Model",
) -> Path:
    y_true, y_prob = _as_arrays(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.4f})", color="#0B3D4F", lw=2)
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Path,
    model_name: str = "Model",
) -> Path:
    y_true, y_prob = _as_arrays(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    prevalence = float(y_true.mean()) if len(y_true) else 0.0
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"{model_name} (PR-AUC={pr_auc:.4f})", color="#C45C26", lw=2)
    plt.axhline(prevalence, color="gray", ls="--", lw=1, label=f"prevalence={prevalence:.3f}")
    plt.xlabel("Recall (melanoma)")
    plt.ylabel("Precision (melanoma)")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="upper right")
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def plot_threshold_pr(
    rows: List[Dict[str, float]],
    save_path: Path,
    selected: Optional[float] = None,
) -> Path:
    thr = [r["threshold"] for r in rows]
    prec = [r["precision"] for r in rows]
    rec = [r["recall"] for r in rows]
    f1 = [r["f1"] for r in rows]
    plt.figure(figsize=(7, 5))
    plt.plot(thr, prec, marker="o", label="Precision")
    plt.plot(thr, rec, marker="o", label="Recall")
    plt.plot(thr, f1, marker="o", label="F1")
    if selected is not None:
        plt.axvline(selected, color="black", ls="--", lw=1, label=f"selected={selected:.2f}")
    plt.xlabel("Decision threshold")
    plt.ylabel("Score")
    plt.title("Precision / Recall / F1 vs threshold")
    plt.legend()
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Path,
    model_name: str = "Model",
    n_bins: int = 10,
) -> Tuple[Path, Dict[str, float]]:
    y_true, y_prob = _as_arrays(y_true, y_prob)
    stats = {
        "brier": float(brier_score_loss(y_true, y_prob)),
        "mean_prob_benign": float(y_prob[y_true == 0].mean()) if np.any(y_true == 0) else 0.0,
        "mean_prob_melanoma": float(y_prob[y_true == 1].mean()) if np.any(y_true == 1) else 0.0,
    }
    try:
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    except ValueError:
        frac_pos, mean_pred = np.array([]), np.array([])

    plt.figure(figsize=(6, 5))
    if len(mean_pred):
        plt.plot(mean_pred, frac_pos, marker="o", label=model_name)
    plt.plot([0, 1], [0, 1], "k--", label="perfect calibration")
    plt.xlabel("Predicted melanoma probability")
    plt.ylabel("Observed melanoma fraction")
    plt.title(f"Reliability curve (Brier={stats['brier']:.4f})")
    plt.legend()
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path, stats


def plot_score_histogram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Path,
) -> Path:
    y_true, y_prob = _as_arrays(y_true, y_prob)
    plt.figure(figsize=(7, 4))
    plt.hist(y_prob[y_true == 0], bins=30, alpha=0.6, label="benign", color="#4C78A8")
    plt.hist(y_prob[y_true == 1], bins=30, alpha=0.6, label="melanoma", color="#C45C26")
    plt.xlabel("Predicted melanoma probability")
    plt.ylabel("Count")
    plt.title("Score distribution")
    plt.legend()
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def plot_training_curves(
    history: dict,
    save_dir: Path,
    model_name: str,
) -> Dict[str, Path]:
    """Plot loss, accuracy, AUC, PR-AUC and recall curves from Keras history."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    specs = [
        ("loss", "Loss", f"{model_name}_loss.png"),
        ("accuracy", "Accuracy", f"{model_name}_accuracy.png"),
        ("auc", "ROC-AUC", f"{model_name}_auc.png"),
        ("pr_auc", "PR-AUC", f"{model_name}_pr_auc.png"),
        ("recall", "Recall", f"{model_name}_recall.png"),
    ]
    for key, ylabel, filename in specs:
        train_s = history.get(key, [])
        val_s = history.get(f"val_{key}", [])
        if not train_s and not val_s:
            continue
        plt.figure(figsize=(7, 4))
        if train_s:
            plt.plot(train_s, label=f"Train {ylabel}")
        if val_s:
            plt.plot(val_s, label=f"Val {ylabel}")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"{model_name} — {ylabel} curves")
        plt.legend()
        plt.tight_layout()
        out = save_dir / filename
        plt.savefig(out, dpi=150)
        plt.close()
        paths[key] = out
    return paths


def best_epoch_from_history(history: dict, key: str = "val_pr_auc") -> int:
    series = history.get(key) or history.get("val_auc") or []
    if not series:
        return 0
    return int(np.argmax(np.asarray(series, dtype=float))) + 1


def full_evaluation_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    output_dir: Path,
    threshold: float = 0.5,
) -> Dict:
    """Generate metrics + plots + sklearn classification report at a given threshold."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    y_true, y_prob = _as_arrays(y_true, y_prob)
    y_pred = (y_prob >= float(threshold)).astype(int)
    metrics = evaluate_predictions(y_true, y_prob, threshold=float(threshold))

    plot_confusion_matrix(y_true, y_pred, output_dir / f"{model_name}_cm.png")
    plot_roc_curve(y_true, y_prob, output_dir / f"{model_name}_roc.png", model_name)
    plot_pr_curve(y_true, y_prob, output_dir / f"{model_name}_pr.png", model_name)
    plot_calibration_curve(
        y_true, y_prob, output_dir / f"{model_name}_calibration.png", model_name
    )
    plot_score_histogram(y_true, y_prob, output_dir / f"{model_name}_scores.png")

    report = classification_report(
        y_true, y_pred, target_names=config.CLASS_NAMES, digits=4, zero_division=0
    )
    report_path = output_dir / f"{model_name}_classification_report.txt"
    report_path.write_text(report, encoding="utf-8")

    metrics_path = output_dir / f"{model_name}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    logger.info("Evaluation artifacts saved to %s", output_dir)
    return {"metrics": metrics, "report": report}
