"""
Model evaluation utilities: metrics, curves, confusion matrix, classification report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import config
from utils.logger import setup_logger

logger = setup_logger("evaluate")


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute classification metrics from probabilities."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob).astype(float).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
    }
    logger.info("Metrics: %s", {k: round(v, 4) for k, v in metrics.items()})
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path,
    labels: Optional[List[str]] = None,
) -> Path:
    labels = labels or config.CLASS_NAMES
    cm = confusion_matrix(y_true, y_pred)
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


def plot_training_curves(
    history: dict,
    save_dir: Path,
    model_name: str,
) -> Tuple[Path, Path]:
    """Plot loss and accuracy curves from Keras history dict."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    loss_path = save_dir / f"{model_name}_loss.png"
    acc_path = save_dir / f"{model_name}_accuracy.png"

    plt.figure(figsize=(7, 4))
    plt.plot(history.get("loss", []), label="Train Loss")
    plt.plot(history.get("val_loss", []), label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} — Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path, dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(history.get("accuracy", []), label="Train Acc")
    plt.plot(history.get("val_accuracy", []), label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} — Accuracy Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(acc_path, dpi=150)
    plt.close()

    return loss_path, acc_path


def full_evaluation_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    output_dir: Path,
) -> Dict:
    """Generate metrics + plots + sklearn classification report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    y_pred = (np.asarray(y_prob).ravel() >= 0.5).astype(int)
    metrics = evaluate_predictions(y_true, y_prob)

    plot_confusion_matrix(y_true, y_pred, output_dir / f"{model_name}_cm.png")
    plot_roc_curve(y_true, y_prob, output_dir / f"{model_name}_roc.png", model_name)

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
