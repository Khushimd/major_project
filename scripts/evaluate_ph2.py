"""Evaluate the production hybrid model on the PH2 images in testing_data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from models.ensemble import EnsemblePredictor, combine_hybrid_probs, resolve_decision_threshold
from preprocessing.image_preprocess import load_and_preprocess
from training.evaluate import evaluate_predictions
from utils.logger import setup_logger

logger = setup_logger("eval_ph2")

IMAGES_DIR = ROOT / "testing_data" / "PH2 Dataset images"
LABELS_PATH = ROOT / "testing_data" / "PH2_dataset.txt"
OUT_PATH = config.MODELS_DIR / "eval_ph2.json"


def parse_labels(path: Path) -> dict[str, int]:
    """Clinical Diagnosis: 0 common nevus, 1 atypical nevus, 2 melanoma."""
    labels: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "||" not in line or "IMD" not in line:
            continue
        parts = [p.strip() for p in line.split("||")]
        if len(parts) < 4:
            continue
        name = parts[1]
        if not name.startswith("IMD"):
            continue
        try:
            clinical = int(parts[3])
        except ValueError:
            continue
        labels[name] = clinical
    return labels


def dermoscopic_path(case_id: str) -> Path:
    return IMAGES_DIR / case_id / f"{case_id}_Dermoscopic_Image" / f"{case_id}.bmp"


def main() -> None:
    labels = parse_labels(LABELS_PATH)
    cases = []
    missing = []
    for case_id, clinical in sorted(labels.items()):
        img = dermoscopic_path(case_id)
        if img.exists():
            cases.append((case_id, clinical, img))
        else:
            missing.append(case_id)
    logger.info("Labeled=%s found=%s missing=%s", len(labels), len(cases), len(missing))
    if not cases:
        raise SystemExit("No PH2 dermoscopic images found.")

    ensemble = EnsemblePredictor(use_tta=False)
    ensemble.load()
    thr = resolve_decision_threshold()
    logger.info("Decision threshold=%.2f", thr)

    y_true = []
    y_clinical = []
    y_eff = []
    y_mob = []
    rows = []
    for i, (case_id, clinical, img) in enumerate(cases, start=1):
        _, processed, _ = load_and_preprocess(str(img))
        batch = np.expand_dims(processed.astype(np.float32), axis=0)
        pred = ensemble.predict(batch, return_individual=True)
        mal = int(clinical == 2)
        p_eff = float(pred["individual"]["efficientnet"])
        p_mob = float(pred["individual"].get("mobilenet") or 0.0)
        p_hyb = float(pred["malignant_prob"])
        y_true.append(mal)
        y_clinical.append(clinical)
        y_eff.append(p_eff)
        y_mob.append(p_mob)
        rows.append(
            {
                "id": case_id,
                "clinical": clinical,
                "true_melanoma": mal,
                "pred": pred["prediction"],
                "hybrid_prob": p_hyb,
                "efficientnet": p_eff,
                "mobilenet": p_mob,
            }
        )
        if i % 25 == 0 or i == len(cases):
            logger.info("Scored %s / %s", i, len(cases))

    y_true_a = np.asarray(y_true, dtype=int)
    y_eff_a = np.asarray(y_eff, dtype=float)
    y_mob_a = np.asarray(y_mob, dtype=float)
    y_hyb_a = combine_hybrid_probs(y_eff_a, y_mob_a)

    clinical_counts = {
        "common_nevus": int(sum(c == 0 for c in y_clinical)),
        "atypical_nevus": int(sum(c == 1 for c in y_clinical)),
        "melanoma": int(sum(c == 2 for c in y_clinical)),
    }
    pred_by_clinical = {}
    for name, code in (("common_nevus", 0), ("atypical_nevus", 1), ("melanoma", 2)):
        idx = [i for i, c in enumerate(y_clinical) if c == code]
        n = len(idx)
        flagged = sum(1 for i in idx if y_hyb_a[i] >= thr)
        pred_by_clinical[name] = {
            "n": n,
            "predicted_melanoma": int(flagged),
            "predicted_non_melanoma": int(n - flagged),
        }

    summary = {
        "dataset": "PH2",
        "n": int(len(cases)),
        "missing": missing,
        "decision_threshold": float(thr),
        "clinical_counts": clinical_counts,
        "positive_class": "melanoma (clinical diagnosis = 2)",
        "efficientnet": evaluate_predictions(y_true_a, y_eff_a, threshold=thr),
        "mobilenet": evaluate_predictions(y_true_a, y_mob_a, threshold=thr),
        "hybrid": evaluate_predictions(y_true_a, y_hyb_a, threshold=thr),
        "predicted_melanoma_by_clinical": pred_by_clinical,
        "cases": rows,
    }
    OUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Saved %s", OUT_PATH)
    print(
        json.dumps(
            {
                "n": summary["n"],
                "clinical_counts": clinical_counts,
                "threshold": thr,
                "efficientnet": summary["efficientnet"],
                "mobilenet": summary["mobilenet"],
                "hybrid": summary["hybrid"],
                "predicted_melanoma_by_clinical": pred_by_clinical,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
