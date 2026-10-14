"""
One frozen external-validation run of Exp2-C on the frozen Derm7pt cohort.

Does NOT retrain, fine-tune, retune the 0.30 threshold, search ensemble
weights, calibrate, redo the audit, or modify production / old-experiment /
frozen-CSV / baseline artifacts.

INFERENCE RUNS: ONCE
TRAINING PERFORMED: NO
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from models.architecture import FocalLoss
from models.ensemble import combine_hybrid_probs
from preprocessing.image_preprocess import ImagePreprocessor
from training.evaluate import (
    evaluate_predictions,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
)
from training.train import _predict_dataset

OUT_DIR = ROOT / "models" / "experiments" / "old_model_derm7pt_external"
MANIFEST = OUT_DIR / "external_manifest.csv"
CACHE_DIR = OUT_DIR / "preprocessed_cache"

EFF_PATH = (
    ROOT
    / "models"
    / "experiments"
    / "internal_optimization"
    / "exp2_loss_comparison"
    / "exp2c_bce_smooth"
    / "efficientnetb0_bce_smooth.keras"
)
MOB_PATH = (
    ROOT
    / "models"
    / "experiments"
    / "internal_optimization"
    / "exp2_loss_comparison"
    / "exp2c_bce_smooth"
    / "mobilenetv2_bce_smooth.keras"
)

WEIGHTS = (0.75, 0.25)
THRESHOLD = 0.30
EXPECTED = {"n": 987, "melanoma": 252, "non_melanoma": 735, "excluded": 24}
EXPECTED_EFF = "02755e9c69e9f885b71696a278f807c9a93c29b88d59f7d252002f8bbbb62f87"
EXPECTED_MOB = "ad049149423ad1a5a73f9b1bd6dabd0765968ae532472b8afd5757eb7f33146f"
FROZEN_CSVS = {
    "manifest": ROOT / "dataset" / "processed" / "melanoma_benign" / "manifest.csv",
    "train": ROOT / "dataset" / "processed" / "melanoma_benign" / "train.csv",
    "val": ROOT / "dataset" / "processed" / "melanoma_benign" / "val.csv",
    "test": ROOT / "dataset" / "processed" / "melanoma_benign" / "test.csv",
}
EXPECTED_CSV_SHA = {
    "manifest": "1b619892e5e1ffa202040e8672deb820bcb99641739f509546320ccdc082882a",
    "train": "73b1b10e28f24d679cb95698e0c3bd6172a9187cd7b7146ec174168c0b89a99a",
    "val": "f6ff80b017b15810bf795089cd474d2cefafbe03cba0c0c3f2065cba9c798226",
    "test": "27d27e79f5ff3566e2c41f7a6f7bd635bd8e01bae2bbb0222b27dd1bde141a5b",
}
PROTECTED_MODELS = [
    ROOT / "models" / "saved" / "efficientnetb0_melanoma.keras",
    ROOT / "models" / "saved" / "mobilenetv2_melanoma.keras",
    ROOT
    / "models"
    / "experiments"
    / "final_melanoma_benign_baseline"
    / "efficientnetb0_final_baseline.keras",
    ROOT
    / "models"
    / "experiments"
    / "final_melanoma_benign_baseline"
    / "mobilenetv2_final_baseline.keras",
    EFF_PATH,
    MOB_PATH,
]
AUDIT_ONLY = {
    MANIFEST,
    OUT_DIR / "audit_report.md",
    OUT_DIR / "audit_report.json",
    OUT_DIR / "derm7pt_audit_report.md",
    OUT_DIR / "derm7pt_audit_report.json",
    OUT_DIR / "derm7pt_external_manifest.csv",
    OUT_DIR / "label_mapping.json",
    OUT_DIR / "overlap_report.csv",
    OUT_DIR / "overlap_report.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _json_dump(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_jsonable), encoding="utf-8")


def _fingerprint(paths: list[Path]) -> dict:
    out = {}
    for path in paths:
        out[str(path)] = {
            "exists": path.is_file(),
            "size": int(path.stat().st_size) if path.is_file() else None,
            "sha256": _sha256(path) if path.is_file() else None,
        }
    return out


def _score_dist(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64).ravel()
    return {
        "n": int(values.size),
        "min": float(np.min(values)) if values.size else None,
        "max": float(np.max(values)) if values.size else None,
        "mean": float(np.mean(values)) if values.size else None,
        "median": float(np.median(values)) if values.size else None,
    }


def load_frozen_cohort() -> pd.DataFrame:
    if not MANIFEST.is_file():
        raise RuntimeError(f"STOP: frozen manifest missing: {MANIFEST}")
    full = pd.read_csv(MANIFEST)
    if len(full) != 1011:
        raise RuntimeError(f"STOP: frozen manifest row count changed: {len(full)}")
    n_excl = int((full["final_evaluation_inclusion"] != "yes").sum())
    if n_excl != EXPECTED["excluded"]:
        raise RuntimeError(f"STOP: excluded count changed: {n_excl}")
    df = full.loc[full["final_evaluation_inclusion"] == "yes"].copy()
    df["binary_label"] = df["binary_label"].astype(int)
    n = len(df)
    n_mel = int((df["binary_label"] == 1).sum())
    n_non = int((df["binary_label"] == 0).sum())
    if n != EXPECTED["n"] or n_mel != EXPECTED["melanoma"] or n_non != EXPECTED["non_melanoma"]:
        raise RuntimeError(f"STOP: frozen cohort counts changed: n={n} mel={n_mel} non={n_non}")
    if int(df["clinic_used_for_evaluation"].eq("no").sum()) != n:
        raise RuntimeError("STOP: clinical images marked for evaluation")
    missing = [p for p in df["derm_abspath"].astype(str) if not Path(p).is_file()]
    if missing:
        raise RuntimeError(f"STOP: {len(missing)} dermoscopic images missing, e.g. {missing[:3]}")
    return df.reset_index(drop=True)


def cache_preprocessed(df: pd.DataFrame) -> pd.DataFrame:
    """Reuse Exp2-C evaluation cache format: process_for_model then JPEG."""
    prep = ImagePreprocessor(target_size=(224, 224), apply_clahe=False, apply_hair_removal=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    rebuilt = 0
    for i, (row_index, src) in enumerate(
        zip(df["derm7pt_row_index"].astype(int), df["derm_abspath"].astype(str))
    ):
        cached = CACHE_DIR / f"{int(row_index):04d}.jpg"
        if cached.is_file():
            paths.append(str(cached))
            continue
        bgr = cv2.imread(src, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(src)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        arr = prep.process_for_model(rgb)
        jpeg = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(str(cached), cv2.cvtColor(jpeg, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        paths.append(str(cached))
        rebuilt += 1
        if rebuilt % 50 == 0:
            print(f"Preprocessed {i + 1}/{len(df)}", flush=True)
    print(f"Preprocessed cache ready ({rebuilt} newly written, {len(df) - rebuilt} reused)", flush=True)
    out = df.copy()
    out["cache_path"] = paths
    return out


def make_ds(df: pd.DataFrame) -> tf.data.Dataset:
    paths = df["cache_path"].astype(str).values
    labels = df["binary_label"].values.astype(np.float32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _decode(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, (224, 224))
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    return ds.map(_decode, num_parallel_calls=tf.data.AUTOTUNE).batch(8).prefetch(tf.data.AUTOTUNE)


def diagnosis_breakdown(df: pd.DataFrame) -> list[dict]:
    rows = []
    for diagnosis, chunk in df.groupby("original_diagnosis", sort=False):
        n = int(len(chunk))
        row = {
            "original_diagnosis": str(diagnosis),
            "n_cases": n,
            "true_melanoma": int((chunk["true_binary_label"] == 1).sum()),
            "true_non_melanoma": int((chunk["true_binary_label"] == 0).sum()),
            "predicted_melanoma": int((chunk["predicted_label"] == 1).sum()),
            "predicted_non_melanoma": int((chunk["predicted_label"] == 0).sum()),
            "mean_ensemble_score": float(chunk["ensemble_score"].mean()),
            "sample_size_note": (
                "descriptive_only_sample_too_small"
                if n < 30
                else "descriptive_counts_and_mean_score_only"
            ),
        }
        rows.append(row)
    return rows


def render_report(payload: dict) -> str:
    m = payload["derm7pt_external"]["metrics"]
    cm = payload["derm7pt_external"]["confusion_matrix"]
    lines = [
        "# Frozen old Exp2-C model — Derm7pt external validation",
        "",
        '"The previously frozen production model was evaluated on the frozen Derm7pt '
        'external cohort without retraining, fine-tuning, threshold optimization, '
        'ensemble-weight optimization, or probability calibration."',
        "",
        "Scores below are model-predicted melanoma probability/score estimates. "
        "They are not calibrated clinical probabilities.",
        "",
        "## 1. Objective",
        "",
        "Obtain one unbiased external-validation result for the already-frozen old "
        "Exp2-C ensemble on the already-frozen Derm7pt dermoscopic cohort.",
        "",
        "## 2. Frozen model specification",
        "",
        "- EfficientNetB0 + MobileNetV2 soft-voting ensemble",
        "- EfficientNet weight: 0.75",
        "- MobileNet weight: 0.25",
        "- Operating threshold: 0.30 (frozen; not searched on Derm7pt)",
        f"- EfficientNet: `{payload['model_files']['efficientnetb0']['path']}`",
        f"- EfficientNet SHA-256: `{payload['model_files']['efficientnetb0']['sha256']}`",
        f"- MobileNet: `{payload['model_files']['mobilenetv2']['path']}`",
        f"- MobileNet SHA-256: `{payload['model_files']['mobilenetv2']['sha256']}`",
        "- Preprocessing: existing `ImagePreprocessor.process_for_model` "
        "(hair removal + light denoise + resize 224 + /255; no CLAHE; no lesion crop; "
        "ImageNet preprocess inside the saved models)",
        "",
        "## 3. OLD INTERNAL TEST (frozen, not recomputed)",
        "",
        "This is **not** the Derm7pt result.",
        "",
        "- Test images: 2,473",
        "- Accuracy: 92.28%",
        "- Sensitivity: 95.24%",
        "- Specificity: 88.86%",
        "- Precision: 90.78%",
        "- F1: 92.96%",
        "- ROC-AUC: 97.99%",
        "- PR-AUC: 98.36%",
        "- TP 1,261 / TN 1,021 / FP 128 / FN 63",
        "",
        "This historical internal result used DICM `class=1` as the positive class, "
        "which is NOT equivalent to a pure melanoma label. Therefore 92.28% is not a "
        "clean melanoma-vs-benign result.",
        "",
        "## 4. DERM7PT EXTERNAL VALIDATION",
        "",
        f"- Evaluated cases: {payload['cohort']['n_evaluated']}",
        f"- Melanoma: {payload['cohort']['n_melanoma']}",
        f"- Non-melanoma: {payload['cohort']['n_non_melanoma']}",
        f"- Excluded (not scored): {payload['cohort']['n_excluded']}",
        "- Image used: dermoscopic only",
        "",
        f"- Accuracy: {m['accuracy']:.4f} ({m['accuracy'] * 100:.2f}%)",
        f"- Sensitivity / Recall: {m['sensitivity']:.4f} ({m['sensitivity'] * 100:.2f}%)",
        f"- Specificity: {m['specificity']:.4f} ({m['specificity'] * 100:.2f}%)",
        f"- Precision: {m['precision']:.4f} ({m['precision'] * 100:.2f}%)",
        f"- F1-score: {m['f1']:.4f}",
        f"- ROC-AUC: {m['auc']:.4f} (continuous ensemble scores)",
        f"- PR-AUC: {m['pr_auc']:.4f} (continuous ensemble scores)",
        "",
        f"- TP: {m['tp']}",
        f"- TN: {m['tn']}",
        f"- FP: {m['fp']}",
        f"- FN: {m['fn']}",
        "",
        "Do not combine this external result with the old internal 92.28% numbers.",
        "",
        "## 5. Confusion matrix",
        "",
        "```",
        "                 Predicted Benign    Predicted Melanoma",
        f"Actual Benign    {cm['tn']:<18} {cm['fp']}",
        f"Actual Melanoma  {cm['fn']:<18} {cm['tp']}",
        "```",
        "",
        "Here “Benign” is the frozen Derm7pt non-melanoma class (nevi, BCC, SK, and "
        "other clearly non-melanoma diagnoses), not a claim that every negative case "
        "is clinically benign.",
        "",
        "## 6. Diagnosis distribution (descriptive)",
        "",
        "These rows are counts and mean model-predicted melanoma scores. Accuracy is "
        "not reported for small categories.",
        "",
        "| Diagnosis | n | Predicted melanoma | Predicted non-melanoma | Mean ensemble score | Note |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in payload["diagnosis_breakdown"]:
        lines.append(
            f"| {row['original_diagnosis']} | {row['n_cases']} | "
            f"{row['predicted_melanoma']} | {row['predicted_non_melanoma']} | "
            f"{row['mean_ensemble_score']:.4f} | {row['sample_size_note']} |"
        )
    lines.extend(
        [
            "",
            "## 7. Limitations",
            "",
            "- External dataset shift: Derm7pt cameras, population, and labels differ from the old development set.",
            "- The old model was trained with DICM `class=1` as the positive class, not pure melanoma.",
            "- Derm7pt negatives include BCC and other non-melanoma diagnoses.",
            "- Patient-level independence could not be completely proven.",
            "- Threshold 0.30 was frozen before this evaluation and was not optimized on Derm7pt.",
            "- This is not a clinical validation study.",
            "",
            "## 8. Reproducibility",
            "",
            f"- Evaluation UTC: {payload['datetime_utc']}",
            f"- Python: {payload['software']['python']}",
            f"- TensorFlow: {payload['software']['tensorflow']}",
            f"- pandas: {payload['software']['pandas']}",
            "- Script: `scripts/eval_old_model_derm7pt_external.py`",
            "- Frozen cohort: `models/experiments/old_model_derm7pt_external/external_manifest.csv`",
            "- Training performed: no",
            "- Threshold search: no",
            "- Ensemble-weight search: no",
            "- Calibration: no",
            "",
            "## 9. STOP",
            "",
            "External validation complete. No further experiment started.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    started = datetime.now(timezone.utc).isoformat()
    print("Snapshotting protected files...", flush=True)
    before_models = _fingerprint(PROTECTED_MODELS)
    before_csvs = {name: _sha256(path) for name, path in FROZEN_CSVS.items()}
    for name, digest in before_csvs.items():
        if digest != EXPECTED_CSV_SHA[name]:
            raise RuntimeError(f"STOP: frozen {name}.csv hash changed before evaluation")
    if before_models[str(EFF_PATH)]["sha256"] != EXPECTED_EFF:
        raise RuntimeError("STOP: EfficientNet hash is not the frozen Exp2-C file")
    if before_models[str(MOB_PATH)]["sha256"] != EXPECTED_MOB:
        raise RuntimeError("STOP: MobileNet hash is not the frozen Exp2-C file")

    audit_before = {path: _sha256(path) for path in AUDIT_ONLY if path.is_file()}

    print("Loading frozen 987-case cohort...", flush=True)
    cohort = load_frozen_cohort()
    cohort = cache_preprocessed(cohort)
    ds = make_ds(cohort)

    print(f"Loading EfficientNet {EFF_PATH}", flush=True)
    eff = tf.keras.models.load_model(str(EFF_PATH), custom_objects={"FocalLoss": FocalLoss}, compile=False)
    y_true, p_eff = _predict_dataset(eff, ds)
    del eff
    tf.keras.backend.clear_session()
    print("EfficientNet inference done", flush=True)

    print(f"Loading MobileNet {MOB_PATH}", flush=True)
    mob = tf.keras.models.load_model(str(MOB_PATH), custom_objects={"FocalLoss": FocalLoss}, compile=False)
    y_true_m, p_mob = _predict_dataset(mob, ds)
    del mob
    tf.keras.backend.clear_session()
    print("MobileNet inference done", flush=True)

    if not np.array_equal(y_true, y_true_m):
        raise RuntimeError("STOP: label order mismatch between backbones")
    if len(y_true) != EXPECTED["n"]:
        raise RuntimeError("STOP: prediction count != 987")
    if not np.array_equal(y_true.astype(int), cohort["binary_label"].values.astype(int)):
        raise RuntimeError("STOP: prediction labels drifted from the frozen manifest")

    p_ens = combine_hybrid_probs(p_eff, p_mob, weights=WEIGHTS, method="soft_voting")
    y_pred = (np.asarray(p_ens).ravel() >= THRESHOLD).astype(int)
    metrics = evaluate_predictions(y_true, p_ens, threshold=THRESHOLD)
    if metrics["threshold"] != THRESHOLD:
        raise RuntimeError("STOP: threshold changed")
    if metrics["tp"] + metrics["fn"] != EXPECTED["melanoma"]:
        raise RuntimeError("STOP: TP+FN != 252")
    if metrics["tn"] + metrics["fp"] != EXPECTED["non_melanoma"]:
        raise RuntimeError("STOP: TN+FP != 735")
    if metrics["tp"] + metrics["tn"] + metrics["fp"] + metrics["fn"] != EXPECTED["n"]:
        raise RuntimeError("STOP: confusion counts do not sum to 987")

    pred = pd.DataFrame(
        {
            "derm7pt_row_index": cohort["derm7pt_row_index"].astype(int).values,
            "case_num": cohort["case_num"].astype(int).values,
            "official_split": cohort["official_split"].astype(str).values,
            "image_filename": cohort["derm_filename"].astype(str).values,
            "derm_relpath": cohort["derm_relpath"].astype(str).values,
            "original_diagnosis": cohort["original_diagnosis"].astype(str).values,
            "true_binary_label": y_true.astype(int),
            "efficientnet_score": np.asarray(p_eff).ravel(),
            "mobilenet_score": np.asarray(p_mob).ravel(),
            "ensemble_score": np.asarray(p_ens).ravel(),
            "threshold": THRESHOLD,
            "predicted_label": y_pred,
        }
    )
    pred["predicted_label_name"] = np.where(pred["predicted_label"] == 1, "melanoma", "non_melanoma")
    pred.to_csv(OUT_DIR / "external_predictions.csv", index=False)

    cm_df = pd.DataFrame(
        {
            "": ["Actual Benign", "Actual Melanoma"],
            "Predicted Benign": [metrics["tn"], metrics["fn"]],
            "Predicted Melanoma": [metrics["fp"], metrics["tp"]],
        }
    )
    cm_df.to_csv(OUT_DIR / "confusion_matrix.csv", index=False)
    plot_confusion_matrix(
        y_true,
        y_pred,
        OUT_DIR / "confusion_matrix.png",
        labels=["Benign", "Melanoma"],
    )
    plot_roc_curve(y_true, p_ens, OUT_DIR / "roc_curve.png", model_name="Frozen Exp2-C ensemble")
    plot_pr_curve(
        y_true,
        p_ens,
        OUT_DIR / "precision_recall_curve.png",
        model_name="Frozen Exp2-C ensemble",
    )

    breakdown = diagnosis_breakdown(pred)
    dist = {
        "overall": _score_dist(p_ens),
        "actual_melanoma": _score_dist(p_ens[y_true == 1]),
        "actual_non_melanoma": _score_dist(p_ens[y_true == 0]),
    }
    payload = {
        "title": "Derm7pt external validation of the frozen old Exp2-C model",
        "methodological_statement": (
            "The previously frozen production model was evaluated on the frozen "
            "Derm7pt external cohort without retraining, fine-tuning, threshold "
            "optimization, ensemble-weight optimization, or probability calibration."
        ),
        "datetime_utc": datetime.now(timezone.utc).isoformat(),
        "started_utc": started,
        "model_files": {
            "efficientnetb0": {"path": str(EFF_PATH), "sha256": EXPECTED_EFF},
            "mobilenetv2": {"path": str(MOB_PATH), "sha256": EXPECTED_MOB},
        },
        "ensemble_weights": {"efficientnet": WEIGHTS[0], "mobilenet": WEIGHTS[1]},
        "threshold": THRESHOLD,
        "preprocessing": (
            "ImagePreprocessor.process_for_model: hair removal + light denoise + "
            "resize 224 + /255; no CLAHE; no lesion crop; ImageNet preprocess inside "
            "the saved models. Cache format matches the original Exp2-C frozen-test "
            "evaluation (JPEG of the processed array, then TF decode/resize/255)."
        ),
        "cohort": {
            "n_input_cases": 1011,
            "n_evaluated": EXPECTED["n"],
            "n_melanoma": EXPECTED["melanoma"],
            "n_non_melanoma": EXPECTED["non_melanoma"],
            "n_excluded": EXPECTED["excluded"],
            "clinical_images_used": False,
        },
        "old_internal_test_quoted_not_recomputed": {
            "test_images": 2473,
            "accuracy": 0.9228,
            "sensitivity": 0.9524,
            "specificity": 0.8886,
            "label_caveat": (
                "DICM class=1 was the positive class. This is not a clean "
                "melanoma-vs-benign result."
            ),
        },
        "derm7pt_external": {
            "metrics": metrics,
            "confusion_matrix": {
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tp": metrics["tp"],
            },
            "predicted_melanoma": int((y_pred == 1).sum()),
            "predicted_non_melanoma": int((y_pred == 0).sum()),
            "score_distribution": dist,
        },
        "diagnosis_breakdown": breakdown,
        "integrity": {
            "inference_runs": 1,
            "training_performed": False,
            "threshold_tuning": False,
            "ensemble_weight_tuning": False,
            "calibration": False,
            "audit_redone": False,
            "old_internal_result_recomputed": False,
            "threshold_used": THRESHOLD,
            "weights_used": list(WEIGHTS),
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    _json_dump(OUT_DIR / "external_metrics.json", payload)
    (OUT_DIR / "external_metrics.md").write_text(render_report(payload) + "\n", encoding="utf-8")

    after_models = _fingerprint(PROTECTED_MODELS)
    after_csvs = {name: _sha256(path) for name, path in FROZEN_CSVS.items()}
    after_audit = {path: _sha256(path) for path in AUDIT_ONLY if path.is_file()}
    if after_models != before_models:
        raise RuntimeError("STOP: a protected model file changed during evaluation")
    if after_csvs != before_csvs:
        raise RuntimeError("STOP: a frozen development CSV changed during evaluation")
    if after_audit != audit_before:
        raise RuntimeError("STOP: an audit/frozen-cohort file changed during evaluation")

    existing_run = {}
    run_path = OUT_DIR / "run_config.json"
    if run_path.is_file():
        existing_run = json.loads(run_path.read_text(encoding="utf-8-sig"))
    run_config = {
        **existing_run,
        "task": "old_model_derm7pt_external_validation",
        "phase_completed": "frozen_external_inference",
        "inference_run": True,
        "inference_runs": 1,
        "training_run": False,
        "audit_redone": False,
        "model_files": payload["model_files"],
        "preprocessing_implementation": payload["preprocessing"],
        "ensemble_weights": payload["ensemble_weights"],
        "threshold": THRESHOLD,
        "threshold_optimization_on_derm7pt": False,
        "ensemble_weight_optimization_on_derm7pt": False,
        "calibration_on_derm7pt": False,
        "n_input_cases": 1011,
        "n_excluded_cases": 24,
        "n_evaluated_cases": 987,
        "datetime_utc": payload["datetime_utc"],
        "audit_datetime_utc": existing_run.get("datetime_utc"),
        "software": payload["software"],
        "old_internal_result_recomputed": False,
        "production_or_live_app_modified": False,
        "old_experiment_modified": False,
        "frozen_csvs_modified": False,
        "baseline_83_4_modified": False,
        "protected_file_hashes_unchanged": True,
    }
    _json_dump(run_path, run_config)

    print(
        "EXTERNAL VALIDATION COMPLETE "
        f"acc={metrics['accuracy']:.4f} sens={metrics['sensitivity']:.4f} "
        f"spec={metrics['specificity']:.4f} f1={metrics['f1']:.4f} "
        f"auc={metrics['auc']:.4f} pr={metrics['pr_auc']:.4f} "
        f"TP={metrics['tp']} TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']}"
    )
    print("STOP. No further experiment started.")


if __name__ == "__main__":
    main()
