"""
Paired PH2 evaluation: frozen 83.42% baseline vs mild-contrast augmentation.

Same 200 PH2 cases, frozen ensemble 0.70/0.30, frozen threshold 0.50.
No PH2 training, threshold search, ensemble search, or calibration.
Does not overwrite production, the frozen baseline, or prior PH2 reports.
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

from models.ensemble import combine_hybrid_probs
from preprocessing.image_preprocess import ImagePreprocessor
from training.evaluate import evaluate_predictions, plot_confusion_matrix, plot_pr_curve, plot_roc_curve
from training.train import _predict_dataset

OUT_DIR = ROOT / "models" / "experiments" / "final_melanoma_benign_mild_contrast_ph2"
CACHE_DIR = OUT_DIR / "preprocessed_cache"
IMAGES_DIR = ROOT / "testing_data" / "PH2 Dataset images"
LABELS_PATH = ROOT / "testing_data" / "PH2_dataset.txt"

BASELINE_DIR = ROOT / "models" / "experiments" / "final_melanoma_benign_baseline"
BASE_EFF = BASELINE_DIR / "efficientnetb0_final_baseline.keras"
BASE_MOB = BASELINE_DIR / "mobilenetv2_final_baseline.keras"

AUG_DIR = ROOT / "models" / "experiments" / "final_melanoma_benign_mild_contrast_augmentation"
AUG_EFF = AUG_DIR / "efficientnetb0_mild_contrast.keras"
AUG_MOB = AUG_DIR / "mobilenetv2_mild_contrast.keras"
AUG_SUMMARY = AUG_DIR / "training_summary.json"

WEIGHTS = (0.70, 0.30)
THRESHOLD = 0.50
EXPECTED = {"n": 200, "melanoma": 40, "strict_benign": 160, "common_nevus": 80, "atypical_nevus": 80}

PROTECTED = [
    ROOT / "models" / "saved" / "efficientnetb0_melanoma.keras",
    ROOT / "models" / "saved" / "mobilenetv2_melanoma.keras",
    ROOT / "models" / "saved" / "inference_config.json",
    BASE_EFF,
    BASE_MOB,
    BASELINE_DIR / "training_summary.json",
    ROOT / "dataset" / "processed" / "final_melanoma_benign" / "train.csv",
    ROOT / "dataset" / "processed" / "final_melanoma_benign" / "val.csv",
    ROOT / "dataset" / "processed" / "final_melanoma_benign" / "test.csv",
    ROOT / "dataset" / "processed" / "final_melanoma_benign" / "manifest.csv",
    ROOT / "models" / "experiments" / "ph2_external_evaluation" / "ph2_external_results.json",
]


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_jsonable), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def snapshot(paths) -> dict:
    out = {}
    for path in paths:
        if path.is_file():
            out[str(path.resolve())] = {"sha256": _sha256(path), "size": path.stat().st_size}
    return out


def parse_ph2() -> pd.DataFrame:
    labels: dict[str, int] = {}
    for line in LABELS_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
        if "||" not in line or "IMD" not in line:
            continue
        parts = [p.strip() for p in line.split("||")]
        if len(parts) < 4 or not parts[1].startswith("IMD"):
            continue
        try:
            clinical = int(parts[3])
        except ValueError:
            continue
        labels[parts[1]] = clinical
    rows = []
    missing = []
    for case_id, clinical in sorted(labels.items()):
        path = IMAGES_DIR / case_id / f"{case_id}_Dermoscopic_Image" / f"{case_id}.bmp"
        if not path.is_file():
            missing.append(case_id)
            continue
        if clinical == 2:
            label, name, subtype = 1, "melanoma", "melanoma"
        elif clinical in (0, 1):
            label, name, subtype = 0, "strict_benign", ("common_nevus" if clinical == 0 else "atypical_nevus")
        else:
            continue
        rows.append(
            {
                "image_id": case_id,
                "clinical_diagnosis": clinical,
                "subtype": subtype,
                "label": label,
                "label_name": name,
                "image_path": str(path.resolve()),
            }
        )
    if missing:
        raise RuntimeError(f"STOP: missing PH2 images: {missing}")
    df = pd.DataFrame(rows)
    got = {
        "n": int(len(df)),
        "melanoma": int((df["label"] == 1).sum()),
        "strict_benign": int((df["label"] == 0).sum()),
        "common_nevus": int((df["subtype"] == "common_nevus").sum()),
        "atypical_nevus": int((df["subtype"] == "atypical_nevus").sum()),
    }
    if got != EXPECTED:
        raise RuntimeError(f"STOP: PH2 cohort {got} != {EXPECTED}")
    return df


def cache_preprocessed(df: pd.DataFrame) -> pd.DataFrame:
    prep = ImagePreprocessor(target_size=(224, 224), apply_clahe=False, apply_hair_removal=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    rebuilt = 0
    for image_id, src in zip(df["image_id"].astype(str), df["image_path"].astype(str)):
        cached = CACHE_DIR / f"{image_id}.jpg"
        if cached.is_file():
            paths.append(str(cached))
            continue
        bgr = cv2.imread(src, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(src)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        longest = max(h, w)
        if longest > 1024:
            scale = 1024.0 / float(longest)
            rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        arr = prep.process_for_model(rgb)
        jpeg = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        ok = cv2.imwrite(str(cached), cv2.cvtColor(jpeg, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            raise RuntimeError(f"Failed to write cache {cached}")
        paths.append(str(cached))
        rebuilt += 1
    print(f"PH2 preprocess cache ready ({rebuilt} newly written)", flush=True)
    out = df.copy()
    out["cache_path"] = paths
    return out


def make_ds(df: pd.DataFrame) -> tf.data.Dataset:
    paths = df["cache_path"].astype(str).values
    labels = df["label"].values.astype(np.float32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _decode(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, (224, 224))
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    return ds.map(_decode, num_parallel_calls=tf.data.AUTOTUNE).batch(8).prefetch(tf.data.AUTOTUNE)


def run_model(eff_path: Path, mob_path: Path, df: pd.DataFrame) -> dict:
    ds = make_ds(df)
    print(f"Predicting EfficientNet {eff_path.name}", flush=True)
    eff = tf.keras.models.load_model(str(eff_path), compile=False)
    y_true, p_eff = _predict_dataset(eff, ds)
    del eff
    tf.keras.backend.clear_session()
    print(f"Predicting MobileNet {mob_path.name}", flush=True)
    mob = tf.keras.models.load_model(str(mob_path), compile=False)
    y_true2, p_mob = _predict_dataset(mob, ds)
    del mob
    tf.keras.backend.clear_session()
    if not np.array_equal(np.asarray(y_true), np.asarray(y_true2)):
        raise RuntimeError("Label order mismatch between backbones")
    ens = combine_hybrid_probs(p_eff, p_mob, weights=WEIGHTS, method="soft_voting")
    return {
        "y_true": np.asarray(y_true).astype(int),
        "p_eff": np.asarray(p_eff).ravel(),
        "p_mob": np.asarray(p_mob).ravel(),
        "ensemble": np.asarray(ens).ravel(),
    }


def cm_frame(metrics: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]],
        index=["true_strict_benign", "true_melanoma"],
        columns=["pred_benign", "pred_melanoma"],
    )


def fmt(m: dict, key: str) -> str:
    v = m[key]
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def observations(base_m: dict, aug_m: dict) -> str:
    return (
        f"Frozen baseline sensitivity {base_m['sensitivity']:.3f} vs mild-contrast {aug_m['sensitivity']:.3f}; "
        f"specificity {base_m['specificity']:.3f} vs {aug_m['specificity']:.3f}; "
        f"precision {base_m['precision']:.3f} vs {aug_m['precision']:.3f}; "
        f"accuracy {base_m['accuracy']:.3f} vs {aug_m['accuracy']:.3f}. "
        f"False negatives: baseline {base_m['fn']}/40, mild-contrast {aug_m['fn']}/40. "
        f"False positives: baseline {base_m['fp']}/160, mild-contrast {aug_m['fp']}/160. "
        "Small PH2 melanoma n=40; treat differences as descriptive, not a ranking. "
        "PH2 was not used to select or replace the frozen baseline."
    )


def write_report(payload: dict, base_m: dict, aug_m: dict) -> None:
    lines = [
        "# PH2: frozen 83.42% baseline vs mild-contrast augmentation",
        "",
        "TRAINING ON PH2: NO",
        "THRESHOLD TUNING ON PH2: NO",
        "ENSEMBLE-WEIGHT TUNING ON PH2: NO",
        "CALIBRATION ON PH2: NO",
        "MODELS MODIFIED: NO",
        "",
        "Scores are model-predicted melanoma probability estimates, not calibrated clinical probabilities.",
        "This is not clinical validation. PH2 was used previously in this project.",
        "",
        "## Cohort",
        "",
        "- 40 melanoma, 160 nevi (80 common + 80 atypical), 200 total",
        "- Same cases for both models",
        "- Ensemble 0.70/0.30, threshold 0.50 (frozen; not searched on PH2)",
        "",
        "## Frozen 83.42% baseline",
        "",
        json.dumps(base_m, indent=2),
        "",
        f"CM TP={base_m['tp']} TN={base_m['tn']} FP={base_m['fp']} FN={base_m['fn']}",
        "",
        "## Mild-contrast augmentation",
        "",
        json.dumps(aug_m, indent=2),
        "",
        f"CM TP={aug_m['tp']} TN={aug_m['tn']} FP={aug_m['fp']} FN={aug_m['fn']}",
        "",
        "## Comparison (same 200 PH2 cases)",
        "",
        "| Metric | Frozen 83.42% baseline | Mild-contrast augmentation |",
        "|---|---:|---:|",
        f"| Accuracy | {fmt(base_m,'accuracy')} | {fmt(aug_m,'accuracy')} |",
        f"| Sensitivity / recall | {fmt(base_m,'sensitivity')} | {fmt(aug_m,'sensitivity')} |",
        f"| Specificity | {fmt(base_m,'specificity')} | {fmt(aug_m,'specificity')} |",
        f"| Precision | {fmt(base_m,'precision')} | {fmt(aug_m,'precision')} |",
        f"| F1 | {fmt(base_m,'f1')} | {fmt(aug_m,'f1')} |",
        f"| ROC-AUC | {fmt(base_m,'auc')} | {fmt(aug_m,'auc')} |",
        f"| PR-AUC | {fmt(base_m,'pr_auc')} | {fmt(aug_m,'pr_auc')} |",
        f"| TP | {base_m['tp']} | {aug_m['tp']} |",
        f"| TN | {base_m['tn']} | {aug_m['tn']} |",
        f"| FP | {base_m['fp']} | {aug_m['fp']} |",
        f"| FN | {base_m['fn']} | {aug_m['fn']} |",
        "",
        payload["observations"],
        "",
        "Not clinically validated. No model was modified from these PH2 results.",
        "",
    ]
    _write_text(OUT_DIR / "ph2_comparison_report.md", "\n".join(lines) + "\n")


def main() -> None:
    started = datetime.now(timezone.utc).isoformat()
    missing = [str(p) for p in (BASE_EFF, BASE_MOB, AUG_EFF, AUG_MOB, AUG_SUMMARY) if not p.is_file()]
    if missing:
        raise RuntimeError("STOP: required model files missing:\n" + "\n".join(missing))
    status_path = AUG_DIR / "logs" / "status.json"
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("stage") != "complete":
            raise RuntimeError(f"STOP: mild-contrast training stage is {status.get('stage')!r}, not complete")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    before = snapshot([p for p in PROTECTED if p.is_file()])
    _json_dump(OUT_DIR / "protected_snapshot_before.json", before)

    df = parse_ph2()
    print(
        f"PH2 cohort n={len(df)} melanoma={(df['label']==1).sum()} "
        f"nevi={(df['label']==0).sum()}",
        flush=True,
    )
    df = cache_preprocessed(df)
    base_pred = run_model(BASE_EFF, BASE_MOB, df)
    aug_pred = run_model(AUG_EFF, AUG_MOB, df)
    if not np.array_equal(base_pred["y_true"], aug_pred["y_true"]):
        raise RuntimeError("STOP: models were not evaluated on the same labels")

    y = base_pred["y_true"]
    base_m = evaluate_predictions(y, base_pred["ensemble"], threshold=THRESHOLD)
    aug_m = evaluate_predictions(y, aug_pred["ensemble"], threshold=THRESHOLD)
    df["baseline_efficientnet_score"] = base_pred["p_eff"]
    df["baseline_mobilenet_score"] = base_pred["p_mob"]
    df["baseline_ensemble_score"] = base_pred["ensemble"]
    df["baseline_predicted"] = (base_pred["ensemble"] >= THRESHOLD).astype(int)
    df["mild_efficientnet_score"] = aug_pred["p_eff"]
    df["mild_mobilenet_score"] = aug_pred["p_mob"]
    df["mild_ensemble_score"] = aug_pred["ensemble"]
    df["mild_predicted"] = (aug_pred["ensemble"] >= THRESHOLD).astype(int)
    df.to_csv(OUT_DIR / "ph2_predictions.csv", index=False)

    _json_dump(OUT_DIR / "baseline_metrics.json", base_m)
    _json_dump(OUT_DIR / "mild_contrast_metrics.json", aug_m)
    cm_frame(base_m).to_csv(OUT_DIR / "baseline_confusion_matrix.csv")
    cm_frame(aug_m).to_csv(OUT_DIR / "mild_contrast_confusion_matrix.csv")
    plot_confusion_matrix(y, df["baseline_predicted"], OUT_DIR / "baseline_cm.png", labels=["Strict benign", "Melanoma"])
    plot_confusion_matrix(y, df["mild_predicted"], OUT_DIR / "mild_contrast_cm.png", labels=["Strict benign", "Melanoma"])
    plot_roc_curve(y, df["baseline_ensemble_score"], OUT_DIR / "baseline_roc.png", "Frozen baseline PH2")
    plot_roc_curve(y, df["mild_ensemble_score"], OUT_DIR / "mild_contrast_roc.png", "Mild-contrast PH2")
    plot_pr_curve(y, df["baseline_ensemble_score"], OUT_DIR / "baseline_pr.png", "Frozen baseline PH2")
    plot_pr_curve(y, df["mild_ensemble_score"], OUT_DIR / "mild_contrast_pr.png", "Mild-contrast PH2")

    comparison = pd.DataFrame(
        [
            {"model": "frozen_83.42_baseline", "evaluation": "PH2", **base_m},
            {"model": "mild_contrast_augmentation", "evaluation": "PH2", **aug_m},
        ]
    )
    comparison.to_csv(OUT_DIR / "comparison_table.csv", index=False)

    payload = {
        "cohort": EXPECTED,
        "ph2_previously_used_in_project": True,
        "ph2_used_for_tuning": False,
        "training_performed": False,
        "ensemble_weights": {"efficientnet": 0.70, "mobilenet": 0.30},
        "threshold": THRESHOLD,
        "baseline_metrics": base_m,
        "mild_contrast_metrics": aug_m,
        "observations": observations(base_m, aug_m),
        "replace_baseline": False,
        "reproducibility": {
            "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "preprocessing": "process_for_model hair+denoise+224+/255; max-side 1024 if needed; no CLAHE; no crop",
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "tensorflow": tf.__version__,
            },
        },
    }
    _json_dump(OUT_DIR / "ph2_results.json", payload)
    write_report(payload, base_m, aug_m)

    after = snapshot([p for p in PROTECTED if p.is_file()])
    if after != before:
        raise RuntimeError("STOP: a protected file changed during PH2 evaluation")
    _json_dump(OUT_DIR / "protected_snapshot_after.json", after)

    print("\nPH2 BASELINE VS MILD-CONTRAST COMPLETE", flush=True)
    print("Cohort: 200 cases (40 melanoma, 160 nevi). Same cases, 0.70/0.30, threshold 0.50.", flush=True)
    print(
        f"BASELINE acc={base_m['accuracy']:.4f} rec={base_m['sensitivity']:.4f} spec={base_m['specificity']:.4f} "
        f"prec={base_m['precision']:.4f} f1={base_m['f1']:.4f} FN={base_m['fn']} FP={base_m['fp']}",
        flush=True,
    )
    print(
        f"MILD     acc={aug_m['accuracy']:.4f} rec={aug_m['sensitivity']:.4f} spec={aug_m['specificity']:.4f} "
        f"prec={aug_m['precision']:.4f} f1={aug_m['f1']:.4f} FN={aug_m['fn']} FP={aug_m['fp']}",
        flush=True,
    )
    print("PH2 TUNING: NO", flush=True)
    print("BASELINE UNCHANGED: YES", flush=True)


if __name__ == "__main__":
    main()
