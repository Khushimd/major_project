"""
Paired PH2 evaluation of two FROZEN models on the same 200-case cohort.

Does not retrain, fine-tune, retune thresholds/weights, calibrate, or modify
production / Exp2-C / baseline / class-weighted / development files.

PH2 was used previously in this project. This is an external evaluation on the
fixed PH2 cohort, not a claim of a never-before-seen dataset.
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

from models.architecture import FocalLoss  # noqa: F401
from models.ensemble import combine_hybrid_probs
from preprocessing.image_preprocess import ImagePreprocessor
from training.evaluate import evaluate_predictions, plot_confusion_matrix, plot_pr_curve, plot_roc_curve
from training.train import _predict_dataset

OUT_DIR = ROOT / "models" / "experiments" / "ph2_external_evaluation"
CACHE_DIR = OUT_DIR / "preprocessed_cache"
IMAGES_DIR = ROOT / "testing_data" / "PH2 Dataset images"
LABELS_PATH = ROOT / "testing_data" / "PH2_dataset.txt"

OLD_EFF = (
    ROOT
    / "models"
    / "experiments"
    / "internal_optimization"
    / "exp2_loss_comparison"
    / "exp2c_bce_smooth"
    / "efficientnetb0_bce_smooth.keras"
)
OLD_MOB = (
    ROOT
    / "models"
    / "experiments"
    / "internal_optimization"
    / "exp2_loss_comparison"
    / "exp2c_bce_smooth"
    / "mobilenetv2_bce_smooth.keras"
)
OLD_WEIGHTS = (0.75, 0.25)
OLD_THRESHOLD = 0.30
OLD_EFF_SHA = "02755e9c69e9f885b71696a278f807c9a93c29b88d59f7d252002f8bbbb62f87"
OLD_MOB_SHA = "ad049149423ad1a5a73f9b1bd6dabd0765968ae532472b8afd5757eb7f33146f"

CW_DIR = ROOT / "models" / "experiments" / "final_melanoma_benign_class_weighted"
CW_EFF = CW_DIR / "efficientnetb0_class_weighted.keras"
CW_MOB = CW_DIR / "mobilenetv2_class_weighted.keras"
CW_SUMMARY = CW_DIR / "training_summary.json"
CW_ENSEMBLE = CW_DIR / "ensemble_config.json"
CW_THRESHOLD = CW_DIR / "threshold_selection.json"

EXPECTED = {"n": 200, "melanoma": 40, "strict_benign": 160, "common_nevus": 80, "atypical_nevus": 80}

PROTECTED = [
    ROOT / "models" / "saved" / "efficientnetb0_melanoma.keras",
    ROOT / "models" / "saved" / "mobilenetv2_melanoma.keras",
    ROOT / "models" / "saved" / "inference_config.json",
    OLD_EFF,
    OLD_MOB,
    ROOT / "models" / "experiments" / "final_melanoma_benign_baseline" / "efficientnetb0_final_baseline.keras",
    ROOT / "models" / "experiments" / "final_melanoma_benign_baseline" / "mobilenetv2_final_baseline.keras",
    CW_EFF,
    CW_MOB,
    CW_SUMMARY,
    ROOT / "dataset" / "processed" / "final_melanoma_benign" / "train.csv",
    ROOT / "dataset" / "processed" / "final_melanoma_benign" / "val.csv",
    ROOT / "dataset" / "processed" / "final_melanoma_benign" / "test.csv",
    ROOT / "dataset" / "processed" / "final_melanoma_benign" / "manifest.csv",
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


def score_dist(values) -> dict:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


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


def class_weighted_config() -> dict:
    missing = [str(p) for p in (CW_EFF, CW_MOB, CW_SUMMARY) if not p.is_file()]
    if missing:
        raise RuntimeError(
            "STOP: class-weighted training is not finished. Missing:\n" + "\n".join(missing)
        )
    summary = json.loads(CW_SUMMARY.read_text(encoding="utf-8"))
    if CW_ENSEMBLE.is_file():
        ens = json.loads(CW_ENSEMBLE.read_text(encoding="utf-8"))
        w_eff = float(ens["efficientnet_weight"])
        w_mob = float(ens["mobilenet_weight"])
    else:
        w_eff = float(summary["ensemble"]["efficientnet_weight"])
        w_mob = float(summary["ensemble"]["mobilenet_weight"])
    if CW_THRESHOLD.is_file():
        thr = float(json.loads(CW_THRESHOLD.read_text(encoding="utf-8"))["threshold"])
    else:
        thr = float(summary["threshold"]["threshold"])
    return {
        "efficientnet_path": str(CW_EFF),
        "mobilenet_path": str(CW_MOB),
        "efficientnet_weight": w_eff,
        "mobilenet_weight": w_mob,
        "threshold": thr,
        "internal_test": summary.get("test", {}).get("ensemble"),
        "class_weights": summary.get("class_weights"),
        "source": str(CW_SUMMARY),
    }


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


def run_model(eff_path: Path, mob_path: Path, df: pd.DataFrame, weights: tuple[float, float]) -> dict:
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
    ens = combine_hybrid_probs(p_eff, p_mob, weights=weights, method="soft_voting")
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


def subtype_rows(df: pd.DataFrame, score_col: str, pred_col: str, model: str, threshold: float) -> list[dict]:
    rows = []
    for subtype in ("common_nevus", "atypical_nevus", "melanoma"):
        chunk = df[df["subtype"] == subtype]
        y = chunk["label"].to_numpy().astype(int)
        p = chunk[pred_col].to_numpy().astype(int)
        s = chunk[score_col].to_numpy()
        dist = score_dist(s)
        rows.append(
            {
                "model": model,
                "subtype": subtype,
                "n": int(len(chunk)),
                "predicted_melanoma": int(p.sum()),
                "predicted_benign": int((p == 0).sum()),
                "tp": int(((y == 1) & (p == 1)).sum()),
                "tn": int(((y == 0) & (p == 0)).sum()),
                "fp": int(((y == 0) & (p == 1)).sum()),
                "fn": int(((y == 1) & (p == 0)).sum()),
                "score_mean": dist["mean"],
                "score_median": dist["median"],
                "score_min": dist["min"],
                "score_max": dist["max"],
                "threshold": threshold,
            }
        )
    return rows


def fmt(m: dict, key: str) -> str:
    v = m[key]
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def write_reports(payload: dict, old_m: dict, cw_m: dict) -> None:
    old_w = payload["old"]
    cw_w = payload["class_weighted"]
    lines = [
        "# PH2 external evaluation of two frozen models",
        "",
        "TRAINING ON PH2: NO",
        "THRESHOLD TUNING ON PH2: NO",
        "ENSEMBLE-WEIGHT TUNING ON PH2: NO",
        "CALIBRATION ON PH2: NO",
        "MODELS MODIFIED: NO",
        "",
        "Scores are **model-predicted melanoma scores / probability estimates**, not calibrated clinical probabilities.",
        "This is **not** clinical validation.",
        "",
        "PH2 was used previously in this project. This run is an external evaluation on the **fixed 200-case PH2 cohort**, not a claim that PH2 is a completely untouched dataset.",
        "",
        "## Cohort",
        "",
        "- Strict melanoma: 40",
        "- Strict benign melanocytic nevi: 160 (80 common + 80 atypical)",
        "- Total: 200",
        "- Same cases for both models",
        "",
        "## Old 92% production model (Exp2-C)",
        "",
        f"- EfficientNet weight: {old_w['efficientnet_weight']:.2f}",
        f"- MobileNet weight: {old_w['mobilenet_weight']:.2f}",
        f"- Threshold: {old_w['threshold']:.2f} (frozen; not searched on PH2)",
        f"- Files: `{old_w['efficientnet_path']}` and `{old_w['mobilenet_path']}`",
        "",
        "Historical internal result (~92.28% accuracy) used DICM class=1 and is **not** a pure melanoma-vs-benign number. Quoted only as historical internal reference.",
        "",
        json.dumps(old_m, indent=2),
        "",
        f"CM TP={old_m['tp']} TN={old_m['tn']} FP={old_m['fp']} FN={old_m['fn']}",
        "",
        "## New class-weighted model",
        "",
        f"- EfficientNet weight: {cw_w['efficientnet_weight']:.2f} (validation-selected)",
        f"- MobileNet weight: {cw_w['mobilenet_weight']:.2f} (validation-selected)",
        f"- Threshold: {cw_w['threshold']:.2f} (validation-selected; not searched on PH2)",
        f"- Files: `{cw_w['efficientnet_path']}` and `{cw_w['mobilenet_path']}`",
        "",
        json.dumps(cw_m, indent=2),
        "",
        f"CM TP={cw_m['tp']} TN={cw_m['tn']} FP={cw_m['fp']} FN={cw_m['fn']}",
        "",
        "## Comparison (same 200 PH2 cases)",
        "",
        "| Metric | Old 92% model | Class-weighted model |",
        "|---|---:|---:|",
        f"| Accuracy | {fmt(old_m,'accuracy')} | {fmt(cw_m,'accuracy')} |",
        f"| Sensitivity | {fmt(old_m,'sensitivity')} | {fmt(cw_m,'sensitivity')} |",
        f"| Specificity | {fmt(old_m,'specificity')} | {fmt(cw_m,'specificity')} |",
        f"| Precision | {fmt(old_m,'precision')} | {fmt(cw_m,'precision')} |",
        f"| F1 | {fmt(old_m,'f1')} | {fmt(cw_m,'f1')} |",
        f"| ROC-AUC | {fmt(old_m,'auc')} | {fmt(cw_m,'auc')} |",
        f"| PR-AUC | {fmt(old_m,'pr_auc')} | {fmt(cw_m,'pr_auc')} |",
        f"| TP | {old_m['tp']} | {cw_m['tp']} |",
        f"| TN | {old_m['tn']} | {cw_m['tn']} |",
        f"| FP | {old_m['fp']} | {cw_m['fp']} |",
        f"| FN | {old_m['fn']} | {cw_m['fn']} |",
        f"| Predicted melanoma | {payload['old_predicted_melanoma']} | {payload['cw_predicted_melanoma']} |",
        f"| Predicted benign | {payload['old_predicted_benign']} | {payload['cw_predicted_benign']} |",
        "",
        "No model is ranked as best. Differences are observed values on this fixed PH2 cohort.",
        "",
        "## Score distributions",
        "",
        "See `score_distribution_summary.csv`. Values are model-predicted melanoma scores.",
        "",
        "## Observations",
        "",
        payload["observations"],
        "",
        "## Limitations",
        "",
        "- PH2 was previously used in this project.",
        "- 200 cases, 40 melanoma: metric variance is high.",
        "- Not clinically validated. Scores are not calibrated clinical probabilities.",
        "- Models were not changed based on these PH2 results.",
        "",
        "## Reproducibility",
        "",
        json.dumps(payload["reproducibility"], indent=2),
        "",
    ]
    _write_text(OUT_DIR / "ph2_external_report.md", "\n".join(lines) + "\n")
    comparison = [
        "# OLD 92% MODEL vs NEW CLASS-WEIGHTED MODEL — PH2",
        "",
        "Fixed 200-case PH2 cohort (40 melanoma, 160 nevi). Same cases for both models.",
        "PH2 was used previously in this project; this is not a never-before-seen external set.",
        "",
        f"Old model: weights {old_w['efficientnet_weight']:.2f}/{old_w['mobilenet_weight']:.2f}, threshold {old_w['threshold']:.2f}",
        f"Class-weighted: weights {cw_w['efficientnet_weight']:.2f}/{cw_w['mobilenet_weight']:.2f}, threshold {cw_w['threshold']:.2f}",
        "",
        "| Metric | Old 92% model | Class-weighted model |",
        "|---|---:|---:|",
        f"| Accuracy | {fmt(old_m,'accuracy')} | {fmt(cw_m,'accuracy')} |",
        f"| Sensitivity | {fmt(old_m,'sensitivity')} | {fmt(cw_m,'sensitivity')} |",
        f"| Specificity | {fmt(old_m,'specificity')} | {fmt(cw_m,'specificity')} |",
        f"| Precision | {fmt(old_m,'precision')} | {fmt(cw_m,'precision')} |",
        f"| F1 | {fmt(old_m,'f1')} | {fmt(cw_m,'f1')} |",
        f"| ROC-AUC | {fmt(old_m,'auc')} | {fmt(cw_m,'auc')} |",
        f"| PR-AUC | {fmt(old_m,'pr_auc')} | {fmt(cw_m,'pr_auc')} |",
        f"| TP | {old_m['tp']} | {cw_m['tp']} |",
        f"| TN | {old_m['tn']} | {cw_m['tn']} |",
        f"| FP | {old_m['fp']} | {cw_m['fp']} |",
        f"| FN | {old_m['fn']} | {cw_m['fn']} |",
        "",
        payload["observations"],
        "",
        "Not clinically validated. No model was modified from these results.",
        "",
    ]
    _write_text(OUT_DIR / "ph2_comparison_report.md", "\n".join(comparison))
    _write_text(
        OUT_DIR / "README.txt",
        "\n".join(
            [
                "Paired PH2 evaluation of two frozen models.",
                "Start with ph2_external_report.md and ph2_comparison_report.md.",
                "PH2 was previously used in this project.",
                "No PH2 tuning was performed.",
                "",
            ]
        ),
    )


def observations(old_m: dict, cw_m: dict) -> str:
    bits = []
    bits.append(
        f"Old model sensitivity {old_m['sensitivity']:.3f} vs class-weighted {cw_m['sensitivity']:.3f}; "
        f"specificity {old_m['specificity']:.3f} vs {cw_m['specificity']:.3f}."
    )
    if old_m["fn"] or cw_m["fn"]:
        bits.append(f"False negatives: old {old_m['fn']}/40, class-weighted {cw_m['fn']}/40.")
    if old_m["fp"] or cw_m["fp"]:
        bits.append(f"False positives: old {old_m['fp']}/160, class-weighted {cw_m['fp']}/160.")
    bits.append("Small PH2 melanoma n=40; treat metric differences as descriptive, not a ranking.")
    return " ".join(bits)


def main() -> None:
    started = datetime.now(timezone.utc).isoformat()
    cw = class_weighted_config()
    if not OLD_EFF.is_file() or not OLD_MOB.is_file():
        raise RuntimeError("STOP: old Exp2-C model files missing")
    if _sha256(OLD_EFF) != OLD_EFF_SHA or _sha256(OLD_MOB) != OLD_MOB_SHA:
        raise RuntimeError("STOP: old Exp2-C hashes do not match the frozen 92.28% files")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    before = snapshot([p for p in PROTECTED if p.is_file()])
    _json_dump(OUT_DIR / "protected_snapshot_before.json", before)

    df = parse_ph2()
    print(
        f"PH2 cohort n={len(df)} melanoma={(df['label']==1).sum()} "
        f"nevi={(df['label']==0).sum()} (common={(df['subtype']=='common_nevus').sum()} "
        f"atypical={(df['subtype']=='atypical_nevus').sum()})",
        flush=True,
    )
    df = cache_preprocessed(df)
    old_pred = run_model(OLD_EFF, OLD_MOB, df, OLD_WEIGHTS)
    cw_pred = run_model(CW_EFF, CW_MOB, df, (cw["efficientnet_weight"], cw["mobilenet_weight"]))
    if not np.array_equal(old_pred["y_true"], cw_pred["y_true"]):
        raise RuntimeError("STOP: models were not evaluated on the same labels")

    y = old_pred["y_true"]
    old_m = evaluate_predictions(y, old_pred["ensemble"], threshold=OLD_THRESHOLD)
    cw_m = evaluate_predictions(y, cw_pred["ensemble"], threshold=float(cw["threshold"]))
    df["old_efficientnet_score"] = old_pred["p_eff"]
    df["old_mobilenet_score"] = old_pred["p_mob"]
    df["old_ensemble_score"] = old_pred["ensemble"]
    df["old_predicted"] = (old_pred["ensemble"] >= OLD_THRESHOLD).astype(int)
    df["cw_efficientnet_score"] = cw_pred["p_eff"]
    df["cw_mobilenet_score"] = cw_pred["p_mob"]
    df["cw_ensemble_score"] = cw_pred["ensemble"]
    df["cw_predicted"] = (cw_pred["ensemble"] >= float(cw["threshold"])).astype(int)
    df.to_csv(OUT_DIR / "ph2_predictions.csv", index=False)

    _json_dump(OUT_DIR / "old_model_metrics.json", old_m)
    _json_dump(OUT_DIR / "class_weighted_model_metrics.json", cw_m)
    cm_frame(old_m).to_csv(OUT_DIR / "old_model_confusion_matrix.csv")
    cm_frame(cw_m).to_csv(OUT_DIR / "class_weighted_confusion_matrix.csv")
    plot_confusion_matrix(y, df["old_predicted"], OUT_DIR / "old_model_cm.png", labels=["Strict benign", "Melanoma"])
    plot_confusion_matrix(y, df["cw_predicted"], OUT_DIR / "class_weighted_cm.png", labels=["Strict benign", "Melanoma"])
    plot_roc_curve(y, df["old_ensemble_score"], OUT_DIR / "old_model_roc.png", "Old Exp2-C PH2")
    plot_roc_curve(y, df["cw_ensemble_score"], OUT_DIR / "class_weighted_roc.png", "Class-weighted PH2")
    plot_pr_curve(y, df["old_ensemble_score"], OUT_DIR / "old_model_pr.png", "Old Exp2-C PH2")
    plot_pr_curve(y, df["cw_ensemble_score"], OUT_DIR / "class_weighted_pr.png", "Class-weighted PH2")

    dist_rows = []
    for model, col in (("old_exp2c", "old_ensemble_score"), ("class_weighted", "cw_ensemble_score")):
        for group, mask in (
            ("all", np.ones(len(df), dtype=bool)),
            ("melanoma", df["label"].to_numpy() == 1),
            ("strict_benign", df["label"].to_numpy() == 0),
            ("common_nevus", df["subtype"].to_numpy() == "common_nevus"),
            ("atypical_nevus", df["subtype"].to_numpy() == "atypical_nevus"),
        ):
            row = {"model": model, "group": group}
            row.update(score_dist(df.loc[mask, col]))
            dist_rows.append(row)
    pd.DataFrame(dist_rows).to_csv(OUT_DIR / "score_distribution_summary.csv", index=False)
    pd.DataFrame(
        subtype_rows(df, "old_ensemble_score", "old_predicted", "old_exp2c", OLD_THRESHOLD)
        + subtype_rows(df, "cw_ensemble_score", "cw_predicted", "class_weighted", float(cw["threshold"]))
    ).to_csv(OUT_DIR / "subtype_results.csv", index=False)

    old_cfg = {
        "efficientnet_path": str(OLD_EFF),
        "mobilenet_path": str(OLD_MOB),
        "efficientnet_weight": OLD_WEIGHTS[0],
        "mobilenet_weight": OLD_WEIGHTS[1],
        "threshold": OLD_THRESHOLD,
        "sha256": {"efficientnet": OLD_EFF_SHA, "mobilenet": OLD_MOB_SHA},
    }
    payload = {
        "cohort": EXPECTED,
        "ph2_previously_used_in_project": True,
        "old": old_cfg,
        "class_weighted": cw,
        "old_metrics": old_m,
        "class_weighted_metrics": cw_m,
        "old_predicted_melanoma": int(df["old_predicted"].sum()),
        "old_predicted_benign": int((df["old_predicted"] == 0).sum()),
        "cw_predicted_melanoma": int(df["cw_predicted"].sum()),
        "cw_predicted_benign": int((df["cw_predicted"] == 0).sum()),
        "score_distributions": {
            "old_all": score_dist(df["old_ensemble_score"]),
            "old_melanoma": score_dist(df.loc[df["label"] == 1, "old_ensemble_score"]),
            "old_benign": score_dist(df.loc[df["label"] == 0, "old_ensemble_score"]),
            "cw_all": score_dist(df["cw_ensemble_score"]),
            "cw_melanoma": score_dist(df.loc[df["label"] == 1, "cw_ensemble_score"]),
            "cw_benign": score_dist(df.loc[df["label"] == 0, "cw_ensemble_score"]),
        },
        "observations": observations(old_m, cw_m),
        "ph2_used_for_tuning": False,
        "training_performed": False,
        "reproducibility": {
            "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "preprocessing": "process_for_model hair+denoise+224+/255; max-side 1024 if needed; no CLAHE; no crop",
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "tensorflow": tf.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        },
    }
    _json_dump(OUT_DIR / "ph2_external_results.json", payload)
    write_reports(payload, old_m, cw_m)

    after = snapshot([p for p in PROTECTED if p.is_file()])
    if after != before:
        raise RuntimeError("STOP: a protected file changed during PH2 evaluation")
    _json_dump(OUT_DIR / "protected_snapshot_after.json", after)

    print("\nPH2 EXTERNAL EVALUATION COMPLETE", flush=True)
    print("Cohort: 200 cases (40 melanoma, 160 nevi). Same cases for both models.", flush=True)
    print(
        f"OLD  weights {OLD_WEIGHTS[0]:.2f}/{OLD_WEIGHTS[1]:.2f} thr {OLD_THRESHOLD:.2f}  "
        f"acc={old_m['accuracy']:.4f} sens={old_m['sensitivity']:.4f} spec={old_m['specificity']:.4f} "
        f"prec={old_m['precision']:.4f} f1={old_m['f1']:.4f} auc={old_m['auc']:.4f} pr={old_m['pr_auc']:.4f} "
        f"TP={old_m['tp']} TN={old_m['tn']} FP={old_m['fp']} FN={old_m['fn']}",
        flush=True,
    )
    print(
        f"CW   weights {cw['efficientnet_weight']:.2f}/{cw['mobilenet_weight']:.2f} thr {cw['threshold']:.2f}  "
        f"acc={cw_m['accuracy']:.4f} sens={cw_m['sensitivity']:.4f} spec={cw_m['specificity']:.4f} "
        f"prec={cw_m['precision']:.4f} f1={cw_m['f1']:.4f} auc={cw_m['auc']:.4f} pr={cw_m['pr_auc']:.4f} "
        f"TP={cw_m['tp']} TN={cw_m['tn']} FP={cw_m['fp']} FN={cw_m['fn']}",
        flush=True,
    )
    print("PH2 TUNING: NO", flush=True)
    print("NEXT TRAINING EXPERIMENT: NO", flush=True)


if __name__ == "__main__":
    main()
