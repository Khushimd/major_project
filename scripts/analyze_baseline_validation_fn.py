"""
Validation-only false-negative analysis of the frozen 83.42% baseline.

Analysis only. Does not retrain, retune, write production/baseline files,
or use the test set for any model-development decision.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import config
from models.ensemble import combine_hybrid_probs
from training.evaluate import evaluate_predictions
from utils.logger import setup_logger

logger = setup_logger("fn_analysis")

DATA_DIR = config.PROCESSED_DATA_DIR / "final_melanoma_benign"
BASELINE_ROOT = config.BASE_DIR / "models" / "experiments" / "final_melanoma_benign_baseline"
CACHE_DIR = BASELINE_ROOT / "preprocessed"
OUT_DIR = config.BASE_DIR / "models" / "experiments" / "final_melanoma_benign_fn_analysis"
ENSEMBLE_WEIGHTS = (0.70, 0.30)
THRESHOLD = 0.50
EXPECTED_VAL = (2438, 1043, 1395)
EXPECTED_CM = {"tp": 858, "tn": 1157, "fp": 238, "fn": 185}
FN_BINS = ((0.00, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.40), (0.40, 0.50))
TP_BINS = ((0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01))
N_CAM = 20
PROTECTED_ROOTS = [
    config.MODELS_DIR.resolve(),
    DATA_DIR.resolve(),
    BASELINE_ROOT.resolve(),
    (config.BASE_DIR / "models" / "experiments" / "final_melanoma_benign_class_weighted").resolve(),
    (config.BASE_DIR / "models" / "experiments" / "final_melanoma_benign_threshold_optimization").resolve(),
    (config.BASE_DIR / "models" / "experiments" / "final_melanoma_benign_moderate_weighting").resolve(),
]


def _jsonable(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _refuse(path: Path) -> None:
    resolved = path.resolve()
    if OUT_DIR.resolve() in resolved.parents or resolved == OUT_DIR.resolve():
        return
    for root in PROTECTED_ROOTS:
        if resolved == root or root in resolved.parents:
            raise RuntimeError(f"Refusing to write protected path: {path}")


def _dump(path: Path, payload) -> None:
    _refuse(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_jsonable), encoding="utf-8")


def _write(path: Path, text: str) -> None:
    _refuse(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def na_series(s: pd.Series) -> pd.Series:
    out = s.copy()
    if out.dtype == object or str(out.dtype) == "string":
        out = out.astype("string")
        out = out.mask(out.isna() | out.str.strip().eq("") | out.str.lower().isin(["nan", "none", "null"]), pd.NA)
    return out


def summarize_numeric(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mean": None, "median": None, "std": None, "min": None, "max": None, "q1": None, "q3": None}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "q1": float(np.percentile(arr, 25)),
        "q3": float(np.percentile(arr, 75)),
    }


def load_and_verify() -> tuple[pd.DataFrame, dict]:
    pred = pd.read_csv(BASELINE_ROOT / "validation_predictions.csv")
    val = pd.read_csv(DATA_DIR / "val.csv")
    pred["image_id"] = pred["image_id"].astype(str)
    val["image_id"] = val["image_id"].astype(str)
    got = (len(val), int((val["label"] == 1).sum()), int((val["label"] == 0).sum()))
    if got != EXPECTED_VAL:
        raise RuntimeError(f"STOP: frozen val counts {got} != {EXPECTED_VAL}")
    rebuilt = combine_hybrid_probs(
        pred["efficientnet_prob"],
        pred["mobilenet_prob"],
        weights=ENSEMBLE_WEIGHTS,
        method="soft_voting",
    )
    if not np.allclose(rebuilt, pred["ensemble_prob"].to_numpy(), atol=1e-6):
        raise RuntimeError("STOP: stored ensemble scores are not 0.70/0.30 soft voting")
    metrics = evaluate_predictions(pred["label"], pred["ensemble_prob"], threshold=THRESHOLD)
    cm = {k: int(metrics[k]) for k in ("tp", "tn", "fp", "fn")}
    if cm != EXPECTED_CM:
        raise RuntimeError(f"STOP: reproduced validation CM {cm} != {EXPECTED_CM}")
    merged = val.merge(pred, on="image_id", how="inner", suffixes=("", "_pred"))
    if "label_pred" in merged.columns:
        if not np.array_equal(merged["label"].to_numpy(), merged["label_pred"].to_numpy()):
            raise RuntimeError("STOP: label mismatch between val.csv and predictions")
        merged = merged.drop(columns=["label_pred"])
    if len(merged) != EXPECTED_VAL[0]:
        raise RuntimeError(f"STOP: merge length {len(merged)} != {EXPECTED_VAL[0]}")
    merged["predicted"] = (merged["ensemble_prob"] >= THRESHOLD).astype(int)
    merged["margin_from_threshold"] = THRESHOLD - merged["ensemble_prob"]
    for col in (
        "source_dataset",
        "diagnosis",
        "benign_malignant",
        "patient_id",
        "lesion_id",
        "sex",
        "anatom_site",
        "image_type",
        "dermoscopic_type",
        "diagnosis_confirm_type",
        "attribution",
    ):
        if col in merged.columns:
            merged[col] = na_series(merged[col])
    return merged, metrics


def original_size(path: str) -> tuple[object, object]:
    try:
        p = Path(str(path))
        if not p.is_file():
            return pd.NA, pd.NA
        with Image.open(p) as img:
            w, h = img.size
        return int(w), int(h)
    except Exception:
        return pd.NA, pd.NA


def cache_stats(image_id: str) -> dict:
    path = CACHE_DIR / f"{image_id}.jpg"
    if not path.is_file():
        return {"cache_width": pd.NA, "cache_height": pd.NA, "brightness": pd.NA, "contrast": pd.NA, "color_variation": pd.NA}
    with Image.open(path) as img:
        rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    gray = rgb.mean(axis=2)
    return {
        "cache_width": int(rgb.shape[1]),
        "cache_height": int(rgb.shape[0]),
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "color_variation": float(rgb.reshape(-1, 3).std(axis=0).mean()),
    }


def pick_across_bins(df: pd.DataFrame, bins, n: int) -> pd.DataFrame:
    parts = []
    nonempty = []
    for lo, hi in bins:
        chunk = df[(df["ensemble_prob"] >= lo) & (df["ensemble_prob"] < hi)].sort_values("image_id")
        if len(chunk):
            nonempty.append((lo, hi, chunk))
    if not nonempty:
        return df.sort_values("image_id").head(n)
    base, extra = divmod(n, len(nonempty))
    for i, (_, _, chunk) in enumerate(nonempty):
        take = min(len(chunk), base + (1 if i < extra else 0))
        if take:
            parts.append(chunk.head(take))
    chosen = pd.concat(parts, ignore_index=True).sort_values("image_id")
    if len(chosen) < n:
        leftover = df[~df["image_id"].isin(chosen["image_id"])].sort_values("image_id")
        chosen = pd.concat([chosen, leftover.head(n - len(chosen))], ignore_index=True)
    return chosen.head(n)


def categorical_compare(fn: pd.DataFrame, tp: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for cohort, frame in (("FN", fn), ("TP", tp)):
        s = na_series(frame[col]) if col in frame.columns else pd.Series(dtype="string")
        missing = int(s.isna().sum())
        vc = s.fillna("NA").value_counts()
        for key, count in vc.items():
            rows.append(
                {
                    "variable": col,
                    "cohort": cohort,
                    "value": str(key),
                    "count": int(count),
                    "percent": 100.0 * int(count) / max(len(frame), 1),
                    "missing_count": missing,
                    "missing_percent": 100.0 * missing / max(len(frame), 1),
                }
            )
    return pd.DataFrame(rows)


def plot_fn_bins(bin_rows: list[dict], path: Path) -> None:
    _refuse(path)
    labels = [r["range"] for r in bin_rows]
    counts = [r["count"] for r in bin_rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, counts, color="#B42318")
    ax.set_xlabel("Validation FN ensemble score")
    ax.set_ylabel("Count")
    ax.set_title("Validation false-negative score bins (threshold 0.50)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_tp_fn_hist(tp: np.ndarray, fn: np.ndarray, path: Path) -> None:
    _refuse(path)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(tp, bins=20, range=(0, 1), alpha=0.55, label=f"TP melanoma n={len(tp)}", color="#0B3D4F")
    ax.hist(fn, bins=20, range=(0, 1), alpha=0.55, label=f"FN melanoma n={len(fn)}", color="#C45C26")
    ax.axvline(THRESHOLD, color="black", ls="--", lw=1, label="threshold 0.50")
    ax.set_xlabel("Ensemble melanoma score")
    ax.set_ylabel("Count")
    ax.set_title("Validation melanoma: TP vs FN scores")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_disagreement(fn: pd.DataFrame, path: Path) -> None:
    _refuse(path)
    fig, ax = plt.subplots(figsize=(6.2, 6))
    ax.scatter(fn["efficientnet_prob"], fn["mobilenet_prob"], s=18, alpha=0.7, c="#C45C26")
    ax.axvline(THRESHOLD, color="gray", ls="--", lw=1)
    ax.axhline(THRESHOLD, color="gray", ls="--", lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("EfficientNet melanoma score")
    ax.set_ylabel("MobileNet melanoma score")
    ax.set_title("Validation FN: model scores")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_site(fn: pd.DataFrame, tp: pd.DataFrame, path: Path) -> None:
    _refuse(path)
    order = (
        pd.concat(
            [
                fn.assign(_c="FN")[["anatom_site", "_c"]],
                tp.assign(_c="TP")[["anatom_site", "_c"]],
            ]
        )
        .assign(anatom_site=lambda d: d["anatom_site"].fillna("NA").astype(str))
    )
    sites = order["anatom_site"].value_counts().index.tolist()
    x = np.arange(len(sites))
    fn_p = [100.0 * ((fn["anatom_site"].fillna("NA").astype(str) == s).mean()) for s in sites]
    tp_p = [100.0 * ((tp["anatom_site"].fillna("NA").astype(str) == s).mean()) for s in sites]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(x - 0.18, fn_p, 0.36, label="FN", color="#C45C26")
    ax.bar(x + 0.18, tp_p, 0.36, label="TP", color="#0B3D4F")
    ax.set_xticks(x)
    ax.set_xticklabels(sites, rotation=30, ha="right")
    ax.set_ylabel("Percent of cohort")
    ax.set_title("Anatomical site: validation FN vs TP melanoma")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load_cache_rgb(image_id: str) -> np.ndarray:
    path = CACHE_DIR / f"{image_id}.jpg"
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def heatmap_center_share(heatmap: np.ndarray) -> float:
    h, w = heatmap.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    rad = 0.35 * min(h, w)
    center = ((yy - cy) ** 2 + (xx - cx) ** 2) <= rad**2
    total = float(heatmap.sum()) + 1e-8
    return float(heatmap[center].sum() / total)


def run_gradcam(fn_sample: pd.DataFrame, tp_sample: pd.DataFrame) -> dict:
    import tensorflow as tf
    from gradcam.gradcam import GradCAM

    eff_path = BASELINE_ROOT / "efficientnetb0_final_baseline.keras"
    model = tf.keras.models.load_model(str(eff_path), compile=False)
    cam = GradCAM(model)
    rows = []
    for cohort, frame in (("FN", fn_sample), ("TP", tp_sample)):
        out_dir = OUT_DIR / "gradcam" / cohort.lower()
        out_dir.mkdir(parents=True, exist_ok=True)
        for _, row in frame.iterrows():
            image_id = str(row["image_id"])
            rgb = load_cache_rgb(image_id)
            batch = (rgb.astype(np.float32) / 255.0)[None, ...]
            heatmap, overlay = cam.generate(batch, rgb)
            side = np.concatenate([rgb, overlay], axis=1)
            save = out_dir / f"{image_id}.png"
            _refuse(save)
            Image.fromarray(side).save(save)
            rows.append(
                {
                    "cohort": cohort,
                    "image_id": image_id,
                    "ensemble_prob": float(row["ensemble_prob"]),
                    "efficientnet_prob": float(row["efficientnet_prob"]),
                    "mobilenet_prob": float(row["mobilenet_prob"]),
                    "center_heatmap_share": heatmap_center_share(np.asarray(heatmap)),
                    "path": str(save),
                }
            )
    del model
    tf.keras.backend.clear_session()
    return {
        "disclaimer": (
            "Grad-CAM highlights image regions that contributed to the model's prediction. "
            "It provides a visual explanation of the regions influencing the model output "
            "and should not be interpreted as definitive evidence of malignancy."
        ),
        "model": "efficientnetb0_final_baseline.keras",
        "n_fn": int((pd.DataFrame(rows)["cohort"] == "FN").sum()),
        "n_tp": int((pd.DataFrame(rows)["cohort"] == "TP").sum()),
        "mean_center_share_fn": float(np.mean([r["center_heatmap_share"] for r in rows if r["cohort"] == "FN"])),
        "mean_center_share_tp": float(np.mean([r["center_heatmap_share"] for r in rows if r["cohort"] == "TP"])),
        "cases": rows,
    }


def categorize(fn: pd.DataFrame, iq_flags: pd.Series) -> pd.DataFrame:
    both = (fn["efficientnet_prob"] < THRESHOLD) & (fn["mobilenet_prob"] < THRESHOLD)
    out = fn[["image_id"]].copy()
    out["borderline_score_fn"] = fn["ensemble_prob"] >= 0.40
    out["very_low_score_fn"] = fn["ensemble_prob"] < 0.20
    out["model_agreement_fn"] = both
    out["model_disagreement_fn"] = ~both
    out["possible_image_quality_related_fn"] = iq_flags.to_numpy()
    out["possible_source_domain_related_fn"] = False
    out["possible_small_subtle_lesion_fn"] = False
    out["no_clear_identifiable_pattern"] = ~(
        out["borderline_score_fn"]
        | out["very_low_score_fn"]
        | out["model_disagreement_fn"]
        | out["possible_image_quality_related_fn"]
    )
    return out


def recommend(evidence: dict) -> dict:
    borderline = evidence["fn_borderline_pct"]
    very_low = evidence["fn_very_low_pct"]
    agree = evidence["fn_both_miss_pct"]
    quality = evidence["fn_quality_flag_pct"]
    source_split = evidence["melanoma_source_nunique"] > 1
    center_gap = evidence["cam_center_tp"] - evidence["cam_center_fn"]
    if very_low >= 45 and agree >= 70 and center_gap >= 0.08:
        return {
            "choice": "C",
            "label": "Lesion-focused preprocessing/localization",
            "reason": (
                "A large share of validation FNs have very low melanoma scores, both models miss them, "
                "and the Grad-CAM sample puts less heatmap mass near the image center than TP cases. "
                "That is the only evidence-supported localization-oriented next test. Do not train it in this run."
            ),
        }
    if (quality >= 25) or (borderline >= 30 and agree >= 70):
        return {
            "choice": "B",
            "label": "Targeted augmentation",
            "reason": (
                "Most validation FNs are missed by both models. Threshold search and moderate class weighting "
                "already failed to keep an acceptable spec/prec band. A next training test, if any, should "
                "target hard melanoma appearance (subtle/low-contrast/borderline scores) via training-time "
                "augmentation only — not TTA-max and not another weight grid. Do not train it in this run."
            ),
        }
    if source_split and evidence.get("source_fn_rate_spread", 0) >= 10:
        return {
            "choice": "D",
            "label": "Source/domain-specific hardening",
            "reason": "Validation melanoma FN rates differed by source enough to justify a domain-focused test. Do not train it in this run.",
        }
    if very_low >= 50 and agree >= 70:
        return {
            "choice": "A",
            "label": "Focal loss",
            "reason": (
                "Many FNs are confidently scored as benign by both models, so a hard-example loss is the "
                "narrowest remaining training change. Do not train it in this run."
            ),
        }
    return {
        "choice": "E",
        "label": "No further training because no actionable pattern was identified",
        "reason": "Validation FN evidence did not isolate one targeted training change with a clear mechanism.",
    }


def write_reports(payload: dict) -> None:
    rec = payload["recommendation"]
    fn_n = payload["n_fn"]
    lines = [
        "# Validation false-negative vs true-positive analysis",
        "",
        "Research analysis only. **Not clinically validated.** Scores are AI probability estimates.",
        "",
        "## 1. Objective",
        "",
        "Understand why the frozen 83.42% baseline misses melanoma on the **validation** set",
        "before deciding whether another targeted training experiment is justified.",
        "",
        "## 2. Frozen model configuration",
        "",
        "EfficientNetB0 0.70 + MobileNetV2 0.30, threshold 0.50, frozen `final_melanoma_benign` splits.",
        "Scores reused from the frozen baseline validation inference artifacts after verifying the CM.",
        "",
        "## 3. Validation cohort",
        "",
        "2438 images; 1043 melanoma; 1395 benign.",
        "",
        "## 4. Number of validation FNs",
        "",
        f"{fn_n} (label=1 and ensemble_prob < 0.50). Reproduced CM TP=858 TN=1157 FP=238 FN=185.",
        "",
        "## 5. FN score distribution",
        "",
        json.dumps(payload["score_distribution"], indent=2),
        "",
        "## 6. EfficientNet vs MobileNet disagreement",
        "",
        json.dumps(payload["disagreement"], indent=2),
        "",
        "## 7. Metadata findings",
        "",
        json.dumps(payload["metadata_findings"], indent=2),
        "",
        "Described as frequency differences only. Not causal.",
        "",
        "## 8. Image-quality findings",
        "",
        json.dumps(payload["image_quality"], indent=2),
        "",
        "Lesion area could not be reliably quantified from the available data.",
        "No clinical segmentation masks were used or invented.",
        "",
        "## 9. Grad-CAM findings",
        "",
        payload["gradcam"]["disclaimer"],
        "",
        json.dumps({k: payload["gradcam"][k] for k in payload["gradcam"] if k != "cases"}, indent=2),
        "",
        "## 10. Main observed failure patterns",
        "",
        json.dumps(payload["failure_patterns"], indent=2),
        "",
        "## 11. Limitations",
        "",
        "- Validation only; test FNs were not inspected for development decisions.",
        "- All validation melanoma is ISIC-DICM-17K, so melanoma FN vs TP cannot show a source contrast.",
        "- Patient IDs are often missing.",
        "- Image stats are from the 224×224 cached pixels the model saw.",
        "- Grad-CAM is not segmentation and is not medical proof.",
        "",
        "## 12. Recommended ONE next intervention",
        "",
        f"**{rec['choice']}. {rec['label']}**",
        "",
        rec["reason"],
        "",
        "This analysis did not train that intervention.",
        "",
        "## 13. Test-set statement",
        "",
        "NO test images were used for model-development decisions.",
        "The frozen test FN count (193) is quoted only as previously published baseline context.",
        "",
    ]
    _write(OUT_DIR / "FN_vs_TP_analysis.md", "\n".join(lines) + "\n")
    readme = [
        "# Frozen baseline validation FN analysis",
        "",
        "Analysis-only experiment. No retraining, no threshold/ensemble retune,",
        "no dashboard/production/baseline overwrite.",
        "",
        f"Validation FNs: {fn_n}. Recommended next intervention: {rec['choice']}. {rec['label']}.",
        "",
        "Start here: `FN_vs_TP_analysis.md`, `fn_analysis_summary.json`.",
        "",
        payload["gradcam"]["disclaimer"],
        "",
        "NO test images were used for model-development decisions. Nothing was trained after this analysis.",
        "",
    ]
    _write(OUT_DIR / "README.md", "\n".join(readme) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "plots").mkdir(parents=True, exist_ok=True)
    merged, val_metrics = load_and_verify()
    logger.info("Validation CM reproduced: %s", EXPECTED_CM)

    melanoma = merged[merged["label"] == 1].copy()
    fn = melanoma[melanoma["predicted"] == 0].copy()
    tp = melanoma[melanoma["predicted"] == 1].copy()
    if len(fn) != EXPECTED_CM["fn"]:
        raise RuntimeError(f"STOP: FN cohort {len(fn)} != {EXPECTED_CM['fn']}")

    sizes = fn["image_path"].map(original_size)
    fn["orig_width"] = [s[0] for s in sizes]
    fn["orig_height"] = [s[1] for s in sizes]
    iq = pd.DataFrame([cache_stats(i) for i in fn["image_id"].astype(str)])
    fn = pd.concat([fn.reset_index(drop=True), iq], axis=1)
    tp_iq = pd.DataFrame([cache_stats(i) for i in tp["image_id"].astype(str)])
    tp = pd.concat([tp.reset_index(drop=True), tp_iq], axis=1)
    tp_sizes = tp["image_path"].map(original_size)
    tp["orig_width"] = [s[0] for s in tp_sizes]
    tp["orig_height"] = [s[1] for s in tp_sizes]

    fn_out = pd.DataFrame(
        {
            "image_id": fn["image_id"],
            "filename": fn["image_path"].map(lambda p: Path(str(p)).name if pd.notna(p) else pd.NA),
            "source_dataset": fn["source_dataset"],
            "patient_id": fn["patient_id"],
            "lesion_id": fn["lesion_id"],
            "diagnosis": fn["diagnosis"],
            "benign_malignant": fn["benign_malignant"],
            "anatom_site": fn["anatom_site"],
            "age_approx": fn["age_approx"],
            "sex": fn["sex"],
            "ensemble_melanoma_score": fn["ensemble_prob"],
            "efficientnet_melanoma_score": fn["efficientnet_prob"],
            "mobilenet_melanoma_score": fn["mobilenet_prob"],
            "margin_from_threshold": fn["margin_from_threshold"],
            "image_width": fn["orig_width"],
            "image_height": fn["orig_height"],
            "diagnosis_confirm_type": fn.get("diagnosis_confirm_type"),
            "image_type": fn.get("image_type"),
        }
    )
    _refuse(OUT_DIR / "validation_false_negatives.csv")
    fn_out.to_csv(OUT_DIR / "validation_false_negatives.csv", index=False, na_rep="NA")

    bin_rows = []
    for lo, hi in FN_BINS:
        mask = (fn["ensemble_prob"] >= lo) & (fn["ensemble_prob"] < hi)
        bin_rows.append(
            {
                "range": f"{lo:.2f}–{hi:.2f}",
                "low": lo,
                "high": hi,
                "count": int(mask.sum()),
                "percent": 100.0 * float(mask.mean()),
            }
        )
    score_dist = {
        "tp": summarize_numeric(tp["ensemble_prob"]),
        "fn": summarize_numeric(fn["ensemble_prob"]),
        "fn_bins": bin_rows,
        "fn_borderline_ge_0.40": int((fn["ensemble_prob"] >= 0.40).sum()),
        "fn_very_low_lt_0.20": int((fn["ensemble_prob"] < 0.20).sum()),
    }
    pd.DataFrame(bin_rows).to_csv(OUT_DIR / "fn_score_distribution.csv", index=False)

    both_miss = (fn["efficientnet_prob"] < THRESHOLD) & (fn["mobilenet_prob"] < THRESHOLD)
    eff_only = (fn["efficientnet_prob"] >= THRESHOLD) & (fn["mobilenet_prob"] < THRESHOLD)
    mob_only = (fn["mobilenet_prob"] >= THRESHOLD) & (fn["efficientnet_prob"] < THRESHOLD)
    disagreement = {
        "fn_both_below_0.50": int(both_miss.sum()),
        "fn_efficientnet_ge_0.50_mobilenet_lt_0.50": int(eff_only.sum()),
        "fn_mobilenet_ge_0.50_efficientnet_lt_0.50": int(mob_only.sum()),
        "fn_mean_abs_score_diff": float((fn["efficientnet_prob"] - fn["mobilenet_prob"]).abs().mean()),
        "tp_mean_efficientnet": float(tp["efficientnet_prob"].mean()),
        "tp_mean_mobilenet": float(tp["mobilenet_prob"].mean()),
        "fn_mean_efficientnet": float(fn["efficientnet_prob"].mean()),
        "fn_mean_mobilenet": float(fn["mobilenet_prob"].mean()),
        "tp_mean_ensemble": float(tp["ensemble_prob"].mean()),
        "fn_mean_ensemble": float(fn["ensemble_prob"].mean()),
    }
    fn.assign(
        abs_model_diff=(fn["efficientnet_prob"] - fn["mobilenet_prob"]).abs(),
        both_below_0_50=both_miss,
        efficientnet_ge_0_50_mobilenet_lt=eff_only,
        mobilenet_ge_0_50_efficientnet_lt=mob_only,
    )[
        [
            "image_id",
            "efficientnet_prob",
            "mobilenet_prob",
            "ensemble_prob",
            "abs_model_diff",
            "both_below_0_50",
            "efficientnet_ge_0_50_mobilenet_lt",
            "mobilenet_ge_0_50_efficientnet_lt",
        ]
    ].to_csv(OUT_DIR / "fn_model_disagreement.csv", index=False)

    meta_tables = []
    for col in ("source_dataset", "anatom_site", "sex", "diagnosis", "diagnosis_confirm_type"):
        meta_tables.append(categorical_compare(fn, tp, col))
    meta_df = pd.concat(meta_tables, ignore_index=True)
    meta_df.to_csv(OUT_DIR / "fn_metadata_analysis.csv", index=False)
    site_fn = fn["anatom_site"].fillna("NA").astype(str).value_counts(normalize=True)
    site_tp = tp["anatom_site"].fillna("NA").astype(str).value_counts(normalize=True)
    site_notes = []
    for site in sorted(set(site_fn.index) | set(site_tp.index)):
        d = 100.0 * (float(site_fn.get(site, 0)) - float(site_tp.get(site, 0)))
        if abs(d) >= 4:
            which = "FN" if d > 0 else "TP"
            site_notes.append(f"FNs were more frequent in {site}" if d > 0 else f"TPs were more frequent in {site} (FN share lower by {abs(d):.1f} pp)")
            if d > 0:
                site_notes[-1] = f"FNs were more frequent in {site} ({100*float(site_fn.get(site,0)):.1f}% of FN vs {100*float(site_tp.get(site,0)):.1f}% of TP)"

    metadata_findings = {
        "all_validation_melanoma_source": "ISIC-DICM-17K",
        "source_contrast_among_melanoma": "None. Every validation melanoma row is ISIC-DICM-17K, so FN vs TP cannot show a source/domain split.",
        "patient_id_missing_fn_percent": float(100.0 * fn["patient_id"].isna().mean()),
        "lesion_id_missing_fn_percent": float(100.0 * fn["lesion_id"].isna().mean()),
        "age_fn": summarize_numeric(fn["age_approx"].to_numpy()),
        "age_tp": summarize_numeric(tp["age_approx"].to_numpy()),
        "sex_fn_percent": (fn["sex"].fillna("NA").astype(str).value_counts(normalize=True) * 100).round(2).to_dict(),
        "sex_tp_percent": (tp["sex"].fillna("NA").astype(str).value_counts(normalize=True) * 100).round(2).to_dict(),
        "site_frequency_notes": site_notes or ["No site differed by 4 percentage points between FN and TP."],
        "missingness": {
            col: {
                "fn_percent": float(100.0 * fn[col].isna().mean()),
                "tp_percent": float(100.0 * tp[col].isna().mean()),
            }
            for col in ("patient_id", "lesion_id", "sex", "age_approx", "anatom_site")
        },
    }

    def _iq(frame):
        return {
            "brightness": summarize_numeric(frame["brightness"].to_numpy()),
            "contrast": summarize_numeric(frame["contrast"].to_numpy()),
            "color_variation": summarize_numeric(frame["color_variation"].to_numpy()),
            "orig_width": summarize_numeric(pd.to_numeric(frame["orig_width"], errors="coerce").to_numpy()),
            "orig_height": summarize_numeric(pd.to_numeric(frame["orig_height"], errors="coerce").to_numpy()),
        }

    image_quality = {
        "pixels_analyzed": "Frozen baseline 224x224 cache (what the model received). Original file size recorded separately when the path existed.",
        "lesion_area": "Lesion area could not be reliably quantified from the available data.",
        "fn": _iq(fn),
        "tp": _iq(tp),
    }
    mel_bright = pd.to_numeric(pd.concat([fn["brightness"], tp["brightness"]]), errors="coerce")
    mel_contrast = pd.to_numeric(pd.concat([fn["contrast"], tp["contrast"]]), errors="coerce")
    b_lo, b_hi = np.nanpercentile(mel_bright, [10, 90])
    c_lo, c_hi = np.nanpercentile(mel_contrast, [10, 90])
    iq_flag = (
        (pd.to_numeric(fn["brightness"], errors="coerce") < b_lo)
        | (pd.to_numeric(fn["brightness"], errors="coerce") > b_hi)
        | (pd.to_numeric(fn["contrast"], errors="coerce") < c_lo)
        | (pd.to_numeric(fn["contrast"], errors="coerce") > c_hi)
    )

    plot_fn_bins(bin_rows, OUT_DIR / "plots" / "fn_score_distribution.png")
    plot_tp_fn_hist(tp["ensemble_prob"].to_numpy(), fn["ensemble_prob"].to_numpy(), OUT_DIR / "plots" / "tp_vs_fn_score_distribution.png")
    plot_disagreement(fn, OUT_DIR / "plots" / "model_disagreement.png")
    plot_site(fn, tp, OUT_DIR / "plots" / "site_fn_vs_tp.png")

    fn_cam = pick_across_bins(fn, FN_BINS, N_CAM)
    tp_cam = pick_across_bins(tp, TP_BINS, N_CAM)
    fn_cam.to_csv(OUT_DIR / "gradcam_fn_sample.csv", index=False)
    tp_cam.to_csv(OUT_DIR / "gradcam_tp_sample.csv", index=False)
    logger.info("Generating Grad-CAM for %s FN and %s TP", len(fn_cam), len(tp_cam))
    gradcam = run_gradcam(fn_cam, tp_cam)
    pd.DataFrame(gradcam["cases"]).to_csv(OUT_DIR / "gradcam_sample_index.csv", index=False)

    cats = categorize(fn, iq_flag)
    cats.to_csv(OUT_DIR / "fn_category_flags.csv", index=False)
    failure_patterns = {
        "borderline_score_fn": int(cats["borderline_score_fn"].sum()),
        "very_low_score_fn": int(cats["very_low_score_fn"].sum()),
        "model_agreement_fn": int(cats["model_agreement_fn"].sum()),
        "model_disagreement_fn": int(cats["model_disagreement_fn"].sum()),
        "possible_image_quality_related_fn": int(cats["possible_image_quality_related_fn"].sum()),
        "possible_source_domain_related_fn": 0,
        "possible_small_subtle_lesion_fn": 0,
        "no_clear_identifiable_pattern": int(cats["no_clear_identifiable_pattern"].sum()),
        "notes": {
            "source_domain": "Not identifiable among validation melanoma (single source).",
            "small_subtle_lesion": "Not used. Lesion area could not be reliably quantified from the available data.",
        },
    }

    answers = {
        "1_most_fns_borderline_around_0.50": score_dist["fn_borderline_ge_0.40"] / len(fn) >= 0.50,
        "1_borderline_count_pct": [score_dist["fn_borderline_ge_0.40"], 100.0 * score_dist["fn_borderline_ge_0.40"] / len(fn)],
        "2_most_fns_confidently_benign": score_dist["fn_very_low_lt_0.20"] / len(fn) >= 0.50,
        "2_very_low_count_pct": [score_dist["fn_very_low_lt_0.20"], 100.0 * score_dist["fn_very_low_lt_0.20"] / len(fn)],
        "3_models_usually_agree_on_fns": disagreement["fn_both_below_0.50"] / len(fn) >= 0.70,
        "3_both_miss": disagreement["fn_both_below_0.50"],
        "4_identifiable_source_domain_patterns": False,
        "5_image_quality_patterns": bool(iq_flag.mean() >= 0.25 and abs(float(image_quality["fn"]["brightness"]["mean"] or 0) - float(image_quality["tp"]["brightness"]["mean"] or 0)) >= 8),
        "6_gradcam_irrelevant_regions": bool((gradcam["mean_center_share_tp"] - gradcam["mean_center_share_fn"]) >= 0.08),
        "7_enough_patterns_to_justify_targeted_retraining": None,
    }
    evidence = {
        "fn_borderline_pct": 100.0 * score_dist["fn_borderline_ge_0.40"] / len(fn),
        "fn_very_low_pct": 100.0 * score_dist["fn_very_low_lt_0.20"] / len(fn),
        "fn_both_miss_pct": 100.0 * disagreement["fn_both_below_0.50"] / len(fn),
        "fn_quality_flag_pct": 100.0 * float(iq_flag.mean()),
        "melanoma_source_nunique": int(melanoma["source_dataset"].nunique(dropna=True)),
        "cam_center_fn": gradcam["mean_center_share_fn"],
        "cam_center_tp": gradcam["mean_center_share_tp"],
    }
    recommendation = recommend(evidence)
    answers["7_enough_patterns_to_justify_targeted_retraining"] = recommendation["choice"] != "E"

    payload = {
        "objective": "Validation-only FN analysis of the frozen 83.42% baseline",
        "retrain": False,
        "test_used_for_development": False,
        "threshold": THRESHOLD,
        "ensemble_weights": {"efficientnet": 0.70, "mobilenet": 0.30},
        "validation_metrics": val_metrics,
        "n_fn": int(len(fn)),
        "n_tp": int(len(tp)),
        "score_distribution": score_dist,
        "disagreement": disagreement,
        "metadata_findings": metadata_findings,
        "image_quality": image_quality,
        "gradcam": gradcam,
        "failure_patterns": failure_patterns,
        "questions": answers,
        "recommendation": recommendation,
        "production_unchanged": True,
        "baseline_unchanged": True,
    }
    _dump(OUT_DIR / "fn_analysis_summary.json", payload)
    write_reports(payload)
    print("FN ANALYSIS COMPLETE", flush=True)
    print(f"Validation FN={len(fn)} TP={len(tp)}", flush=True)
    print(
        f"FN bins 0-0.1/0.1-0.2/0.2-0.3/0.3-0.4/0.4-0.5 = "
        + "/".join(str(r["count"]) for r in bin_rows),
        flush=True,
    )
    print(
        f"Both miss={disagreement['fn_both_below_0.50']} "
        f"Eff>=0.5={disagreement['fn_efficientnet_ge_0.50_mobilenet_lt_0.50']} "
        f"Mob>=0.5={disagreement['fn_mobilenet_ge_0.50_efficientnet_lt_0.50']}",
        flush=True,
    )
    print(f"RECOMMENDATION: {recommendation['choice']}. {recommendation['label']}", flush=True)
    print("TEST USED FOR DEVELOPMENT: NO", flush=True)
    print("TRAINING STARTED AFTER ANALYSIS: NO", flush=True)


if __name__ == "__main__":
    main()
