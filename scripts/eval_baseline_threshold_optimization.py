"""
Validation-only threshold search on the frozen 83.42% baseline ensemble.

Does not retrain. Does not load or rewrite model weights. Does not write
production, the frozen baseline folder, or the class-weighted experiment.
Test scores are stored but not evaluated until one threshold is locked.
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

import config
from models.ensemble import combine_hybrid_probs
from training.evaluate import evaluate_predictions, plot_confusion_matrix
from utils.logger import setup_logger

logger = setup_logger("baseline_threshold_opt")

DATA_DIR = config.PROCESSED_DATA_DIR / "final_melanoma_benign"
BASELINE_ROOT = config.BASE_DIR / "models" / "experiments" / "final_melanoma_benign_baseline"
CLASS_WEIGHTED_ROOT = config.BASE_DIR / "models" / "experiments" / "final_melanoma_benign_class_weighted"
OUT_DIR = config.BASE_DIR / "models" / "experiments" / "final_melanoma_benign_threshold_optimization"
EXPECTED = {
    "train": (11379, 4703, 6676),
    "val": (2438, 1043, 1395),
    "test": (2437, 1041, 1396),
}
ENSEMBLE_WEIGHTS = (0.70, 0.30)
BASELINE_THRESHOLD = 0.50
THRESHOLD_GRID = [round(x, 2) for x in np.arange(0.20, 0.70 + 1e-9, 0.01)]
CONSTRAINTS = {
    "specificity": 0.82,
    "precision": 0.78,
    "f1": 0.80,
    "accuracy": 0.82,
}
FROZEN_TEST_BASELINE = {
    "accuracy": 0.8342224045958145,
    "sensitivity": 0.8146013448607109,
    "specificity": 0.8488538681948424,
    "precision": 0.8007554296506137,
    "f1": 0.8076190476190476,
    "auc": 0.9126790142826079,
    "pr_auc": 0.8884757664753524,
    "tp": 848,
    "tn": 1185,
    "fp": 211,
    "fn": 193,
    "threshold": 0.50,
}
PROTECTED_ROOTS = [
    config.MODELS_DIR.resolve(),
    DATA_DIR.resolve(),
    BASELINE_ROOT.resolve(),
    CLASS_WEIGHTED_ROOT.resolve(),
    (config.BASE_DIR / "models" / "experiments" / "final_melanoma_benign_baseline_tta").resolve(),
]
PROTECTED_FILES = {
    config.EFFICIENTNET_PATH.resolve(),
    config.MOBILENET_PATH.resolve(),
    config.INFERENCE_CONFIG_PATH.resolve(),
    (BASELINE_ROOT / "efficientnetb0_final_baseline.keras").resolve(),
    (BASELINE_ROOT / "mobilenetv2_final_baseline.keras").resolve(),
    (BASELINE_ROOT / "training_summary.json").resolve(),
    (BASELINE_ROOT / "training_config.json").resolve(),
}


def _jsonable(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _refuse_protected(path: Path) -> None:
    resolved = path.resolve()
    if resolved in PROTECTED_FILES:
        raise RuntimeError(f"Refusing to write protected file: {path}")
    if OUT_DIR.resolve() in resolved.parents or resolved == OUT_DIR.resolve():
        return
    for root in PROTECTED_ROOTS:
        if resolved == root or root in resolved.parents:
            raise RuntimeError(f"Refusing to write protected path: {path}")


def _json_dump(path: Path, payload) -> None:
    _refuse_protected(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_jsonable), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    _refuse_protected(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_protected() -> dict:
    extra = []
    for csv_name in ("train.csv", "val.csv", "test.csv", "manifest.csv"):
        extra.append(DATA_DIR / csv_name)
    extra.extend(
        [
            BASELINE_ROOT / "efficientnetb0_final_baseline.keras",
            BASELINE_ROOT / "mobilenetv2_final_baseline.keras",
            BASELINE_ROOT / "training_summary.json",
            BASELINE_ROOT / "validation_predictions.csv",
            BASELINE_ROOT / "test_predictions.csv",
        ]
    )
    out = {}
    for path in sorted(set(PROTECTED_FILES) | {p.resolve() for p in extra if p.is_file()}):
        if path.is_file():
            out[str(path)] = {"sha256": _sha256(path), "size": path.stat().st_size}
    return out


def assert_unchanged(before: dict) -> None:
    after = snapshot_protected()
    for key, meta in before.items():
        if after.get(key) != meta:
            raise RuntimeError(f"Protected file changed: {key}")


def verify_frozen_splits() -> None:
    for name in ("train", "val", "test"):
        df = pd.read_csv(DATA_DIR / f"{name}.csv")
        got = (len(df), int((df["label"] == 1).sum()), int((df["label"] == 0).sum()))
        if got != EXPECTED[name]:
            raise RuntimeError(f"STOP: frozen {name} counts {got} != {EXPECTED[name]}")


def load_split_scores(split: str) -> pd.DataFrame:
    src = BASELINE_ROOT / f"{'validation' if split == 'val' else 'test'}_predictions.csv"
    if not src.is_file():
        raise FileNotFoundError(f"Missing frozen baseline scores: {src}")
    df = pd.read_csv(src)
    need = {"image_id", "label", "efficientnet_prob", "mobilenet_prob", "ensemble_prob"}
    missing = need - set(df.columns)
    if missing:
        raise RuntimeError(f"STOP: {src.name} missing columns {sorted(missing)}")
    rebuilt = combine_hybrid_probs(
        df["efficientnet_prob"].to_numpy(),
        df["mobilenet_prob"].to_numpy(),
        weights=ENSEMBLE_WEIGHTS,
        method="soft_voting",
    )
    if not np.allclose(rebuilt, df["ensemble_prob"].to_numpy(), atol=1e-6, rtol=0):
        raise RuntimeError(f"STOP: {split} ensemble scores do not match 0.70/0.30 soft voting")
    out = pd.DataFrame(
        {
            "image_id": df["image_id"].astype(str),
            "label": df["label"].astype(int),
            "efficientnet_prob": df["efficientnet_prob"].astype(float),
            "mobilenet_prob": df["mobilenet_prob"].astype(float),
            "ensemble_prob": df["ensemble_prob"].astype(float),
        }
    )
    got = (len(out), int((out["label"] == 1).sum()), int((out["label"] == 0).sum()))
    if got != EXPECTED[split]:
        raise RuntimeError(f"STOP: {split} prediction counts {got} != {EXPECTED[split]}")
    ids = set(pd.read_csv(DATA_DIR / f"{split}.csv")["image_id"].astype(str))
    if set(out["image_id"]) != ids:
        raise RuntimeError(f"STOP: {split} prediction image_ids do not match frozen CSV")
    return out


def constraint_failures(row: dict) -> list[str]:
    failed = []
    for key, minimum in CONSTRAINTS.items():
        if float(row[key]) < minimum - 1e-12:
            failed.append(f"{key} {row[key]:.4f} < {minimum:.2f}")
    return failed


def select_constrained(rows: list[dict]) -> dict:
    feasible = [r for r in rows if not constraint_failures(r)]
    if not feasible:
        return {
            "selected": None,
            "feasible": False,
            "n_feasible": 0,
            "rule": (
                "No validation threshold in 0.20-0.70 (step 0.01) satisfied "
                "specificity>=0.82, precision>=0.78, F1>=0.80, and accuracy>=0.82. "
                "Constraints were not relaxed."
            ),
        }
    chosen = max(
        feasible,
        key=lambda r: (
            float(r["sensitivity"]),
            -int(r["fn"]),
            float(r["f1"]),
            float(r["specificity"]),
            float(r["precision"]),
        ),
    )
    return {
        "selected": chosen,
        "feasible": True,
        "n_feasible": len(feasible),
        "feasible_thresholds": [float(r["threshold"]) for r in feasible],
        "rule": (
            "Among validation thresholds meeting specificity>=0.82, precision>=0.78, "
            "F1>=0.80, accuracy>=0.82: highest sensitivity, then lowest FN, "
            "then highest F1, then highest specificity, then highest precision."
        ),
    }


def useful_near_baseline(rows: list[dict], at_half: dict) -> list[dict]:
    useful = []
    for row in rows:
        if float(row["threshold"]) == BASELINE_THRESHOLD:
            continue
        fn_down = int(row["fn"]) < int(at_half["fn"])
        spec_ok = float(row["specificity"]) >= 0.82
        prec_ok = float(row["precision"]) >= 0.78
        if fn_down and spec_ok and prec_ok:
            useful.append(row)
    return useful


def score_dist(values, labels=None, cls=None) -> dict:
    arr = np.asarray(values, dtype=float).ravel()
    if labels is not None and cls is not None:
        arr = arr[np.asarray(labels).ravel() == cls]
    if arr.size == 0:
        return {}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
    }


def three_zone_note(val_df: pd.DataFrame, selected_thr: float | None) -> dict:
    y = val_df["label"].to_numpy()
    p = val_df["ensemble_prob"].to_numpy()
    benign = score_dist(p, y, 0)
    melanoma = score_dist(p, y, 1)
    overlap = float(benign["p90"]) > float(melanoma["p10"])
    supported = overlap and selected_thr is not None
    note = {
        "supported_as_interface_idea_only": supported,
        "not_clinical_thresholds": True,
        "cannot_rule_out_melanoma": True,
        "validation_benign": benign,
        "validation_melanoma": melanoma,
        "overlap_benign_p90_vs_melanoma_p10": overlap,
    }
    if supported:
        low = round(min(0.35, max(0.20, selected_thr - 0.15)), 2)
        high = round(min(0.70, selected_thr + 0.05), 2)
        note["possible_interface_bands_validation_only"] = {
            "lower_suspicion": f"ensemble_prob < {low:.2f}",
            "intermediate_clinical_review": f"{low:.2f} <= ensemble_prob < {high:.2f}",
            "higher_suspicion": f"ensemble_prob >= {high:.2f}",
            "disclaimer": (
                "Descriptive score bands from validation overlap only. "
                "Not clinical cutoffs. Intermediate cases still require clinician review. "
                "A low score does not rule out melanoma."
            ),
        }
    else:
        note["possible_interface_bands_validation_only"] = None
    return note


def plot_tradeoff(rows: list[dict], selected: float | None, save_path: Path) -> None:
    _refuse_protected(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    thr = [r["threshold"] for r in rows]
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    ax = axes[0]
    ax.plot(thr, [r["sensitivity"] for r in rows], label="Sensitivity", color="#C45C26")
    ax.plot(thr, [r["specificity"] for r in rows], label="Specificity", color="#0B3D4F")
    ax.plot(thr, [r["precision"] for r in rows], label="Precision", color="#2E7D4F")
    ax.plot(thr, [r["f1"] for r in rows], label="F1", color="#6B4C9A")
    ax.plot(thr, [r["accuracy"] for r in rows], label="Accuracy", color="#4A4A4A", ls="--")
    ax.axhline(0.82, color="gray", lw=0.8, ls=":", label="0.82 constraint")
    ax.axhline(0.78, color="#2E7D4F", lw=0.6, ls=":", alpha=0.7)
    ax.axvline(BASELINE_THRESHOLD, color="black", ls="--", lw=1, label="current 0.50")
    if selected is not None:
        ax.axvline(selected, color="#C45C26", ls="-", lw=1.2, label=f"selected {selected:.2f}")
    ax.set_ylabel("Metric")
    ax.set_ylim(0.55, 1.0)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("Validation threshold trade-off (frozen 0.70/0.30 ensemble)")
    ax2 = axes[1]
    ax2.plot(thr, [r["fn"] for r in rows], label="FN (missed melanoma)", color="#B42318")
    ax2.plot(thr, [r["fp"] for r in rows], label="FP (false melanoma)", color="#175CD3")
    ax2.axvline(BASELINE_THRESHOLD, color="black", ls="--", lw=1)
    if selected is not None:
        ax2.axvline(selected, color="#C45C26", ls="-", lw=1.2)
    ax2.set_xlabel("Decision threshold")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def fmt_pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def fmt_delta(new: float, old: float, as_pp: bool = True) -> str:
    delta = float(new) - float(old)
    if as_pp:
        return f"{100.0 * delta:+.2f} pp"
    return f"{delta:+.4f}"


def interpret(test_m: dict, val_selected: dict | None, feasible: bool) -> dict:
    old = FROZEN_TEST_BASELINE
    if not feasible or val_selected is None:
        return {
            "verdict": "no constrained threshold was selected; 0.50 remains the operating point",
            "better_than_0_50": False,
            "replace_baseline": False,
            "detail": (
                "No validation threshold met all preferred constraints. "
                "Threshold optimization alone did not yield a selectable operating point. "
                "The frozen 83.42% baseline at 0.50 is unchanged."
            ),
        }
    fn_drop = int(old["fn"]) - int(test_m["fn"])
    spec_drop = float(old["specificity"]) - float(test_m["specificity"])
    prec_drop = float(old["precision"]) - float(test_m["precision"])
    acc_drop = float(old["accuracy"]) - float(test_m["accuracy"])
    aggressive = spec_drop > 0.04 or prec_drop > 0.04 or float(test_m["specificity"]) < 0.80 or float(test_m["precision"]) < 0.76
    balanced = (
        fn_drop >= 5
        and float(test_m["specificity"]) >= 0.82
        and float(test_m["precision"]) >= 0.78
        and float(test_m["f1"]) >= 0.80
        and float(test_m["accuracy"]) >= 0.82
        and not aggressive
    )
    if balanced:
        verdict = "balanced operating-point improvement on the internal frozen test set"
        better = True
        detail = (
            "The validation-selected threshold reduced false negatives while keeping "
            "specificity, precision, F1, and accuracy within the preferred bands. "
            "This is an operating-point change, not a new model. Do not replace the "
            "frozen baseline automatically."
        )
    elif fn_drop > 0 and aggressive:
        verdict = "aggressive sensitivity-oriented operating point, not a superior model"
        better = False
        detail = (
            "False negatives fell, but specificity and/or precision dropped enough that "
            "this should not be called an improvement over 0.50."
        )
    elif fn_drop <= 0:
        verdict = "threshold optimization alone did not provide a satisfactory FN reduction"
        better = False
        detail = (
            "The constrained validation threshold did not reduce frozen-test false negatives "
            "in a meaningful way versus 0.50."
        )
    else:
        verdict = "trade-off, not a clear improvement over 0.50"
        better = False
        detail = (
            "Some metrics moved in the desired direction and others did not. "
            "Keep the frozen 83.42% baseline at 0.50."
        )
    return {
        "verdict": verdict,
        "better_than_0_50": better,
        "replace_baseline": False,
        "detail": detail,
        "fn_reduction": fn_drop,
        "specificity_drop": spec_drop,
        "precision_drop": prec_drop,
        "accuracy_drop": acc_drop,
        "aggressive": aggressive,
    }


def write_report(payload: dict) -> None:
    val_rows = payload["validation_threshold_rows"]
    at_half = payload["validation_at_0_50"]
    sel = payload["selection"]
    chosen = sel.get("selected")
    test_m = payload.get("test")
    interp = payload["interpretation"]
    useful = payload["useful_validation_rows"]
    max_sens = payload["highest_sensitivity_validation"]
    zone = payload["three_zone"]

    def row_line(r: dict) -> str:
        mark = ""
        if abs(float(r["threshold"]) - BASELINE_THRESHOLD) < 1e-9:
            mark = "  ← current 0.50"
        if chosen is not None and abs(float(r["threshold"]) - float(chosen["threshold"])) < 1e-9:
            mark += "  ← selected"
        if abs(float(r["threshold"]) - float(max_sens["threshold"])) < 1e-9:
            mark += "  ← max sensitivity"
        return (
            f"| {r['threshold']:.2f} | {r['accuracy']:.4f} | {r['sensitivity']:.4f} | "
            f"{r['specificity']:.4f} | {r['precision']:.4f} | {r['f1']:.4f} | "
            f"{r['tp']} | {r['tn']} | {r['fp']} | {r['fn']} |{mark}"
        )

    lines = [
        "# Frozen baseline threshold-optimization experiment",
        "",
        "Research / decision-support evaluation only. **Not clinically validated.**",
        "Scores are AI probability estimates, not calibrated clinical probabilities.",
        "This experiment does not diagnose benignity and does not replace the 83.42% baseline.",
        "",
        "## 1. Objective",
        "",
        "Ask whether a different decision threshold on the existing 0.70/0.30 ensemble",
        "can reduce false negatives without an unacceptable drop in specificity, precision,",
        "F1, or accuracy. Models were not retrained. TTA was not used.",
        "",
        "## 2. Frozen setup (unchanged)",
        "",
        "- Dataset: `dataset/processed/final_melanoma_benign/`",
        "- Validation 2438 (1043 melanoma / 1395 benign); test 2437 (1041 / 1396)",
        "- EfficientNetB0 0.70 + MobileNetV2 0.30 soft voting",
        "- Current operating threshold: 0.50",
        "- Scores reused from frozen baseline inference artifacts (no weight reload)",
        "",
        "## 3. Protocol",
        "",
        "1. Store validation and test ensemble scores.",
        "2. Search thresholds 0.20-0.70 step 0.01 on **validation only**.",
        "3. Apply preferred constraints, then lock one threshold.",
        "4. Evaluate the frozen test set **once**.",
        "5. No post-test retuning.",
        "",
        "## 4. Validation metrics at current 0.50",
        "",
        json.dumps(at_half, indent=2),
        "",
        f"Validation ROC-AUC {payload['validation_auc']:.4f}; PR-AUC {payload['validation_pr_auc']:.4f} ",
        "(unchanged across thresholds).",
        "",
        "## 5. Constrained selection",
        "",
        f"Feasible validation thresholds: {sel['n_feasible']}",
        "",
        sel["rule"],
        "",
    ]
    if chosen is None:
        lines += [
            "**No threshold satisfied all preferred constraints.** Constraints were not relaxed.",
            "",
            "Failed constraints at every candidate are listed in `selected_threshold.json`.",
            "",
        ]
    else:
        lines += [
            f"**Selected validation threshold: {chosen['threshold']:.2f}**",
            "",
            json.dumps(chosen, indent=2),
            "",
        ]
    lines += [
        "## 6. Highest-sensitivity validation threshold (not selected unless it met constraints)",
        "",
        json.dumps(max_sens, indent=2),
        "",
        "This row is reported for trade-off context only.",
        "",
        "## 7. Validation thresholds with FN ↓ vs 0.50 and spec/prec still reasonably high",
        "",
    ]
    if useful:
        lines.append("| Threshold | Acc | Sens | Spec | Prec | F1 | TP | TN | FP | FN |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        lines.extend(row_line(r) for r in useful)
        lines.append("")
    else:
        lines += ["None in the 0.20-0.70 grid.", ""]
    lines += [
        "## 8. Full validation trade-off table",
        "",
        "| Threshold | Acc | Sens | Spec | Prec | F1 | TP | TN | FP | FN |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(row_line(r) for r in val_rows)
    lines += ["", "## 9. Frozen test evaluation", ""]
    if test_m is None:
        lines += [
            "Test was **not** evaluated because no constrained validation threshold was selected.",
            "",
        ]
    else:
        lines += [
            f"One evaluation at locked threshold {payload['locked_threshold']:.2f}.",
            "",
            json.dumps(test_m, indent=2),
            "",
            f"Test confusion matrix: TP {test_m['tp']} / TN {test_m['tn']} / FP {test_m['fp']} / FN {test_m['fn']}",
            "",
            f"ROC-AUC {test_m['auc']:.4f}; PR-AUC {test_m['pr_auc']:.4f} ",
            "(same ranking metrics as the frozen 0.50 baseline, as expected).",
            "",
            "## 10. Baseline vs optimized threshold (frozen test)",
            "",
            "| Metric | Baseline 0.50 | Optimized | Change |",
            "|---|---:|---:|---:|",
            f"| Accuracy | {fmt_pct(FROZEN_TEST_BASELINE['accuracy'])} | {fmt_pct(test_m['accuracy'])} | {fmt_delta(test_m['accuracy'], FROZEN_TEST_BASELINE['accuracy'])} |",
            f"| Sensitivity | {fmt_pct(FROZEN_TEST_BASELINE['sensitivity'])} | {fmt_pct(test_m['sensitivity'])} | {fmt_delta(test_m['sensitivity'], FROZEN_TEST_BASELINE['sensitivity'])} |",
            f"| Specificity | {fmt_pct(FROZEN_TEST_BASELINE['specificity'])} | {fmt_pct(test_m['specificity'])} | {fmt_delta(test_m['specificity'], FROZEN_TEST_BASELINE['specificity'])} |",
            f"| Precision | {fmt_pct(FROZEN_TEST_BASELINE['precision'])} | {fmt_pct(test_m['precision'])} | {fmt_delta(test_m['precision'], FROZEN_TEST_BASELINE['precision'])} |",
            f"| F1 | {FROZEN_TEST_BASELINE['f1']:.4f} | {test_m['f1']:.4f} | {fmt_delta(test_m['f1'], FROZEN_TEST_BASELINE['f1'], as_pp=False)} |",
            f"| ROC-AUC | {FROZEN_TEST_BASELINE['auc']:.4f} | {test_m['auc']:.4f} | {fmt_delta(test_m['auc'], FROZEN_TEST_BASELINE['auc'], as_pp=False)} |",
            f"| PR-AUC | {FROZEN_TEST_BASELINE['pr_auc']:.4f} | {test_m['pr_auc']:.4f} | {fmt_delta(test_m['pr_auc'], FROZEN_TEST_BASELINE['pr_auc'], as_pp=False)} |",
            f"| TP | {FROZEN_TEST_BASELINE['tp']} | {test_m['tp']} | {int(test_m['tp']) - int(FROZEN_TEST_BASELINE['tp']):+d} |",
            f"| TN | {FROZEN_TEST_BASELINE['tn']} | {test_m['tn']} | {int(test_m['tn']) - int(FROZEN_TEST_BASELINE['tn']):+d} |",
            f"| FP | {FROZEN_TEST_BASELINE['fp']} | {test_m['fp']} | {int(test_m['fp']) - int(FROZEN_TEST_BASELINE['fp']):+d} |",
            f"| FN | {FROZEN_TEST_BASELINE['fn']} | {test_m['fn']} | {int(test_m['fn']) - int(FROZEN_TEST_BASELINE['fn']):+d} |",
            "",
            f"FN reduction = {int(FROZEN_TEST_BASELINE['fn']) - int(test_m['fn'])}",
            "",
        ]
    lines += [
        "## 11. Interpretation",
        "",
        f"**{interp['verdict']}**",
        "",
        interp["detail"],
        "",
        f"Better than 0.50 as a balanced operating point: {interp['better_than_0_50']}",
        f"Replace frozen baseline: {interp['replace_baseline']}",
        "",
        "## 12. Optional three-zone interface note",
        "",
        json.dumps(zone, indent=2),
        "",
        "## 13. What was not modified",
        "",
        "- `models/saved/`",
        "- `models/experiments/final_melanoma_benign_baseline/`",
        "- `models/experiments/final_melanoma_benign_class_weighted/`",
        "- frozen CSVs, production inference config, dashboard",
        "",
        "Script: `scripts/eval_baseline_threshold_optimization.py`.",
        "",
    ]
    _write_text(OUT_DIR / "THRESHOLD_OPTIMIZATION_REPORT.md", "\n".join(lines) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    before = snapshot_protected()
    verify_frozen_splits()

    val_df = load_split_scores("val")
    test_df = load_split_scores("test")
    _refuse_protected(OUT_DIR / "validation_predictions.csv")
    _refuse_protected(OUT_DIR / "test_predictions.csv")
    val_df.to_csv(OUT_DIR / "validation_predictions.csv", index=False)
    test_df.to_csv(OUT_DIR / "test_predictions.csv", index=False)
    logger.info("Stored validation and test scores. Threshold search uses validation only.")

    y_val = val_df["label"].to_numpy()
    p_val = val_df["ensemble_prob"].to_numpy()
    rows = [evaluate_predictions(y_val, p_val, threshold=t) for t in THRESHOLD_GRID]
    at_half = next(r for r in rows if abs(float(r["threshold"]) - BASELINE_THRESHOLD) < 1e-9)
    max_sens = max(rows, key=lambda r: (float(r["sensitivity"]), -int(r["fn"])))
    useful = useful_near_baseline(rows, at_half)
    selection = select_constrained(rows)
    chosen = selection.get("selected")

    pd.DataFrame(rows).to_csv(OUT_DIR / "validation_threshold_metrics.csv", index=False)
    failed_by_thr = {f"{r['threshold']:.2f}": constraint_failures(r) for r in rows}
    selected_payload = {
        "baseline_threshold": BASELINE_THRESHOLD,
        "ensemble_weights": {"efficientnet": 0.70, "mobilenet": 0.30},
        "grid": THRESHOLD_GRID,
        "constraints": CONSTRAINTS,
        "validation_at_0_50": at_half,
        "selection": selection,
        "highest_sensitivity_validation": max_sens,
        "constraint_failures_by_threshold": failed_by_thr,
        "test_used_for_selection": False,
        "tta": False,
        "retrain": False,
    }
    _json_dump(OUT_DIR / "selected_threshold.json", selected_payload)
    logger.info("Validation threshold locked. test_used_for_selection=False")

    test_metrics = None
    locked = None
    if chosen is not None:
        locked = float(chosen["threshold"])
        y_test = test_df["label"].to_numpy()
        p_test = test_df["ensemble_prob"].to_numpy()
        test_metrics = evaluate_predictions(y_test, p_test, threshold=locked)
        y_pred = (p_test >= locked).astype(int)
        plot_confusion_matrix(
            y_test,
            y_pred,
            OUT_DIR / "confusion_matrix.png",
            labels=["Benign", "Melanoma"],
        )
        test_out = test_df.copy()
        test_out["predicted"] = y_pred
        test_out.to_csv(OUT_DIR / "test_predictions.csv", index=False)
        _json_dump(
            OUT_DIR / "final_test_metrics.json",
            {
                "threshold": locked,
                "evaluated_once": True,
                "post_test_tuning": False,
                "metrics": test_metrics,
                "frozen_baseline_test_at_0_50": FROZEN_TEST_BASELINE,
                "deltas": {
                    "fn_reduction": int(FROZEN_TEST_BASELINE["fn"]) - int(test_metrics["fn"]),
                    "sensitivity": float(test_metrics["sensitivity"]) - float(FROZEN_TEST_BASELINE["sensitivity"]),
                    "specificity": float(test_metrics["specificity"]) - float(FROZEN_TEST_BASELINE["specificity"]),
                    "precision": float(test_metrics["precision"]) - float(FROZEN_TEST_BASELINE["precision"]),
                    "f1": float(test_metrics["f1"]) - float(FROZEN_TEST_BASELINE["f1"]),
                    "accuracy": float(test_metrics["accuracy"]) - float(FROZEN_TEST_BASELINE["accuracy"]),
                    "auc": float(test_metrics["auc"]) - float(FROZEN_TEST_BASELINE["auc"]),
                    "pr_auc": float(test_metrics["pr_auc"]) - float(FROZEN_TEST_BASELINE["pr_auc"]),
                },
            },
        )
    else:
        _json_dump(
            OUT_DIR / "final_test_metrics.json",
            {
                "threshold": None,
                "evaluated_once": False,
                "reason": "No constrained validation threshold; test not used to invent a cutoff.",
                "frozen_baseline_test_at_0_50": FROZEN_TEST_BASELINE,
            },
        )

    plot_tradeoff(rows, locked if locked is not None else BASELINE_THRESHOLD, OUT_DIR / "threshold_tradeoff_plot.png")
    interp = interpret(test_metrics or {}, chosen, bool(selection["feasible"]))
    payload = {
        "validation_threshold_rows": rows,
        "validation_at_0_50": at_half,
        "validation_auc": float(at_half["auc"]),
        "validation_pr_auc": float(at_half["pr_auc"]),
        "selection": selection,
        "highest_sensitivity_validation": max_sens,
        "useful_validation_rows": useful,
        "locked_threshold": locked,
        "test": test_metrics,
        "interpretation": interp,
        "three_zone": three_zone_note(val_df, locked if locked is not None else BASELINE_THRESHOLD),
    }
    write_report(payload)
    assert_unchanged(before)
    _json_dump(OUT_DIR / "production_snapshot_after.json", snapshot_protected())

    print("THRESHOLD OPTIMIZATION COMPLETE", flush=True)
    print(f"Validation feasible thresholds: {selection['n_feasible']}", flush=True)
    if chosen is None:
        print("Selected threshold: NONE (constraints not met; not relaxed)", flush=True)
    else:
        print(f"Selected validation threshold: {chosen['threshold']:.2f}", flush=True)
        print(
            f"VAL acc={chosen['accuracy']:.4f} sens={chosen['sensitivity']:.4f} "
            f"spec={chosen['specificity']:.4f} prec={chosen['precision']:.4f} "
            f"f1={chosen['f1']:.4f} FN={chosen['fn']}",
            flush=True,
        )
        print(
            f"TEST acc={test_metrics['accuracy']:.4f} sens={test_metrics['sensitivity']:.4f} "
            f"spec={test_metrics['specificity']:.4f} prec={test_metrics['precision']:.4f} "
            f"f1={test_metrics['f1']:.4f} FN={test_metrics['fn']}",
            flush=True,
        )
        print(
            f"FN reduction vs frozen 193: {int(FROZEN_TEST_BASELINE['fn']) - int(test_metrics['fn'])}",
            flush=True,
        )
    print(f"INTERPRETATION: {interp['verdict']}", flush=True)
    print("BASELINE UNCHANGED: YES", flush=True)
    print("PRODUCTION UNCHANGED: YES", flush=True)
    print("TEST USED FOR TUNING: NO", flush=True)
    print("POST-TEST TUNING: NO", flush=True)


if __name__ == "__main__":
    main()
