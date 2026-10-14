"""
Prepare ISIC-DICM-17K manifests and a 70/15/15 grouped split.

Does not train models or touch production weights.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from training.data_integrity import (
    audit_dicm_split_detail,
    audit_old_dataset_overlap,
    audit_split_leakage,
    save_leakage_report,
)
from training.dataset_loader import MelanomaDataset


def _split_block(name: str, df: pd.DataFrame) -> dict:
    n = len(df)
    mel = int((df["label"] == 1).sum())
    non = int((df["label"] == 0).sum())
    return {
        "total": n,
        "melanoma": mel,
        "non_melanoma": non,
        "melanoma_pct": round(100.0 * mel / max(n, 1), 2),
        "non_melanoma_pct": round(100.0 * non / max(n, 1), 2),
    }


def main() -> None:
    out_dir = Path(config.ISIC_DICM_SPLIT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = MelanomaDataset(dataset_key="isic_dicm_17k")
    frame = loader.load_metadata()
    manifest_path = out_dir / "manifest.csv"
    frame.to_csv(manifest_path, index=False)

    train, val, test = loader._split_isic_dicm_70_15_15(save=True)
    leakage = audit_split_leakage(train, val, test, check_near_duplicates=False)
    detail = audit_dicm_split_detail(train, val, test)
    overlap = audit_old_dataset_overlap(frame["image_id"].astype(str))

    save_leakage_report(leakage, path=out_dir / "leakage_report.json")
    (out_dir / "split_detail.json").write_text(json.dumps(detail, indent=2), encoding="utf-8")
    (out_dir / "old_dataset_overlap.json").write_text(json.dumps(overlap, indent=2), encoding="utf-8")

    missing_images = int((~frame["image_path"].map(lambda p: Path(str(p)).is_file())).sum())
    missing_labels = int(frame["label"].isna().sum())
    report = {
        "dataset": "ISIC-DICM-17K",
        "image_dir": str(config.ISIC_DICM_IMAGE_DIR),
        "label_source": [
            str(config.ISIC_DICM_LABEL_DIR / "train-set-metadata.csv"),
            str(config.ISIC_DICM_LABEL_DIR / "valid-set-metadata.csv"),
        ],
        "label_column": "class",
        "total_images": int(len(frame)),
        "melanoma": int((frame["label"] == 1).sum()),
        "non_melanoma": int((frame["label"] == 0).sum()),
        "training": _split_block("train", train),
        "validation": _split_block("val", val),
        "test": _split_block("test", test),
        "grouping": {
            "priority": ["patient_id", "lesion_id", "isic_id"],
            "groups": int(frame["group_id"].nunique()),
            "patients_filled": int((frame["patient_id"].astype(str).str.strip() != "").sum()),
            "lesions_filled": int((frame["lesion_id"].astype(str).str.strip() != "").sum()),
        },
        "leakage": {
            "patient_overlap": detail["patient_overlap"],
            "lesion_overlap": detail["lesion_overlap"],
            "image_overlap": detail["image_overlap"],
            "clean": detail["clean"],
            "detail": detail,
        },
        "old_dataset_overlap": overlap,
        "missing_images": missing_images,
        "missing_labels": missing_labels,
        "seed": config.SEED,
        "split_dir": str(out_dir),
        "old_splits_untouched": str(config.SPLIT_DIR),
        "production_models_untouched": [
            str(config.EFFICIENTNET_PATH),
            str(config.MOBILENET_PATH),
        ],
        "active_dataset_unchanged": config.ACTIVE_DATASET,
        "preprocessing": (
            "Same as live inference process_for_model: hair removal + light denoise "
            "+ resize 224 + scale to [0,1]. CLAHE is not applied (inference skips it). "
            "EfficientNetB0/MobileNetV2 still apply ImageNet preprocess inside the model."
        ),
        "augmentation_train_only": (
            "Existing mild dermoscopy aug: horizontal/vertical flip, small rotation, "
            "small zoom, small translation, moderate contrast/brightness. "
            "No oversampling. No val/test aug. BCE later at train time."
        ),
    }
    (out_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "ISIC-DICM-17K dataset validation report",
        f"Total images: {report['total_images']}",
        f"Melanoma: {report['melanoma']}",
        f"Non-melanoma: {report['non_melanoma']}",
        "",
        "Training:",
        f"- total {report['training']['total']}",
        f"- melanoma {report['training']['melanoma']}",
        f"- non-melanoma {report['training']['non_melanoma']}",
        f"- melanoma % {report['training']['melanoma_pct']}",
        "",
        "Validation:",
        f"- total {report['validation']['total']}",
        f"- melanoma {report['validation']['melanoma']}",
        f"- non-melanoma {report['validation']['non_melanoma']}",
        f"- melanoma % {report['validation']['melanoma_pct']}",
        "",
        "Test:",
        f"- total {report['test']['total']}",
        f"- melanoma {report['test']['melanoma']}",
        f"- non-melanoma {report['test']['non_melanoma']}",
        f"- melanoma % {report['test']['melanoma_pct']}",
        "",
        f"Patient overlap: {detail['patient_overlap']}",
        f"Lesion overlap: {detail['lesion_overlap']}",
        f"Duplicate/image overlap: {detail['image_overlap']}",
        f"Missing images: {missing_images}",
        f"Missing labels: {missing_labels}",
        f"SIIM overlap (unused): {overlap['siim_isic_2020']['overlap_n']}",
        f"HAM10000 overlap (unused): {overlap['ham10000']['overlap_n']}",
    ]
    (out_dir / "validation_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote manifests to {out_dir}")
    print("Training was not started. Production models were not changed.")


if __name__ == "__main__":
    main()
