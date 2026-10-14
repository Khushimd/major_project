"""ISIC-DICM-17K loader uses official class labels and a leakage-free split."""

import numpy as np
import pandas as pd

from training.data_integrity import audit_dicm_split_detail
from training.dataset_loader import MelanomaDataset, normalize_isic_id


def test_normalize_isic_id_from_renamed_filename():
    assert normalize_isic_id("ISIC2019_0025105_mel.jpg") == "ISIC_0025105"
    assert normalize_isic_id("ISIC2020_8951909_mel.jpg") == "ISIC_8951909"
    assert normalize_isic_id("ISIC_8951909") == "ISIC_8951909"
    assert normalize_isic_id("ISIC0000002") == "ISIC_0000002"


def test_official_class_labels_and_grouped_split(tmp_path, monkeypatch):
    import cv2

    raw = tmp_path / "images"
    raw.mkdir()
    labels = tmp_path / "labels"
    labels.mkdir()
    rows = []
    for i in range(20):
        label = i % 2
        isic = f"ISIC_{i:07d}"
        path = raw / f"{isic}.jpg"
        color = (8, 10, 20) if label else (140, 160, 180)
        cv2.imwrite(str(path), np.full((32, 32, 3), color, dtype=np.uint8))
        rows.append(
            {
                "split": "train" if i < 16 else "valid",
                "image": f"ISIC2019_{i:07d}_{'mel' if label else 'nonmel'}.jpg",
                "isic_id": isic,
                "class": label,
                "patient_id": f"IP_{i // 2}",
                "lesion_id": f"IL_{i}",
            }
        )
    frame = pd.DataFrame(rows)
    frame[frame["split"] == "train"].drop(columns=["split"]).to_csv(labels / "train-set-metadata.csv", index=False)
    frame[frame["split"] == "valid"].drop(columns=["split"]).to_csv(labels / "valid-set-metadata.csv", index=False)

    import config

    monkeypatch.setattr(config, "ISIC_DICM_IMAGE_DIR", raw)
    monkeypatch.setattr(config, "ISIC_DICM_LABEL_DIR", labels)
    monkeypatch.setattr(config, "ISIC_DICM_SPLIT_DIR", tmp_path / "splits")
    monkeypatch.setitem(config.DATASETS["isic_dicm_17k"], "image_dir", str(raw))
    monkeypatch.setitem(config.DATASETS["isic_dicm_17k"], "raw_dir", str(labels))

    loader = MelanomaDataset(dataset_key="isic_dicm_17k")
    df = loader.load_metadata()
    assert len(df) == 20
    assert int(df["label"].sum()) == 10
    assert set(df["label"].unique()) <= {0, 1}
    train, val, test = loader.split(save=True)
    assert len(train) + len(val) + len(test) == 20
    audit = audit_dicm_split_detail(train, val, test)
    assert audit["patient_overlap"] == 0
    assert audit["lesion_overlap"] == 0
    assert audit["image_overlap"] == 0
    assert (tmp_path / "splits" / "train.csv").exists()
