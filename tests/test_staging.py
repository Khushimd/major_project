"""
Unit / smoke tests for Melanoma CDSS core modules.
Run: python -m pytest tests/ -q
     python tests/test_staging.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from staging.ajcc_staging import AJCCStagingEngine, ClinicalParameters, predict_ajcc_stage
from preprocessing.image_preprocess import ImagePreprocessor


def test_ajcc_stage_0():
    r = predict_ajcc_stage(0.0, False, 0, False, False)
    assert r["stage"] == "Stage 0"


def test_ajcc_stage_ia():
    r = predict_ajcc_stage(0.5, False, 0, False, False)
    assert r["stage"] == "Stage IA"


def test_ajcc_stage_iv_metastasis():
    r = predict_ajcc_stage(1.0, False, 1, False, True)
    assert r["stage"] == "Stage IV"
    assert r["risk_level"] == "Critical"


def test_ajcc_stage_iii_nodes():
    r = predict_ajcc_stage(1.2, True, 2, True, False)
    assert r["stage"] == "Stage III"


def test_ajcc_thick_ulcerated():
    engine = AJCCStagingEngine()
    params = ClinicalParameters(
        breslow_thickness_mm=4.5,
        ulceration=True,
        mitotic_rate=3.0,
        lymph_node_involvement=False,
        metastasis=False,
    )
    result = engine.predict(params)
    assert result.stage == "Stage IIC"
    assert result.t_category == "T4b"


def test_preprocessor_pipeline():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, size=(300, 280, 3), dtype=np.uint8)
    prep = ImagePreprocessor(target_size=(224, 224))
    out = prep.process(img, return_uint8=False)
    assert out.shape == (224, 224, 3)
    assert out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_quality_check():
    sharp = np.zeros((128, 128, 3), dtype=np.uint8)
    sharp[40:90, 40:90] = 255
    report = ImagePreprocessor.check_quality(sharp)
    assert report.resolution == (128, 128)
    assert isinstance(report.warnings, list)


if __name__ == "__main__":
    test_ajcc_stage_0()
    test_ajcc_stage_ia()
    test_ajcc_stage_iv_metastasis()
    test_ajcc_stage_iii_nodes()
    test_ajcc_thick_ulcerated()
    test_preprocessor_pipeline()
    test_quality_check()
    print("All smoke tests passed.")
