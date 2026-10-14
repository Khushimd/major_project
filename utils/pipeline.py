"""
End-to-end inference pipeline for Melanoma CDSS.

Upload → Preprocess → Hybrid predict → Grad-CAM → (AJCC later) → PDF
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import tensorflow as tf

import config
from database.db import get_db
from gradcam.gradcam import GradCAM, generate_gradcam_if_malignant
from models.ensemble import EnsemblePredictor
from preprocessing.image_preprocess import ImagePreprocessor, load_and_preprocess
from reports.pdf_generator import DiagnosticReportGenerator
from segmentation.unet import LesionSegmenter
from staging.ajcc_staging import AJCCStagingEngine, ClinicalParameters
from utils.helpers import file_sha256, generate_unique_id, timestamp_now
from utils.logger import setup_logger

logger = setup_logger("pipeline")


class MelanomaPipeline:
    """Orchestrates the full clinical decision support workflow."""

    def __init__(self) -> None:
        self.preprocessor = ImagePreprocessor()
        self.ensemble = EnsemblePredictor(use_tta=False)
        self.segmenter = LesionSegmenter()
        self.staging_engine = AJCCStagingEngine()
        self.report_gen = DiagnosticReportGenerator()
        self._loaded = False

    def load_models(self) -> None:
        """Lazy-load hybrid models once."""
        if not self._loaded:
            tf.keras.utils.set_random_seed(config.SEED)
            self.ensemble.load()
            self._loaded = True
            logger.info("Hybrid inference models ready (EfficientNet + MobileNet).")

    def run(
        self,
        image_path: str,
        patient: Dict[str, Any],
        clinical: Optional[Dict[str, Any]] = None,
        doctor_notes: str = "",
        owner_user_id: Optional[int] = None,
        assigned_doctor_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute classification (+ optional staging if clinical provided).

        Uses the trained hybrid models on the standard preprocess.
        Scores are computed live (not reused from cache).
        """
        self.load_models()
        patient_id = patient.get("patient_id") or generate_unique_id("PID")
        patient["patient_id"] = patient_id

        image_hash = file_sha256(image_path)

        original_rgb, processed, quality = load_and_preprocess(image_path)
        batch = np.expand_dims(processed.astype(np.float32), axis=0)

        mask = self.segmenter.segment(original_rgb)
        seg_overlay = self.segmenter.overlay_mask(original_rgb, mask)
        seg_path = config.RESULTS_DIR / f"{patient_id}_segmentation.jpg"
        cv2.imwrite(str(seg_path), cv2.cvtColor(seg_overlay, cv2.COLOR_RGB2BGR))

        prep_path = config.RESULTS_DIR / f"{patient_id}_preprocessed.jpg"
        prep_uint8 = (processed * 255).astype(np.uint8)
        cv2.imwrite(str(prep_path), cv2.cvtColor(prep_uint8, cv2.COLOR_RGB2BGR))

        pred = self.ensemble.predict(batch)
        pred["from_cache"] = False
        logger.info(
            "Prediction=%s conf=%.3f mal=%.3f thr=%.3f model=%s",
            pred["prediction"],
            pred["confidence"],
            pred["malignant_prob"],
            pred.get("decision_threshold"),
            pred["model_used"],
        )

        gradcam_path = None
        gradcam_info = {"generated": False, "message": ""}
        if str(pred.get("prediction") or "").lower() == "malignant":
            save_path = str(config.RESULTS_DIR / f"{patient_id}_gradcam.jpg")
            gradcam_path, gradcam_info = self._generate_gradcam(
                pred["prediction"], batch, original_rgb, save_path
            )

        # Staging only when clinical values are explicitly provided
        staging = None
        if pred["prediction"] == "Malignant" and clinical:
            params = ClinicalParameters.from_form(clinical)
            staging = self.staging_engine.predict(params).to_dict()

        report_id = generate_unique_id("RPT")
        report_path = self.report_gen.generate(
            patient=patient,
            prediction=pred,
            staging=staging,
            gradcam_path=gradcam_path,
            lesion_image_path=image_path,
            doctor_notes=doctor_notes,
            report_id=report_id,
        )

        record = {
            "patient_id": patient_id,
            "patient_name": patient.get("name"),
            "age": patient.get("age"),
            "sex": patient.get("sex"),
            "image_path": image_path,
            "image_hash": image_hash,
            "prediction": pred["prediction"],
            "confidence": pred["confidence"],
            "malignant_prob": pred["malignant_prob"],
            "benign_prob": pred["benign_prob"],
            "model_used": pred["model_used"],
            "ajcc_stage": staging.get("stage") if staging else None,
            "risk_level": staging.get("risk_level") if staging else None,
            "t_category": staging.get("t_category") if staging else None,
            "n_category": staging.get("n_category") if staging else None,
            "m_category": staging.get("m_category") if staging else None,
            "staging_details": staging.get("details") if staging else None,
            "staging_recommendation": staging.get("recommendation") if staging else None,
            "staging_follow_up": staging.get("follow_up") if staging else None,
            "gradcam_path": gradcam_path,
            "report_path": str(report_path),
            "doctor_notes": doctor_notes,
            "breslow_thickness": (
                float(clinical.get("breslow_thickness_mm", 0)) if clinical else None
            ),
            "ulceration": (
                1
                if clinical
                and str(clinical.get("ulceration", "")).lower()
                in {"yes", "true", "1", "on"}
                else 0
            ),
            "mitotic_rate": float(clinical.get("mitotic_rate", 0)) if clinical else None,
            "lymph_node_involvement": (
                1
                if clinical
                and str(clinical.get("lymph_node_involvement", "")).lower()
                in {"yes", "true", "1", "on"}
                else 0
            ),
            "metastasis": (
                1
                if clinical
                and str(clinical.get("metastasis", "")).lower()
                in {"yes", "true", "1", "on"}
                else 0
            ),
            "created_at": timestamp_now(),
            "owner_user_id": owner_user_id,
            "assigned_doctor_id": assigned_doctor_id,
            "review_status": "pending_review" if owner_user_id else None,
            "report_code": report_id,
        }
        db_id = get_db().insert_prediction(record)

        return {
            "db_id": db_id,
            "patient": patient,
            "prediction": pred,
            "staging": staging,
            "gradcam": gradcam_info,
            "gradcam_path": gradcam_path,
            "segmentation_path": str(seg_path),
            "preprocessed_path": str(prep_path),
            "report_path": str(report_path),
            "report_id": report_id,
            "image_hash": image_hash,
            "quality": {
                "is_acceptable": quality.is_acceptable,
                "blur_score": quality.blur_score,
                "brightness": quality.brightness,
                "resolution": quality.resolution,
                "warnings": quality.warnings,
            },
            "disclaimer": config.MEDICAL_DISCLAIMER,
            "timestamp": record["created_at"],
        }

    def _generate_gradcam(
        self,
        prediction: str,
        batch: np.ndarray,
        original_rgb: np.ndarray,
        save_path: str,
    ) -> tuple:
        info = {"generated": False, "message": "Grad-CAM unavailable (model not loaded)."}
        models = [m for m in (self.ensemble.mobilenet, self.ensemble.efficientnet) if m is not None]
        for cam_model in models:
            try:
                info = generate_gradcam_if_malignant(
                    prediction,
                    cam_model,
                    batch,
                    original_rgb,
                    save_path=save_path,
                )
                if Path(save_path).exists():
                    return save_path, info
            except Exception as exc:
                logger.exception("Grad-CAM failed: %s", exc)
                info = {"generated": False, "message": f"Grad-CAM unavailable ({exc})."}
        return None, info

    def ensure_gradcam(self, record: Dict[str, Any]) -> Optional[str]:
        """Create or restyle a melanoma heatmap for an already-saved case."""
        if str(record.get("prediction") or "").lower() not in {"malignant", "melanoma"}:
            return None
        image_path = record.get("image_path")
        if not image_path or not Path(str(image_path)).exists():
            return None
        pid = record.get("patient_id") or generate_unique_id("PID")
        save_path = str(record.get("gradcam_path") or (config.RESULTS_DIR / f"{pid}_gradcam.jpg"))
        stamp = Path(save_path + f".{GradCAM.OVERLAY_STYLE_VERSION}")
        if Path(save_path).exists() and stamp.exists():
            return save_path

        original_rgb, processed, _ = load_and_preprocess(str(image_path))
        self.load_models()
        batch = np.expand_dims(processed.astype(np.float32), axis=0)
        path, _ = self._generate_gradcam("Malignant", batch, original_rgb, save_path)
        if path and record.get("id"):
            get_db().update_prediction(int(record["id"]), {"gradcam_path": path})
            record["gradcam_path"] = path
        return path

    def apply_ajcc_staging(
        self,
        pred_id: int,
        clinical: Dict[str, Any],
        doctor_notes: str = "",
    ) -> Dict[str, Any]:
        """Apply AJCC rules to an existing malignant prediction and refresh PDF."""
        record = get_db().get_prediction(pred_id)
        if not record:
            raise ValueError("Prediction not found.")
        if str(record.get("prediction", "")).lower() != "malignant":
            raise ValueError("AJCC staging is only available for malignant predictions.")

        params = ClinicalParameters.from_form(clinical)
        staging = self.staging_engine.predict(params).to_dict()

        patient = {
            "patient_id": record.get("patient_id"),
            "name": record.get("patient_name"),
            "age": record.get("age"),
            "sex": record.get("sex"),
        }
        prediction = {
            "prediction": record.get("prediction"),
            "confidence": record.get("confidence"),
            "malignant_prob": record.get("malignant_prob"),
            "benign_prob": record.get("benign_prob"),
            "model_used": record.get("model_used"),
        }
        notes = doctor_notes or record.get("doctor_notes") or ""
        report_id = generate_unique_id("RPT")
        report_path = self.report_gen.generate(
            patient=patient,
            prediction=prediction,
            staging=staging,
            gradcam_path=record.get("gradcam_path"),
            lesion_image_path=record.get("image_path"),
            doctor_notes=notes,
            report_id=report_id,
        )

        get_db().update_prediction(
            pred_id,
            {
                "ajcc_stage": staging.get("stage"),
                "risk_level": staging.get("risk_level"),
                "t_category": staging.get("t_category"),
                "n_category": staging.get("n_category"),
                "m_category": staging.get("m_category"),
                "staging_details": staging.get("details"),
                "staging_recommendation": staging.get("recommendation"),
                "staging_follow_up": staging.get("follow_up"),
                "report_path": str(report_path),
                "doctor_notes": notes,
                "breslow_thickness": float(clinical.get("breslow_thickness_mm") or 0),
                "ulceration": 1
                if str(clinical.get("ulceration", "")).lower()
                in {"yes", "true", "1", "on"}
                else 0,
                "mitotic_rate": float(clinical.get("mitotic_rate") or 0),
                "lymph_node_involvement": 1
                if str(clinical.get("lymph_node_involvement", "")).lower()
                in {"yes", "true", "1", "on"}
                else 0,
                "metastasis": 1
                if str(clinical.get("metastasis", "")).lower()
                in {"yes", "true", "1", "on"}
                else 0,
            },
        )
        return {"staging": staging, "report_path": str(report_path), "report_id": report_id}


pipeline = MelanomaPipeline()
