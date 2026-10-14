"""Small Flask helpers for clinic dashboards."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from flask import url_for

ROOT = Path(__file__).resolve().parent.parent


def to_static_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    static_root = (ROOT / "static").resolve()
    try:
        rel = p.resolve().relative_to(static_root)
        return url_for("static", filename=str(rel).replace("\\", "/"))
    except (ValueError, OSError):
        return None


def display_prediction(label: Optional[str]) -> str:
    """Map stored Benign/Malignant labels to Melanoma / Benign."""
    raw = str(label or "").strip().lower()
    if raw in {"malignant", "melanoma"}:
        return "Melanoma"
    if raw in {"benign", "non-melanoma", "non_melanoma"}:
        return "Benign"
    return str(label or "—")


def is_melanoma(label: Optional[str]) -> bool:
    raw = str(label or "").strip().lower()
    return raw in {"malignant", "melanoma"}


def display_model_name(model_used: Optional[str] = None) -> str:
    """Production scorer is the EfficientNet + MobileNet hybrid."""
    return "Hybrid (EfficientNet + MobileNet)"


def clinician_report_fields(
    doctor: Optional[Dict[str, Any]],
    fallback_name: str = "",
) -> Dict[str, str]:
    """Hospital, name, and contact shown on patient-facing reports."""
    empty = "—"
    if not doctor:
        return {
            "name": str(fallback_name or empty),
            "phone": empty,
            "email": empty,
            "hospital": empty,
            "qualification": empty,
        }
    profile = doctor.get("profile") or {}
    return {
        "name": str(doctor.get("full_name") or fallback_name or empty),
        "phone": str(doctor.get("phone") or empty),
        "email": str(doctor.get("email") or empty),
        "hospital": str(profile.get("hospital") or empty),
        "qualification": str(profile.get("specialization") or empty),
    }
