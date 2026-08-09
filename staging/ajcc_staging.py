"""
AJCC 8th Edition Melanoma Staging — rule-based clinical decision logic.

Uses Breslow thickness, ulceration, mitotic rate, lymph node involvement,
and distant metastasis to assign pathologic/clinical stage groups.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Union

import config
from utils.logger import setup_logger

logger = setup_logger("staging")


@dataclass
class ClinicalParameters:
    """Patient clinical parameters for AJCC staging."""

    breslow_thickness_mm: float
    ulceration: bool
    mitotic_rate: float  # mitoses per mm²
    lymph_node_involvement: bool
    metastasis: bool
    num_positive_nodes: int = 0

    @classmethod
    def from_form(cls, data: Dict[str, Any]) -> "ClinicalParameters":
        """Parse form / JSON payload into ClinicalParameters."""

        def _bool(val: Any) -> bool:
            if isinstance(val, bool):
                return val
            if val is None:
                return False
            return str(val).strip().lower() in {"yes", "y", "true", "1", "on"}

        thickness = float(data.get("breslow_thickness_mm") or data.get("breslow") or 0)
        mitotic = float(data.get("mitotic_rate") or data.get("mitotic") or 0)
        nodes = int(data.get("num_positive_nodes") or 0)
        return cls(
            breslow_thickness_mm=thickness,
            ulceration=_bool(data.get("ulceration")),
            mitotic_rate=mitotic,
            lymph_node_involvement=_bool(data.get("lymph_node_involvement") or data.get("lymph_nodes")),
            metastasis=_bool(data.get("metastasis")),
            num_positive_nodes=nodes,
        )


@dataclass
class StagingResult:
    """AJCC staging output with clinical guidance."""

    stage: str
    t_category: str
    n_category: str
    m_category: str
    risk_level: str
    recommendation: str
    follow_up: str
    details: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AJCCStagingEngine:
    """
    Rule-based AJCC 8th Edition melanoma stage predictor.

    Simplified TNM grouping suitable for a CDSS educational / decision-support
    prototype. Clinical use must be validated by a qualified physician.
    """

    # Follow-up schedules (guideline-inspired, educational)
    FOLLOW_UP: Dict[str, str] = {
        "Stage 0": "Clinical exam every 6–12 months for 5 years; then annually. Sun protection counseling.",
        "Stage IA": "Clinical exam every 6–12 months for 5 years; then annually. Consider imaging if symptomatic.",
        "Stage IB": "Clinical exam every 3–6 months for 2 years, then every 3–12 months through year 5; then annually.",
        "Stage IIA": "Clinical exam every 3–6 months for 2 years, then every 3–12 months through year 5; imaging as indicated.",
        "Stage IIB": "Clinical exam every 3–6 months for 2 years, then every 3–12 months through year 5; consider surveillance imaging.",
        "Stage IIC": "Clinical exam every 3–6 months for 2 years; multidisciplinary review; adjuvant therapy evaluation.",
        "Stage III": "Multidisciplinary oncology care; exams every 3–6 months; nodal basin ultrasound / imaging per protocol.",
        "Stage IV": "Systemic therapy evaluation (immunotherapy / targeted therapy); palliative support as needed; frequent imaging.",
    }

    RECOMMENDATIONS: Dict[str, str] = {
        "Stage 0": "Wide local excision with appropriate margins. Excellent prognosis.",
        "Stage IA": "Wide local excision. Sentinel lymph node biopsy (SLNB) generally not indicated for very thin lesions without adverse features.",
        "Stage IB": "Wide local excision. Discuss SLNB. Dermatologic follow-up mandatory.",
        "Stage IIA": "Wide local excision + consider SLNB. Discuss adjuvant options if high-risk features.",
        "Stage IIB": "Wide local excision + SLNB. Strongly consider adjuvant systemic therapy referral.",
        "Stage IIC": "Wide local excision + SLNB. Referral to medical oncology for adjuvant immunotherapy consideration.",
        "Stage III": "Complete lymphadenectomy or nodal management per guidelines. Systemic adjuvant therapy strongly recommended.",
        "Stage IV": "Urgent oncology referral. Systemic therapy (PD-1 inhibitors, BRAF/MEK if BRAF-mutant). Consider clinical trials.",
    }

    def predict(self, params: ClinicalParameters) -> StagingResult:
        """
        Predict AJCC stage from clinical parameters.

        Priority rules:
        1. Distant metastasis → Stage IV
        2. Regional lymph node involvement → Stage III
        3. Otherwise → Stage 0 / I / II based on T category
        """
        logger.info(
            "Staging: Breslow=%.2fmm ulceration=%s mitotic=%.1f nodes=%s mets=%s",
            params.breslow_thickness_mm,
            params.ulceration,
            params.mitotic_rate,
            params.lymph_node_involvement,
            params.metastasis,
        )

        t_cat = self._t_category(params)
        n_cat = self._n_category(params)
        m_cat = "M1" if params.metastasis else "M0"

        if params.metastasis:
            stage = "Stage IV"
        elif params.lymph_node_involvement or params.num_positive_nodes > 0:
            stage = "Stage III"
        else:
            stage = self._stage_from_t(t_cat, params)

        risk = config.RISK_LEVELS.get(stage, "Unknown")
        result = StagingResult(
            stage=stage,
            t_category=t_cat,
            n_category=n_cat,
            m_category=m_cat,
            risk_level=risk,
            recommendation=self.RECOMMENDATIONS.get(stage, "Consult oncology."),
            follow_up=self.FOLLOW_UP.get(stage, "Follow institutional melanoma protocol."),
            details=self._describe(params, t_cat, n_cat, m_cat, stage),
        )
        logger.info("Predicted stage: %s (risk=%s)", stage, risk)
        return result

    def _t_category(self, p: ClinicalParameters) -> str:
        """
        AJCC 8th Edition T category from Breslow thickness and ulceration.

        Tis: melanoma in situ (Breslow ~ 0 / not invasive)
        T1a: ≤1.0 mm, without ulceration, and mitotic rate < 1/mm² (simplified)
        T1b: ≤1.0 mm with ulceration OR mitotic rate ≥ 1/mm² OR 0.8–1.0 mm
        T2a/b: >1.0–2.0 mm ± ulceration
        T3a/b: >2.0–4.0 mm ± ulceration
        T4a/b: >4.0 mm ± ulceration
        """
        t = p.breslow_thickness_mm
        u = p.ulceration
        mito = p.mitotic_rate

        if t <= 0:
            return "Tis"

        if t <= 1.0:
            # AJCC 8: T1a = <0.8 mm without ulceration; T1b = 0.8–1.0 or ulcerated ≤1.0
            if t < 0.8 and not u and mito < 1.0:
                return "T1a"
            return "T1b"

        if t <= 2.0:
            return "T2b" if u else "T2a"
        if t <= 4.0:
            return "T3b" if u else "T3a"
        return "T4b" if u else "T4a"

    def _n_category(self, p: ClinicalParameters) -> str:
        if not p.lymph_node_involvement and p.num_positive_nodes <= 0:
            return "N0"
        n = p.num_positive_nodes
        if n <= 1:
            return "N1"
        if n <= 3:
            return "N2"
        return "N3"

    def _stage_from_t(self, t_cat: str, p: ClinicalParameters) -> str:
        """Map T category (N0 M0) to anatomic stage group."""
        mapping = {
            "Tis": "Stage 0",
            "T1a": "Stage IA",
            "T1b": "Stage IB",
            "T2a": "Stage IB",
            "T2b": "Stage IIA",
            "T3a": "Stage IIA",
            "T3b": "Stage IIB",
            "T4a": "Stage IIB",
            "T4b": "Stage IIC",
        }
        return mapping.get(t_cat, "Stage IB")

    @staticmethod
    def _describe(
        p: ClinicalParameters,
        t: str,
        n: str,
        m: str,
        stage: str,
    ) -> str:
        return (
            f"TNM: {t} {n} {m} → {stage}. "
            f"Breslow {p.breslow_thickness_mm:.2f} mm; "
            f"Ulceration={'Yes' if p.ulceration else 'No'}; "
            f"Mitotic rate={p.mitotic_rate:.1f}/mm²; "
            f"Nodal involvement={'Yes' if p.lymph_node_involvement else 'No'}; "
            f"Distant metastasis={'Yes' if p.metastasis else 'No'}."
        )


def predict_ajcc_stage(
    breslow_thickness_mm: float,
    ulceration: Union[bool, str],
    mitotic_rate: float,
    lymph_node_involvement: Union[bool, str],
    metastasis: Union[bool, str],
    num_positive_nodes: int = 0,
) -> Dict[str, Any]:
    """Convenience function returning staging result as a dictionary."""
    params = ClinicalParameters.from_form(
        {
            "breslow_thickness_mm": breslow_thickness_mm,
            "ulceration": ulceration,
            "mitotic_rate": mitotic_rate,
            "lymph_node_involvement": lymph_node_involvement,
            "metastasis": metastasis,
            "num_positive_nodes": num_positive_nodes,
        }
    )
    return AJCCStagingEngine().predict(params).to_dict()
