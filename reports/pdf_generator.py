"""
PDF diagnostic report generator for Melanoma CDSS.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)

import config
from utils.logger import setup_logger

logger = setup_logger("reports")


class DiagnosticReportGenerator:
    """Generate a professional multi-section PDF diagnostic report."""

    def __init__(self, output_dir: Path = config.REPORTS_DIR) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.styles = self._build_styles()

    def _build_styles(self) -> dict:
        base = getSampleStyleSheet()
        styles = {
            "title": ParagraphStyle(
                "ReportTitle",
                parent=base["Heading1"],
                fontSize=15,
                leading=19,
                textColor=colors.HexColor("#0B3D4F"),
                alignment=TA_CENTER,
                spaceAfter=3,
                fontName="Helvetica-Bold",
            ),
            "dept": ParagraphStyle(
                "HospitalDept",
                parent=base["Normal"],
                fontSize=10,
                textColor=colors.HexColor("#4A6670"),
                alignment=TA_CENTER,
                spaceAfter=2,
            ),
            "subtitle": ParagraphStyle(
                "Subtitle",
                parent=base["Normal"],
                fontSize=10,
                textColor=colors.HexColor("#4A6670"),
                alignment=TA_CENTER,
                spaceAfter=12,
            ),
            "heading": ParagraphStyle(
                "SectionHead",
                parent=base["Heading2"],
                fontSize=12,
                textColor=colors.HexColor("#0B3D4F"),
                spaceBefore=14,
                spaceAfter=6,
                fontName="Helvetica-Bold",
            ),
            "heading_first": ParagraphStyle(
                "SectionHeadFirst",
                parent=base["Heading2"],
                fontSize=12,
                textColor=colors.HexColor("#0B3D4F"),
                spaceBefore=8,
                spaceAfter=6,
                fontName="Helvetica-Bold",
            ),
            "body": ParagraphStyle(
                "Body",
                parent=base["Normal"],
                fontSize=10,
                leading=14,
                alignment=TA_JUSTIFY,
            ),
            "disclaimer": ParagraphStyle(
                "Disclaimer",
                parent=base["Normal"],
                fontSize=8,
                textColor=colors.HexColor("#8B1E1E"),
                alignment=TA_CENTER,
                leading=11,
            ),
            "notes": ParagraphStyle(
                "Notes",
                parent=base["Normal"],
                fontSize=10,
                leading=16,
                textColor=colors.HexColor("#333333"),
            ),
            "label": ParagraphStyle(
                "Label",
                parent=base["Normal"],
                fontSize=9,
                textColor=colors.HexColor("#555555"),
            ),
        }
        return styles

    def generate(
        self,
        patient: Dict[str, Any],
        prediction: Dict[str, Any],
        staging: Optional[Dict[str, Any]] = None,
        gradcam_path: Optional[str] = None,
        lesion_image_path: Optional[str] = None,
        doctor_notes: str = "",
        report_id: Optional[str] = None,
        verification: Optional[Dict[str, Any]] = None,
        clinical_review: Optional[Dict[str, Any]] = None,
        include_model_details: bool = True,
        prescription: Optional[Dict[str, Any]] = None,
        clinician: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Build and write the PDF report.

        Returns:
            Path to the generated PDF file.
        """
        report_id = report_id or f"RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        pdf_name = f"{report_id}.pdf" if include_model_details else f"{report_id}-patient.pdf"
        out_path = self.output_dir / pdf_name

        clinician = clinician or {}
        hospital = str(clinician.get("hospital") or "").strip()
        if not hospital or hospital == "—":
            hospital = "Smart Skin Health"

        doc = SimpleDocTemplate(
            str(out_path),
            pagesize=A4,
            leftMargin=18 * mm,
            rightMargin=18 * mm,
            topMargin=16 * mm,
            bottomMargin=16 * mm,
            title=f"{hospital} — Consultation Report — {report_id}",
        )

        story = []
        story.append(Paragraph(self._xml(hospital), self.styles["title"]))
        story.append(Paragraph(
            "Department of Dermatology &amp; Dermato-Oncology",
            self.styles["dept"],
        ))
        story.append(Paragraph(
            "Skin Lesion Consultation Report",
            self.styles["subtitle"],
        ))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0B3D4F")))
        story.append(Spacer(1, 6))

        story.append(Paragraph("Consulting doctor", self.styles["heading_first"]))
        story.append(self._kv_table([
            ["Doctor", str(clinician.get("name") or "—")],
            ["Doctor phone", str(clinician.get("phone") or "—")],
            ["Doctor email", str(clinician.get("email") or "—")],
            ["Education", str(clinician.get("qualification") or "—")],
        ]))
        story.append(Spacer(1, 8))

        story.append(Paragraph(
            "AI-assisted decision support — not a substitute for clinical diagnosis.",
            self.styles["disclaimer"],
        ))
        story.append(Spacer(1, 8))

        story.append(Paragraph("1. Patient Details", self.styles["heading"]))
        patient_rows = [
            ["Report ID", report_id],
            ["Patient ID", str(patient.get("patient_id", "N/A"))],
            ["Patient Name", str(patient.get("name", "N/A"))],
            ["Phone", str(patient.get("phone", "N/A") or "N/A")],
            ["Age", str(patient.get("age", "N/A"))],
            ["Sex", str(patient.get("sex", "N/A"))],
            ["Date / time of analysis", datetime.now().strftime("%Y-%m-%d %H:%M")],
        ]
        story.append(self._kv_table(patient_rows))

        from utils.web import display_model_name, display_prediction, is_melanoma

        pred_label = prediction.get("prediction", "N/A")
        display_label = display_prediction(pred_label)
        melanoma = is_melanoma(pred_label)

        if include_model_details:
            story.append(Paragraph("2. AI-assisted prediction", self.styles["heading"]))
            threshold = prediction.get("decision_threshold")
            if threshold is None:
                from models.ensemble import resolve_decision_threshold

                threshold = resolve_decision_threshold()
            pred_rows = [
                ["AI prediction", display_label],
                ["Melanoma probability", f"{float(prediction.get('malignant_prob', 0))*100:.1f}%"],
                ["Decision threshold", f"{float(threshold):.2f}"],
                ["Model used", display_model_name(prediction.get("model_used"))],
            ]
            story.append(self._kv_table(pred_rows))
            story.append(Paragraph(
                "This result is intended to support clinical assessment and does not replace professional diagnosis.",
                self.styles["body"],
            ))

        # Images
        img_heading = "2. Imaging" if not include_model_details else "3. Imaging"
        story.append(Paragraph(img_heading, self.styles["heading"]))
        if lesion_image_path and Path(lesion_image_path).exists():
            story.extend(self._safe_image(lesion_image_path, "Dermoscopic image"))
        if melanoma and gradcam_path and Path(gradcam_path).exists():
            story.extend(self._safe_image(gradcam_path, "Grad-CAM heatmap"))
            story.append(Paragraph(
                "Grad-CAM highlights image regions that contributed to the model's prediction. "
                "It represents model attention and should not be interpreted as definitive evidence of malignancy.",
                self.styles["body"],
            ))

        # AJCC Staging
        stage_heading = "3. Clinical stage assessment" if not include_model_details else "4. Clinical stage assessment"
        story.append(Paragraph(stage_heading, self.styles["heading"]))
        if include_model_details:
            story.append(Paragraph(
                "Based on the clinical/pathology information entered by the doctor. "
                "The image classifier does not independently determine AJCC stage.",
                self.styles["body"],
            ))
        if staging and (melanoma or not include_model_details) and staging.get("stage"):
            stage_rows = [
                ["AJCC Stage", staging.get("stage", "N/A")],
                ["TNM", f"{staging.get('t_category','')} {staging.get('n_category','')} {staging.get('m_category','')}"],
                ["Risk Level", staging.get("risk_level", "N/A")],
                ["Details", staging.get("details", "")],
            ]
            story.append(self._kv_table(stage_rows))
            story.append(Spacer(1, 6))
            story.append(Paragraph("Clinical recommendation", self.styles["heading"]))
            story.append(Paragraph(staging.get("recommendation", "") or "—", self.styles["body"]))
            story.append(Paragraph("Suggested follow-up", self.styles["heading"]))
            story.append(Paragraph(staging.get("follow_up", "") or "—", self.styles["body"]))
        elif include_model_details and melanoma:
            story.append(Paragraph(
                "Malignant classification confirmed. AJCC staging pending clinical parameters "
                "(Breslow thickness, ulceration, mitotic rate, lymph nodes, metastasis).",
                self.styles["body"],
            ))
            story.append(Paragraph("5. Clinical Recommendation", self.styles["heading"]))
            story.append(Paragraph(
                "Complete AJCC clinical form in the CDSS to generate stage-specific guidance.",
                self.styles["body"],
            ))
            story.append(Paragraph("6. Suggested Follow-up", self.styles["heading"]))
            story.append(Paragraph(
                "Urgent dermatology / oncology review pending formal staging.",
                self.styles["body"],
            ))
        elif include_model_details:
            story.append(Paragraph(
                "AJCC staging not applicable (AI prediction is Benign).",
                self.styles["body"],
            ))
            story.append(Paragraph("5. Clinical Recommendation", self.styles["heading"]))
            story.append(Paragraph(
                "Routine dermatologic monitoring. Reassurance with sun-safety education. "
                "Re-evaluate if lesion changes in size, color, or border irregularity.",
                self.styles["body"],
            ))
            story.append(Paragraph("6. Suggested Follow-up", self.styles["heading"]))
            story.append(Paragraph(
                "Annual skin examination or sooner if clinical concern arises.",
                self.styles["body"],
            ))
        else:
            story.append(Paragraph(
                "Clinical stage was not recorded for this visit.",
                self.styles["body"],
            ))

        # Doctor notes
        story.append(Paragraph("7. Doctor Notes", self.styles["heading"]))
        notes = doctor_notes.strip() if doctor_notes else "_" * 80
        story.append(Paragraph(notes.replace("\n", "<br/>"), self.styles["notes"]))
        story.append(Spacer(1, 20))
        if clinical_review:
            story.append(Paragraph("8. Doctor's assessment", self.styles["heading"]))
            story.append(self._kv_table([
                ["Clinical impression", clinical_review.get("final_diagnosis") or "—"],
                ["Additional observations", clinical_review.get("clinical_observations") or "—"],
                ["Recommendations", clinical_review.get("recommendations") or "—"],
                ["Follow-up notes", clinical_review.get("follow_up_notes") or "—"],
                ["Lesion location", clinical_review.get("lesion_location") or "—"],
            ]))
        if prescription and (prescription.get("items") or prescription.get("notes")):
            story.append(Paragraph("Prescription", self.styles["heading"]))
            story.append(Paragraph(
                "Issued by the treating doctor. The AI does not prescribe medicines.",
                self.styles["body"],
            ))
            if prescription.get("diagnosis"):
                story.append(Paragraph(
                    f"<b>Diagnosis:</b> {prescription.get('diagnosis')}",
                    self.styles["body"],
                ))
            rx_rows = [["Medicine", "Dosage", "Frequency", "Duration", "Instructions"]]
            for item in prescription.get("items") or []:
                rx_rows.append(
                    [
                        item.get("medicine") or "—",
                        item.get("dosage") or "—",
                        item.get("frequency") or "—",
                        item.get("duration") or "—",
                        item.get("instructions") or "—",
                    ]
                )
            if len(rx_rows) > 1:
                table = Table(rx_rows, colWidths=[1.4 * inch, 1.0 * inch, 1.1 * inch, 1.0 * inch, 2.0 * inch])
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8F1F5")),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#0B3D4F")),
                    ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#B0C4CE")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]))
                story.append(Spacer(1, 6))
                story.append(table)
            if prescription.get("notes"):
                story.append(Spacer(1, 6))
                story.append(Paragraph(
                    f"<b>Notes:</b> {prescription.get('notes')}",
                    self.styles["body"],
                ))
        if verification:
            story.append(Paragraph("Doctor verification", self.styles["heading"]))
            story.append(self._kv_table([
                ["Verified by", verification.get("doctor_name") or "—"],
                ["Hospital", verification.get("hospital") or clinician.get("hospital") or "—"],
                ["Doctor phone", verification.get("phone") or clinician.get("phone") or "—"],
                ["Doctor ID", str(verification.get("doctor_id") or "—")],
                ["Date/time", verification.get("verified_at") or "—"],
                ["Status", "DOCTOR VERIFIED"],
            ]))
        story.append(Paragraph(
            "Physician Signature: ________________________ &nbsp;&nbsp;&nbsp; Date: ______________",
            self.styles["body"],
        ))
        story.append(Spacer(1, 16))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
        story.append(Paragraph(
            "AI-assisted decision support — not a substitute for clinical diagnosis.",
            self.styles["disclaimer"],
        ))

        doc.build(story)
        logger.info("PDF report written: %s", out_path)
        return out_path

    @staticmethod
    def _xml(text: str) -> str:
        return (
            str(text or "")
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    def _kv_table(self, rows: list) -> Table:
        data = [[Paragraph(f"<b>{k}</b>", self.styles["label"]),
                 Paragraph(str(v), self.styles["body"])] for k, v in rows]
        table = Table(data, colWidths=[2.2 * inch, 4.3 * inch])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E8F1F5")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#0B3D4F")),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#B0C4CE")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        return table

    def _safe_image(self, path: str, caption: str) -> list:
        elements = [Paragraph(f"<b>{caption}</b>", self.styles["label"]), Spacer(1, 4)]
        try:
            img = Image(path, width=3.2 * inch, height=3.2 * inch, kind="proportional")
            elements.append(img)
            elements.append(Spacer(1, 8))
        except Exception as exc:
            logger.warning("Could not embed image %s: %s", path, exc)
            elements.append(Paragraph(f"[Image unavailable: {path}]", self.styles["body"]))
        return elements
